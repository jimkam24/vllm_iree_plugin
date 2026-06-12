"""
Path A1 Benchmark
==================
Compares pure vLLM PP vs native+IREEWorker (SDPA attention) PP.

Config 1: Pure vLLM PP — both ranks native gpu_worker (measured via LLM API)
Config 4: Path A1 — rank 0 native (Triton), rank 1 IREEWorker (SDPA)

Metrics: TTFT latency, memory, throughput at multiple sequence lengths.

Usage:
    cd /vllm_iree/vllm_test
    PP_BENCH_CONFIG=1 python3 benchmark_path_a1.py   # pure vLLM baseline
    PP_BENCH_CONFIG=4 python3 benchmark_path_a1.py   # Path A1
"""

import os
import sys
import time
import json
import statistics

CONFIG = os.environ.get("PP_BENCH_CONFIG", "4")

print(f"\n{'='*60}")
print(f"Path A1 Benchmark — Config {CONFIG}")
print(f"{'='*60}")

# ── Common settings ───────────────────────────────────────────────────────────
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["VLLM_PP_LAYER_PARTITION"] = "14,2"
os.environ["IREE_GPU_ASSIGNMENT"] = "1,0"   # rank0→V100S, rank1→A2

# Prompt lengths to test
PROMPT_LENGTHS = [5, 13, 50, 100]
N_WARMUP = 3
N_MEASURE = 10

PROMPTS = {
    5:   "The capital of France is",           # exactly 5 tokens
    13:  "The capital of France is Paris and the capital of Germany is",
    50:  "The capital of France is Paris. " * 5,
    100: "The capital of France is Paris. " * 10,
}

if CONFIG == "1":
    # Pure vLLM PP via LLM API
    os.environ["VLLM_PLUGINS"] = ""
    config_name = "Pure vLLM PP (native + native)"
elif CONFIG == "4":
    # Path A1: native rank 0 + IREEWorker rank 1 with SDPA
    os.environ["VLLM_PLUGINS"] = "iree"
    os.environ["IREE_WORKER_RANKS"] = "1"
    os.environ["IREE_USE_VLLM_MODEL"] = "1"
    config_name = "Path A1 (native Triton + IREEWorker SDPA)"
else:
    print(f"Unknown config {CONFIG}. Use PP_BENCH_CONFIG=1 or 4.")
    sys.exit(1)

print(f"Config: {config_name}")

# ── Config 1: Pure vLLM PP via LLM API ───────────────────────────────────────
if CONFIG == "1":
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    llm = LLM(
        model="meta-llama/Llama-3.2-1B",
        dtype="float32",
        max_model_len=512,
        enforce_eager=True,
        pipeline_parallel_size=2,
        tensor_parallel_size=1,
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(max_tokens=1)

    results = {}
    for prompt_len in PROMPT_LENGTHS:
        # Build prompt of exact token length
        token_ids = list(range(100, 100 + prompt_len))
        prompt = tokenizer.decode(token_ids)

        # Warmup
        for _ in range(N_WARMUP):
            llm.generate([prompt], sampling_params)

        # Measure
        times = []
        for _ in range(N_MEASURE):
            t0 = time.perf_counter()
            out = llm.generate([prompt], sampling_params)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        token = out[0].outputs[0].token_ids[0]
        mean_ms = statistics.mean(times)
        p50_ms = statistics.median(times)
        tput = N_MEASURE / sum(t / 1000 for t in times)

        results[prompt_len] = {
            "mean_ms": round(mean_ms, 3),
            "p50_ms": round(p50_ms, 3),
            "throughput": round(tput, 3),
            "token": token,
        }
        print(f"\n  Prompt length {prompt_len:4d} tokens: "
              f"mean={mean_ms:.2f}ms p50={p50_ms:.2f}ms "
              f"tput={tput:.2f}tok/s token={token}")

# ── Config 4: Path A1 via HybridExecutor ─────────────────────────────────────
elif CONFIG == "4":
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.core.sched.output import SchedulerOutput, CachedRequestData, NewRequestData
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    from vllm_plugin.executor.hybrid_executor import HybridExecutor

    print("\n[1] Building VllmConfig...")
    engine_args = EngineArgs(
        model="meta-llama/Llama-3.2-1B",
        dtype="float32",
        max_model_len=512,
        max_num_seqs=2,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        distributed_executor_backend=(
            "vllm_plugin.executor.hybrid_executor.HybridExecutor"
        ),
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.parallel_config.pipeline_parallel_size = 2
    vllm_config.parallel_config.world_size = 2

    print("[2] Instantiating HybridExecutor...")
    executor = HybridExecutor(vllm_config)

    print("[3] Initializing KV cache...")
    kv_cache_specs = executor.get_kv_cache_specs()
    available_gpu_memory = executor.determine_available_memory()
    kv_cache_configs = get_kv_cache_configs(
        vllm_config, kv_cache_specs, available_gpu_memory
    )
    executor.initialize_from_config(kv_cache_configs)

    def blocks_for_len(n_tokens, block_size=16):
        """Return block_ids tuple covering n_tokens with given block_size."""
        n_blocks = (n_tokens + block_size - 1) // block_size
        # Use blocks starting from index 0
        return (list(range(n_blocks)),)

    def make_scheduler_output(req_id, token_ids):
        new_req = NewRequestData(
            req_id=req_id,
            prompt_token_ids=token_ids,
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_ids=blocks_for_len(len(token_ids)),
            num_computed_tokens=0,
            lora_request=None,
            mm_features=[],
            prompt_embeds=None,
        )
        return SchedulerOutput(
            scheduled_new_reqs=[new_req],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={req_id: len(token_ids)},
            total_num_scheduled_tokens=len(token_ids),
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            preempted_req_ids=None,
            has_structured_output_requests=False,
            pending_structured_output_tokens=False,
            num_invalid_spec_tokens=None,
            kv_connector_metadata=None,
            ec_connector_metadata=None,
        )

    def measure_memory_mb():
        def get_mem(worker):
            import torch
            return torch.cuda.memory_allocated(0) / 1024 / 1024
        results = executor.collective_rpc(get_mem)
        return {"rank0_mb": results[0], "rank1_mb": results[1]}

    results = {}
    for prompt_len in PROMPT_LENGTHS:
        token_ids = list(range(100, 100 + prompt_len))

        # Warmup
        for i in range(N_WARMUP):
            sched = make_scheduler_output(f"warmup-{prompt_len}-{i}", token_ids)
            _ = executor.execute_model(sched)

        # Measure
        times = []
        last_token = None
        for i in range(N_MEASURE):
            sched = make_scheduler_output(f"bench-{prompt_len}-{i}", token_ids)
            t0 = time.perf_counter()
            output = executor.execute_model(sched)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            if output is not None and output.sampled_token_ids:
                last_token = output.sampled_token_ids[0][0]

        mean_ms = statistics.mean(times)
        p50_ms = statistics.median(times)
        tput = N_MEASURE / sum(t / 1000 for t in times)

        results[prompt_len] = {
            "mean_ms": round(mean_ms, 3),
            "p50_ms": round(p50_ms, 3),
            "throughput": round(tput, 3),
            "token": last_token,
        }
        print(f"\n  Prompt length {prompt_len:4d} tokens: "
              f"mean={mean_ms:.2f}ms p50={p50_ms:.2f}ms "
              f"tput={tput:.2f}tok/s token={last_token}")

    mem = measure_memory_mb()
    print(f"\n  Memory: rank0={mem['rank0_mb']:.1f}MB rank1={mem['rank1_mb']:.1f}MB "
          f"total={mem['rank0_mb']+mem['rank1_mb']:.1f}MB")
    executor.shutdown()

# ── Save results ──────────────────────────────────────────────────────────────
out_path = f"/tmp/benchmark_path_a1_config{CONFIG}.json"
with open(out_path, "w") as f:
    json.dump({
        "config": CONFIG,
        "config_name": config_name,
        "results_by_prompt_len": {str(k): v for k, v in results.items()},
    }, f, indent=2)

print(f"\n{'='*60}")
print(f"BENCHMARK COMPLETE — {config_name}")
print(f"{'='*60}")
print(f"\n{'Prompt':>10} {'Mean ms':>10} {'P50 ms':>10} {'Tok/s':>10} {'Token':>8}")
print("-"*52)
for plen, r in results.items():
    correct = "✅" if r["token"] == 12366 else "❌"
    print(f"{plen:>10} {r['mean_ms']:>10.2f} {r['p50_ms']:>10.2f} "
          f"{r['throughput']:>10.2f} {r['token']:>6} {correct}")
print(f"\nResults saved to {out_path}")