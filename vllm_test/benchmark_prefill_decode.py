"""
Prefill + Decode Benchmark
===========================
Measures TTFT (prefill) and per-token decode latency for three configurations:

  Config 1: vLLM-vLLM   — both ranks native gpu_worker (via LLM API)
  Config 2: vLLM-custom  — rank 0 native, rank 1 IREEWorker+SDPA (Path A1)
  Config 3: custom-custom — both ranks IREEWorker+HF (Path B, IREE-IREE)

Usage:
    cd /vllm_iree/vllm_test
    PP_BENCH_CONFIG=1 python3 benchmark_prefill_decode.py
    PP_BENCH_CONFIG=2 python3 benchmark_prefill_decode.py
    PP_BENCH_CONFIG=3 python3 benchmark_prefill_decode.py
"""

import os, sys, time, json, statistics

CONFIG = os.environ.get("PP_BENCH_CONFIG", "2")

print(f"\n{'='*65}")
print(f"Prefill + Decode Benchmark — Config {CONFIG}")
print(f"{'='*65}")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["VLLM_PP_LAYER_PARTITION"] = "14,2"
os.environ["IREE_GPU_ASSIGNMENT"] = "0,1"

PROMPT = "The capital of France is"
N_WARMUP = 3
N_MEASURE = 10
N_DECODE = 10       # decode steps per measurement
PROMPT_LENS = [5, 50, 100]

if CONFIG == "1":
    os.environ["VLLM_PLUGINS"] = ""
    config_name = "vLLM-vLLM (native + native)"
elif CONFIG == "2":
    os.environ["VLLM_PLUGINS"] = "iree"
    os.environ["IREE_WORKER_RANKS"] = "1"
    os.environ["IREE_USE_VLLM_MODEL"] = "1"
    os.environ["IREE_CUDA_ARCH"] = "sm_70"
    config_name = "vLLM-custom (native Triton + IREEWorker SDPA)"
elif CONFIG == "3":
    os.environ["VLLM_PLUGINS"] = "iree"
    os.environ["IREE_WORKER_RANKS"] = "0,1"   # both ranks IREEWorker
    os.environ["IREE_USE_VLLM_MODEL"] = "1"   # both use vLLM model + SDPA
    os.environ["IREE_CUDA_ARCH"] = "sm_70"
    config_name = "custom-custom (IREEWorker SDPA + IREEWorker SDPA)"
elif CONFIG == "1r":
    os.environ["VLLM_PLUGINS"] = "iree"
    os.environ["IREE_WORKER_RANKS"] = "99"   # no IREE workers — both native
    os.environ["IREE_USE_VLLM_MODEL"] = "1"  # native workers still use vLLM model
    config_name = "vLLM-vLLM (HybridExecutor, Ray)"
    out_path = "/tmp/bench_pd_config1r.json"
elif CONFIG == "4":
    os.environ["VLLM_PLUGINS"] = "iree"
    os.environ["IREE_WORKER_RANKS"] = "0"
    os.environ["IREE_USE_VLLM_MODEL"] = "1"
    os.environ["IREE_CUDA_ARCH"] = "sm_70"
    config_name = "vLLM-custom ( IREEWorker SDPA + native Triton)"
elif CONFIG == "5":  # Config 2 + CPU FFN
    os.environ["VLLM_PLUGINS"] = "iree"
    os.environ["IREE_WORKER_RANKS"] = "1"
    os.environ["IREE_USE_VLLM_MODEL"] = "1"
    os.environ["IREE_USE_CPU_FFN"] = "1"
    os.environ["IREE_USE_FFN"] = "1"
    os.environ["IREE_CUDA_ARCH"] = "sm_86"
    config_name = "vLLM-custom (native Triton + IREEWorker CPU IREE FFN)"
elif CONFIG == "6":  # Config 2 + CPU FFN
    os.environ["VLLM_PLUGINS"] = "iree"
    os.environ["IREE_WORKER_RANKS"] = "1"
    os.environ["IREE_USE_VLLM_MODEL"] = "1"
    os.environ["IREE_USE_CPU_FFN"] = "1"
    os.environ["IREE_CUDA_ARCH"] = "sm_86"
    config_name = "vLLM-custom (native Triton + IREEWorker CPU FFN)"
elif CONFIG == "7":  # Config 2 + CPU FFN
    os.environ["VLLM_PLUGINS"] = "iree"
    os.environ["IREE_WORKER_RANKS"] = "1"
    os.environ["IREE_USE_VLLM_MODEL"] = "1"
    os.environ["IREE_USE_FFN"] = "1"
    os.environ["IREE_CUDA_ARCH"] = "sm_86"
    config_name = "vLLM-custom (native Triton + IREEWorker GPU FFN)"
else:
    print(f"Unknown config {CONFIG}. Use PP_BENCH_CONFIG=1, 2, or 3.")
    sys.exit(1)

print(f"Config: {config_name}\n")

# ── Config 1: Pure vLLM via LLM API ──────────────────────────────────────────
if CONFIG == "1":
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="meta-llama/Llama-3.2-1B",
        dtype="float32",
        max_model_len=512,
        enforce_eager=True,
        pipeline_parallel_size=2,
        disable_log_stats=True,
    )

    results = {}
    for plen in PROMPT_LENS:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        base = tok.encode(PROMPT)
        tokens = (base * (plen // len(base) + 1))[:plen]
        prompt_text = tok.decode(tokens)

        # Warmup
        for _ in range(N_WARMUP):
            llm.generate([prompt_text], SamplingParams(max_tokens=1))

        # Prefill (max_tokens=1)
        prefill_times = []
        for _ in range(N_MEASURE):
            t0 = time.perf_counter()
            out = llm.generate([prompt_text], SamplingParams(max_tokens=1))
            t1 = time.perf_counter()
            prefill_times.append((t1 - t0) * 1000)

        # Decode (max_tokens=N_DECODE, measure per-token)
        decode_times = []
        for _ in range(N_MEASURE):
            t0 = time.perf_counter()
            out = llm.generate([prompt_text], SamplingParams(max_tokens=N_DECODE))
            t1 = time.perf_counter()
            decode_times.append((t1 - t0) * 1000 / N_DECODE)

        results[plen] = {
            "prefill_mean": round(statistics.mean(prefill_times), 2),
            "prefill_p50": round(statistics.median(prefill_times), 2),
            "decode_per_token_mean": round(statistics.mean(decode_times), 2),
            "decode_per_token_p50": round(statistics.median(decode_times), 2),
        }
        print(f"  Prompt {plen:4d} tokens | "
              f"TTFT: {results[plen]['prefill_mean']:.2f}ms | "
              f"Decode/tok: {results[plen]['decode_per_token_mean']:.2f}ms")

# ── Configs 2 & 3: HybridExecutor ────────────────────────────────────────────
else:
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm_plugin.executor.hybrid_executor import HybridExecutor
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    from vllm.v1.core.sched.output import (
        SchedulerOutput, CachedRequestData, NewRequestData
    )
    from vllm.sampling_params import SamplingParams
    from transformers import AutoTokenizer

    engine_args = EngineArgs(
        model="meta-llama/Llama-3.2-1B",
        dtype="float32",
        max_model_len=512,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        distributed_executor_backend=(
            "vllm_plugin.executor.hybrid_executor.HybridExecutor"
        ),
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.parallel_config.pipeline_parallel_size = 2
    vllm_config.parallel_config.world_size = 2

    executor = HybridExecutor(vllm_config)
    kv_specs = executor.get_kv_cache_specs()
    avail = executor.determine_available_memory()
    kv_configs = get_kv_cache_configs(vllm_config, kv_specs, avail)
    executor.initialize_from_config(kv_configs)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    block_size = 16
    max_blocks = (512 + block_size - 1) // block_size

    def make_prefill_sched(req_id, token_ids, block_ids):
        req = NewRequestData(
            req_id=req_id,
            prompt_token_ids=token_ids,
            sampling_params=SamplingParams(max_tokens=N_DECODE),
            pooling_params=None,
            block_ids=(block_ids,),
            num_computed_tokens=0,
            lora_request=None,
            mm_features=[],
            prompt_embeds=None,
        )
        return SchedulerOutput(
            scheduled_new_reqs=[req],
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

    def make_decode_sched(req_id, n_computed, block_ids, last_token, all_tokens):
        cached = CachedRequestData(
            req_ids=[req_id],
            resumed_req_ids=set(),
            new_token_ids=[[last_token]],
            all_token_ids={req_id: len(all_tokens)},
            new_block_ids=[([],)],
            num_computed_tokens=[n_computed],
            num_output_tokens=[n_computed - len(all_tokens) + 1
                               if n_computed > len(all_tokens) - 1 else 1],
        )
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached,
            num_scheduled_tokens={req_id: 1},
            total_num_scheduled_tokens=1,
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

    results = {}
    req_counter = 0

    for plen in PROMPT_LENS:
        base = tokenizer.encode(PROMPT)
        tokens = (base * (plen // len(base) + 1))[:plen]
        block_ids = list(range(max_blocks))

        # Warmup prefill
        for i in range(N_WARMUP):
            rid = f"warmup-{plen}-{i}"
            executor.execute_model(make_prefill_sched(rid, tokens, block_ids))

        # Measure prefill
        prefill_times = []
        for i in range(N_MEASURE):
            rid = f"pfill-{plen}-{i}-{req_counter}"
            req_counter += 1
            t0 = time.perf_counter()
            out = executor.execute_model(
                make_prefill_sched(rid, tokens, block_ids)
            )
            t1 = time.perf_counter()
            prefill_times.append((t1 - t0) * 1000)

        # Measure decode: run prefill then N_DECODE decode steps
        decode_step_times = []
        for i in range(N_MEASURE):
            rid = f"dec-{plen}-{i}-{req_counter}"
            req_counter += 1
            # Prefill
            out = executor.execute_model(
                make_prefill_sched(rid, tokens, block_ids)
            )
            last_token = out.sampled_token_ids[0][0] if out else 0
            all_tokens = list(tokens) + [last_token]
            n_computed = len(tokens)
            step_times = []
            for step in range(N_DECODE):
                t0 = time.perf_counter()
                out = executor.execute_model(
                    make_decode_sched(rid, n_computed, block_ids,
                                      last_token, all_tokens)
                )
                t1 = time.perf_counter()
                step_times.append((t1 - t0) * 1000)
                if out and out.sampled_token_ids:
                    last_token = out.sampled_token_ids[0][0]
                    all_tokens.append(last_token)
                n_computed += 1
            decode_step_times.append(statistics.mean(step_times))

        results[plen] = {
            "prefill_mean": round(statistics.mean(prefill_times), 2),
            "prefill_p50": round(statistics.median(prefill_times), 2),
            "decode_per_token_mean": round(statistics.mean(decode_step_times), 2),
            "decode_per_token_p50": round(statistics.median(decode_step_times), 2),
        }
        print(f"  Prompt {plen:4d} tokens | "
              f"TTFT: {results[plen]['prefill_mean']:.2f}ms | "
              f"Decode/tok: {results[plen]['decode_per_token_mean']:.2f}ms")

    def get_mem(worker):
        import torch
        return torch.cuda.memory_allocated(0) / 1024 / 1024
    mem = executor.collective_rpc(get_mem)
    print(f"\n  Memory: rank0={mem[0]:.0f}MB rank1={mem[1]:.0f}MB "
          f"total={mem[0]+mem[1]:.0f}MB")
    executor.shutdown()

# ── Save + print summary ──────────────────────────────────────────────────────
out_path = f"/tmp/bench_pd_config{CONFIG}.json"
with open(out_path, "w") as f:
    json.dump({
        "config": CONFIG,
        "config_name": config_name,
        "n_decode_steps": N_DECODE,
        "results": {str(k): v for k, v in results.items()},
    }, f, indent=2)

print(f"\n{'='*65}")
print(f"BENCHMARK COMPLETE — {config_name}")
print(f"{'='*65}")
print(f"\n{'Prompt':>8} {'TTFT mean':>12} {'TTFT p50':>10} "
      f"{'Decode/tok':>12} {'Decode p50':>12}")
print("-"*58)
for plen, r in results.items():
    print(f"{plen:>8} {r['prefill_mean']:>12.2f} {r['prefill_p50']:>10.2f} "
          f"{r['decode_per_token_mean']:>12.2f} "
          f"{r['decode_per_token_p50']:>12.2f}")
print(f"\nResults saved to {out_path}")