"""
Batch Size Benchmark
=====================
Measures TTFT and decode/tok across batch sizes for all configurations.

Batch sizes: 1, 4, 16, 64
Prompt length: fixed at 50 tokens
Decode steps: 10

Configs:
  1r: vLLM-vLLM via HybridExecutor (Ray baseline)
  2:  vLLM-custom (Triton rank0 + SDPA rank1)
  3:  custom-custom (SDPA rank0 + SDPA rank1)
  4:  custom-vLLM (SDPA rank0 + Triton rank1)

Usage:
    cd /vllm_iree/vllm_test
    PP_BENCH_CONFIG=1r python3 benchmark_batch.py
"""

import os, sys, time, json, statistics

CONFIG = os.environ.get("PP_BENCH_CONFIG", "2")

print(f"\n{'='*65}")
print(f"Batch Size Benchmark — Config {CONFIG}")
print(f"{'='*65}")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["VLLM_PP_LAYER_PARTITION"] = "14,2"
os.environ["IREE_GPU_ASSIGNMENT"] = "1,0"
os.environ["IREE_CUDA_ARCH"] = "sm_70"

PROMPT_LEN = 50
BATCH_SIZES = [1, 4, 16, 64]
N_WARMUP = 2
N_MEASURE = 5
N_DECODE = 10

if CONFIG == "1r":
    os.environ["VLLM_PLUGINS"] = "iree"
    os.environ["IREE_WORKER_RANKS"] = "99"
    os.environ["IREE_USE_VLLM_MODEL"] = "1"
    config_name = "vLLM-vLLM (HybridExecutor, Ray baseline)"
elif CONFIG == "2":
    os.environ["VLLM_PLUGINS"] = "iree"
    os.environ["IREE_WORKER_RANKS"] = "1"
    os.environ["IREE_USE_VLLM_MODEL"] = "1"
    config_name = "vLLM-custom (Triton rank0 + SDPA rank1)"
elif CONFIG == "3":
    os.environ["VLLM_PLUGINS"] = "iree"
    os.environ["IREE_WORKER_RANKS"] = "0,1"
    os.environ["IREE_USE_VLLM_MODEL"] = "1"
    config_name = "custom-custom (SDPA rank0 + SDPA rank1)"
elif CONFIG == "4":
    os.environ["VLLM_PLUGINS"] = "iree"
    os.environ["IREE_WORKER_RANKS"] = "0"
    os.environ["IREE_USE_VLLM_MODEL"] = "1"
    config_name = "custom-vLLM (SDPA rank0 + Triton rank1)"
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
    print(f"Unknown config {CONFIG}. Use 1r, 2, 3, or 4.")
    sys.exit(1)

print(f"Config: {config_name}")
print(f"Prompt length: {PROMPT_LEN} tokens, Decode steps: {N_DECODE}\n")

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
    max_num_seqs=128,
    max_num_batched_tokens=8192, 
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
max_blocks = (512 + block_size - 1) // block_size  # 32 blocks per req

base_tokens = tokenizer.encode("The capital of France is")
prompt_tokens = (base_tokens * (PROMPT_LEN // len(base_tokens) + 1))[:PROMPT_LEN]

req_counter = 0

def make_sched(base_req_id, batch_size):
    new_reqs = []
    num_tokens = {}
    for b in range(batch_size):
        rid = f"{base_req_id}-b{b}"
        block_start = b * max_blocks
        block_ids = list(range(block_start, block_start + max_blocks))
        req = NewRequestData(
            req_id=rid,
            prompt_token_ids=prompt_tokens,
            sampling_params=SamplingParams(max_tokens=N_DECODE),
            pooling_params=None,
            block_ids=(block_ids,),
            num_computed_tokens=0,
            lora_request=None,
            mm_features=[],
            prompt_embeds=None,
        )
        new_reqs.append(req)
        num_tokens[rid] = len(prompt_tokens)
    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_tokens,
        total_num_scheduled_tokens=len(prompt_tokens) * batch_size,
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

def make_decode_sched(base_req_id, batch_size, n_computed, last_tokens, all_tokens_lens):
    req_ids = [f"{base_req_id}-b{b}" for b in range(batch_size)]
    cached = CachedRequestData(
        req_ids=req_ids,
        resumed_req_ids=set(),
        new_token_ids=[[last_tokens[b]] for b in range(batch_size)],
        all_token_ids={f"{base_req_id}-b{b}": all_tokens_lens[b]
                       for b in range(batch_size)},
        new_block_ids=[([],) for _ in range(batch_size)],
        num_computed_tokens=[n_computed] * batch_size,
        num_output_tokens=[max(1, n_computed - PROMPT_LEN + 1)] * batch_size,
    )
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={f"{base_req_id}-b{b}": 1
                               for b in range(batch_size)},
        total_num_scheduled_tokens=batch_size,
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

for batch_size in BATCH_SIZES:
    print(f"  Batch {batch_size:3d}...", end="", flush=True)

    # Warmup
    for i in range(N_WARMUP):
        rid = f"w-bs{batch_size}-{i}-{req_counter}"; req_counter += 1
        try: executor.execute_model(make_sched(rid, batch_size))
        except Exception as e: print(f"\n  Warmup failed: {e}"); break

    # Prefill
    prefill_times = []
    for i in range(N_MEASURE):
        rid = f"p-bs{batch_size}-{i}-{req_counter}"; req_counter += 1
        t0 = time.perf_counter()
        try:
            out = executor.execute_model(make_sched(rid, batch_size))
            prefill_times.append((time.perf_counter() - t0) * 1000)
        except Exception as e:
            print(f"\n  Prefill failed: {e}"); break

    if not prefill_times:
        print(" FAILED"); continue

    # Decode
    decode_times = []
    for i in range(N_MEASURE):
        rid = f"d-bs{batch_size}-{i}-{req_counter}"; req_counter += 1
        out = executor.execute_model(make_sched(rid, batch_size))
        last_tok = out.sampled_token_ids[0][0] if out and out.sampled_token_ids else 0
        last_tokens = [last_tok] * batch_size
        all_lens = [PROMPT_LEN + 1] * batch_size
        n_computed = PROMPT_LEN
        step_t = []
        for _ in range(N_DECODE):
            t0 = time.perf_counter()
            try:
                out = executor.execute_model(
                    make_decode_sched(rid, batch_size, n_computed, last_tokens, all_lens))
                step_t.append((time.perf_counter() - t0) * 1000)
                if out and out.sampled_token_ids:
                    last_tokens = [out.sampled_token_ids[0][0]] * batch_size
                for b in range(batch_size): all_lens[b] += 1
                n_computed += 1
            except Exception as e:
                print(f"\n  Decode failed: {e}"); break
        if step_t: decode_times.append(statistics.mean(step_t))

    mp = statistics.mean(prefill_times)
    md = statistics.mean(decode_times) if decode_times else 0
    pt = (batch_size * PROMPT_LEN) / (mp / 1000)
    dt = batch_size / (md / 1000) if md > 0 else 0
    results[batch_size] = {
        "prefill_ms": round(mp, 2), "decode_ms": round(md, 2),
        "prefill_tok_s": round(pt, 1), "decode_req_s": round(dt, 1),
    }
    print(f" TTFT={mp:.1f}ms  Dec={md:.1f}ms/tok  "
          f"PrefTput={pt:.0f}tok/s  DecTput={dt:.1f}req/s")

out_path = f"/tmp/bench_batch_config{CONFIG}.json"
with open(out_path, "w") as f:
    json.dump({"config": CONFIG, "config_name": config_name,
               "prompt_len": PROMPT_LEN, "results":
               {str(k): v for k, v in results.items()}}, f, indent=2)

print(f"\n{'='*65}")
print(f"DONE — {config_name}")
print(f"{'='*65}")
print(f"\n{'Batch':>6} {'TTFT ms':>10} {'Dec ms/tok':>12} "
      f"{'Pref tok/s':>12} {'Dec req/s':>11}")
print("-"*55)
for bs, r in results.items():
    print(f"{bs:>6} {r['prefill_ms']:>10.1f} {r['decode_ms']:>12.1f} "
          f"{r['prefill_tok_s']:>12.0f} {r['decode_req_s']:>11.1f}")
print(f"\nSaved to {out_path}")
executor.shutdown()