"""
PP Configuration Benchmark
===========================
Measures latency, memory, and throughput for three pipeline parallelism configurations:

  Config 1: Pure vLLM PP     — both ranks native gpu_worker.Worker
  Config 2: IREE-IREE PP     — both ranks IREEWorker with IREE vmfb dispatch
  Config 3: PyTorch-PyTorch  — both ranks IREEWorker with PyTorch dispatch (no IREE)

Metrics:
  - TTFT (Time To First Token): latency of a single prefill forward pass
  - Memory: GPU memory used per rank during inference
  - Throughput: tokens/second over N repeated forward passes

Usage:
    cd /vllm_iree/vllm_test
    python3 benchmark_pp_configs.py

Results saved to /tmp/benchmark_results.json and printed as a table.
"""

import os
import sys
import time
import json
import gc
import statistics

# ── Select configuration via CONFIG env var ───────────────────────────────────
CONFIG = os.environ.get("PP_BENCH_CONFIG", "1")

print(f"\n{'='*60}")
print(f"PP Configuration Benchmark — Config {CONFIG}")
print(f"{'='*60}")

# ── Common settings ───────────────────────────────────────────────────────────
os.environ["VLLM_PLUGINS"] = "iree"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["VLLM_PP_LAYER_PARTITION"] = "14,2"
os.environ["IREE_GPU_ASSIGNMENT"] = "1,0"   # rank0→V100S(GPU1), rank1→A2(GPU0)
os.environ["IREE_CUDA_ARCH"] = "sm_70"      # compile target (V100S)

if CONFIG == "1":
    # Pure vLLM PP: rank 99 is never a valid rank → both get NativeWorkerWithSend
    os.environ["IREE_WORKER_RANKS"] = "99"
    config_name = "Pure vLLM PP (native + native)"
elif CONFIG == "2":
    # IREE-IREE PP: both ranks IREEWorker with IREE vmfb
    os.environ["IREE_WORKER_RANKS"] = "0,1"
    os.environ.pop("IREE_USE_PYTORCH", None)
    config_name = "IREE-IREE PP (iree + iree)"
elif CONFIG == "3":
    # PyTorch-PyTorch PP: both ranks IREEWorker but using PyTorch dispatch
    os.environ["IREE_WORKER_RANKS"] = "0,1"
    os.environ["IREE_USE_PYTORCH"] = "1"
    config_name = "PyTorch-PyTorch PP (pytorch + pytorch)"
else:
    print(f"Unknown config {CONFIG}. Set PP_BENCH_CONFIG=1, 2, or 3.")
    sys.exit(1)

print(f"Config: {config_name}")

# ── Imports (after env vars set) ──────────────────────────────────────────────
import torch
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.core.sched.output import SchedulerOutput, CachedRequestData, NewRequestData
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm_plugin.executor.hybrid_executor import HybridExecutor

# ── Benchmark settings ────────────────────────────────────────────────────────
N_WARMUP = 3       # warmup forward passes (not measured)
N_MEASURE = 10     # measured forward passes for latency/throughput
# PROMPT_TOKENS = [791, 6864, 315, 9822, 374]  # "The capital of France is"
PROMPT_TOKENS = list(range(100, 150))  # 50 tokens

# ── Build VllmConfig ──────────────────────────────────────────────────────────
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
print("  OK")

# ── Instantiate HybridExecutor ────────────────────────────────────────────────
print("\n[2] Instantiating HybridExecutor...")
executor = HybridExecutor(vllm_config)
print(f"  Ray workers: {len(executor.workers)}")

# ── KV cache init ─────────────────────────────────────────────────────────────
print("\n[3] Initializing KV cache...")
kv_cache_specs = executor.get_kv_cache_specs()
available_gpu_memory = executor.determine_available_memory()
kv_cache_configs = get_kv_cache_configs(vllm_config, kv_cache_specs, available_gpu_memory)
executor.initialize_from_config(kv_cache_configs)
print("  OK")

# ── Helper: build SchedulerOutput ────────────────────────────────────────────
def make_scheduler_output(req_id: str, token_ids: list[int]) -> SchedulerOutput:
    new_req = NewRequestData(
        req_id=req_id,
        prompt_token_ids=token_ids,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_ids=([0, 1, 2],),
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

# ── Helper: measure GPU memory per rank ──────────────────────────────────────
def measure_memory_mb() -> dict:
    """Returns GPU memory allocated (MB) per rank via collective_rpc."""
    def get_mem(worker):
        import torch
        device = torch.device("cuda:0")
        return torch.cuda.memory_allocated(device) / 1024 / 1024
    results = executor.collective_rpc(get_mem)
    return {"rank0_mb": results[0], "rank1_mb": results[1]}

# ── Warmup ────────────────────────────────────────────────────────────────────
print(f"\n[4] Warmup ({N_WARMUP} passes)...")
for i in range(N_WARMUP):
    sched = make_scheduler_output(f"warmup-{i}", PROMPT_TOKENS)
    _ = executor.execute_model(sched)
print("  Warmup complete")

# ── Measure memory after warmup ───────────────────────────────────────────────
print("\n[5] Measuring GPU memory...")
mem = measure_memory_mb()
print(f"  Rank 0: {mem['rank0_mb']:.1f} MB")
print(f"  Rank 1: {mem['rank1_mb']:.1f} MB")
print(f"  Total:  {mem['rank0_mb'] + mem['rank1_mb']:.1f} MB")

# ── Latency measurement ───────────────────────────────────────────────────────
print(f"\n[6] Measuring latency ({N_MEASURE} passes)...")
latencies_ms = []

for i in range(N_MEASURE):
    sched = make_scheduler_output(f"bench-{i}", PROMPT_TOKENS)
    t0 = time.perf_counter()
    output = executor.execute_model(sched)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000
    latencies_ms.append(latency_ms)
    
    if i == 0:
        # Verify correctness on first measured pass
        if output is not None and output.sampled_token_ids:
            token = output.sampled_token_ids[0][0]
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
            decoded = tokenizer.decode([token])
            print(f"  First token: {token} = '{decoded}' {'✅' if token == 12366 else '❌ WRONG'}")

mean_ms = statistics.mean(latencies_ms)
std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0
p50_ms = statistics.median(latencies_ms)
p90_ms = sorted(latencies_ms)[int(0.9 * len(latencies_ms))]
min_ms = min(latencies_ms)
max_ms = max(latencies_ms)

print(f"\n  Latency over {N_MEASURE} passes:")
print(f"    Mean:   {mean_ms:.2f} ms")
print(f"    Std:    {std_ms:.2f} ms")
print(f"    P50:    {p50_ms:.2f} ms")
print(f"    P90:    {p90_ms:.2f} ms")
print(f"    Min:    {min_ms:.2f} ms")
print(f"    Max:    {max_ms:.2f} ms")

# ── Throughput measurement ────────────────────────────────────────────────────
print(f"\n[7] Measuring throughput...")
N_TOKENS = 50  # number of forward passes for throughput
t_start = time.perf_counter()
for i in range(N_TOKENS):
    sched = make_scheduler_output(f"tput-{i}", PROMPT_TOKENS)
    _ = executor.execute_model(sched)
t_end = time.perf_counter()
elapsed = t_end - t_start
throughput = N_TOKENS / elapsed
print(f"  {N_TOKENS} passes in {elapsed:.2f}s = {throughput:.2f} forward_passes/sec")
print(f"  Effective throughput: {throughput:.2f} tokens/sec (1 token per pass)")

# ── Results summary ───────────────────────────────────────────────────────────
results = {
    "config": CONFIG,
    "config_name": config_name,
    "layer_split": os.environ["VLLM_PP_LAYER_PARTITION"],
    "latency_ms": {
        "mean": round(mean_ms, 3),
        "std": round(std_ms, 3),
        "p50": round(p50_ms, 3),
        "p90": round(p90_ms, 3),
        "min": round(min_ms, 3),
        "max": round(max_ms, 3),
        "all": [round(x, 3) for x in latencies_ms],
    },
    "memory_mb": {
        "rank0": round(mem["rank0_mb"], 1),
        "rank1": round(mem["rank1_mb"], 1),
        "total": round(mem["rank0_mb"] + mem["rank1_mb"], 1),
    },
    "throughput_tokens_per_sec": round(throughput, 3),
    "n_warmup": N_WARMUP,
    "n_measure": N_MEASURE,
}

# Save results
results_path = f"/tmp/benchmark_config{CONFIG}.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[8] Results saved to {results_path}")

# ── Cleanup ───────────────────────────────────────────────────────────────────
print("\n[9] Shutdown...")
executor.shutdown()

print(f"\n{'='*60}")
print(f"BENCHMARK COMPLETE — Config {CONFIG}: {config_name}")
print(f"  Mean latency: {mean_ms:.2f} ms")
print(f"  Memory total: {mem['rank0_mb'] + mem['rank1_mb']:.1f} MB")
print(f"  Throughput:   {throughput:.2f} tokens/sec")
print(f"{'='*60}")