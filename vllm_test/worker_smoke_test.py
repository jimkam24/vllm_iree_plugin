"""
Worker lifecycle smoke test.

Tests the IREEWorker lifecycle in isolation, without starting the full
vLLM engine. Calls each lifecycle method in order and verifies no crashes.

Run from /vllm_iree/vllm_test/:
    python3 worker_smoke_test.py
"""

import os
os.environ["VLLM_PLUGINS"] = "iree"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

import torch
from vllm.engine.arg_utils import EngineArgs

print("=" * 60)
print("IREEWorker Lifecycle Smoke Test")
print("=" * 60)

# ── 1. Build VllmConfig via EngineArgs ────────────────────────────────────────
print("\n[1] Building VllmConfig...")
engine_args = EngineArgs(
    model="meta-llama/Llama-3.2-1B",
    dtype="float32",
    max_model_len=512,           # small — this is a smoke test
    max_num_seqs=2,
    enforce_eager=True,          # no cuda graphs
    gpu_memory_utilization=0.3,  # leave room for other processes
    worker_cls="vllm_plugin.worker.worker.IREEWorker",
)
vllm_config = engine_args.create_engine_config()
print("  VllmConfig OK")
print("  max_model_len:", vllm_config.model_config.max_model_len)
print("  worker_cls:", vllm_config.parallel_config.worker_cls)

# ── 2. Instantiate worker ─────────────────────────────────────────────────────
print("\n[2] Instantiating IREEWorker...")
from vllm_plugin.worker.worker import IREEWorker

worker = IREEWorker(
    vllm_config=vllm_config,
    local_rank=0,
    rank=0,
    distributed_init_method="env://",
    is_driver_worker=True,
)
print("  IREEWorker instantiated OK")

# ── 3. init_device ────────────────────────────────────────────────────────────
print("\n[3] Calling init_device()...")
worker.init_device()
print("  device:", worker.device)
print("  init_device OK")

# ── 4. load_model ─────────────────────────────────────────────────────────────
print("\n[4] Calling load_model()...")
worker.load_model()
print("  load_model OK")

# ── 5. determine_available_memory ─────────────────────────────────────────────
print("\n[5] Calling determine_available_memory()...")
mem = worker.determine_available_memory()
print(f"  available memory: {mem / 1e9:.2f} GB")
print("  determine_available_memory OK")

# ── 6. initialize_cache ───────────────────────────────────────────────────────
print("\n[6] Calling initialize_cache()...")
worker.initialize_cache(num_gpu_blocks=64, num_cpu_blocks=0)
print("  num_gpu_blocks:", worker.cache_config.num_gpu_blocks)
print("  initialize_cache OK")

# ── 7. get_kv_cache_spec ──────────────────────────────────────────────────────
print("\n[7] Calling get_kv_cache_spec()...")
spec = worker.get_kv_cache_spec()
print("  kv_cache_spec keys:", list(spec.keys()))
print("  get_kv_cache_spec OK")

# ── 8. compile_or_warm_up_model ───────────────────────────────────────────────
print("\n[8] Calling compile_or_warm_up_model()...")
worker.compile_or_warm_up_model()
print("  compile_or_warm_up_model OK")

# ── 9. check_health ───────────────────────────────────────────────────────────
print("\n[9] Calling check_health()...")
worker.check_health()
print("  check_health OK")

print("\n" + "=" * 60)
print("ALL LIFECYCLE STEPS PASSED")
print("=" * 60)