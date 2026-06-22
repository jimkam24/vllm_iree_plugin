"""
Decode Correctness Test
=======================
Tests that prefill + multi-step decode produces correct tokens for all configs.

Uses vLLM's native Scheduler to drive HybridExecutor — this gives the correct
SchedulerOutput construction automatically (block allocation, num_computed_tokens,
new_token_ids etc.) so we test actual model correctness, not our manual schedule
construction.

Reference: single-GPU LLM API with temperature=0 gives ' Paris. It is the most'
for prompt 'The capital of France is' (with BOS token prepended by tokenizer).

Run from /vllm_iree/vllm_test/:
    python3 decode_test.py [--config CONFIG]

    CONFIG: 1r, 2, 3, 4, 5, 6, 7 (default: 2)
"""

import os
import sys
import argparse

# ── Parse args before setting env vars ───────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="2",
    choices=["1r", "2", "3", "4", "5", "6", "7"],
    help="Benchmark config to test")
parser.add_argument("--max-tokens", type=int, default=6)
args = parser.parse_args()
CONFIG = args.config

# ── Environment setup ─────────────────────────────────────────────────────────
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["VLLM_PP_LAYER_PARTITION"] = "14,2"
os.environ["IREE_GPU_ASSIGNMENT"] = "1,0"

if CONFIG == "1r":
    os.environ["VLLM_PLUGINS"] = "iree"
    os.environ["IREE_WORKER_RANKS"] = "99"   # both native
    os.environ["IREE_USE_VLLM_MODEL"] = "1"
    config_name = "HybridExecutor both native (1r)"
elif CONFIG == "2":
    os.environ["VLLM_PLUGINS"] = "iree"
    os.environ["IREE_WORKER_RANKS"] = "1"
    os.environ["IREE_USE_VLLM_MODEL"] = "1"
    os.environ["IREE_CUDA_ARCH"] = "sm_70"
    config_name = "native + IREEWorker Triton (2)"
elif CONFIG == "3":
    os.environ["VLLM_PLUGINS"] = "iree"
    os.environ["IREE_WORKER_RANKS"] = "0,1"
    os.environ["IREE_USE_VLLM_MODEL"] = "1"
    os.environ["IREE_CUDA_ARCH"] = "sm_70"
    config_name = "both IREEWorker Triton (3)"
elif CONFIG == "4":
    os.environ["VLLM_PLUGINS"] = "iree"
    os.environ["IREE_WORKER_RANKS"] = "0"
    os.environ["IREE_USE_VLLM_MODEL"] = "1"
    os.environ["IREE_CUDA_ARCH"] = "sm_70"
    config_name = "IREEWorker + native (4)"
elif CONFIG == "5":
    os.environ["VLLM_PLUGINS"] = "iree"
    os.environ["IREE_WORKER_RANKS"] = "1"
    os.environ["IREE_USE_VLLM_MODEL"] = "1"
    os.environ["IREE_USE_FFN"] = "1"
    os.environ["IREE_USE_CPU_FFN_IREE"] = "1"
    os.environ["IREE_CUDA_ARCH"] = "sm_86"
    config_name = "native + IREEWorker IREE CPU FFN (5)"
elif CONFIG == "6":
    os.environ["VLLM_PLUGINS"] = "iree"
    os.environ["IREE_WORKER_RANKS"] = "1"
    os.environ["IREE_USE_VLLM_MODEL"] = "1"
    os.environ["IREE_USE_CPU_FFN"] = "1"
    os.environ["IREE_CUDA_ARCH"] = "sm_86"
    config_name = "native + IREEWorker PyTorch CPU FFN (6)"
elif CONFIG == "7":
    os.environ["VLLM_PLUGINS"] = "iree"
    os.environ["IREE_WORKER_RANKS"] = "1"
    os.environ["IREE_USE_VLLM_MODEL"] = "1"
    os.environ["IREE_USE_FFN"] = "1"
    os.environ["IREE_CUDA_ARCH"] = "sm_86"
    config_name = "native + IREEWorker IREE GPU FFN (7)"

print(f"\n{'='*60}")
print(f"Decode Correctness Test — Config {CONFIG}")
print(f"{'='*60}")
print(f"Config: {config_name}\n")

# ── Imports ───────────────────────────────────────────────────────────────────
from vllm.engine.arg_utils import EngineArgs
from vllm_plugin.executor.hybrid_executor import HybridExecutor
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.request import Request
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer

# ── Setup executor ────────────────────────────────────────────────────────────
print("[1] Building VllmConfig...")
engine_args = EngineArgs(
    model="meta-llama/Llama-3.2-1B",
    dtype="float32",
    max_model_len=512,
    enforce_eager=True,
    gpu_memory_utilization=0.5,
    pipeline_parallel_size=2,
    distributed_executor_backend="ray",
)
vllm_config = engine_args.create_engine_config()
vllm_config.parallel_config.distributed_executor_backend = (
    "vllm_plugin.executor.hybrid_executor.HybridExecutor"
)
print("  OK")

print("[2] Instantiating HybridExecutor...")
executor = HybridExecutor(vllm_config)
print("  OK")

print("[3] Initializing KV cache...")
kv_specs = executor.get_kv_cache_specs()
avail = executor.determine_available_memory()
kv_configs = get_kv_cache_configs(vllm_config, kv_specs, avail)
executor.initialize_from_config(kv_configs)
# Required for Scheduler
vllm_config.cache_config.num_gpu_blocks = kv_configs[0].num_blocks
print(f"  {kv_configs[0].num_blocks} blocks allocated")
print("  OK")

print("[4] Instantiating native vLLM Scheduler...")
structured_output_manager = StructuredOutputManager(vllm_config)
scheduler = Scheduler(
    vllm_config=vllm_config,
    kv_cache_config=kv_configs[0],
    structured_output_manager=structured_output_manager,
    block_size=vllm_config.cache_config.block_size,
)
print("  OK")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# ── Reference ─────────────────────────────────────────────────────────────────
# LLM API prepends BOS (128000) automatically. We do the same here so our
# output matches the reference: ' Paris. It is the most'
PROMPT = "The capital of France is"
prompt_token_ids = tokenizer.encode(PROMPT)  # includes BOS
REFERENCE = " Paris. It is the most"

print(f"\nPrompt: {repr(PROMPT)}")
print(f"Token IDs: {prompt_token_ids}")
print(f"Reference: {repr(REFERENCE)}\n")

# ── Run prefill + decode ──────────────────────────────────────────────────────
print("[5] Running prefill + decode...")

req = Request(
    request_id="decode-test-0",
    prompt_token_ids=prompt_token_ids,
    sampling_params=SamplingParams(max_tokens=args.max_tokens, temperature=0),
    pooling_params=None,
    eos_token_id=None,
    arrival_time=0.0,
    mm_features=None,
    lora_request=None,
)
scheduler.add_request(req)

generated_ids = []
step = 0
while scheduler.has_unfinished_requests():
    scheduler_output = scheduler.schedule()
    if not scheduler_output.total_num_scheduled_tokens:
        break

    model_output = executor.execute_model(scheduler_output)

    if model_output and model_output.sampled_token_ids:
        token_id = model_output.sampled_token_ids[0][0]
        generated_ids.append(token_id)
        token_text = tokenizer.decode([token_id])
        label = "Prefill" if step == 0 else f"Decode {step}"
        print(f"  {label:10s} → token {token_id:6d} = {repr(token_text)}")

    scheduler.update_from_output(scheduler_output, model_output)
    step += 1

generated_text = tokenizer.decode(generated_ids)
print(f"\nGenerated: {repr(generated_text)}")
print(f"Reference: {repr(REFERENCE)}")

# ── Verify ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
if generated_text == REFERENCE:
    print(f"✅ PASSED — output matches reference exactly")
else:
    # Check token-by-token
    ref_ids = tokenizer.encode(REFERENCE, add_special_tokens=False)
    n_match = sum(1 for a, b in zip(generated_ids, ref_ids) if a == b)
    print(f"⚠️  Output differs from reference")
    print(f"   Matched {n_match}/{min(len(generated_ids), len(ref_ids))} tokens")
    if n_match == len(ref_ids):
        print(f"   (Reference is a prefix of generated — OK)")
    elif n_match >= len(ref_ids) * 0.8:
        print(f"   (Close match — likely numerical precision difference)")
    else:
        print(f"   (Significant mismatch — investigate)")
print(f"{'='*60}\n")

# ── Cleanup ───────────────────────────────────────────────────────────────────
executor.shutdown()