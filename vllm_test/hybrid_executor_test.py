"""
HybridExecutor smoke test — with correct KV cache init sequence.

Replicates the EngineCore._initialize_kv_caches() sequence manually
so we can test execute_model without the full engine.

Run from /vllm_iree/vllm_test/:
    python3 hybrid_executor_test.py
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["VLLM_PLUGINS"] = "iree"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["IREE_CUDA_ARCH"] = "sm_86"
os.environ["IREE_WORKER_RANKS"] = "1"  # select which ranks run iree
os.environ["VLLM_PP_LAYER_PARTITION"] = "14,2" # layer split to ranks
os.environ["IREE_GPU_ASSIGNMENT"]="1,0" # gpu assignment to ranks
# os.environ["IREE_FORCE_RECOMPILE"] = "1"
os.environ["IREE_USE_VLLM_MODEL"] = "1"
os.environ["IREE_USE_FFN"] = "1" # FFN with IREE (GPU default or with CPU paired with use cpu ffn iree)
os.environ["IREE_USE_CPU_FFN"] = "0" # FFN on CPU no IREE!!
os.environ["IREE_USE_CPU_FFN_IREE"] = "1" # FFN on CPU with IREE !
# os.environ["IREE_USE_COMPILE"] = "1"


# Reminder: GPU 0 in pci bus id is A2

from vllm.engine.arg_utils import EngineArgs
from vllm.v1.core.sched.output import SchedulerOutput, CachedRequestData, NewRequestData
from vllm.sampling_params import SamplingParams
from vllm.config.compilation import CUDAGraphMode

print("=" * 60)
print("HybridExecutor Smoke Test")
print("=" * 60)

# ── 1. Build VllmConfig ───────────────────────────────────────────────────────
print("\n[1] Building VllmConfig...")
engine_args = EngineArgs(
    model="meta-llama/Llama-3.2-1B",
    dtype="float32",
    max_model_len=512,
    max_num_seqs=4,
    enforce_eager=True,
    gpu_memory_utilization=0.4,
    pipeline_parallel_size=2,
    distributed_executor_backend="ray",  # ← tell vLLM it's Ray
)
vllm_config = engine_args.create_engine_config()
# Override PP after config creation to bypass vLLM's executor check
# vllm_config.parallel_config.pipeline_parallel_size = 2
# vllm_config.parallel_config.world_size = 2
# print("  OK")
vllm_config.parallel_config.distributed_executor_backend = \
    "vllm_plugin.executor.hybrid_executor.HybridExecutor"
vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

# ── 2. Instantiate HybridExecutor (creates + inits both workers) ──────────────
print("\n[2] Instantiating HybridExecutor...")
print("  (loads both workers — fast if .vmfb cached)")
from vllm_plugin.executor.hybrid_executor import HybridExecutor
executor = HybridExecutor(vllm_config)
print(f"  Ray workers: {len(executor.workers)}")
print("  OK")

def check_gpu(worker):
    import torch
    return f"rank={getattr(getattr(worker,'worker',worker),'rank','?')} GPU={torch.cuda.get_device_name(0)}"
results = executor.collective_rpc(check_gpu)
print("GPU assignment:", results)

# After executor = HybridExecutor(vllm_config)
print("\n[2b] Memory BEFORE KV cache (weights only):")
def get_mem_before(worker):
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()
    alloc = torch.cuda.memory_allocated(0) / 1024**3
    free, total = torch.cuda.mem_get_info(0)
    name = torch.cuda.get_device_name(0)
    return f"{name}: allocated={alloc:.3f}GB free={free/1024**3:.3f}GB"
mem = executor.collective_rpc(get_mem_before)
for m in mem:
    print(f"  {m}")


# ── 3. KV cache initialization (replicates EngineCore._initialize_kv_caches) ──
print("\n[3] Initializing KV cache...")

# Step 3a: get KV cache specs from all workers
kv_cache_specs = executor.get_kv_cache_specs()
print(f"  kv_cache_specs from {len(kv_cache_specs)} workers")

# Step 3b: profile available memory on all workers
available_gpu_memory = executor.determine_available_memory()
print(f"  available_gpu_memory: {[f'{m/1e9:.2f}GB' for m in available_gpu_memory]}")



# After [3] KV cache initialization, add:
print("\n[3b] Memory usage per rank:")
def get_mem(worker):
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()
    alloc = torch.cuda.memory_allocated(0) / 1024**3
    name = torch.cuda.get_device_name(0)
    return f"{name}: {alloc:.3f}GB allocated"
mem = executor.collective_rpc(get_mem)
for m in mem:
    print(f"  {m}")






# Step 3c: compute KV cache config from specs + available memory
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
kv_cache_configs = get_kv_cache_configs(
    vllm_config, kv_cache_specs, available_gpu_memory
)
print(f"  kv_cache_configs computed: {len(kv_cache_configs)} configs")

# Step 3d: allocate KV cache + warmup on all workers
executor.initialize_from_config(kv_cache_configs)
print("  initialize_from_config OK")
print("  KV cache initialization complete")

# ── 4. Build fake SchedulerOutput ─────────────────────────────────────────────
print("\n[4] Building fake SchedulerOutput...")
prompt_token_ids = [791, 6864, 315, 9822, 374]  # "The capital of France is"

new_req = NewRequestData(
    req_id="hybrid-test-0",
    prompt_token_ids=prompt_token_ids,
    sampling_params=SamplingParams(max_tokens=10),
    pooling_params=None,
    block_ids=([0, 1, 2],),
    num_computed_tokens=0,
    lora_request=None,
    mm_features=[],
    prompt_embeds=None,
)

# num_common_prefix_blocks needs one entry per KV cache group
# For our stub with one group ("iree_attn"), use [0]
# For the native worker with real attention groups, use [0] as well
# (0 = no common prefix blocks for this fresh request)
scheduler_output = SchedulerOutput(
    scheduled_new_reqs=[new_req],
    scheduled_cached_reqs=CachedRequestData.make_empty(),
    num_scheduled_tokens={"hybrid-test-0": len(prompt_token_ids)},
    total_num_scheduled_tokens=len(prompt_token_ids),
    finished_req_ids=set(),
    free_encoder_mm_hashes=[],
    scheduled_spec_decode_tokens={},
    scheduled_encoder_inputs={},
    num_common_prefix_blocks=[0],   # fix: was [], needs one entry per KV cache group
    preempted_req_ids=None,
    has_structured_output_requests=False,
    pending_structured_output_tokens=False,
    num_invalid_spec_tokens=None,
    kv_connector_metadata=None,
    ec_connector_metadata=None,
)
print("  OK")

# ── 5. execute_model ──────────────────────────────────────────────────────────
print("\n[5] Calling execute_model()...")
output = executor.execute_model(scheduler_output)
print("  output type:", type(output).__name__)
print("  req_ids:", output.req_ids)
print("  sampled_token_ids:", output.sampled_token_ids)

if output.sampled_token_ids:
    token_id = output.sampled_token_ids[0][0]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    next_token = tokenizer.decode([token_id])
    print(f"  next token: '{next_token}'")

# ── 6. check_health + shutdown ────────────────────────────────────────────────
print("\n[6] check_health()...")
executor.check_health()
print("  OK")

print("\n[7] shutdown()...")
executor.shutdown()
print("  OK")

print("\n" + "=" * 60)
print("HybridExecutor smoke test PASSED")
print("=" * 60)