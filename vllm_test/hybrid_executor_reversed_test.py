"""
HybridExecutor reversed role test.

IREE is rank 0 (first rank), native vLLM Worker is rank 1 (last rank).
Tests that the send path works when IREE is the first pipeline stage.

Note: This tests communication wiring only. Real partial forward pass
(Step 3b) is not implemented yet — IREE still runs the full model
internally and sends a stub tensor to rank 1.

Run from /vllm_iree/vllm_test/:
    python3 hybrid_executor_reversed_test.py
"""

import os
os.environ["VLLM_PLUGINS"] = "iree"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# IREE is rank 0, native is rank 1
os.environ["IREE_WORKER_RANKS"] = "0"
# Layer partition: IREE gets first 6 layers, native gets last 10
os.environ["VLLM_PP_LAYER_PARTITION"] = "6,10"
os.environ["IREE_GPU_ASSIGNMENT"]="1,0"

from vllm.engine.arg_utils import EngineArgs
from vllm.v1.core.sched.output import SchedulerOutput, CachedRequestData, NewRequestData
from vllm.sampling_params import SamplingParams

# At the top of hybrid_executor_reversed_test.py, after imports:
from vllm.v1.worker import gpu_worker
orig_check = gpu_worker.Worker.check_health
def patched_check(self):
    from vllm.distributed.parallel_state import get_pp_group
    pp = get_pp_group()
    with open(f"/tmp/pp_native_rank_{self.rank}.txt", "w") as f:
        f.write(f"rank={self.rank} pp.rank_in_group={pp.rank_in_group} pp.world_size={pp.world_size} is_last={pp.is_last_rank} ranks={pp.ranks}\n")
    orig_check(self)
gpu_worker.Worker.check_health = patched_check

print("=" * 60)
print("HybridExecutor Reversed Role Test")
print("  IREE    = rank 0 (first rank, sends activations)")
print("  Native  = rank 1 (last rank,  returns output)")
print("=" * 60)

# ── 1. Build VllmConfig ───────────────────────────────────────────────────────
print("\n[1] Building VllmConfig...")
engine_args = EngineArgs(
    model="meta-llama/Llama-3.2-1B",
    dtype="float32",
    max_model_len=512,
    max_num_seqs=2,
    enforce_eager=True,
    gpu_memory_utilization=0.6,
    distributed_executor_backend=(
        "vllm_plugin.executor.hybrid_executor.HybridExecutor"
    ),
)
vllm_config = engine_args.create_engine_config()
vllm_config.parallel_config.pipeline_parallel_size = 2
vllm_config.parallel_config.world_size = 2
print("  OK")

# ── 2. Instantiate HybridExecutor ─────────────────────────────────────────────
print("\n[2] Instantiating HybridExecutor...")
print("  (IREE_WORKER_RANKS=0: IREE is first rank)")
from vllm_plugin.executor.hybrid_executor import HybridExecutor
executor = HybridExecutor(vllm_config)
print(f"  Ray workers: {len(executor.workers)}")
print("  OK")

# ── 3. KV cache initialization ────────────────────────────────────────────────
print("\n[3] Initializing KV cache...")
kv_cache_specs = executor.get_kv_cache_specs()
print(f"  kv_cache_specs from {len(kv_cache_specs)} workers")

available_gpu_memory = executor.determine_available_memory()
print(f"  available_gpu_memory: {[f'{m/1e9:.2f}GB' for m in available_gpu_memory]}")

from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
kv_cache_configs = get_kv_cache_configs(
    vllm_config, kv_cache_specs, available_gpu_memory
)
print(f"  kv_cache_configs computed: {len(kv_cache_configs)} configs")

executor.initialize_from_config(kv_cache_configs)
print("  initialize_from_config OK")
print("  KV cache initialization complete")

# ── 4. Build fake SchedulerOutput ─────────────────────────────────────────────
print("\n[4] Building fake SchedulerOutput...")
prompt_token_ids = [791, 6864, 315, 9822, 374]  # "The capital of France is"

new_req = NewRequestData(
    req_id="reversed-test-0",
    prompt_token_ids=prompt_token_ids,
    sampling_params=SamplingParams(max_tokens=10),
    pooling_params=None,
    block_ids=([0, 1, 2],),
    num_computed_tokens=0,
    lora_request=None,
    mm_features=[],
    prompt_embeds=None,
)

scheduler_output = SchedulerOutput(
    scheduled_new_reqs=[new_req],
    scheduled_cached_reqs=CachedRequestData.make_empty(),
    num_scheduled_tokens={"reversed-test-0": len(prompt_token_ids)},
    total_num_scheduled_tokens=len(prompt_token_ids),
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
print("  OK")

print("\n[4.5] Checking PP group state on all workers...")
def get_runner_info(worker):
    from vllm.distributed.parallel_state import get_pp_group
    pp = get_pp_group()
    # Also directly call model_runner.execute_model equivalent check
    mr = getattr(worker, 'worker', worker).model_runner if hasattr(worker, 'worker') else worker.model_runner
    bp = getattr(mr, 'broadcast_pp_output', 'N/A')
    return (f"rank={getattr(getattr(worker,'worker',worker),'rank','?')} "
            f"broadcast_pp_output={bp} "
            f"is_last={pp.is_last_rank} "
            f"pp.rank_in_group={pp.rank_in_group} "
            f"pp.ranks={pp.ranks}")

results = executor.collective_rpc(get_runner_info)
print("Runner info:", results)

# ── 5. execute_model ──────────────────────────────────────────────────────────
print("\n[5] Calling execute_model()...")
output = executor.execute_model(scheduler_output)
print("  output type:", type(output).__name__)

if output is not None:
    print("  req_ids:", output.req_ids)
    print("  sampled_token_ids:", output.sampled_token_ids)

    if output.sampled_token_ids:
        token_id = output.sampled_token_ids[0][0]
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        next_token = tokenizer.decode([token_id])
        print(f"  next token: '{next_token}'")
else:
    print("  WARNING: output is None")

# ── 6. check_health + shutdown ────────────────────────────────────────────────
print("\n[6] check_health()...")
executor.check_health()
print("  OK")

print("\n[7] shutdown()...")
executor.shutdown()
print("  OK")

print("\n" + "=" * 60)
print("HybridExecutor reversed role test COMPLETE")
print("  Note: Step 3b (real activation passing) not yet implemented.")
print("  IREE still runs full model; native receives stub activations.")
print("=" * 60)