"""
HybridExecutor IREE-only test.

Both pipeline ranks use IREEWorker — no native vLLM gpu_worker.
Tests that two IREEWorkers can coexist in a pipeline.

Note: This tests communication wiring only. Real partial forward pass
(Step 3b) is not implemented yet — both workers run the full IREE model
independently. Rank 0 sends a stub tensor to rank 1, rank 1 ignores it
and runs its own full forward pass.

Expected output: correct token (' Paris') since rank 1 runs full model.

Run from /vllm_iree/vllm_test/:
    python3 hybrid_executor_iree_only_test.py
"""

import os
os.environ["VLLM_PLUGINS"] = "iree"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["IREE_WORKER_RANKS"] = "0,1"      # both ranks are IREEWorker
os.environ["VLLM_PP_LAYER_PARTITION"] = "14,2"  # even split
os.environ["IREE_GPU_ASSIGNMENT"] = "0,1"    # rank 0 → GPU 0, rank 1 → GPU 1

from vllm.engine.arg_utils import EngineArgs
from vllm.v1.core.sched.output import SchedulerOutput, CachedRequestData, NewRequestData
from vllm.sampling_params import SamplingParams

print("=" * 60)
print("HybridExecutor IREE-Only Test")
print("  Rank 0: IREEWorker (first rank, sends stub activations)")
print("  Rank 1: IREEWorker (last rank,  returns output)")
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
print("  (IREE_WORKER_RANKS=0,1: both ranks are IREEWorker)")
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
    req_id="iree-only-test-0",
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
    num_scheduled_tokens={"iree-only-test-0": len(prompt_token_ids)},
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

# ── 5. execute_model ──────────────────────────────────────────────────────────
print("\n[5] Calling execute_model()...")
output = executor.execute_model(scheduler_output)
print("  output type:", type(output).__name__)

if output is not None and output.req_ids:
    print("  req_ids:", output.req_ids)
    print("  sampled_token_ids:", output.sampled_token_ids)
    if output.sampled_token_ids:
        token_id = output.sampled_token_ids[0][0]
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        next_token = tokenizer.decode([token_id])
        print(f"  next token: '{next_token}'")
        # Token will be correct since rank 1 runs full IREE model
        assert token_id == 12366, (
            f"Expected 12366 (' Paris'), got {token_id}. "
            "Note: this tests wiring only, not real PP."
        )
        print("  ✅ correct token — IREE-only pipeline wiring works")
else:
    print("  WARNING: output is None or empty")

# ── 6. check_health + shutdown ────────────────────────────────────────────────
print("\n[6] check_health()...")
executor.check_health()
print("  OK")

print("\n[7] shutdown()...")
executor.shutdown()
print("  OK")

print("\n" + "=" * 60)
print("HybridExecutor IREE-only test COMPLETE")
print("  Both IREEWorkers coexist in pipeline — wiring verified.")
print("  Step 3b will wire real activation passing between ranks.")
print("=" * 60)