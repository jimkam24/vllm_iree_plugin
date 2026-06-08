"""
execute_model smoke test.

Verifies that IREEModelRunner.execute_model() produces output of the
correct shape when called with a real token sequence.

Run from /vllm_iree/vllm_test/:
    python3 execute_model_test.py
"""

import os
os.environ["VLLM_PLUGINS"] = "iree"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

import torch
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.core.sched.output import SchedulerOutput, CachedRequestData
from vllm.v1.core.sched.output import NewRequestData
from vllm.sampling_params import SamplingParams

print("=" * 60)
print("execute_model Smoke Test")
print("=" * 60)

# ── Build config ──────────────────────────────────────────────────────────────
print("\n[1] Building VllmConfig...")
engine_args = EngineArgs(
    model="meta-llama/Llama-3.2-1B",
    dtype="float32",
    max_model_len=512,
    max_num_seqs=2,
    enforce_eager=True,
    gpu_memory_utilization=0.3,
    worker_cls="vllm_plugin.worker.worker.IREEWorker",
)
vllm_config = engine_args.create_engine_config()
print("  OK")

# ── Instantiate and init worker ───────────────────────────────────────────────
print("\n[2] Init worker lifecycle...")
from vllm_plugin.worker.worker import IREEWorker

worker = IREEWorker(
    vllm_config=vllm_config,
    local_rank=0,
    rank=0,
    distributed_init_method="env://",
    is_driver_worker=True,
)
worker.init_device()
worker.load_model()
worker.initialize_cache(num_gpu_blocks=64, num_cpu_blocks=0)
worker.compile_or_warm_up_model()
print("  OK")

# ── Build a fake SchedulerOutput ──────────────────────────────────────────────
print("\n[3] Building fake SchedulerOutput...")

# "The capital of France is" tokenized approximately
prompt_token_ids = [791, 6864, 315, 9822, 374]

new_req = NewRequestData(
    req_id="test-req-0",
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
    num_scheduled_tokens={"test-req-0": len(prompt_token_ids)},
    total_num_scheduled_tokens=len(prompt_token_ids),
    finished_req_ids=set(),
    free_encoder_mm_hashes=[],
    scheduled_spec_decode_tokens={},
    scheduled_encoder_inputs={},
    num_common_prefix_blocks=[],
    preempted_req_ids=None,
    has_structured_output_requests=False,
    pending_structured_output_tokens=False,
    num_invalid_spec_tokens=None,
    kv_connector_metadata=None,
    ec_connector_metadata=None,
)
print("  OK")

# ── Call execute_model ────────────────────────────────────────────────────────
print("\n[4] Calling execute_model()...")
output = worker.execute_model(scheduler_output)
print("  output type:", type(output).__name__)
print("  req_ids:", output.req_ids)
print("  sampled_token_ids:", output.sampled_token_ids)

# Verify output shape
assert len(output.req_ids) == 1, f"Expected 1 req, got {len(output.req_ids)}"
assert len(output.sampled_token_ids) == 1, \
    f"Expected 1 sampled list, got {len(output.sampled_token_ids)}"
assert len(output.sampled_token_ids[0]) == 1, \
    f"Expected 1 token, got {len(output.sampled_token_ids[0])}"

token_id = output.sampled_token_ids[0][0]
assert isinstance(token_id, int), f"Token ID should be int, got {type(token_id)}"
assert 0 <= token_id < 128256, f"Token ID {token_id} out of vocab range"

print(f"  next token id: {token_id}")

# Decode the token
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
next_token = tokenizer.decode([token_id])
print(f"  next token text: '{next_token}'")

print("\n" + "=" * 60)
print("execute_model PASSED")
print("  IREE dispatch produced a valid next token.")
print("=" * 60)