import os
os.environ['VLLM_PLUGINS'] = 'iree'
os.environ['IREE_WORKER_RANKS'] = '1'
os.environ['IREE_USE_VLLM_MODEL'] = '1'
os.environ['VLLM_PP_LAYER_PARTITION'] = '14,2'
os.environ['IREE_GPU_ASSIGNMENT'] = '1,0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
from vllm.engine.arg_utils import EngineArgs
from vllm_plugin.executor.hybrid_executor import HybridExecutor
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.core.sched.output import SchedulerOutput, CachedRequestData, NewRequestData
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer

engine_args = EngineArgs(
    model='meta-llama/Llama-3.2-1B', dtype='float32',
    max_model_len=512, enforce_eager=True, gpu_memory_utilization=0.5,
    distributed_executor_backend='vllm_plugin.executor.hybrid_executor.HybridExecutor',
)
vllm_config = engine_args.create_engine_config()
vllm_config.parallel_config.pipeline_parallel_size = 2
vllm_config.parallel_config.world_size = 2

executor = HybridExecutor(vllm_config)
kv_specs = executor.get_kv_cache_specs()
avail = executor.determine_available_memory()
kv_configs = get_kv_cache_configs(vllm_config, kv_specs, avail)
executor.initialize_from_config(kv_configs)

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
prompt = 'The capital of France is'
prompt_tokens = tokenizer.encode(prompt)
print(f'Prompt: {prompt_tokens} ({len(prompt_tokens)} tokens)')

block_size = 16
n_blocks = (512 + block_size - 1) // block_size
block_ids = list(range(n_blocks))

# Step 1: Prefill
new_req = NewRequestData(
    req_id='decode-test',
    prompt_token_ids=prompt_tokens,
    sampling_params=SamplingParams(max_tokens=10),
    pooling_params=None,
    block_ids=(block_ids,),
    num_computed_tokens=0,
    lora_request=None,
    mm_features=[],
    prompt_embeds=None,
)
def make_sched(new_reqs, cached, num_tokens_dict):
    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=num_tokens_dict,
        total_num_scheduled_tokens=sum(num_tokens_dict.values()),
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

out = executor.execute_model(make_sched([new_req], CachedRequestData.make_empty(), {'decode-test': len(prompt_tokens)}))
prefill_token = out.sampled_token_ids[0][0] if out else None
print(f'Prefill: {prefill_token} = {repr(tokenizer.decode([prefill_token]))}')

# Steps 2+: Decode
generated = [prefill_token]
all_tokens = prompt_tokens + generated
current_computed = len(prompt_tokens)  # tokens computed after prefill

for step in range(5):
    # CachedRequestData fields: req_ids, resumed_req_ids, new_token_ids,
    # all_token_ids, new_block_ids, num_computed_tokens, num_output_tokens
    cached = CachedRequestData(
        req_ids=['decode-test'],
        resumed_req_ids=set(),          # empty set, not list
        new_token_ids=[[generated[-1]]],
        all_token_ids={'decode-test': len(all_tokens)},  # dict of req_id -> len
        new_block_ids=[([],)],          # ← tuple of empty lists, one per KV group
        num_computed_tokens=[current_computed],
        num_output_tokens=[len(generated)],
    )
    out = executor.execute_model(make_sched([], cached, {'decode-test': 1}))
    token = out.sampled_token_ids[0][0] if out else None
    generated.append(token)
    all_tokens.append(token)
    current_computed += 1
    print(f'Step {step+1}: {token} = {repr(tokenizer.decode([token]) if token else None)}')

print(f'Generated: {repr(tokenizer.decode(generated))}')
executor.shutdown()