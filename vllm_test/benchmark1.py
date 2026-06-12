import os, time
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['VLLM_PLUGINS'] = ''  # no plugin

from vllm import LLM, SamplingParams

llm = LLM(
    model='meta-llama/Llama-3.2-1B',
    dtype='float32',
    max_model_len=512,
    enforce_eager=True,
    tensor_parallel_size=1,
    pipeline_parallel_size=2,
)

# prompts = ['The capital of France is']
prompt_tokens = list(range(100, 150))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
prompts = tok.decode(prompt_tokens)
sampling_params = SamplingParams(max_tokens=1)

# Warmup
for _ in range(3):
    llm.generate(prompts, sampling_params)

# Measure
times = []
for _ in range(10):
    t0 = time.perf_counter()
    out = llm.generate(prompts, sampling_params)
    t1 = time.perf_counter()
    times.append((t1-t0)*1000)

import statistics
print('Token:', out[0].outputs[0].token_ids[0])
print(f'Mean latency: {statistics.mean(times):.2f} ms')
print(f'P50: {statistics.median(times):.2f} ms')
print(f'Throughput: {10/sum(t/1000 for t in times):.2f} tok/s')