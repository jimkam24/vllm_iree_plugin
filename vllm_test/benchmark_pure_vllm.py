import os, time, statistics
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['VLLM_PLUGINS'] = ''
os.environ['VLLM_PP_LAYER_PARTITION'] = '14,2'
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')

llm = LLM(model='meta-llama/Llama-3.2-1B', dtype='float32',
    max_model_len=512, max_num_seqs=4, gpu_memory_utilization=0.4,
    enforce_eager=True, pipeline_parallel_size=2, distributed_executor_backend="ray")

PROMPT = 'The capital of France is'
base = tok.encode(PROMPT)
params_1 = SamplingParams(max_tokens=1, temperature=0)
params_10 = SamplingParams(max_tokens=10, temperature=0)

for plen in [5, 50, 100]:
    tokens = (base * (plen // len(base) + 1))[:plen]
    prompt_text = tok.decode(tokens)

    # Warmup
    for _ in range(5): llm.generate([prompt_text], params_1)

    # TTFT
    ttft_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        llm.generate([prompt_text], params_1)
        ttft_times.append((time.perf_counter()-t0)*1000)

    # Decode per token
    decode_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        llm.generate([prompt_text], params_10)
        decode_times.append((time.perf_counter()-t0)*1000 / 10)

    print(f'  Prompt {plen:3d} tokens | TTFT: {statistics.mean(ttft_times):.2f}ms | Decode/tok: {statistics.mean(decode_times):.2f}ms')

print('COMPLETE — vLLM native PP compiled 14+2')
del llm