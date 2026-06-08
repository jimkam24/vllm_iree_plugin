import os
os.environ["VLLM_PLUGINS"] = "iree"
os.environ["VLLM_PP_LAYER_PARTITION"] = "10,6"

from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    dtype="float32",
    max_model_len=512,
    enforce_eager=True,
    gpu_memory_utilization=0.3,
    distributed_executor_backend="vllm_plugin.executor.hybrid_executor.HybridExecutor",
)

# Override PP after init
llm.llm_engine.vllm_config.parallel_config.pipeline_parallel_size = 2

outputs = llm.generate(
    ["The capital of France is"],
    SamplingParams(max_tokens=5, temperature=0.0),
)
print("Output:", outputs[0].outputs[0].text)