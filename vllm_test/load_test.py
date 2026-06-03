# 1. Platform
from vllm_plugin.platform import IREEPlatform
print('IREEPlatform OK:', IREEPlatform._enum)

# 2. Attention backend
from vllm_plugin.attention.attention import IREEAttentionBackend, IREEAttentionBackendImpl, IREEAttentionMetadataBuilder
print('IREEAttentionBackend OK:', IREEAttentionBackend.get_name())

# 3. Worker
from vllm_plugin.worker.worker import IREEWorker
print('IREEWorker OK')

# 4. Model runner
from vllm_plugin.worker.model_runner import IREEModelRunner
print('IREEModelRunner OK')

# 5. vLLM discovers the plugin via entry points
from vllm.platforms import current_platform
print('current_platform:', current_platform)