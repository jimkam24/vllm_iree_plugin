"""
IREEPlatform — vLLM out-of-tree platform plugin for IREE backends.
 
Tells vLLM how to interact with our IREE runtime instead of CUDA.
Entry point registered via pyproject.toml / setup.cfg under:
    vllm.platform_plugins = iree = vllm_iree.platform:IREEPlatform
"""
 
import torch
from vllm.platforms import Platform, PlatformEnum
 
 
class IREEPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "cuda"
    device_type: str = "cuda"
    simple_compile_backend: str = "eager"   # disable torch.compile


    # identifier for the custom backend
    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return f"iree/cuda:{device_id}" 

    # TODO: in IREE pin memory probably is available
    # since IREE manages its own memory we disable it for now
    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    # TODO WARNING: does IREE support torch.inference_mode, do we keep this?
    @classmethod
    def inference_mode(cls):
        return torch.inference_mode()

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        pass

    # will be called before config is built, patch the arg patcher
    @classmethod
    def pre_register_and_update(cls, parser=None) -> None:
        pass  # TODO: add arg parser patches here later (IREE-specific CLI flags here later if needed)

    # here we set worker class, tweak scheduler, block size, compilation mode etc
    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        from vllm.config import CompilationMode
        """
        Called once after the full VllmConfig is built.
        Use this to:
          - point vLLM at our worker class
          - disable CUDA-specific compilation paths
          - set block size and other backend constraints
        """

        # 1. Route all workers to IREEWorker.
        if vllm_config.parallel_config.worker_cls == "auto":
            vllm_config.parallel_config.worker_cls = "vllm_plugin.worker.IREEWorker"

        # 2. Disable CUDA-specific compilation
        from vllm.config import CompilationMode
        vllm_config.compilation_config.mode = CompilationMode.NONE

        # 3. Block size — 16 is a safe default; TODO: revisit when profiling KV cache.
        if vllm_config.cache_config is not None:
            vllm_config.cache_config.block_size = 16

    # returns the attention backend string path
    @classmethod
    def get_attn_backend_cls(cls, 
                             selected_backend,  # AttentionBackendEnum
                             attn_selector_config, # AttentionSelectorConfig
                             ) -> str:
        """
        Return the dotted import path of our attention backend.
        vLLM will import this class and use it for all attention layers.
 
        attn_selector_config fields available if needed later:
            .head_size, .dtype, .kv_cache_dtype, .block_size,
            .use_mla, .has_sink, .use_sparse, .use_mm_prefix, .attn_type
        """

        return "vllm_plugin.attention.IREEAttentionBackend"
    
def get_platform_cls_qualname() -> str:
    return "vllm_plugin.platform.IREEPlatform"

# Future hooks (not needed for now)

# get_device_communicator_cls   — distributed comms (TODO: multi-device)
# get_static_graph_wrapper_cls  — graph capture wrapper (not used with IREE)
# get_current_memory_usage      — memory profiling hook
# get_punica_wrapper            — only if we add LoRA support
# get_compile_backend           — custom torch.compile backend
# apply_config_platform_defaults — fine-tune cudagraph sizes
# support_hybrid_kv_cache       — feature flag
# support_static_graph_mode     — feature flag
