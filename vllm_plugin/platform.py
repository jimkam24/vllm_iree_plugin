"""
IREEPlatform — vLLM out-of-tree platform plugin for IREE backends.
 
Tells vLLM how to interact with our IREE runtime instead of CUDA.
Entry point registered via pyproject.toml / setup.cfg under:
    vllm.platform_plugins = iree = vllm_iree.platform:IREEPlatform
"""
 
import os
import torch
from vllm.platforms import Platform, PlatformEnum
 
 
class IREEPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "cuda"
    device_type: str = "cuda"
    ray_device_key: str = "GPU"                        
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"  
    dist_backend: str = "nccl"    


    # identifier for the custom backend
    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return f"iree/cuda:{device_id}" 

    # since IREE manages its own memory we disable it for now
    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def inference_mode(cls):
        return torch.inference_mode()

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        pass

    # will be called before config is built, patch the arg patcher
    @classmethod
    def pre_register_and_update(cls, parser=None) -> None:
        pass

    # here we set worker class, tweak scheduler, block size, compilation mode etc
    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        from vllm.config import CompilationMode

        my_rank = int(os.environ.get("MY_PP_RANK", "0"))
        iree_ranks_str = os.environ.get("IREE_WORKER_RANKS", "1")
        # Handle both "0,1" (driver format) and "01" (digit-encoded worker format)
        if "," in iree_ranks_str:
            iree_ranks = set(int(x.strip()) for x in iree_ranks_str.split(",") if x.strip().isdigit())
        else:
            iree_ranks = set(int(c) for c in iree_ranks_str if c.isdigit())
        is_hybrid = "MY_PP_RANK" in os.environ

        if is_hybrid and my_rank not in iree_ranks:
            # Native CUDA rank — use gpu_worker, don't touch compilation
            if vllm_config.parallel_config.worker_cls == "auto":
                vllm_config.parallel_config.worker_cls = (
                    "vllm_plugin.worker.native_wrapper.NativeWorkerWithSend"
                )
            return  # ← early return, no IREE settings

        # IREE rank (or single-worker mode) — use IREEWorker + custom attention
        if vllm_config.parallel_config.worker_cls == "auto":
            vllm_config.parallel_config.worker_cls = (
                "vllm_plugin.worker.worker.IREEWorker"
            )
        vllm_config.compilation_config.mode = CompilationMode.NONE
        if vllm_config.cache_config is not None:
            vllm_config.cache_config.block_size = 16
        
    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config) -> str:
        return "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    
    @classmethod
    def set_device(cls, device: torch.device) -> None:
        my_rank = int(os.environ.get("MY_PP_RANK", "0"))
        # Decode: each character is one rank digit
        # e.g. "01" → {0, 1}, "1" → {1}, "012" → {0, 1, 2}
        iree_ranks_str = os.environ.get("IREE_WORKER_RANKS", "1")
        iree_ranks = set(
            int(x.strip())
            for x in iree_ranks_str.split(",")
            if x.strip().isdigit()
        )
        is_hybrid = "MY_PP_RANK" in os.environ

        if is_hybrid and my_rank not in iree_ranks:
            # Native CUDA worker — CUDA_VISIBLE_DEVICES already restricts
            # to one GPU, which always appears as cuda:0 within this process.
            torch.cuda.set_device(0)
        else:
            # IREE worker — IREE manages its own device context.
            pass
            
    @classmethod
    def is_cuda_alike(cls) -> bool:
        # We run on CUDA hardware in Phase 1, so CUDA-alike ops are valid.
        return True
    
    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype) -> None:
        # Allow all dtypes — the native CUDA worker validates its own dtype
        # separately. We don't restrict at the platform level.
        pass

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        from vllm.platforms.interface import DeviceCapability
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(device_id)
            return DeviceCapability(major=major, minor=minor)
        return None

    @classmethod
    def get_current_memory_usage(cls, device=None) -> float:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            return torch.cuda.max_memory_allocated(device)
        return 0.0
    
def get_platform_cls_qualname() -> str:
    pp_rank = os.environ.get("IREE_PP_RANK", "")
    # In hybrid mode, only activate IREEPlatform on the IREE rank.
    # If IREE_PP_RANK is not set, activate unconditionally (single-worker mode).
    if pp_rank == "" or pp_rank == "1":
        return "vllm_plugin.platform.IREEPlatform"
    return None  # rank 0 falls through to default CUDA platform

# Future hooks (not needed for now)

# get_device_communicator_cls   — distributed comms (TODO: multi-device)
# get_static_graph_wrapper_cls  — graph capture wrapper (not used with IREE)
# get_current_memory_usage      — memory profiling hook
# get_punica_wrapper            — only if we add LoRA support
# get_compile_backend           — custom torch.compile backend
# apply_config_platform_defaults — fine-tune cudagraph sizes
# support_hybrid_kv_cache       — feature flag
# support_static_graph_mode     — feature flag
