"""
IREEModelRunner — prepares inputs, runs the IREE forward pass, returns outputs.

Phase 1: exports a PyTorch model slice to MLIR via iree-turbine, compiles to
         .vmfb, and dispatches inference through the IREE runtime.
Phase 2: swap IREE compile target to amd-aie, rest unchanged.

Key features:
  - Layer partitioning via VLLM_PP_LAYER_PARTITION or even split
  - .vmfb cached on disk, reloaded on subsequent runs
  - force_recompile flag for full control
  - causal mask patch for transformers >= 4.56.0 compatibility
  
  
Step 3b replacement functions for model_runner.py.
 
Replace the following in model_runner.py:
  - _export_and_compile()  → rank-aware version below
  - _run_iree()            → two-input version below
  - execute_model()        → wires intermediate_tensors below
 
Four wrapper variants based on PP rank position:
  FullModel   (first+last): input_ids → logits         (single worker)
  FirstRank   (first only): input_ids → hidden_states  (sends to next)
  MiddleRank  (neither):    hidden_states, pos → hidden (passes through)
  LastRank    (last only):  hidden_states, pos → logits (final output)

"""

import os
import torch
import torch.nn as nn
import torch.export as torch_export
import numpy as np
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_model_runner import CommonAttentionMetadata

logger = init_logger(__name__)

# IREE compile target for Phase 1.
# Change to "amd-aie" for Phase 2 (NPU).
IREE_TARGET = os.environ.get("IREE_TARGET", "cuda")

# LLVM target arch per IREE target.
IREE_TARGET_ARCH = {
    "cuda": os.environ.get("IREE_CUDA_ARCH", "sm_70"),  # A2 = sm_86, V100S = sm_70
    "amd-aie": "aie",
}

# ── Rank-aware model wrappers ─────────────────────────────────────────────────
 
class FullModelWrapper(nn.Module):
    """Single rank owns all layers. input_ids → logits."""
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model
 
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids)
        return out.logits
 
 
class FirstRankWrapper(nn.Module):
    """
    First rank (not last). Owns embedding + layers[0:layer_end].
    input_ids → hidden_states  (sent to next rank via NCCL)
    """
    def __init__(self, hf_model, layer_start, layer_end):
        super().__init__()
        self.embed_tokens = hf_model.model.embed_tokens
        self.layers = nn.ModuleList(list(hf_model.model.layers)[layer_start:layer_end])
        self.rotary_emb = hf_model.model.rotary_emb

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.rotary_emb(hidden, pos_ids)
        # Build causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        causal_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=hidden.dtype, device=input_ids.device)
        causal_mask = causal_mask.masked_fill(~mask, torch.finfo(hidden.dtype).min)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=causal_mask,
                          position_ids=pos_ids, position_embeddings=pos_emb)
        return hidden
 
 
class MiddleRankWrapper(nn.Module):
    """
    Middle rank (not first, not last). Owns layers[layer_start:layer_end].
    hidden_states, position_ids → hidden_states
    """
    def __init__(self, hf_model, layer_start, layer_end):
        super().__init__()
        self.layers = nn.ModuleList(list(hf_model.model.layers)[layer_start:layer_end])
        self.rotary_emb = hf_model.model.rotary_emb

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        seq_len = position_ids.shape[1]
        pos_emb = self.rotary_emb(hidden_states, position_ids)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        causal_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=hidden_states.dtype, device=hidden_states.device)
        causal_mask = causal_mask.masked_fill(~mask, torch.finfo(hidden_states.dtype).min)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=causal_mask,
                                  position_ids=position_ids, position_embeddings=pos_emb)
        return hidden_states
 
class LastRankWrapper(nn.Module):
    """
    Last rank (not first). Owns layers[layer_start:layer_end] + norm + lm_head.
    hidden_states, position_ids → logits
    """
    def __init__(self, hf_model, layer_start, layer_end):
        super().__init__()
        self.layers = nn.ModuleList(list(hf_model.model.layers)[layer_start:layer_end])
        self.rotary_emb = hf_model.model.rotary_emb
        self.norm = hf_model.model.norm
        self.lm_head = hf_model.lm_head

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        seq_len = position_ids.shape[1]
        pos_emb = self.rotary_emb(hidden_states, position_ids)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        causal_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=hidden_states.dtype, device=hidden_states.device)
        causal_mask = causal_mask.masked_fill(~mask, torch.finfo(hidden_states.dtype).min)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=causal_mask,
                                  position_ids=position_ids, position_embeddings=pos_emb)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)
def _build_wrapper(
    hf_model,
    layer_start: int,
    layer_end: int,
    is_first: bool,
    is_last: bool,
) -> nn.Module:
    """Select the correct wrapper based on PP rank position."""
    if is_first and is_last:
        return FullModelWrapper(hf_model)
    elif is_first:
        return FirstRankWrapper(hf_model, layer_start, layer_end)
    elif is_last:
        return LastRankWrapper(hf_model, layer_start, layer_end)
    else:
        return MiddleRankWrapper(hf_model, layer_start, layer_end)
 


# ── Layer partition ───────────────────────────────────────────────────────────

def _compute_layer_range(
    num_layers: int,
    pp_rank: int,
    pp_world_size: int,
) -> tuple[int, int]:
    """
    Compute which transformer layers this PP rank owns.

    Priority:
      1. VLLM_PP_LAYER_PARTITION env var — explicit per-rank counts
         e.g. VLLM_PP_LAYER_PARTITION=10,6 means rank 0 gets 10, rank 1 gets 6
      2. Even split — last rank absorbs remainder

    Returns:
        (layer_start, layer_end) — half-open range [start, end)
    """
    partition_str = os.environ.get("VLLM_PP_LAYER_PARTITION", "")
    if partition_str:
        splits = [int(x.strip()) for x in partition_str.split(",")]
        if len(splits) != pp_world_size:
            raise ValueError(
                f"VLLM_PP_LAYER_PARTITION has {len(splits)} entries "
                f"but pp_world_size={pp_world_size}"
            )
        if sum(splits) != num_layers:
            raise ValueError(
                f"VLLM_PP_LAYER_PARTITION sums to {sum(splits)} "
                f"but model has {num_layers} layers"
            )
        layer_start = sum(splits[:pp_rank])
        layer_end = layer_start + splits[pp_rank]
    else:
        # Even split — last rank absorbs remainder
        base = num_layers // pp_world_size
        remainder = num_layers % pp_world_size
        layer_start = pp_rank * base + min(pp_rank, remainder)
        layer_end = layer_start + base + (1 if pp_rank < remainder else 0)

    logger.info(
        "IREEModelRunner: PP rank %d/%d owns layers [%d, %d)",
        pp_rank, pp_world_size, layer_start, layer_end,
    )
    return layer_start, layer_end


# ── IREE export + compile ─────────────────────────────────────────────────────

def _patch_causal_mask():
    """
    Patch create_causal_mask in llama modeling to bypass the vmap-based
    implementation in transformers >= 4.56.0 which torch.export cannot trace.
    Returns (original_fn, llama_module) so the caller can restore it.
    """
    import transformers.models.llama.modeling_llama as llama_module

    original = llama_module.create_causal_mask

    def traceable_causal_mask(
        config, input_embeds, past_key_values_length=0,
        sliding_window=None, cache_position=None,
        attention_mask=None, **kwargs
    ):
        batch_size, seq_len = input_embeds.shape[:2]
        full_len = seq_len + past_key_values_length
        mask = torch.tril(
            torch.ones((seq_len, full_len), dtype=torch.bool,
                       device=input_embeds.device)
        )
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        additive = torch.zeros_like(mask, dtype=input_embeds.dtype)
        additive = additive.masked_fill(
            ~mask, torch.finfo(input_embeds.dtype).min
        )
        return additive

    llama_module.create_causal_mask = traceable_causal_mask
    return original, llama_module


def _restore_causal_mask(original_fn, llama_module):
    llama_module.create_causal_mask = original_fn


def _export_and_compile(
    model_name: str,
    vmfb_path: str,
    mlir_path: str,
    vllm_config,
    layer_start: int,
    layer_end: int,
    is_first: bool,
    is_last: bool,
    target: str = "cuda",
    target_arch: str = "sm_70",
) -> None:
    """
    Export model slice to MLIR via iree-turbine and compile to .vmfb.
 
    Wrapper selection:
      FullModel   (is_first + is_last): input_ids → logits
      FirstRank   (is_first only):      input_ids → hidden_states
      MiddleRank  (neither):            (hidden_states, pos_ids) → hidden_states
      LastRank    (is_last only):       (hidden_states, pos_ids) → logits
 
    Dynamic shapes: seq_len is dynamic for all wrappers.
    """
    import iree.turbine.aot as aot
    import iree.compiler as iree_compiler
    from transformers import AutoModelForCausalLM
 
    logger_msg = (
        f"layers [{layer_start},{layer_end}) "
        f"is_first={is_first} is_last={is_last}"
    )
 
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).cpu().eval()
 
    wrapper = _build_wrapper(hf_model, layer_start, layer_end, is_first, is_last)
    wrapper = wrapper.eval()
    
    # Free the full HF model — wrapper already holds references to its slice.
    # Unused layers (not referenced by wrapper) will be freed by GC.
    import gc
    hidden_size = hf_model.config.hidden_size  # 2048 for Llama 1B
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
 

 
    # Build example inputs and dynamic shapes based on wrapper type
    seq_len_dim = torch_export.Dim("seq_len", min=1, max=4096)
 
    if is_first and is_last:
        # FullModel: input_ids only
        example_args = (torch.zeros((1, 6), dtype=torch.long),)
        dynamic_shapes = {"input_ids": {1: seq_len_dim}}
 
    elif is_first:
        # FirstRank: input_ids only
        example_args = (torch.zeros((1, 6), dtype=torch.long),)
        dynamic_shapes = {"input_ids": {1: seq_len_dim}}
 
    else:
        # MiddleRank or LastRank: hidden_states + position_ids
        example_args = (
            torch.zeros((1, 6, hidden_size), dtype=torch.float32),
            torch.arange(6, dtype=torch.long).unsqueeze(0),
        )
        dynamic_shapes = {
            "hidden_states": {1: seq_len_dim},
            "position_ids":  {1: seq_len_dim},
        }
 
    # Patch causal mask before export
    original_fn, llama_module = _patch_causal_mask()
    try:
        exported = aot.export(
            wrapper,
            args=example_args,
            dynamic_shapes=dynamic_shapes,
        )
        exported.save_mlir(mlir_path)
    finally:
        _restore_causal_mask(original_fn, llama_module)
 
    iree_compiler.tools.compile_file(
        mlir_path,
        output_file=vmfb_path,
        target_backends=[target],
        extra_args=[
            f"--iree-cuda-target={target_arch}",
            "--iree-input-type=torch",
        ],
    )
    
    return wrapper

def _run_iree_first_or_full(iree_config, iree_fn, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Dispatch for FullModel or FirstRank wrappers.
    Input: input_ids [1, seq_len]
    Output: logits [1, seq_len, vocab] or hidden_states [1, seq_len, hidden]
    """
    import iree.runtime as ireert
    input_np = input_ids.cpu().numpy().astype(np.int64).reshape(1, -1)
    iree_input = ireert.asdevicearray(iree_config.device, input_np)
    result = iree_fn(iree_input)
    return torch.from_numpy(np.array(result))
 
 
def _run_iree_middle_or_last(
    iree_config,
    iree_fn,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Dispatch for MiddleRank or LastRank wrappers.
    Inputs: hidden_states [1, seq_len, hidden], position_ids [1, seq_len]
    Output: hidden_states or logits
    """
    import iree.runtime as ireert
    h_np = hidden_states.cpu().numpy().astype(np.float32)
    p_np = position_ids.cpu().numpy().astype(np.int64)
    iree_h = ireert.asdevicearray(iree_config.device, h_np)
    iree_p = ireert.asdevicearray(iree_config.device, p_np)
    result = iree_fn(iree_h, iree_p)
    return torch.from_numpy(np.array(result))


def _load_vmfb(vmfb_path: str, iree_device: str):
    """Load a compiled .vmfb artifact into the IREE runtime."""
    import iree.runtime as ireert

    config = ireert.Config(iree_device)
    ctx = ireert.SystemContext(config=config)
    with open(vmfb_path, "rb") as f:
        vmfb = f.read()
    vm_module = ireert.VmModule.copy_buffer(ctx.instance, vmfb)
    ctx.add_vm_module(vm_module)
    main_fn = ctx.modules.module["main"]
    logger.info("IREEModelRunner: loaded .vmfb from %s", vmfb_path)
    return config, main_fn


# ── Model runner ──────────────────────────────────────────────────────────────

class IREEModelRunner:
    """
    Sits between IREEWorker and the IREE runtime.

    Responsibilities:
      - Load weights and compile to .vmfb (or reload cached artifact)
      - Track per-request state
      - Build input tensors from SchedulerOutput each step
      - Dispatch inference through IREE runtime
      - Return ModelRunnerOutput to the worker
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device = None,
        force_recompile: bool = False,
        vmfb_dir: str = "/tmp/iree_artifacts",
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device = device or torch.device("cuda:0")
        self.force_recompile = force_recompile
        self.vmfb_dir = vmfb_dir

        # Set after load_model()
        self.model: nn.Module | None = None
        self.iree_fn = None
        self.iree_config = None
        self.layer_start: int = 0
        self.layer_end: int = 0

        # req_id -> {prompt_token_ids, num_computed_tokens}
        self.requests: dict[str, dict] = {}

        os.makedirs(self.vmfb_dir, exist_ok=True)

    # ── Artifact paths ────────────────────────────────────────────────────────

    def _artifact_name(self) -> str:
        """
        Unique artifact name encoding model + layer range + target.
        Example: meta_llama_llama_3_2_1b_layers_0_16_cuda
        """
        model_short = (
            self.model_config.model
            .replace("/", "_")
            .replace("-", "_")
            .lower()
        )
        return (
            f"{model_short}"
            f"_layers_{self.layer_start}_{self.layer_end}"
            f"_{IREE_TARGET}"
        )

    def _vmfb_path(self) -> str:
        return os.path.join(self.vmfb_dir, self._artifact_name() + ".vmfb")

    def _mlir_path(self) -> str:
        return os.path.join(self.vmfb_dir, self._artifact_name() + ".mlir")

    # ── Layer partition ───────────────────────────────────────────────────────

    def _resolve_layer_range(self) -> tuple[int, int]:
        num_layers = self.model_config.hf_config.num_hidden_layers
        # Check if HybridExecutor injected the rank directly
        additional = getattr(self.vllm_config, 'additional_config', {}) or {}
        if 'hybrid_pp_rank' in additional:
            pp_rank = additional['hybrid_pp_rank']
            pp_world_size = additional['hybrid_pp_world_size']
        else:
            try:
                from vllm.distributed.parallel_state import get_pp_group
                pp_group = get_pp_group()
                pp_rank = pp_group.rank_in_group
                pp_world_size = pp_group.world_size
            except Exception:
                pp_rank = 0
                pp_world_size = 1
        return _compute_layer_range(num_layers, pp_rank, pp_world_size)

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """
        Load weights, determine layer range, compile or reload .vmfb.
        """
        logger.info(
            "IREEModelRunner: loading %s ...", self.model_config.model
        )
        self.model = get_model(vllm_config=self.vllm_config)
        self.model.eval()
        logger.info("IREEModelRunner: weights loaded.")

        self.layer_start, self.layer_end = self._resolve_layer_range()
        
        from vllm.distributed.parallel_state import get_pp_group
        try:
            pp = get_pp_group()
            self.is_first_rank = pp.is_first_rank
            self.is_last_rank = pp.is_last_rank
        except Exception:
            self.is_first_rank = True
            self.is_last_rank = True

        if os.environ.get("IREE_USE_FFN", "0") == "1":
            self._compile_mlp_iree()
            return  # skip full layer vmfb loading — not needed for Path A2
        elif os.environ.get("IREE_USE_CPU_FFN", "0") == "1":
            self._patch_mlp_cpu()

        vmfb_path = self._vmfb_path()
        mlir_path = self._mlir_path()

        if os.path.exists(vmfb_path) and not self.force_recompile:
            logger.info(
                "IREEModelRunner: found cached .vmfb at %s, skipping compile.",
                vmfb_path,
            )
        else:
            if self.force_recompile and os.path.exists(vmfb_path):
                logger.info(
                    "IREEModelRunner: force_recompile=True, recompiling..."
                )
            _pt_wrapper = _export_and_compile(
                model_name=self.model_config.model,
                vmfb_path=vmfb_path,
                mlir_path=mlir_path,
                vllm_config=self.vllm_config,
                layer_start=self.layer_start,
                layer_end=self.layer_end,
                is_first=self.is_first_rank,
                is_last=self.is_last_rank,
                target=IREE_TARGET,
                target_arch=IREE_TARGET_ARCH.get(IREE_TARGET, "sm_70"),
            )
            # Store wrapper for torch.compile path — same HF model, no second load
            if os.environ.get("IREE_USE_PYTORCH", "0") == "1" and _pt_wrapper is not None:
                import gc
                _pt_wrapper = _pt_wrapper.to(self.device).eval()
                self._pt_compiled = torch.compile(_pt_wrapper, mode="reduce-overhead")
                del _pt_wrapper
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("IREEModelRunner: torch.compile wrapper ready.")

        iree_device_str = f"{IREE_TARGET}://0"
        self.iree_config, self.iree_fn = _load_vmfb(vmfb_path, iree_device_str)
        logger.info("IREEModelRunner: IREE runtime ready.")
        
        
        # Build torch.compile wrapper for Config 3 baseline
        if os.environ.get("IREE_USE_PYTORCH", "0") == "1":
            if not hasattr(self, '_pt_compiled'):
                # vmfb was cached — need to load HF model once with lock
                import fcntl, gc
                lock_path = "/tmp/iree_pt_load.lock"
                with open(lock_path, 'w') as lf:
                    fcntl.flock(lf, fcntl.LOCK_EX)
                    try:
                        from transformers import AutoModelForCausalLM
                        # hf_model_pt = AutoModelForCausalLM.from_pretrained(
                        #     self.model_config.model,
                        #     torch_dtype=torch.float32,
                        #     attn_implementation="eager",
                        # ).to(self.device).eval()
                        # pt_wrapper = _build_wrapper(
                        #     hf_model_pt,
                        #     self.layer_start, self.layer_end,
                        #     self.is_first_rank, self.is_last_rank,
                        # ).to(self.device).eval()
                        # del hf_model_pt
                        
                        hf_model_pt = AutoModelForCausalLM.from_pretrained(
                            self.model_config.model,
                            torch_dtype=torch.float32,
                            attn_implementation="eager",
                        ).cpu().eval()                    # ← load to CPU first
                        pt_wrapper = _build_wrapper(
                            hf_model_pt, self.layer_start, self.layer_end,
                            self.is_first_rank, self.is_last_rank,
                        ).eval()
                        del hf_model_pt                   # ← free full model before GPU move
                        gc.collect()
                        pt_wrapper = pt_wrapper.to(self.device)  # ← only slice goes to GPU
                        
                    
                        torch.cuda.empty_cache()
                        self._pt_compiled = torch.compile(
                            pt_wrapper, mode="reduce-overhead"
                        )
                        del pt_wrapper
                        gc.collect()
                        torch.cuda.empty_cache()
                        logger.info("IREEModelRunner: torch.compile wrapper ready (cached vmfb path).")
                    finally:
                        fcntl.flock(lf, fcntl.LOCK_UN)
        

    # ── KV cache interface ────────────────────────────────────────────────────

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        hf_config = self.model_config.hf_config
        head_size = hf_config.hidden_size // hf_config.num_attention_heads
        num_kv_heads = hf_config.num_key_value_heads
        block_size = self.cache_config.block_size

        specs = {}
        for layer_idx in range(self.layer_start, self.layer_end):
            layer_name = f"model.layers.{layer_idx}.self_attn.attn"
            specs[layer_name] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=torch.float32,
            )
        return specs

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        from vllm.v1.worker.gpu_model_runner import bind_kv_cache

        self.kv_cache_config = kv_cache_config
        self.kv_caches: list[torch.Tensor] = []

        # Allocate KV cache tensors directly from the spec
        # Shape: [2, num_blocks, block_size, num_kv_heads, head_size]
        hf_config = self.model_config.hf_config
        head_size = hf_config.hidden_size // hf_config.num_attention_heads
        num_kv_heads = hf_config.num_key_value_heads
        block_size = self.cache_config.block_size
        num_blocks = kv_cache_config.num_blocks

        kv_caches: dict[str, torch.Tensor] = {}
        for layer_idx in range(self.layer_start, self.layer_end):
            layer_name = f"model.layers.{layer_idx}.self_attn.attn"
            kv_tensor = torch.zeros(
                2, num_blocks, block_size, num_kv_heads, head_size,
                dtype=torch.float32,
                device=self.device,
            )
            kv_caches[layer_name] = kv_tensor

        forward_context = self.vllm_config.compilation_config.static_forward_context

        bind_kv_cache(
            kv_caches=kv_caches,
            forward_context=forward_context,
            runner_kv_caches=self.kv_caches,
        )
        logger.info(
            "IREEModelRunner: KV cache initialized and bound for layers %d-%d "
            "(%d layers, %d blocks).",
            self.layer_start, self.layer_end, len(kv_caches), num_blocks,
        )

    # ── Warmup / profiling ────────────────────────────────────────────────────

    def _dummy_run(self, num_tokens: int) -> None:
        """Warmup forward pass through PyTorch (not IREE)."""
        assert self.model is not None, "load_model() must be called first"
        input_ids = torch.zeros(
            num_tokens, dtype=torch.long, device=self.device
        )
        positions = torch.arange(
            num_tokens, dtype=torch.long, device=self.device
        )
        with torch.no_grad(), set_forward_context(None, self.vllm_config):
            self.model(input_ids=input_ids, positions=positions)

    def profile_run(self) -> None:
        self._dummy_run(self.scheduler_config.max_num_batched_tokens)

    def warm_up(self) -> None:
        logger.info("IREEModelRunner: warmup complete (artifact pre-compiled).")
        # No dummy run needed — vmfb compiled during load_model()

    # ── Per-step state management ─────────────────────────────────────────────

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        for req in scheduler_output.scheduled_new_reqs:
            block_ids_flat = []
            if req.block_ids:
                ids = (req.block_ids[0]
                    if isinstance(req.block_ids[0], (list, tuple))
                    else req.block_ids)
                block_ids_flat = list(ids)
            self.requests[req.req_id] = {
                "prompt_token_ids": req.prompt_token_ids,
                "num_computed_tokens": 0,
                "block_ids": block_ids_flat,  # ← store for decode
            }
            
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            if req_id in self.requests:
                self.requests[req_id]["num_computed_tokens"] = (
                    cached.num_computed_tokens[i]
                )

    # ── Input preparation ─────────────────────────────────────────────────────

    def _prepare_inputs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids: list[int] = []
        positions: list[int] = []

        for req in scheduler_output.scheduled_new_reqs:
            state = self.requests[req.req_id]
            start = state["num_computed_tokens"]
            chunk = req.prompt_token_ids[start:]
            token_ids.extend(chunk)
            positions.extend(range(start, start + len(chunk)))

        cached = scheduler_output.scheduled_cached_reqs
        for req_id in cached.req_ids:
            state = self.requests.get(req_id)
            if state is None:
                continue
            n = state["num_computed_tokens"]
            token_ids.append(
                state["prompt_token_ids"][n]
                if n < len(state["prompt_token_ids"]) else 0
            )
            positions.append(n)

        input_ids = torch.tensor(
            token_ids, dtype=torch.long, device=self.device
        )
        pos = torch.tensor(positions, dtype=torch.long, device=self.device)
        return input_ids, pos
    
    def release_vllm_model(self) -> None:
        """Free the vLLM model after warmup — IREE vmfb handles inference."""
        if self.model is not None:
            import gc
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("IREEModelRunner: vLLM model released, GPU memory freed.")

    # ── IREE dispatch ─────────────────────────────────────────────────────────

    # def _run_iree(
    #     self,
    #     input_ids: torch.Tensor,
    #     hidden_states: torch.Tensor | None = None,
    #     position_ids: torch.Tensor | None = None,
    # ) -> torch.Tensor:
    #     if self.is_first_rank or (self.is_first_rank and self.is_last_rank):
    #         return _run_iree_first_or_full(
    #             self.iree_config, self.iree_fn, input_ids
    #         )
    #     else:
    #         return _run_iree_middle_or_last(
    #             self.iree_config, self.iree_fn, hidden_states, position_ids
    #         )

    def _run_iree(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Config 3: torch.compile dispatch
        if os.environ.get("IREE_USE_PYTORCH", "0") == "1":
            with torch.no_grad():
                if self.is_first_rank:
                    # input_ids may be 1D [seq_len] — add batch dim
                    ids = input_ids.reshape(1, -1) if input_ids.dim() == 1 else input_ids
                    return self._pt_compiled(ids)
                else:
                    h = hidden_states.unsqueeze(0) if hidden_states.dim() == 2 else hidden_states
                    p = position_ids.unsqueeze(0) if position_ids.dim() == 1 else position_ids
                    return self._pt_compiled(h.to(self.device), p.to(self.device))
        # Config 2: IREE vmfb dispatch
        if self.is_first_rank or (self.is_first_rank and self.is_last_rank):
            return _run_iree_first_or_full(self.iree_config, self.iree_fn, input_ids)
        else:
            return _run_iree_middle_or_last(
                self.iree_config, self.iree_fn, hidden_states, position_ids
            )
            
            
    def _execute_vllm_model(
        self,
        scheduler_output: SchedulerOutput,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
    ) -> ModelRunnerOutput | object:
        from vllm.forward_context import set_forward_context
        from vllm.distributed.parallel_state import get_pp_group
        from vllm_plugin.attention.attention import IREEAttentionMetadata

        num_tokens = input_ids.shape[0]
        block_size = self.cache_config.block_size
        max_blocks = (
            self.vllm_config.model_config.max_model_len + block_size - 1
        ) // block_size

        # Determine prefill vs decode
        # Prefill: new requests being processed for the first time
        # Decode: cached requests generating next token
        new_reqs = scheduler_output.scheduled_new_reqs
        cached_reqs = scheduler_output.scheduled_cached_reqs
        num_new = len(new_reqs)
        num_cached = len(cached_reqs.req_ids)
        num_reqs = num_new + num_cached

        # is_prefill: true if any request has query_len > 1
        is_prefill = (num_tokens > num_reqs)

        # Build slot_mapping, block_table, seq_lens
        slot_mapping_list: list[int] = []
        seq_lens_list: list[int] = []
        block_table = torch.zeros(
            num_reqs, max_blocks, dtype=torch.int32, device=self.device
        )

        req_idx = 0
        for req in new_reqs:
            state = self.requests.get(req.req_id, {})
            start = state.get("num_computed_tokens", 0)
            query_len = len(req.prompt_token_ids) - start
            seq_len = len(req.prompt_token_ids)  # full sequence length
            seq_lens_list.append(seq_len)
            # slot_mapping: sequential slots starting from 0 for prefill
            for s in range(start, start + query_len):
                slot_mapping_list.append(s)
            # block_table from request's block_ids
            if req.block_ids:
                ids = (req.block_ids[0]
                    if isinstance(req.block_ids[0], (list, tuple))
                    else req.block_ids)
                for i, bid in enumerate(list(ids)[:max_blocks]):
                    block_table[req_idx, i] = int(bid)
            req_idx += 1

        for i, req_id in enumerate(cached_reqs.req_ids):
            state = self.requests.get(req_id, {})
            n_computed = state.get("num_computed_tokens", 0)
            # seq_len = all tokens computed so far + 1 new token
            seq_len = n_computed + 1
            seq_lens_list.append(seq_len)
            # slot for the new token = n_computed position
            slot_mapping_list.append(n_computed)
            # block_table: use blocks stored during prefill in request state
            stored_block_ids = state.get("block_ids", [])
            for j, bid in enumerate(stored_block_ids[:max_blocks]):
                block_table[req_idx, j] = int(bid)
            req_idx += 1

        slot_mapping = torch.tensor(
            slot_mapping_list, dtype=torch.int64, device=self.device
        )
        seq_lens = torch.tensor(
            seq_lens_list, dtype=torch.int32, device=self.device
        )

        iree_meta = IREEAttentionMetadata(
            num_actual_tokens=num_tokens,
            seq_lens=seq_lens,
            slot_mapping=slot_mapping,
            block_tables=block_table,
            is_prefill=is_prefill,
        )

        forward_context = self.vllm_config.compilation_config.static_forward_context
        attn_metadata = {layer_name: iree_meta for layer_name in forward_context}

        with torch.no_grad(), set_forward_context(
            attn_metadata=attn_metadata,
            vllm_config=self.vllm_config,
            num_tokens=num_tokens,
        ):
            output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
            )

        pp = get_pp_group()
        if not pp.is_last_rank:
            return output  # IntermediateTensors — worker sends via NCCL

        hidden_states = (output if isinstance(output, torch.Tensor)
                        else output.tensors.get("hidden_states", output))
        logits = self.model.compute_logits(hidden_states)

        # For decode: sample last token only; for prefill: also last token
        next_token_id = int(torch.argmax(logits[-1, :]).item())
        req_ids = list(self.requests.keys())
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=[[next_token_id]] * len(req_ids),
        )


    def _build_common_attn_metadata(
        self,
        scheduler_output: SchedulerOutput,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> "CommonAttentionMetadata":

        num_tokens = input_ids.shape[0]
        num_reqs = len(scheduler_output.scheduled_new_reqs) + len(
            scheduler_output.scheduled_cached_reqs.req_ids
        )

        # query_start_loc: [0, num_tokens] for single request
        query_start_loc = torch.tensor(
            [0, num_tokens], dtype=torch.int32, device=self.device
        )
        query_start_loc_cpu = query_start_loc.cpu()

        seq_lens = torch.tensor(
            [num_tokens], dtype=torch.int32, device=self.device
        )

        # slot_mapping: sequential slots for prefill
        slot_mapping = torch.arange(
            num_tokens, dtype=torch.int64, device=self.device
        )

        # block_table: get from scheduled request block_ids
        block_size = self.cache_config.block_size
        max_blocks = (512 + block_size - 1) // block_size  # max_model_len / block_size
        block_table = torch.zeros(
            1, max_blocks, dtype=torch.int32, device=self.device
        )
        if scheduler_output.scheduled_new_reqs:
            req = scheduler_output.scheduled_new_reqs[0]
            block_ids = req.block_ids[0] if req.block_ids else []
            for i, bid in enumerate(block_ids[:max_blocks]):
                block_table[0, i] = bid

        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=num_tokens,
            max_seq_len=num_tokens,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=True,
        )

    # ── Main inference entry point ────────────────────────────────────────────

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors=None,
    ) -> ModelRunnerOutput:
        if not scheduler_output.total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT

        self._update_states(scheduler_output)
        input_ids, positions = self._prepare_inputs(scheduler_output)
        
        if os.environ.get("IREE_USE_VLLM_MODEL", "0") == "1":
            return self._execute_vllm_model(
                scheduler_output, input_ids, positions, intermediate_tensors
            )

        if intermediate_tensors is not None and not self.is_first_rank:
            hidden_states = intermediate_tensors.get("hidden_states")
            if hidden_states is None:
                hidden_states = next(iter(intermediate_tensors.values()))
            if not isinstance(hidden_states, torch.Tensor):
                hidden_states = torch.from_numpy(np.array(hidden_states)).float()
            else:
                hidden_states = hidden_states.float()
            logits = self._run_iree(
                input_ids=input_ids,
                hidden_states=hidden_states.unsqueeze(0) if hidden_states.dim() == 2 else hidden_states,
                position_ids=positions.unsqueeze(0) if positions.dim() == 1 else positions,
            )
                
        
        else:
            # First rank or full model — use input_ids
            logits = self._run_iree(input_ids=input_ids)

        if self.is_last_rank:
            next_token_id = int(torch.argmax(logits[0, -1, :]).item())
            req_ids = list(self.requests.keys())
            return ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={r: i for i, r in enumerate(req_ids)},
                sampled_token_ids=[[next_token_id]] * len(req_ids),
            )
        else:
            # Return hidden states for next rank
            return {"hidden_states": logits}  # logits = hidden_states for non-last rank
        
    def _compile_mlp_iree(self) -> None:
        import iree.turbine.aot as aot
        import iree.compiler as iree_compiler
        import iree.runtime as ireert
        import numpy as np

        # Determine target: GPU CUDA or CPU llvm-cpu
        use_cpu = os.environ.get("IREE_USE_CPU_FFN_IREE", "0") == "1"
        if use_cpu:
            iree_target_backend = "llvm-cpu"
            iree_device_str = "local-task"
            target_arch = "host"
            extra_args = [
                "--iree-input-type=torch",
                "--iree-opt-level=O3",
                "--iree-llvmcpu-target-cpu=host",
                "--iree-llvmcpu-target-cpu-features=host",
                "--iree-llvmcpu-enable-vector-contract-custom-kernels",
                "--iree-llvmcpu-reassociate-fp-reductions",
            ]
            suffix = "cpu"
        else:
            iree_target_backend = IREE_TARGET
            iree_device_str = f"{IREE_TARGET}://0"
            target_arch = IREE_TARGET_ARCH.get(IREE_TARGET, "sm_70")
            extra_args = [
                f"--iree-cuda-target={target_arch}",
                "--iree-input-type=torch",
            ]
            suffix = target_arch

        def make_iree_forward(fn, cfg):
            def iree_mlp_forward(x: torch.Tensor) -> torch.Tensor:
                x_cpu = x.detach().cpu()
                iree_in = ireert.asdevicearray(cfg.device, x_cpu.numpy())
                out_np = np.array(fn(iree_in))
                return torch.from_numpy(out_np).to(x.device, x.dtype)
            return iree_mlp_forward

        seq_dim = torch.export.Dim("seq_len", min=1, max=4096)
        hidden = self.model_config.hf_config.hidden_size
        example = torch.randn(5, hidden)

        for layer_idx in range(self.layer_start, self.layer_end):
            model_short = self.model_config.model.replace(
                "/", "_").replace("-", "_").lower()
            vmfb_path = os.path.join(
                self.vmfb_dir,
                f"mlp_{model_short}_layer{layer_idx}_{suffix}.vmfb"
            )
            mlir_path = vmfb_path.replace(".vmfb", ".mlir")

            layer = self.model.model.layers[layer_idx]
            wrapper = _MLPWrapper(layer.mlp).cpu().eval()

            if not os.path.exists(vmfb_path) or self.force_recompile:
                logger.info("IREEModelRunner: compiling MLP layer %d (%s) -> %s",
                        layer_idx, suffix, vmfb_path)
                exported = aot.export(
                    wrapper, args=(example,),
                    dynamic_shapes={"x": {0: seq_dim}},
                )
                exported.save_mlir(mlir_path)
                iree_compiler.tools.compile_file(
                    mlir_path, output_file=vmfb_path,
                    target_backends=[iree_target_backend],
                    extra_args=extra_args,
                )
                logger.info("IREEModelRunner: MLP layer %d compiled.", layer_idx)
            else:
                logger.info("IREEModelRunner: cached MLP layer %d at %s",
                        layer_idx, vmfb_path)

            iree_config = ireert.Config(iree_device_str)
            ctx = ireert.SystemContext(config=iree_config)
            with open(vmfb_path, "rb") as f:
                vmfb = f.read()
            vm_module = ireert.VmModule.copy_buffer(ctx.instance, vmfb)
            ctx.add_vm_module(vm_module)
            iree_fn = ctx.modules.module["main"]

            layer.mlp._original_forward = layer.mlp.forward
            layer.mlp.forward = make_iree_forward(iree_fn, iree_config)
            logger.info("IREEModelRunner: IREE FFN (%s) patched on layer %d",
                    suffix, layer_idx)

        logger.info(
            "IREEModelRunner: IREE FFN dispatch (%s) active for layers %d-%d.",
            suffix, self.layer_start, self.layer_end - 1,
        )
        
        
    

    def _free_unused_layers(self) -> None:
        """Free GPU memory by replacing unused transformer layers with Identity."""
        import gc
        num_layers = len(self.model.model.layers)
        freed = 0
        for i in list(range(0, self.layer_start)) + \
                list(range(self.layer_end, num_layers)):
            self.model.model.layers[i] = torch.nn.Identity()
            freed += 1
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(
            "IREEModelRunner: freed %d unused layers, keeping [%d, %d). "
            "Call torch.cuda.memory_allocated() to verify.",
            freed, self.layer_start, self.layer_end,
        )
        
        
    def _patch_mlp_cpu(self) -> None:
        """
        Move FFN weights to CPU and patch mlp.forward to run entirely on CPU.
        Uses F.silu instead of vLLM's SiluAndMul to avoid GPU dependency.
        """
        import gc
        import torch.nn.functional as F

        def make_cpu_forward(mlp_module):
            # Extract weights once — move to CPU
            # gate_up_proj is fused [2*inter, hidden] — split manually
            w_gate_up = mlp_module.gate_up_proj.weight.data.cpu()  # [16384, 2048]
            w_down = mlp_module.down_proj.weight.data.cpu()        # [2048, 8192]
            inter = w_down.shape[1]  # 8192

            w_gate = w_gate_up[:inter].t()   # [2048, 8192]
            w_up   = w_gate_up[inter:].t()   # [2048, 8192]
            w_down_t = w_down.t()            # [8192, 2048]

            def cpu_mlp_forward(x: torch.Tensor) -> torch.Tensor:
                x_cpu = x.detach().cpu()
                gate = F.silu(x_cpu @ w_gate)   # [n, 8192]
                up   = x_cpu @ w_up             # [n, 8192]
                out  = (gate * up) @ w_down_t   # [n, 2048]
                return out.to(x.device, x.dtype)
            return cpu_mlp_forward

        for layer_idx in range(self.layer_start, self.layer_end):
            layer = self.model.model.layers[layer_idx]
            layer.mlp._original_forward = layer.mlp.forward
            layer.mlp.forward = make_cpu_forward(layer.mlp)
            logger.info("IREEModelRunner: CPU PyTorch FFN patched on layer %d",
                    layer_idx)

        gc.collect()
        torch.cuda.empty_cache()
        logger.info(
            "IREEModelRunner: CPU PyTorch FFN dispatch active for layers %d-%d.",
            self.layer_start, self.layer_end - 1,
        )
# ── IREE FFN dispatch ─────────────────────────────────────────────────────────

class _MLPWrapper(torch.nn.Module):
    """
    Plain nn.Module wrapping vLLM's fused LlamaMLP for IREE export.
    Splits gate_up_proj into separate gate and up projections.
    Shared across all layers — same shape for all Llama 1B layers.
    """
    def __init__(self, mlp_vllm):
        super().__init__()
        import torch.nn as nn
        w_gate_up = mlp_vllm.gate_up_proj.weight.data  # [2*inter, hidden]
        w_down    = mlp_vllm.down_proj.weight.data      # [hidden, inter]
        hidden    = w_down.shape[0]
        inter     = w_down.shape[1]
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj   = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.gate_proj.weight.data = w_gate_up[:inter].clone()
        self.up_proj.weight.data   = w_gate_up[inter:].clone()
        self.down_proj.weight.data = w_down.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up   = self.up_proj(x)
        return self.down_proj(gate * up)


