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
 
    hidden_size = hf_model.config.hidden_size  # 2048 for Llama 1B
 
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
            _export_and_compile(
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

        iree_device_str = f"{IREE_TARGET}://0"
        self.iree_config, self.iree_fn = _load_vmfb(vmfb_path, iree_device_str)
        logger.info("IREEModelRunner: IREE runtime ready.")

    # ── KV cache interface ────────────────────────────────────────────────────

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return {
            "iree_attn": FullAttentionSpec(
                self.cache_config.block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float16,
            )
        }

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """TODO (Step 3): allocate and bind IREE-side KV buffers."""
        pass

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
            self.requests[req.req_id] = {
                "prompt_token_ids": req.prompt_token_ids,
                "num_computed_tokens": 0,
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

    # ── IREE dispatch ─────────────────────────────────────────────────────────

    def _run_iree(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.is_first_rank or (self.is_first_rank and self.is_last_rank):
            return _run_iree_first_or_full(
                self.iree_config, self.iree_fn, input_ids
            )
        else:
            return _run_iree_middle_or_last(
                self.iree_config, self.iree_fn, hidden_states, position_ids
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

        if intermediate_tensors is not None and not self.is_first_rank:
            # Receive real hidden states from previous rank
            hidden_states = intermediate_tensors.get("hidden_states")
            
            hs = intermediate_tensors.get("hidden_states")
            res = intermediate_tensors.get("residual")
            combined = (hs + res) if res is not None else hs
            import sys
            print(f"[IREE recv] hs mean={hs.float().mean():.6f}", file=sys.stderr, flush=True)
            if res is not None:
                print(f"[IREE recv] res mean={res.float().mean():.6f}", file=sys.stderr, flush=True)
                print(f"[IREE recv] combined mean={combined.float().mean():.6f}", file=sys.stderr, flush=True)
            # Quick sanity: what token would HF layers 14-15 produce from this?
            from transformers import AutoModelForCausalLM, AutoConfig
            import torch.nn as nn
            hf_config = AutoConfig.from_pretrained('meta-llama/Llama-3.2-1B')
            hf_config._attn_implementation = 'eager'
            hf_m = AutoModelForCausalLM.from_pretrained(
                'meta-llama/Llama-3.2-1B', config=hf_config, torch_dtype=torch.float32
            ).to(combined.device).eval()
            pos_ids = torch.arange(combined.shape[0], device=combined.device).unsqueeze(0)
            h = combined.unsqueeze(0).float() if combined.dim() == 2 else combined.float()
            pos_emb = hf_m.model.rotary_emb(h, pos_ids)
            seq_len = h.shape[1]
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=h.device))
            cm = torch.zeros(1,1,seq_len,seq_len,dtype=h.dtype,device=h.device)
            cm = cm.masked_fill(~mask.unsqueeze(0).unsqueeze(0), torch.finfo(h.dtype).min)
            with torch.no_grad():
                for layer in hf_m.model.layers[14:]:
                    h = layer(h, attention_mask=cm, position_ids=pos_ids, position_embeddings=pos_emb)
                h = hf_m.model.norm(h)
                logits = hf_m.lm_head(h)
                token = int(torch.argmax(logits[0,-1,:]).item())
            print(f"[IREE recv] HF completion token from received tensor: {token}", file=sys.stderr, flush=True)
            
            import sys
            print(f"[IREE rank1] received hidden_states shape: {hidden_states.shape}, "
                f"dtype: {hidden_states.dtype}, "
                f"mean: {hidden_states.float().mean().item():.4f}", 
                file=sys.stderr, flush=True)
                    
            
            if hidden_states is None:
                # fallback: try first value
                hidden_states = next(iter(intermediate_tensors.values()))
            hidden_states = torch.from_numpy(
                np.array(hidden_states)
            ).float() if not isinstance(hidden_states, torch.Tensor) else hidden_states.float()
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