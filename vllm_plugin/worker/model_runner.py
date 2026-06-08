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
"""

import os
import torch
import torch.nn as nn
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
    model_name: str,        # e.g. "meta-llama/Llama-3.2-1B"
    vmfb_path: str,
    mlir_path: str,
    vllm_config,
    target: str = "cuda",
    target_arch: str = "sm_70",
) -> None:
    """
    Export model to MLIR via iree-turbine and compile to .vmfb.
    Uses dynamic shapes so the artifact works for any sequence length.
    """
    import iree.turbine.aot as aot
    import iree.compiler as iree_compiler
    import torch.export as torch_export
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        attn_implementation="eager",   # decompose attention, no SDPA op
    ).cpu().eval()

    class ForwardWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m

        def forward(self, input_ids):
            out = self.model(input_ids=input_ids)
            return out.logits

    wrapper = ForwardWrapper(hf_model).eval()

    # Example input for tracing — seq_len=6 is arbitrary
    example_input_ids = torch.zeros((1, 6), dtype=torch.long)

    dynamic_shapes = {
        "input_ids": {1: torch_export.Dim("seq_len", min=1, max=4096)},
    }

    # Patch causal mask before export, restore in finally
    original_fn, llama_module = _patch_causal_mask()
    try:
        exported = aot.export(
            wrapper,
            args=(example_input_ids,),
            dynamic_shapes=dynamic_shapes,
        )
        exported.save_mlir(mlir_path)
        logger.info("IREEModelRunner: MLIR saved to %s", mlir_path)
    finally:
        _restore_causal_mask(original_fn, llama_module)

    logger.info(
        "IREEModelRunner: compiling to .vmfb (target=%s arch=%s)...",
        target, target_arch,
    )
    iree_compiler.tools.compile_file(
        mlir_path,
        output_file=vmfb_path,
        target_backends=[target],
        extra_args=[
            f"--iree-cuda-target={target_arch}",
            "--iree-input-type=torch",
        ],
    )
    logger.info("IREEModelRunner: .vmfb saved to %s", vmfb_path)


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
            example_input = torch.zeros((1, 6), dtype=torch.long)
            _export_and_compile(
                model_name=self.model_config.model,
                vmfb_path=vmfb_path,
                mlir_path=mlir_path,
                vllm_config=self.vllm_config,
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
        """.vmfb already compiled in load_model() — nothing to do."""
        logger.info("IREEModelRunner: warmup complete (artifact pre-compiled).")

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

    def _run_iree(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Dispatch a forward pass through the IREE runtime.
        Returns logits as a CPU torch.Tensor [1, seq_len, vocab_size].
        """
        import iree.runtime as ireert
        input_np = input_ids.cpu().numpy().astype(np.int64).reshape(1, -1)
        iree_input = ireert.asdevicearray(self.iree_config.device, input_np)
        logits_iree = self.iree_fn(iree_input)
        return torch.from_numpy(np.array(logits_iree))

    # ── Main inference entry point ────────────────────────────────────────────

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors=None,  # activations from previous PP rank
    ) -> ModelRunnerOutput:
        if not scheduler_output.total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT

        self._update_states(scheduler_output)
        input_ids, _positions = self._prepare_inputs(scheduler_output)

        if intermediate_tensors is not None:
            # TODO (Step 3b): use intermediate_tensors as input to IREE
            # instead of input_ids. For now we still run full model via
            # IREE independently — this just unblocks the deadlock.
            pass

        logits = self._run_iree(input_ids)
        next_token_id = int(torch.argmax(logits[0, -1, :]).item())

        req_ids = list(self.requests.keys())
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=[[next_token_id]] * len(req_ids),
        )