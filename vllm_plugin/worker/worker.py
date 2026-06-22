"""
IREEWorker — extends native vLLM gpu_worker with IREE/CPU FFN dispatch.

All attention, KV cache, metadata, PP boundary, and torch.compile handled
natively by gpu_worker + gpu_model_runner.

IREEWorker only patches FFN layers after model loading for heterogeneous
dispatch to IREE (GPU/CPU) or plain CPU (BLAS).

Phase 1: FFN dispatch to IREE cuda or llvm-cpu target.
Phase 2: change IREE_TARGET to amd-aie for NPU dispatch.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vllm.v1.worker.gpu_worker import Worker as _NativeWorker
from vllm.logger import init_logger

logger = init_logger(__name__)

IREE_TARGET = os.environ.get("IREE_TARGET", "cuda")
IREE_TARGET_ARCH = {
    "cuda": os.environ.get("IREE_CUDA_ARCH", "sm_86"),
    "amd-aie": "aie",
}


class IREEWorker(_NativeWorker):
    """
    Native gpu_worker extended with optional IREE/CPU FFN dispatch.

    Without FFN flags: identical to native gpu_worker (Triton attention,
    native gpu_model_runner, PP boundary handled natively).

    With IREE_USE_FFN=1: FFN layers patched to dispatch through IREE.
    With IREE_USE_CPU_FFN=1: FFN layers patched to run on CPU (BLAS).
    """

    def init_device(self) -> None:
        # Native init — no custom attention, use Triton
        super().init_device()
        logger.info("IREEWorker: device initialised (%s)", self.device)

    def load_model(self) -> None:
        """Load model via native gpu_model_runner, then patch FFN if requested."""
        super().load_model()
        logger.info("IREEWorker: model loaded via native gpu_model_runner.")

        if os.environ.get("IREE_USE_FFN", "0") == "1":
            self._compile_mlp_iree()
        elif os.environ.get("IREE_USE_CPU_FFN", "0") == "1":
            self._patch_mlp_cpu()

    def _get_model(self):
        """Access the underlying vLLM model from gpu_model_runner."""
        return self.model_runner.model

    def _get_layer_range(self):
        """Get layer range owned by this PP rank (set by vLLM's make_layers)."""
        model = self._get_model()
        return model.model.start_layer, model.model.end_layer

    def _compile_mlp_iree(self) -> None:
        """Export FFN to IREE vmfb and patch mlp.forward for dispatch."""
        import iree.turbine.aot as aot
        import iree.compiler as iree_compiler
        import iree.runtime as ireert

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
            target_arch = IREE_TARGET_ARCH.get(IREE_TARGET, "sm_86")
            extra_args = [
                f"--iree-cuda-target={target_arch}",
                "--iree-input-type=torch",
            ]
            suffix = target_arch

        model = self._get_model()
        layer_start, layer_end = self._get_layer_range()
        hidden = self.model_config.hf_config.hidden_size
        vmfb_dir = os.environ.get("IREE_VMFB_DIR", "/tmp/iree_artifacts")
        os.makedirs(vmfb_dir, exist_ok=True)

        seq_dim = torch.export.Dim("seq_len", min=1, max=4096)
        example = torch.randn(5, hidden)

        def make_iree_forward(fn, cfg):
            def iree_mlp_forward(x: torch.Tensor) -> torch.Tensor:
                x_cpu = x.detach().cpu()
                iree_in = ireert.asdevicearray(cfg.device, x_cpu.numpy())
                out_np = np.array(fn(iree_in))
                return torch.from_numpy(out_np).to(x.device, x.dtype)
            return iree_mlp_forward

        for layer_idx in range(layer_start, layer_end):
            model_short = self.model_config.model.replace(
                "/", "_").replace("-", "_").lower()
            vmfb_path = os.path.join(
                vmfb_dir,
                f"mlp_{model_short}_layer{layer_idx}_{suffix}.vmfb"
            )
            mlir_path = vmfb_path.replace(".vmfb", ".mlir")

            layer = model.model.layers[layer_idx]
            wrapper = _MLPWrapper(layer.mlp).cpu().eval()

            force_recompile = bool(os.environ.get("IREE_FORCE_RECOMPILE", ""))
            if not os.path.exists(vmfb_path) or force_recompile:
                logger.info("IREEWorker: compiling MLP layer %d (%s) -> %s",
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
                logger.info("IREEWorker: MLP layer %d compiled.", layer_idx)
            else:
                logger.info("IREEWorker: cached MLP layer %d at %s",
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
            logger.info("IREEWorker: IREE FFN (%s) patched on layer %d",
                       suffix, layer_idx)

        logger.info("IREEWorker: IREE FFN dispatch (%s) active for layers %d-%d.",
                   suffix, layer_start, layer_end - 1)

    def _patch_mlp_cpu(self) -> None:
        """Move FFN weights to CPU and patch mlp.forward to run on CPU (BLAS)."""
        import gc

        model = self._get_model()
        layer_start, layer_end = self._get_layer_range()

        def make_cpu_forward(mlp_module):
            w_gate_up = mlp_module.gate_up_proj.weight.data.cpu()
            w_down = mlp_module.down_proj.weight.data.cpu()
            inter = w_down.shape[1]
            w_gate = w_gate_up[:inter].t()
            w_up   = w_gate_up[inter:].t()
            w_down_t = w_down.t()

            def cpu_mlp_forward(x: torch.Tensor) -> torch.Tensor:
                x_cpu = x.detach().cpu()
                gate = F.silu(x_cpu @ w_gate)
                up   = x_cpu @ w_up
                out  = (gate * up) @ w_down_t
                return out.to(x.device, x.dtype)
            return cpu_mlp_forward

        for layer_idx in range(layer_start, layer_end):
            layer = model.model.layers[layer_idx]
            layer.mlp._original_forward = layer.mlp.forward
            layer.mlp.forward = make_cpu_forward(layer.mlp)
            logger.info("IREEWorker: CPU FFN patched on layer %d", layer_idx)

        gc.collect()
        torch.cuda.empty_cache()
        logger.info("IREEWorker: CPU FFN active for layers %d-%d.",
                   layer_start, layer_end - 1)
        
    def execute_model(self, scheduler_output):
        output = super().execute_model(scheduler_output)
        if self.model_runner.input_batch is not None:
            for req in scheduler_output.scheduled_new_reqs:
                req_idx = self.model_runner.input_batch.req_id_to_index.get(req.req_id)
                if req_idx is not None:
                    n_tokens = len(req.prompt_token_ids)
                    self.model_runner.input_batch.num_tokens_no_spec[req_idx] = n_tokens
        return output

    def check_health(self) -> None:
        from vllm.distributed.parallel_state import get_pp_group
        pp = get_pp_group()
        logger.info(
            "IREEWorker check_health: rank=%d is_first=%s is_last=%s",
            self.rank, pp.is_first_rank, pp.is_last_rank,
        )


# ── MLP wrapper for IREE export ───────────────────────────────────────────────

class _MLPWrapper(nn.Module):
    """Plain nn.Module for IREE AOT export — splits fused gate_up_proj."""

    def __init__(self, mlp_vllm):
        super().__init__()
        w_gate_up = mlp_vllm.gate_up_proj.weight.data
        w_down    = mlp_vllm.down_proj.weight.data
        hidden    = w_down.shape[0]
        inter     = w_down.shape[1]
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj   = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.gate_proj.weight.data = w_gate_up[:inter].clone()
        self.up_proj.weight.data   = w_gate_up[inter:].clone()
        self.down_proj.weight.data = w_down.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up   = self.up_proj(x)
        return self.down_proj(gate * up)