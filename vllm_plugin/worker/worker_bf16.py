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
import time
from vllm_plugin.profiling.rank_latency_profiling import PROFILING_ENABLED

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
        from iree.turbine.aot import externalize_module_parameters, save_module_parameters
        import ml_dtypes

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
                "--iree-llvmcpu-target-cpu-features=+bf16",
                "--iree-opt-data-tiling",
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
                orig_device = x.device          # cuda:0
                orig_dtype = x.dtype            # bf16
                x_cpu = x.detach().cpu()
                np_in = x_cpu.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
                iree_in = ireert.asdevicearray(cfg.device, np_in)
                out_np = np.array(fn(iree_in))
                # out = torch.from_numpy(out_np).to(torch.bfloat16)
                out = torch.from_numpy(out_np).to(
                device=orig_device, dtype=orig_dtype             # back to cuda:0, bf16
                )
                return out
            return iree_mlp_forward

        for layer_idx in range(layer_start, layer_end):
            model_short = self.model_config.model.replace(
                "/", "_").replace("-", "_").lower()
            vmfb_path = os.path.join(
                vmfb_dir,
                f"mlp_{model_short}_layer{layer_idx}_{suffix}.vmfb"
            )
            mlir_path = vmfb_path.replace(".vmfb", ".mlir")
            param_path = vmfb_path.replace(".vmfb", ".irpa")

            layer = model.model.layers[layer_idx]
            wrapper = _MLPWrapper(layer.mlp).cpu().eval()

            force_recompile = bool(os.environ.get("IREE_FORCE_RECOMPILE", ""))
            
            if not os.path.exists(vmfb_path) or not os.path.exists(param_path) or force_recompile:
                logger.info("IREEWorker: compiling MLP layer %d (%s) -> %s",
                           layer_idx, suffix, vmfb_path)
                
                example = example.to(torch.bfloat16)
                
                externalize_module_parameters(wrapper)
                
                exported = aot.export(
                    wrapper, args=(example,),
                    dynamic_shapes={"x": {0: seq_dim}},
                )
                
                save_module_parameters(param_path, wrapper)
                
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
            
            # 1. Load externalized parameters FIRST
            param_index = ireert.ParameterIndex()
            param_index.load(param_path)            
            param_provider = param_index.create_provider(scope="model")  # scope must match externalize default

            io_params_module = ireert.create_io_parameters_module(
                ctx.instance, param_provider
            )
            ctx.add_vm_module(io_params_module)
            
            
            with open(vmfb_path, "rb") as f:
                vmfb = f.read()
            vm_module = ireert.VmModule.copy_buffer(ctx.instance, vmfb)
            ctx.add_vm_module(vm_module)
            iree_fn = ctx.modules.module["main"]

            mem_before = torch.cuda.memory_allocated(0)

            layer.mlp._original_forward = layer.mlp.forward
            layer.mlp.forward = make_iree_forward(iree_fn, iree_config)

            # record the size of the GPU tensors we're about to release
            gate_up_bytes = (layer.mlp.gate_up_proj.weight.data.numel()
                             * layer.mlp.gate_up_proj.weight.data.element_size())
            down_bytes = (layer.mlp.down_proj.weight.data.numel()
                          * layer.mlp.down_proj.weight.data.element_size())

            # release GPU-resident FFN weights — the IREE vmfb owns its own copy
            layer.mlp.gate_up_proj.weight.data = torch.empty(0, device='cuda')
            layer.mlp.down_proj.weight.data = torch.empty(0, device='cuda')

            mem_after = torch.cuda.memory_allocated(0)

            logger.info(
                "IREEWorker: IREE FFN (%s) patched on layer %d | "
                "weights freed: %.1f MB (gate_up %.1f + down %.1f) | "
                "allocated %.1f MB -> %.1f MB (delta %.1f MB)",
                suffix, layer_idx,
                (gate_up_bytes + down_bytes) / 1024**2,
                gate_up_bytes / 1024**2,
                down_bytes / 1024**2,
                mem_before / 1024**2,
                mem_after / 1024**2,
                (mem_before - mem_after) / 1024**2,
            )

        import gc
        gc.collect()
        torch.cuda.empty_cache()

        free_after, _ = torch.cuda.mem_get_info(0)
        logger.info(
            "IREEWorker: IREE FFN dispatch (%s) active for layers %d-%d | "
            "allocated now %.1f MB | driver free now %.1f MB",
            suffix, layer_start, layer_end - 1,
            torch.cuda.memory_allocated(0) / 1024**2,
            free_after / 1024**2,
        )

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

            mem_before = torch.cuda.memory_allocated(0)

            layer.mlp._original_forward = layer.mlp.forward
            layer.mlp.forward = make_cpu_forward(layer.mlp)

            # record the size of the GPU tensors we're about to release
            gate_up_bytes = layer.mlp.gate_up_proj.weight.data.numel() * layer.mlp.gate_up_proj.weight.data.element_size()
            down_bytes = layer.mlp.down_proj.weight.data.numel() * layer.mlp.down_proj.weight.data.element_size()

            # release the GPU-resident FFN weights — the CPU module owns its own copies
            layer.mlp.gate_up_proj.weight.data = torch.empty(0, device='cuda')
            layer.mlp.down_proj.weight.data = torch.empty(0, device='cuda')

            mem_after = torch.cuda.memory_allocated(0)

            logger.info(
                "IREEWorker: CPU FFN patched on layer %d | "
                "weights to free: %.1f MB (gate_up %.1f + down %.1f) | "
                "allocated %.1f MB -> %.1f MB (delta %.1f MB)",
                layer_idx,
                (gate_up_bytes + down_bytes) / 1024**2,
                gate_up_bytes / 1024**2,
                down_bytes / 1024**2,
                mem_before / 1024**2,
                mem_after / 1024**2,
                (mem_before - mem_after) / 1024**2,
            )

        gc.collect()
        torch.cuda.empty_cache()

        # final snapshot after empty_cache returns pages to the driver
        free_after, _ = torch.cuda.mem_get_info(0)
        logger.info(
            "IREEWorker: CPU FFN done, layers %d-%d | "
            "allocated now %.1f MB | driver free now %.1f MB",
            layer_start, layer_end - 1,
            torch.cuda.memory_allocated(0) / 1024**2,
            free_after / 1024**2,
        )
        logger.info("IREEWorker: CPU FFN active for layers %d-%d.",
                   layer_start, layer_end - 1)
        
    def execute_model(self, scheduler_output):
        
        if PROFILING_ENABLED:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        
        output = super().execute_model(scheduler_output)
        if self.model_runner.input_batch is not None:
            for req in scheduler_output.scheduled_new_reqs:
                req_idx = self.model_runner.input_batch.req_id_to_index.get(req.req_id)
                if req_idx is not None:
                    n_tokens = len(req.prompt_token_ids)
                    self.model_runner.input_batch.num_tokens_no_spec[req_idx] = n_tokens
    
        if PROFILING_ENABLED:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._last_step_latency_ms = (time.perf_counter() - t0) * 1e3
            
        return output
    
    def get_last_step_latency_ms(self) -> float:
        return getattr(self, "_last_step_latency_ms", 0.0)

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
        y = self.down_proj(gate * up)
        return y.to(torch.float32)