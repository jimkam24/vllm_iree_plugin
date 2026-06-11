"""
HybridExecutor — Ray-based heterogeneous pipeline-parallel executor.

Extends RayDistributedExecutor to dispatch different worker classes
per pipeline rank:

  Rank 0: native vLLM Worker (V100S, CUDA) — transformer layers [0, split)
  Rank 1: IREEWorker (A2, IREE/CUDA)       — transformer layers [split, N)

The layer split is controlled by VLLM_PP_LAYER_PARTITION env var.
Cross-rank activation transfer uses gloo (torch.distributed).

Usage — set via EngineArgs:
    distributed_executor_backend = "vllm_plugin.executor.hybrid_executor.HybridExecutor"
    pipeline_parallel_size = 2

Phase 1: both GPUs on same node, Ray manages process isolation.
Phase 2: rank 1 points at NPU via IREE amd-aie target, same executor code.
"""

import os
from typing import Any

from vllm.logger import init_logger
from vllm.v1.executor.ray_executor import RayDistributedExecutor, RayWorkerWrapper
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

# Worker class paths — change rank 1 to IREEWorker
# _RANK0_WORKER_CLS = "vllm.v1.worker.gpu_worker.Worker"
_RANK0_WORKER_CLS = "vllm_plugin.worker.native_wrapper.NativeWorkerWithSend"
_RANK1_WORKER_CLS = "vllm_plugin.worker.worker.IREEWorker"


class HybridExecutor(RayDistributedExecutor):
    """
    Heterogeneous pipeline-parallel executor.

    Inherits all Ray infrastructure from RayDistributedExecutor.
    Only overrides worker class assignment per rank.

    Each rank runs in its own Ray actor (separate process), so:
    - Rank 0 loads CUDAPlatform naturally
    - Rank 1 loads IREEPlatform naturally
    - No platform stub conflicts
    - gloo rendezvous works correctly across processes
    """

    supports_pp: bool = True

    def _init_workers_ray(
        self,
        placement_group: "PlacementGroup",
        **ray_remote_kwargs,
    ) -> None:
        """
        Override to inject per-rank worker classes.

        Strategy: call super() to create all Ray actors and run the
        full worker init sequence, but patch each rank's kwargs to
        use the correct worker_cls before init_worker is called.

        We do this by overriding the collective_rpc("init_worker") call
        by temporarily monkey-patching _init_workers_ray to intercept
        the all_kwargs list.
        """
        # Store original collective_rpc so we can intercept init_worker
        original_collective_rpc = self.collective_rpc

        def patched_collective_rpc(method, timeout=None, args=(), kwargs=None,
                                    non_block=False, **kw):
            # Parse IREE worker ranks once at the top — supports multiple IREE ranks
            iree_worker_ranks = set(
                int(x.strip())
                for x in os.environ.get("IREE_WORKER_RANKS", "1").split(",")
            )

            if method == "update_environment_variables" and args:
                all_env_vars = list(args[0])
                gpu_assignment = os.environ.get("IREE_GPU_ASSIGNMENT", "")
                gpu_indices = [x.strip() for x in gpu_assignment.split(",")] if gpu_assignment else None

                for rank, env_dict in enumerate(all_env_vars):
                    if gpu_indices and rank < len(gpu_indices):
                        env_dict["CUDA_VISIBLE_DEVICES"] = gpu_indices[rank]
                    else:
                        env_dict["CUDA_VISIBLE_DEVICES"] = str(rank)
                    env_dict["MY_PP_RANK"] = str(rank)
                    # Send the full set string so workers can reconstruct it
                    env_dict["IREE_WORKER_RANKS"] = os.environ.get("IREE_WORKER_RANKS", "1")
                args = (all_env_vars,)
                logger.info(
                    "HybridExecutor: injected per-rank CUDA_VISIBLE_DEVICES into "
                    "update_environment_variables"
                )

            if method == "init_worker" and args:
                all_kwargs = list(args[0])
                from vllm.v1.attention.backends.registry import AttentionBackendEnum
                import copy

                for rank, wk in enumerate(all_kwargs):
                    logger.info(
                        "init_worker rank %d worker_cls=%s",
                        rank,
                        wk["vllm_config"].parallel_config.worker_cls,
                    )

                logger.info("HybridExecutor: injecting per-rank worker classes...")

                for rank, worker_kwargs in enumerate(all_kwargs):
                    if rank in iree_worker_ranks:
                        worker_cls = _RANK1_WORKER_CLS  # IREEWorker
                    else:
                        worker_cls = _RANK0_WORKER_CLS  # native gpu_worker

                    cfg = copy.deepcopy(worker_kwargs["vllm_config"])
                    cfg.parallel_config.worker_cls = worker_cls
                    cfg.parallel_config.distributed_executor_backend = "ray"

                    if rank in iree_worker_ranks:
                        cfg.attention_config.backend = AttentionBackendEnum.CUSTOM
                    else:
                        cfg.attention_config.backend = AttentionBackendEnum.TRITON_ATTN

                    if cfg.additional_config is None:
                        cfg.additional_config = {}
                    cfg.additional_config["hybrid_pp_rank"] = rank
                    cfg.additional_config["hybrid_pp_world_size"] = len(all_kwargs)

                    all_kwargs[rank] = {
                        **worker_kwargs,
                        "vllm_config": cfg,
                        "local_rank": 0,
                        "is_driver_worker": (rank == len(all_kwargs) - 1),
                    }
                    logger.info(
                        "  rank %d -> %s (layers: hybrid_pp_rank=%d/%d)",
                        rank, worker_cls, rank, len(all_kwargs),
                    )

                args = (all_kwargs,)

            return original_collective_rpc(
                method, timeout=timeout, args=args,
                kwargs=kwargs, non_block=non_block, **kw
            )

        # Temporarily replace collective_rpc during super()._init_workers_ray
        self.collective_rpc = patched_collective_rpc
        try:
            super()._init_workers_ray(placement_group, **ray_remote_kwargs)      
        finally:
            # Always restore original collective_rpc
            self.collective_rpc = original_collective_rpc
            

        logger.info("HybridExecutor: both workers initialised via Ray.")

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        non_block: bool = False,
    ) -> ModelRunnerOutput | None:
        """
        Pipeline-parallel forward pass.

        Current (Phase 1 baseline):
          Both workers run full forward independently.
          Output comes from rank 0 (driver worker).

        TODO (Step 3b — activation passing):
          1. Rank 0 runs layers [0, split) → produces intermediate activations
          2. Transfer activations rank0 → rank1 via gloo
          3. Rank 1 runs layers [split, N) → produces logits + sampled tokens
          4. Return rank1 output

        Profiling hooks for Step 4 go here:
          - t_rank0 = time rank0 forward
          - t_transfer = time gloo transfer
          - t_rank1 = time rank1 forward
          - record to metrics collector
        """
        import time
        t_start = time.perf_counter()

        # Phase 1: forward pass on all workers
        outputs = self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
        )

        # Phase 2: sample tokens on all workers
        # gpu_model_runner uses async two-phase execution:
        # execute_model() stores state and returns None,
        # sample_tokens() reads state and returns ModelRunnerOutput.
        # We must call sample_tokens() on all workers after execute_model().
        sample_outputs = self.collective_rpc(
            "sample_tokens",
            args=(None,),  # grammar_output=None (no structured output)
        )

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.debug("HybridExecutor.execute_model: %.2f ms", t_elapsed_ms)

        # Return the first non-empty output from sample_tokens.
        # The last PP rank produces the final ModelRunnerOutput.
        for output in sample_outputs:
            if output is not None and output.req_ids:
                return output
        return None

    # ── Profiling hooks (Step 4) ──────────────────────────────────────────────
    #
    # _measure_rank_latencies(outputs)   — per-rank forward pass timing
    # _measure_gloo_transfer(src, dst)   — activation transfer overhead
    # _decompose_ttft(t_rank0, t_xfer, t_rank1) — TTFT split across ranks