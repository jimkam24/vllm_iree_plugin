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
_RANK0_WORKER_CLS = "vllm.v1.worker.gpu_worker.Worker"
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
            # Intercept only the init_worker call
            if method == "init_worker" and args:
                all_kwargs = list(args[0])
                
                for rank, wk in enumerate(all_kwargs):
                    logger.info(
                        "init_worker rank %d worker_cls=%s",
                        rank,
                        wk["vllm_config"].parallel_config.worker_cls,
                    )
                        
                logger.info(
                    "HybridExecutor: injecting per-rank worker classes..."
                )
                for rank, worker_kwargs in enumerate(all_kwargs):
                    if rank == 0:
                        worker_cls = _RANK0_WORKER_CLS
                    else:
                        worker_cls = _RANK1_WORKER_CLS

                    # Clone config and set worker_cls for this rank
                    import copy
                    cfg = copy.deepcopy(worker_kwargs["vllm_config"])
                    cfg.parallel_config.worker_cls = worker_cls

                    # Tell gpu_worker it's running under Ray so it skips the
                    # local_world_size <= visible_device_count assertion.
                    # The assertion only fires when backend is not "ray".
                    cfg.parallel_config.distributed_executor_backend = "ray"

                    # Embed PP rank for layer partition in IREEModelRunner
                    if cfg.additional_config is None:
                        cfg.additional_config = {}
                    cfg.additional_config["hybrid_pp_rank"] = rank
                    cfg.additional_config["hybrid_pp_world_size"] = len(all_kwargs)

                    all_kwargs[rank] = {**worker_kwargs, "vllm_config": cfg}
                    logger.info(
                        "  rank %d -> %s (layers: hybrid_pp_rank=%d/%d)",
                        rank, worker_cls, rank, len(all_kwargs),
                    )

                args = (all_kwargs,)
                
                
            if method == "update_environment_variables" and args:
                # Inject per-rank CUDA_VISIBLE_DEVICES before Ray sets env vars
                all_env_vars = list(args[0])
                iree_worker_rank = int(os.environ.get("IREE_WORKER_RANK", "1"))
                for rank, env_dict in enumerate(all_env_vars):
                    env_dict["CUDA_VISIBLE_DEVICES"] = str(rank)
                    env_dict["MY_PP_RANK"] = str(rank)
                    env_dict["IREE_WORKER_RANK"] = str(iree_worker_rank)
                args = (all_env_vars,)
                logger.info(
                    "HybridExecutor: injected per-rank CUDA_VISIBLE_DEVICES "
                    "into update_environment_variables"
                )

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
            
        if hasattr(self, '_env_vars_for_all_workers'):
            iree_worker_rank = int(os.environ.get("IREE_WORKER_RANK", "1"))
            
            # Override CUDA_VISIBLE_DEVICES per rank so each worker sees only its GPU.
            # Without this, all workers on the same node see all GPUs and NCCL
            # reports duplicate GPU error.
            for rank, env_dict in enumerate(self._env_vars_for_all_workers):
                env_dict["CUDA_VISIBLE_DEVICES"] = str(rank)  # rank 0 → GPU 0, rank 1 → GPU 1
                env_dict["MY_PP_RANK"] = str(rank)
                env_dict["IREE_WORKER_RANK"] = str(iree_worker_rank)
            
            self.collective_rpc(
                "update_environment_variables",
                args=(self._get_env_vars_to_be_updated(),),
            )
            logger.info(
                "HybridExecutor: per-rank GPU isolation set — "
                "rank 0 → CUDA_VISIBLE_DEVICES=0, rank 1 → CUDA_VISIBLE_DEVICES=1"
            )

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

        outputs = self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
        )

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.debug("HybridExecutor.execute_model: %.2f ms", t_elapsed_ms)

        # Rank 0 (first PP rank) returns None — it sends activations to rank 1.
        # Rank 1 (last PP rank) returns the final ModelRunnerOutput.
        # outputs list is ordered by rank, so outputs[1] is IREEWorker's output.
        for output in reversed(outputs):
            if output is not None:
                return output
        return None

    # ── Profiling hooks (Step 4) ──────────────────────────────────────────────
    #
    # _measure_rank_latencies(outputs)   — per-rank forward pass timing
    # _measure_gloo_transfer(src, dst)   — activation transfer overhead
    # _decompose_ttft(t_rank0, t_xfer, t_rank1) — TTFT split across ranks