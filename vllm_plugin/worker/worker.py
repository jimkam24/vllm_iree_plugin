"""
IREEWorker — vLLM worker that dispatches inference to the IREE runtime.
 
vLLM calls this class at every step of the engine loop:
  init_device -> load_model -> determine_available_memory ->
  initialize_cache -> compile_or_warm_up_model -> execute_model (loop)
"""
 
import torch
import os
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.utils.torch_utils import set_random_seed
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.v1.worker.worker_base import WorkerBase
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput, EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.logger import init_logger
 
from vllm_plugin.worker.model_runner import IREEModelRunner
 
logger = init_logger(__name__)
 
 
class IREEWorker(WorkerBase):
    """
    Execution engine that vLLM talks to at every inference step.
    For Phase 1 this runs on a single GPU via IREE targeting cuda.
    Phase 2 will swap the IREE compile target to something else (?).
    """


    # create model runner and store config
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        # model_runner is created here so other methods can reference it,
        # but load_model() is where weights are actually loaded.
        self.model_runner = IREEModelRunner(
            self.vllm_config,
            force_recompile=bool(os.environ.get("IREE_FORCE_RECOMPILE", "")),
            vmfb_dir=os.environ.get("IREE_VMFB_DIR", "/tmp/iree_artifacts"),
        )

    def init_device(self) -> None:
        import os
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        init_distributed_environment(
            world_size=self.parallel_config.world_size,
            rank=self.rank,
            distributed_init_method=self.distributed_init_method,
            local_rank=self.local_rank,
            backend="nccl",
        )

        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
        )

        set_random_seed(self.model_config.seed)
        # With CUDA_VISIBLE_DEVICES=<rank>, the worker sees only 1 GPU
        # which always appears as cuda:0 within its own process.
        self.device = torch.device("cuda:0")
        logger.info("IREEWorker: device initialised (%s)", self.device)
    
    def load_model(self) -> None:
        """Load model weights and compile to .vmfb via iree-turbine."""
        self.model_runner.load_model()


    def determine_available_memory(self) -> int:
        """
        Return available memory in bytes for KV cache allocation.
        IREE manages GPU memory internally so we can't profile it the
        same way the native worker does. Instead we return the actual
        free GPU memory on our device, which gives get_kv_cache_configs
        a realistic value to work with.
        """
        free, total = torch.cuda.mem_get_info(0)  # cuda:0 within this process
        # Apply gpu_memory_utilization fraction, then subtract model weights
        usable = int(free * self.cache_config.gpu_memory_utilization)
        return max(usable, 0)
 

    def initialize_cache(
        self, num_gpu_blocks: int, num_cpu_blocks: int
    ) -> None:
        """Store block counts that the scheduler computed."""
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks


    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Allocate the actual KV cache using the config the engine computed.
        TODO (Phase 1 Step 3): wire up IREE-side KV buffer allocation here.
        """
        pass


    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Delegate KV cache layout description to the model runner."""
        return self.model_runner.get_kv_cache_spec()


    def compile_or_warm_up_model(self) -> None:
        """
        Trigger any ahead-of-time compilation or warmup runs.
        The .vmfb is already compiled during load_model(); this is a
        placeholder for any additional warmup forward passes needed later.
        """
        self.model_runner.warm_up()

   
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput | None:
        if not scheduler_output.total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT

        from vllm.distributed.parallel_state import get_pp_group
        from vllm.v1.outputs import SamplerOutput
        pp_group = get_pp_group()
        intermediate_tensors = None

        # ── Receive from previous rank (if not first) ─────────────────────
        if not pp_group.is_first_rank:
            from vllm.distributed.parallel_state import get_tp_group
            tensor_dict = pp_group.recv_tensor_dict(
                all_gather_group=get_tp_group(),
                all_gather_tensors={},
            )
            intermediate_tensors = tensor_dict  # pass to model runner later

        # ── Run forward pass ──────────────────────────────────────────────
        output = self.model_runner.execute_model(
            scheduler_output,
            intermediate_tensors=intermediate_tensors,
        )

        # ── Send to next rank (if not last) ───────────────────────────────
        # output is IntermediateTensors when we are not the last rank
        if not pp_group.is_last_rank:
            assert isinstance(output, dict), (
                "Expected IntermediateTensors dict from model runner "
                "when not last PP rank"
            )
            pp_group.send_tensor_dict(output, all_gather_group=None)
            return None  # intermediate ranks don't return final output

        # ── Last rank returns final output ────────────────────────────────
        return output if self.is_driver_worker else None
    
    def sample_tokens(self, grammar_output: object) -> object:
        # Required by WorkerBase interface; sampling happens inside
        # execute_model for now.
        return EMPTY_MODEL_RUNNER_OUTPUT
    
    #TODO: ping IREE worker to see if it still is ok
    def check_health(self) -> None:
        from vllm.distributed.parallel_state import get_pp_group
        pp = get_pp_group()
        logger.info(
            "check_health: rank=%d pp.ranks=%s pp.world_size=%d "
            "is_first=%s is_last=%s",
            self.rank, pp.ranks, pp.world_size,
            pp.is_first_rank, pp.is_last_rank,
        )


    

# Future (Phase 3 HybridExecutor)
#
# _init_worker_distributed_environment  — explicit multi-device init
# profile                               — torch profiler hook
