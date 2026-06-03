"""
IREEWorker — vLLM worker that dispatches inference to the IREE runtime.
 
vLLM calls this class at every step of the engine loop:
  init_device -> load_model -> determine_available_memory ->
  initialize_cache -> compile_or_warm_up_model -> execute_model (loop)
"""
 
import torch
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
        self.model_runner = IREEModelRunner(self.vllm_config)

    def init_device(self) -> None:
        import os
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        init_distributed_environment(
            world_size=self.parallel_config.world_size,
            rank=self.rank,
            distributed_init_method=self.distributed_init_method,
            local_rank=self.local_rank,
            backend="gloo",
        )

        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
        )

        set_random_seed(self.model_config.seed)
        self.device = torch.device("cuda:0")
        logger.info("IREEWorker: device initialised (%s)", self.device)
    
    def load_model(self) -> None:
        """Load model weights and compile to .vmfb via iree-turbine."""
        self.model_runner.load_model()


    def determine_available_memory(self) -> int:
        """
        Return available memory in bytes for KV cache allocation.
        IREE manages GPU memory internally, so we return a synthetic
        value that keeps vLLM's scheduler happy.
        The factor of 2 gives the scheduler headroom for its null-block.
        """
        return (
            4                                   # bytes per token (fp32)
            * self.model_config.max_model_len
            * self.scheduler_config.max_num_seqs
            * 2                                 # headroom factor
        )
 

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
        """
        Called at every engine step.
        Non-driver workers return None — only the driver collects output.
        """
        if not scheduler_output.total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT
 
        output = self.model_runner.execute_model(scheduler_output)
        return output if self.is_driver_worker else None
 
    def sample_tokens(self, grammar_output: object) -> object:
        # Required by WorkerBase interface; sampling happens inside
        # execute_model for now.
        return EMPTY_MODEL_RUNNER_OUTPUT
    
    def check_health(self) -> None:
        # TODO: ping iree.runtime to verify the device is still alive.
        return


    

# Future (Phase 3 HybridExecutor)
#
# _init_worker_distributed_environment  — explicit multi-device init
# profile                               — torch profiler hook
