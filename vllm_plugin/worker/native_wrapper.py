
from vllm.v1.worker.gpu_worker import Worker as _NativeWorker
import os
import time
from vllm_plugin.profiling.rank_latency_profiling import PROFILING_ENABLED
import torch


class NativeWorkerWithSend(_NativeWorker):
    """
    Thin subclass of gpu_worker.Worker used as rank 0 in HybridExecutor.
    Rank 0 runs layers [0, split) and sends IntermediateTensors to rank 1
    via NCCL — handled natively by gpu_worker.execute_model.
    """
    def init_device(self) -> None:
        super().init_device()
        
        
        
    def execute_model(self, scheduler_output):
        if PROFILING_ENABLED:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

        output = super().execute_model(scheduler_output)

        if PROFILING_ENABLED:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._last_step_latency_ms = (time.perf_counter() - t0) * 1e3

        return output

    def get_last_step_latency_ms(self) -> float:
        return getattr(self, "_last_step_latency_ms", 0.0)