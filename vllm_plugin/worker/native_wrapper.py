
from vllm.v1.worker.gpu_worker import Worker as _NativeWorker
import os


class NativeWorkerWithSend(_NativeWorker):
    """
    Thin subclass of gpu_worker.Worker used as rank 0 in HybridExecutor.
    Rank 0 runs layers [0, split) and sends IntermediateTensors to rank 1
    via NCCL — handled natively by gpu_worker.execute_model.
    """
    def init_device(self) -> None:
        super().init_device()