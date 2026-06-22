
from vllm.v1.worker.gpu_worker import Worker as _NativeWorker
import os


class NativeWorkerWithSend(_NativeWorker):
    """
    Thin subclass of gpu_worker.Worker used as rank 0 in HybridExecutor.
    """
    def init_device(self) -> None:
        super().init_device()