import sys
from vllm.v1.worker.gpu_worker import Worker as _NativeWorker


class NativeWorkerWithSend(_NativeWorker):
    """
    Thin subclass of gpu_worker.Worker used as rank 0 in HybridExecutor.

    For Path B (IREE-IREE): execute_model_state.hidden_states would need
    to be sent here — but Path B uses IREEWorker for both ranks anyway.

    For Path A1 (IREE_USE_VLLM_MODEL=1): gpu_worker.Worker already sends
    IntermediateTensors via NCCL internally. No override needed.
    """
    pass