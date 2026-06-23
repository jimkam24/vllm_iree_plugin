"""
Definitive rank → physical-GPU probe.

Each worker reports its own PCI bus ID. We match that against nvidia-smi
ground truth, so there is NO ambiguity from CUDA_VISIBLE_DEVICES reindexing.

nvidia-smi ground truth (from the shell):
    index 0  NVIDIA A2     bus 00000000:3B:00.0   15 GB
    index 1  V100S         bus 00000000:AF:00.0   32 GB

Run from /vllm_iree/vllm_test/ with the SAME env your benchmarks use:
    IREE_GPU_ASSIGNMENT=1,0 IREE_WORKER_RANKS=1 VLLM_PP_LAYER_PARTITION=14,2 \
        python3 probe_rank_gpu.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("VLLM_PLUGINS", "iree")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("IREE_CUDA_ARCH", "sm_86")
os.environ.setdefault("IREE_WORKER_RANKS", "1")
os.environ.setdefault("VLLM_PP_LAYER_PARTITION", "14,2")
os.environ.setdefault("IREE_GPU_ASSIGNMENT", "1,0")

from vllm.engine.arg_utils import EngineArgs
from vllm.config.compilation import CUDAGraphMode

engine_args = EngineArgs(
    model="meta-llama/Llama-3.2-1B",
    dtype="float32",
    max_model_len=512,
    max_num_seqs=4,
    enforce_eager=True,
    gpu_memory_utilization=0.4,
    pipeline_parallel_size=2,
    distributed_executor_backend="ray",
)
vllm_config = engine_args.create_engine_config()
vllm_config.parallel_config.distributed_executor_backend = \
    "vllm_plugin.executor.hybrid_executor.HybridExecutor"
vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

from vllm_plugin.executor.hybrid_executor import HybridExecutor
executor = HybridExecutor(vllm_config)


def probe(worker):
    """Report this worker's own view of its single visible GPU."""
    import os as _os
    import torch

    # What CUDA_VISIBLE_DEVICES did this worker process receive?
    cvd = _os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")

    # The bus ID of the GPU this process is actually running on.
    # get_device_properties(0) — index 0 is the ONLY visible device here,
    # but the bus ID is physical and cannot be reindexed.
    props = torch.cuda.get_device_properties(0)
    # PCI bus id via NVML-backed pynvml if available, else from properties
    bus_id = None
    try:
        bus_id = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    except Exception:
        bus_id = "<no pci attrs on properties>"

    return {
        "CUDA_VISIBLE_DEVICES": cvd,
        "name": props.name,
        "total_gb": round(props.total_memory / 1024**3, 1),
        "bus_id": bus_id,
    }


results = executor.collective_rpc(probe)

print("\n" + "=" * 68)
print("  RANK → PHYSICAL GPU (matched by PCI bus ID)")
print("=" * 68)
print("  nvidia-smi truth:  A2 = bus ...3B (15GB),  V100S = bus ...AF (32GB)")
print("  " + "-" * 64)
for rank, r in enumerate(results):
    print(f"  rank {rank}:")
    print(f"    CUDA_VISIBLE_DEVICES = {r['CUDA_VISIBLE_DEVICES']}")
    print(f"    name reported        = {r['name']}")
    print(f"    total memory         = {r['total_gb']} GB")
    print(f"    PCI bus id           = {r['bus_id']}")
print("=" * 68)

executor.shutdown()