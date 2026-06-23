"""
Print the physical GPU order under CUDA_DEVICE_ORDER=PCI_BUS_ID, plus the
rank → physical-GPU mapping that HybridExecutor's IREE_GPU_ASSIGNMENT produces.

Run from /vllm_iree/vllm_test/:
    python3 check_pci_order.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch

print("=" * 60)
print("Physical GPU order (CUDA_DEVICE_ORDER=PCI_BUS_ID)")
print("=" * 60)

n = torch.cuda.device_count()
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    total_gb = props.total_memory / 1024**3
    print(f"  cuda:{i}  {props.name:<28} "
          f"{total_gb:6.1f}GB  sm_{props.major}{props.minor}")

# ── how IREE_GPU_ASSIGNMENT maps ranks to these physical indices ──────────────
assignment = os.environ.get("IREE_GPU_ASSIGNMENT", "")
iree_ranks = os.environ.get("IREE_WORKER_RANKS", "1")
partition = os.environ.get("VLLM_PP_LAYER_PARTITION", "")

print()
print("=" * 60)
print("Rank → GPU mapping (from env)")
print("=" * 60)
print(f"  IREE_GPU_ASSIGNMENT   = {assignment!r}")
print(f"  IREE_WORKER_RANKS     = {iree_ranks!r}")
print(f"  VLLM_PP_LAYER_PARTITION = {partition!r}")
print()

if assignment:
    gpu_idx = [x.strip() for x in assignment.split(",")]
    iree_set = set(
        int(x) for x in iree_ranks.split(",") if x.strip().isdigit()
    )
    parts = [x.strip() for x in partition.split(",")] if partition else None

    for rank, phys in enumerate(gpu_idx):
        phys_i = int(phys)
        name = torch.cuda.get_device_properties(phys_i).name
        total_gb = torch.cuda.get_device_properties(phys_i).total_memory / 1024**3
        worker = "IREEWorker (offload-capable)" if rank in iree_set else "NativeWorkerWithSend"
        layers = f"{parts[rank]} layers" if parts and rank < len(parts) else "?"
        print(f"  rank {rank}  →  cuda:{phys_i}  {name:<28} "
              f"{total_gb:6.1f}GB  |  {layers:<10}  |  {worker}")
else:
    print("  IREE_GPU_ASSIGNMENT not set — rank i maps to cuda:i by default")

print("=" * 60)