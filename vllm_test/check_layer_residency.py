"""
Check whether each PP worker loads the FULL model or only its assigned layers.

vLLM's gpu_model_runner loads all weights, then make_layers only RUNS
[start_layer, end_layer). Layers outside that range may still be materialized
(wasting HBM) as PPMissingLayer or as real-but-unused modules — this probe
tells us which, per rank.

Run with the verified env:
    IREE_GPU_ASSIGNMENT=0,1 IREE_WORKER_RANKS=1 VLLM_PP_LAYER_PARTITION=14,2 \
        python3 check_layer_residency.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("VLLM_PLUGINS", "iree")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("IREE_CUDA_ARCH", "sm_86")
os.environ.setdefault("IREE_WORKER_RANKS", "1")
os.environ.setdefault("VLLM_PP_LAYER_PARTITION", "14,2")
os.environ.setdefault("IREE_GPU_ASSIGNMENT", "0,1")

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


def probe_layers(worker):
    """Inspect each decoder layer: is it a real module with weights, or a stub?"""
    import torch

    model = worker.model_runner.model
    inner = model.model  # LlamaModel
    start = inner.start_layer
    end = inner.end_layer

    layers_info = []
    real_weight_bytes = 0
    stub_count = 0
    real_count = 0

    for idx, layer in enumerate(inner.layers):
        # PPMissingLayer is vLLM's stub for layers this rank doesn't own.
        cls_name = type(layer).__name__
        is_stub = cls_name == "PPMissingLayer"

        # Sum the bytes of any parameters actually materialized on this layer.
        layer_bytes = 0
        has_real_params = False
        for p in layer.parameters(recurse=True):
            n = p.numel()
            if n > 0:
                layer_bytes += n * p.element_size()
                has_real_params = True

        if is_stub or not has_real_params:
            stub_count += 1
        else:
            real_count += 1
            real_weight_bytes += layer_bytes

        layers_info.append({
            "idx": idx,
            "cls": cls_name,
            "bytes_mb": round(layer_bytes / 1024**2, 1),
            "owned": start <= idx < end,
        })

    total_alloc = torch.cuda.memory_allocated(0) / 1024**3

    return {
        "device": torch.cuda.get_device_name(0),
        "owns_range": [start, end - 1],
        "num_layers_total": len(inner.layers),
        "real_layers": real_count,
        "stub_layers": stub_count,
        "real_weight_gb": round(real_weight_bytes / 1024**3, 3),
        "total_allocated_gb": round(total_alloc, 3),
        "layers": layers_info,
    }


results = executor.collective_rpc(probe_layers)

print("\n" + "=" * 70)
print("  LAYER RESIDENCY PER RANK")
print("=" * 70)
for rank, r in enumerate(results):
    owned = r["owns_range"]
    n_owned = owned[1] - owned[0] + 1
    print(f"\n[rank {rank}] {r['device']}")
    print(f"  owns layers        : {owned[0]}–{owned[1]}  ({n_owned} layers)")
    print(f"  total layer slots  : {r['num_layers_total']}")
    print(f"  REAL layers loaded : {r['real_layers']}")
    print(f"  stub layers        : {r['stub_layers']}")
    print(f"  real weight memory : {r['real_weight_gb']} GB")
    print(f"  total allocated    : {r['total_allocated_gb']} GB")
    # Verdict
    if r["real_layers"] > n_owned:
        extra = r["real_layers"] - n_owned
        print(f"  ⚠ LOADS FULL MODEL: {extra} extra layers materialized "
              f"beyond the {n_owned} this rank runs")
    elif r["real_layers"] == n_owned:
        print(f"  ✓ loads ONLY its {n_owned} owned layers")
    else:
        print(f"  ? fewer real layers than owned — unexpected, inspect below")

    # Per-layer detail: show which non-owned layers carry weights
    leaked = [L for L in r["layers"] if not L["owned"] and L["bytes_mb"] > 0]
    if leaked:
        idxs = ", ".join(str(L["idx"]) for L in leaked)
        mb = sum(L["bytes_mb"] for L in leaked)
        print(f"  non-owned layers holding weights: [{idxs}]  (~{mb:.0f} MB)")
print("=" * 70)

executor.shutdown()