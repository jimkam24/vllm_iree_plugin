"""
Memory analysis harness — rank 1 FFN offload vs no offload.

GOAL
────
Isolate exactly what FFN offload buys on rank 1 (the A2, layers 14-15):
  - GPU memory freed by moving FFN weights off-device
  - The resulting gain in KV cache token capacity

This is the thesis's core memory claim. Latency/throughput benchmarks
already exist; this quantifies the *memory* win, which is the actual
contribution of FFN offload.

WHY THE OLD MEASUREMENT WAS NOT ENOUGH
──────────────────────────────────────
1. torch.cuda.memory_allocated() only counts PyTorch's caching allocator.
   IREE allocates through its own device buffers (ireert.asdevicearray),
   invisible to memory_allocated(). For IREE configs this UNDERSTATES usage.
   → Fix: use torch.cuda.mem_get_info() (driver-level free/total) as the
     ground truth. It sees IREE buffers, CUDA context, everything.

2. The old test measured "freed GB" but never converted it to KV token
   capacity — which is the number that proves the thesis.
   → Fix: compute bytes_per_token from the KVCacheSpec and divide.

3. "Memory BEFORE KV cache (weights only)" was only true if no profiling
   forward had run. determine_available_memory() runs a forward that leaves
   activation peaks cached, polluting a later "weights only" reading.
   → Fix: take the weights-only snapshot immediately after load_model(),
     before any profiling forward. Use a dedicated RPC.

MEASUREMENT POINTS (all on rank 1)
──────────────────────────────────
  A. driver_free_after_load   — mem_get_info free, right after model load
                                 (weights + CUDA context resident, no KV, no
                                 profiling activations)
  B. torch_allocated_after_load — memory_allocated, same moment (PyTorch view)
  C. available_for_kv         — determine_available_memory()[rank1]
                                 (vLLM's own profiled KV budget)
  D. bytes_per_kv_token       — derived from KVCacheSpec on rank 1
  E. kv_token_capacity        — C / D  (the headline number)

RUN
───
Two passes, identical config except the FFN env vars:
    # Pass 1 — baseline, no offload
    IREE_USE_FFN=0 IREE_USE_CPU_FFN=0 python3 memory_analysis.py --tag no_ffn

    # Pass 2 — IREE CPU FFN offload (config 5)
    IREE_USE_FFN=1 IREE_USE_CPU_FFN_IREE=1 python3 memory_analysis.py --tag cpu_ffn

Each pass appends one row to IREE_MEMORY_CSV (default /tmp/memory_analysis.csv).
After both passes, run:
    python3 memory_analysis.py --compare
to print the rank-1 diff (freed memory + KV capacity gain).
"""

import argparse
import csv
import os
import sys

# ── env setup (mirror hybrid_executor_test.py) ────────────────────────────────
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("VLLM_PLUGINS", "iree")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("IREE_CUDA_ARCH", "sm_86")
os.environ.setdefault("IREE_WORKER_RANKS", "1")
os.environ.setdefault("VLLM_PP_LAYER_PARTITION", "14,2")
os.environ.setdefault("IREE_GPU_ASSIGNMENT", "1,0")

_CSV = os.environ.get("IREE_MEMORY_CSV", "/tmp/memory_analysis.csv")
_HEADER = [
    "tag", "rank", "device_name", "is_iree_worker",
    "driver_free_after_load_gb",   # A — ground truth at load (allocator-cached)
    "driver_free_after_forward_gb",  # A2 — after profiling forward + empty_cache
    "torch_allocated_after_load_gb",  # B — PyTorch view (misses IREE)
    "available_for_kv_gb",         # C — vLLM profiled KV budget
    "bytes_per_kv_token",          # D
    "kv_token_capacity",           # E = C / D
    "gpu_total_gb",
]

# Which ranks run the IREE worker (and therefore CAN do FFN offload).
# Parsed from the same env var the executor uses, so the report always
# matches the actual deployment — whether offload is on rank 0, rank 1, or both.
_IREE_WORKER_RANKS = set(
    int(x.strip())
    for x in os.environ.get("IREE_WORKER_RANKS", "1").split(",")
    if x.strip().isdigit()
)


# ══════════════════════════════════════════════════════════════════════════════
# RPCs — run inside the worker process so they see that rank's real device
# ══════════════════════════════════════════════════════════════════════════════

def _rpc_mem_after_load(worker):
    """Snapshot driver-level + torch-level memory. Run right after load_model."""
    import torch
    free, total = torch.cuda.mem_get_info(0)
    alloc = torch.cuda.memory_allocated(0)
    return {
        "driver_free_gb": free / 1024**3,
        "torch_allocated_gb": alloc / 1024**3,
        "total_gb": total / 1024**3,
        "device_name": torch.cuda.get_device_name(0),
    }


def _rpc_bytes_per_kv_token(worker):
    """
    Derive bytes per KV token from this rank's KVCacheSpec.

    A page_size_bytes / block_size gives per-token bytes for one layer group;
    summing across the rank's groups gives total per-token KV cost on this rank.
    Robust to vLLM version differences via getattr fallbacks.
    """
    try:
        runner = worker.model_runner
        specs = runner.get_kv_cache_spec()  # dict: layer_name -> KVCacheSpec
    except Exception as exc:
        return {"bytes_per_token": 0.0, "error": f"spec unavailable: {exc}"}

    total_bytes_per_token = 0.0
    n_layers = 0
    for layer_name, spec in specs.items():
        block_size = getattr(spec, "block_size", None)
        page_bytes = getattr(spec, "page_size_bytes", None)
        if block_size and page_bytes:
            total_bytes_per_token += page_bytes / block_size
            n_layers += 1

    return {
        "bytes_per_token": float(total_bytes_per_token),
        "n_kv_layers_this_rank": n_layers,
    }


def _rpc_mem_after_forward(worker):
    """
    Snapshot driver-level free memory AFTER the profiling forward has run.

    determine_available_memory() triggers a profiling forward in which the
    offloaded FFN executes on CPU — so its weights are not GPU-resident at
    peak. We empty_cache() first to return freed pages to the driver, so
    mem_get_info reflects true post-offload residency, not allocator caching.
    This is the snapshot that makes driver-free agree with the KV-budget gain.
    """
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(0)
    return {"driver_free_after_forward_gb": free / 1024**3}


def _rpc_ffn_patched(worker):
    """
    Report whether this rank's MLP layers are actually FFN-offload-patched.

    The offload code sets layer.mlp._original_forward when it patches.
    Its presence is ground truth that the offload engaged on THIS rank —
    no need to trust env vars or which-rank-is-which reasoning.
    """
    try:
        model = worker.model_runner.model
        start = model.model.start_layer
        end = model.model.end_layer
    except Exception as exc:
        return {"patched": None, "error": f"model unavailable: {exc}"}

    patched, total = 0, 0
    for idx in range(start, end):
        try:
            mlp = model.model.layers[idx].mlp
        except Exception:
            continue
        total += 1
        if hasattr(mlp, "_original_forward"):
            patched += 1

    return {
        "layers_owned": total,
        "layers_patched": patched,
        "fully_patched": (total > 0 and patched == total),
        "layer_range": [start, end - 1],
    }


# ══════════════════════════════════════════════════════════════════════════════
# main measurement pass
# ══════════════════════════════════════════════════════════════════════════════

def run_pass(tag: str) -> None:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.config.compilation import CUDAGraphMode

    print("=" * 60)
    print(f"Memory analysis pass — tag={tag}")
    print(f"  IREE_USE_FFN={os.environ.get('IREE_USE_FFN','0')} "
          f"IREE_USE_CPU_FFN={os.environ.get('IREE_USE_CPU_FFN','0')} "
          f"IREE_USE_CPU_FFN_IREE={os.environ.get('IREE_USE_CPU_FFN_IREE','0')}")
    print("=" * 60)

    engine_args = EngineArgs(
        model="meta-llama/Llama-3.2-1B",
        dtype="float32",
        max_model_len=512,
        max_num_seqs=4,
        enforce_eager=True,
        gpu_memory_utilization=0.2,
        pipeline_parallel_size=2,
        distributed_executor_backend="ray",
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.parallel_config.distributed_executor_backend = \
        "vllm_plugin.executor.hybrid_executor.HybridExecutor"
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    from vllm_plugin.executor.hybrid_executor import HybridExecutor
    executor = HybridExecutor(vllm_config)

    n_ranks = len(executor.workers)

    # ── A & B: memory right after load, BEFORE any profiling forward ──────────
    # Clean "weights + context, no KV, no activation peak" point, all ranks.
    after_load = executor.collective_rpc(_rpc_mem_after_load)

    # ── D: bytes per KV token, all ranks ──────────────────────────────────────
    bpt = executor.collective_rpc(_rpc_bytes_per_kv_token)

    # ── C: vLLM's own profiled KV budget, all ranks ───────────────────────────
    available = executor.determine_available_memory()

    # ── A2: driver-free AFTER the profiling forward (offload now reflected) ────
    after_forward = executor.collective_rpc(_rpc_mem_after_forward)

    # ── verify FFN offload actually engaged, per rank ─────────────────────────
    patched = executor.collective_rpc(_rpc_ffn_patched)

    # ── per-rank report + CSV rows ────────────────────────────────────────────
    if not os.path.exists(_CSV):
        with open(_CSV, "w", newline="") as f:
            csv.writer(f).writerow(_HEADER)

    for rank in range(n_ranks):
        r_load = after_load[rank]
        r_bpt = bpt[rank]
        bytes_per_token = r_bpt.get("bytes_per_token", 0.0)
        r_avail_bytes = available[rank]
        r_avail_gb = r_avail_bytes / 1e9
        kv_capacity = (r_avail_bytes / bytes_per_token) if bytes_per_token > 0 else 0.0
        is_iree = rank in _IREE_WORKER_RANKS

        role = "IREE worker (offload-capable)" if is_iree else "native worker"
        print(f"\n[rank {rank}] {r_load['device_name']} — {role}")
        p = patched[rank]
        if p.get("fully_patched"):
            print(f"  FFN offload            = ENGAGED ✓ "
                  f"({p['layers_patched']}/{p['layers_owned']} layers, "
                  f"range {p['layer_range']})")
        elif p.get("layers_patched", 0) > 0:
            print(f"  FFN offload            = PARTIAL ⚠ "
                  f"({p['layers_patched']}/{p['layers_owned']} layers)")
        else:
            print(f"  FFN offload            = not engaged "
                  f"(native FFN on GPU)")
        print(f"  driver free after load = {r_load['driver_free_gb']:.3f} GB  (allocator-cached)")
        print(f"  driver free post-fwd   = {after_forward[rank]['driver_free_after_forward_gb']:.3f} GB  (true residency)")
        print(f"  torch allocated        = {r_load['torch_allocated_gb']:.3f} GB  (PyTorch view)")
        print(f"  available for KV       = {r_avail_gb:.3f} GB")
        print(f"  bytes/KV token         = {bytes_per_token:.1f} "
              f"(layers on rank: {r_bpt.get('n_kv_layers_this_rank','?')})")
        print(f"  KV token capacity      = {kv_capacity:,.0f} tokens")

        with open(_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                tag, rank, r_load["device_name"], int(is_iree),
                f"{r_load['driver_free_gb']:.4f}",
                f"{after_forward[rank]['driver_free_after_forward_gb']:.4f}",
                f"{r_load['torch_allocated_gb']:.4f}",
                f"{r_avail_gb:.4f}",
                f"{bytes_per_token:.1f}",
                f"{kv_capacity:.0f}",
                f"{r_load['total_gb']:.4f}",
            ])

    print(f"\n  → appended {n_ranks} rows tag={tag} to {_CSV}")

    executor.shutdown()


# ══════════════════════════════════════════════════════════════════════════════
# compare two passes
# ══════════════════════════════════════════════════════════════════════════════

def compare() -> None:
    if not os.path.exists(_CSV):
        print(f"No CSV at {_CSV} — run passes first.")
        return

    # rows[tag][rank] = row dict
    rows: dict[str, dict[int, dict]] = {}
    with open(_CSV, newline="") as f:
        for row in csv.DictReader(f):
            rows.setdefault(row["tag"], {})[int(row["rank"])] = row

    if "no_ffn" not in rows or "cpu_ffn" not in rows:
        print("Need both tags present: no_ffn and cpu_ffn.")
        print(f"Have: {list(rows.keys())}")
        return

    base, off = rows["no_ffn"], rows["cpu_ffn"]
    ranks = sorted(set(base) & set(off))

    def g(r, k): return float(r[k])

    print("\n" + "=" * 64)
    print("  FFN offload vs no offload — per rank")
    print("=" * 64)

    for rank in ranks:
        b, o = base[rank], off[rank]
        dev = b.get("device_name", f"rank{rank}")
        is_iree = b.get("is_iree_worker", "0") == "1"

        freed_load = g(o, "driver_free_after_load_gb") - g(b, "driver_free_after_load_gb")
        freed_fwd = g(o, "driver_free_after_forward_gb") - g(b, "driver_free_after_forward_gb")
        avail_d = g(o, "available_for_kv_gb") - g(b, "available_for_kv_gb")
        kv_b = g(b, "kv_token_capacity")
        kv_o = g(o, "kv_token_capacity")
        kv_d = kv_o - kv_b

        marker = "  ← offload-capable" if is_iree else ""
        print(f"\n  ┌─ rank {rank}: {dev}{marker}")
        print(f"  │  {'metric':<26}{'no FFN':>12}{'CPU FFN':>12}{'delta':>12}")
        print(f"  │  {'-'*60}")
        print(f"  │  {'driver free (at load)':<26}"
              f"{g(b,'driver_free_after_load_gb'):>10.3f}GB"
              f"{g(o,'driver_free_after_load_gb'):>10.3f}GB{freed_load:>+10.3f}GB")
        print(f"  │  {'driver free (post-forward)':<26}"
              f"{g(b,'driver_free_after_forward_gb'):>10.3f}GB"
              f"{g(o,'driver_free_after_forward_gb'):>10.3f}GB{freed_fwd:>+10.3f}GB")
        print(f"  │  {'available for KV':<26}"
              f"{g(b,'available_for_kv_gb'):>10.3f}GB"
              f"{g(o,'available_for_kv_gb'):>10.3f}GB{avail_d:>+10.3f}GB")
        print(f"  │  {'KV token capacity':<26}"
              f"{kv_b:>12,.0f}{kv_o:>12,.0f}{kv_d:>+12,.0f}")
        if kv_b > 0:
            print(f"  │  KV gain: {kv_d:+,.0f} tokens ({100*kv_d/kv_b:+.1f}%), "
                  f"GPU freed (post-fwd): {freed_fwd:+.3f} GB")
        # Sanity flag: the non-offload rank should barely move
        if not is_iree and abs(freed_fwd) > 0.05:
            print(f"  │  ⚠ note: this rank is NOT offload-capable but moved "
                  f"{freed_fwd:+.3f}GB post-forward — check for noise or shared state")
    print("\n" + "=" * 64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="label for this pass (e.g. no_ffn, cpu_ffn)")
    ap.add_argument("--compare", action="store_true",
                    help="print rank-1 diff from existing CSV")
    args = ap.parse_args()

    if args.compare:
        compare()
    elif args.tag:
        run_pass(args.tag)
    else:
        print("Use --tag <label> to run a pass, or --compare to diff.")
        sys.exit(1)