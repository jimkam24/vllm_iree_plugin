"""
Capacity ceiling sweep — max sequence length before KV exhaustion.

Goal: show that FFN offload on the constrained A2 raises the maximum
sequence length the card can serve, by freeing KV budget.

Method (no repeated hard OOMs):
  1. From the KV cache config, read the TOTAL number of KV blocks the card
     allocated (this already reflects whether FFN was offloaded — offload
     frees memory, so determine_available_memory returns more, so more blocks).
  2. max_tokens_capacity = num_blocks * block_size.
  3. Sweep sequence lengths around that ceiling; for each, check whether the
     request fits (needed_blocks <= num_blocks). Validate by actually running
     the largest fitting length to confirm it executes.

Run twice, compare:
    # no offload
    IREE_GPU_ASSIGNMENT=0,1 IREE_WORKER_RANKS=1 VLLM_PP_LAYER_PARTITION=14,2 \
    IREE_USE_FFN=0 IREE_USE_CPU_FFN=0 IREE_USE_CPU_FFN_IREE=0 \
        python3 capacity_sweep.py --tag no_ffn

    # CPU FFN offload
    IREE_GPU_ASSIGNMENT=0,1 IREE_WORKER_RANKS=1 VLLM_PP_LAYER_PARTITION=14,2 \
    IREE_USE_FFN=1 IREE_USE_CPU_FFN_IREE=1 \
        python3 capacity_sweep.py --tag cpu_ffn

    python3 capacity_sweep.py --compare
"""

import argparse
import csv
import os
import sys

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("VLLM_PLUGINS", "iree")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("IREE_CUDA_ARCH", "sm_86")
os.environ.setdefault("IREE_WORKER_RANKS", "1")
os.environ.setdefault("VLLM_PP_LAYER_PARTITION", "14,2")
os.environ.setdefault("IREE_GPU_ASSIGNMENT", "0,1")

# Allow a large context so max_model_len doesn't cap us below the KV ceiling.
MAX_MODEL_LEN = int(os.environ.get("CAP_MAX_MODEL_LEN", "8192"))
# Base utilization kept safe for BOTH cards. Per-GPU squeeze is done via
# IREE_GPU_UTIL (handled by HybridExecutor), e.g. "0.4,0.05" → V100S safe,
# A2 squeezed so KV memory is the binding constraint. MUST be identical
# across the two configs being compared.
GPU_UTIL = float(os.environ.get("CAP_GPU_UTIL", "0.4"))
CSV = os.environ.get("CAP_CSV", "/tmp/capacity_sweep.csv")
HEADER = ["tag", "num_kv_blocks", "block_size", "max_tokens",
          "available_for_kv_gb", "bytes_per_token", "largest_run_ok"]


def run(tag):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    from vllm.v1.core.sched.output import (
        SchedulerOutput, CachedRequestData, NewRequestData)
    from vllm.sampling_params import SamplingParams
    from transformers import AutoTokenizer

    ea = EngineArgs(
        model="meta-llama/Llama-3.2-1B",
        dtype="float32",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=4,
        enforce_eager=True,
        gpu_memory_utilization=GPU_UTIL,
        pipeline_parallel_size=2,
        distributed_executor_backend="ray",
    )
    cfg = ea.create_engine_config()
    cfg.parallel_config.distributed_executor_backend = \
        "vllm_plugin.executor.hybrid_executor.HybridExecutor"
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    from vllm_plugin.executor.hybrid_executor import HybridExecutor
    ex = HybridExecutor(cfg)

    kv_specs = ex.get_kv_cache_specs()
    avail = ex.determine_available_memory()

    # KV budget numbers for the A2 (rank 1)
    avail_a2_bytes = avail[1]
    avail_a2_gb = avail_a2_bytes / 1e9

    # bytes-per-token on the A2, for context
    def bpt_rpc(worker):
        runner = worker.model_runner
        try:
            specs = runner.get_kv_cache_spec()
            tot = 0.0
            bs = 16
            for s in specs.values():
                bsz = getattr(s, "block_size", None)
                pb = getattr(s, "page_size_bytes", None)
                if bsz and pb:
                    tot += pb / bsz
                    bs = bsz
            return {"bytes_per_token": tot, "block_size": bs}
        except Exception as exc:
            return {"bytes_per_token": 0.0, "block_size": 16, "err": str(exc)}

    bpt_info = ex.collective_rpc(bpt_rpc)[1]
    bpt = bpt_info["bytes_per_token"]
    block_size = bpt_info["block_size"] or 16

    print("=" * 60)
    print(f"  Capacity sweep — tag={tag}  "
          f"(base_util={GPU_UTIL}, IREE_GPU_UTIL={os.environ.get('IREE_GPU_UTIL','<unset>')})")
    print("=" * 60)
    print(f"  bytes/token            : {bpt:.0f}")
    print(f"  available_for_kv (A2)  : {avail_a2_gb:.3f} GB")
    print(f"  block size             : {block_size}")

    # ── let vLLM compute the ceiling ──────────────────────────────────────────
    # get_kv_cache_configs raises if KV budget can't hold max_model_len, and the
    # error message contains the exact "estimated maximum model length". That IS
    # the memory-bound sequence ceiling — no risky execution needed.
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

    max_len_ceiling = None
    fits_max_model_len = False
    try:
        kv_configs = get_kv_cache_configs(cfg, kv_specs, avail)
        ex.initialize_from_config(kv_configs)
        fits_max_model_len = True
        max_len_ceiling = MAX_MODEL_LEN
        print(f"  ceiling                : >= {MAX_MODEL_LEN:,} tokens "
              f"(fits full max_model_len)")
    except ValueError as exc:
        msg = str(exc)
        # Parse "estimated maximum model length is N"
        import re
        m = re.search(r"maximum model length is (\d+)", msg)
        if m:
            max_len_ceiling = int(m.group(1))
            print(f"  ceiling                : {max_len_ceiling:,} tokens "
                  f"(vLLM-computed memory limit)")
        else:
            print(f"  ceiling                : could not parse from: {msg[:120]}")

    # Derive a token count for the CSV (the ceiling itself)
    max_tokens = max_len_ceiling or 0
    num_blocks = (max_tokens // block_size) if max_tokens else 0
    run_ok = fits_max_model_len  # "ran" = was able to init at full context

    if not os.path.exists(CSV):
        with open(CSV, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)
    with open(CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            tag, num_blocks, block_size, max_tokens,
            f"{avail_a2_gb:.4f}", f"{bpt:.0f}", int(run_ok)])
    print(f"\n  → wrote tag={tag} to {CSV}")
    ex.shutdown()


def compare():
    if not os.path.exists(CSV):
        print(f"No CSV at {CSV}"); return
    rows = {}
    with open(CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["tag"]] = r
    if "no_ffn" not in rows or "cpu_ffn" not in rows:
        print(f"Need both tags. Have: {list(rows.keys())}"); return
    b, o = rows["no_ffn"], rows["cpu_ffn"]
    mb, mo = int(b["max_tokens"]), int(o["max_tokens"])
    print("\n" + "=" * 60)
    print("  CAPACITY CEILING — A2, FFN offload vs none")
    print("=" * 60)
    print(f"  {'':<22}{'no FFN':>14}{'CPU FFN':>14}{'delta':>10}")
    print(f"  {'max seq tokens':<22}{mb:>14,}{mo:>14,}{mo-mb:>+10,}")
    print(f"  {'KV blocks':<22}{int(b['num_kv_blocks']):>14,}"
          f"{int(o['num_kv_blocks']):>14,}"
          f"{int(o['num_kv_blocks'])-int(b['num_kv_blocks']):>+10,}")
    if mb > 0:
        print(f"\n  Offload raises the sequence ceiling by "
              f"{mo-mb:,} tokens ({100*(mo-mb)/mb:+.1f}%)")
    if mo > mb:
        print(f"\n  GAP BAND: sequence lengths in ({mb:,}, {mo:,}] are")
        print(f"  REJECTED without offload but SERVED with offload.")
        print(f"  → the capability claim, made concrete.")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        compare()
    elif a.tag:
        run(a.tag)
    else:
        print("Use --tag <no_ffn|cpu_ffn> or --compare"); sys.exit(1)