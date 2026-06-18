"""
Benchmark Analysis Script
==========================
Reads benchmark JSON results and produces a detailed breakdown of:
  - Absolute latencies (prefill TTFT, decode per-token)
  - Slowdowns relative to Ray-based vLLM-vLLM baseline (Config 1-ray)
  - Component breakdown estimates

Expected JSON files (from benchmark_prefill_decode.py):
  /tmp/bench_pd_config1.json   — vLLM-vLLM via LLM API (no Ray)
  /tmp/bench_pd_config1r.json  — vLLM-vLLM via HybridExecutor (with Ray)
  /tmp/bench_pd_config2.json   — vLLM-custom (Path A1)
  /tmp/bench_pd_config3.json   — custom-custom (both SDPA)

Usage:
    python3 benchmark_analysis.py
"""

import json
import os

RESULT_FILES = {
    "1-api": "/tmp/bench_pd_config1.json",
    "1-ray": "/tmp/bench_pd_config1r.json",
    "2":     "/tmp/bench_pd_config2.json",
    "3":     "/tmp/bench_pd_config3.json",
}

SHORT_NAMES = {
    "1-api": "vLLM-vLLM (LLM API, no Ray)",
    "1-ray": "vLLM-vLLM (HybridExecutor, Ray)",
    "2":     "vLLM-custom (Triton+SDPA)",
    "3":     "custom-custom (SDPA+SDPA)",
}

def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

data = {k: load(v) for k, v in RESULT_FILES.items()}
available = {k: v for k, v in data.items() if v is not None}

if not available:
    print("No result files found. Run benchmark_prefill_decode.py first.")
    exit(1)

# Get common prompt lengths
all_lens = set()
for d in available.values():
    all_lens.update(int(k) for k in d["results"].keys())
prompt_lens = sorted(all_lens)

print("\n" + "="*75)
print("BENCHMARK ANALYSIS — Heterogeneous PP Timing Breakdown")
print("="*75)

# ── Table 1: Absolute TTFT ───────────────────────────────────────────────────
print("\n[1] TTFT (Prefill Latency) — ms")
print(f"{'Prompt':>8}", end="")
for k in sorted(available.keys()):
    print(f"  {k:>12}", end="")
print()
print("-" * (8 + 14 * len(available)))
for plen in prompt_lens:
    print(f"{plen:>8}", end="")
    for k in sorted(available.keys()):
        r = available[k]["results"].get(str(plen))
        val = f"{r['prefill_mean']:.2f}" if r else "N/A"
        print(f"  {val:>12}", end="")
    print()

# ── Table 2: Absolute Decode/tok ─────────────────────────────────────────────
print("\n[2] Decode Per-Token Latency — ms/tok")
print(f"{'Prompt':>8}", end="")
for k in sorted(available.keys()):
    print(f"  {k:>12}", end="")
print()
print("-" * (8 + 14 * len(available)))
for plen in prompt_lens:
    print(f"{plen:>8}", end="")
    for k in sorted(available.keys()):
        r = available[k]["results"].get(str(plen))
        val = f"{r['decode_per_token_mean']:.2f}" if r else "N/A"
        print(f"  {val:>12}", end="")
    print()

# ── Table 3: Slowdown vs Ray baseline ────────────────────────────────────────
baseline_key = "1-ray" if "1-ray" in available else "1-api"
baseline_name = SHORT_NAMES.get(baseline_key, baseline_key)
print(f"\n[3] TTFT Slowdown vs '{baseline_name}'")
print(f"{'Prompt':>8}", end="")
for k in sorted(available.keys()):
    if k != baseline_key:
        print(f"  {k:>12}", end="")
print()
print("-" * (8 + 14 * (len(available) - 1)))
for plen in prompt_lens:
    base_r = available[baseline_key]["results"].get(str(plen))
    if not base_r:
        continue
    base_ttft = base_r["prefill_mean"]
    print(f"{plen:>8}", end="")
    for k in sorted(available.keys()):
        if k == baseline_key:
            continue
        r = available[k]["results"].get(str(plen))
        if r:
            slowdown = r["prefill_mean"] / base_ttft
            print(f"  {slowdown:>11.2f}x", end="")
        else:
            print(f"  {'N/A':>12}", end="")
    print()

print(f"\n[4] Decode Slowdown vs '{baseline_name}'")
print(f"{'Prompt':>8}", end="")
for k in sorted(available.keys()):
    if k != baseline_key:
        print(f"  {k:>12}", end="")
print()
print("-" * (8 + 14 * (len(available) - 1)))
for plen in prompt_lens:
    base_r = available[baseline_key]["results"].get(str(plen))
    if not base_r:
        continue
    base_dec = base_r["decode_per_token_mean"]
    print(f"{plen:>8}", end="")
    for k in sorted(available.keys()):
        if k == baseline_key:
            continue
        r = available[k]["results"].get(str(plen))
        if r:
            slowdown = r["decode_per_token_mean"] / base_dec
            print(f"  {slowdown:>11.2f}x", end="")
        else:
            print(f"  {'N/A':>12}", end="")
    print()

# ── Table 4: Ray overhead isolation ──────────────────────────────────────────
if "1-api" in available and "1-ray" in available:
    print("\n[5] Ray Overhead (Config 1-ray minus Config 1-api)")
    print(f"{'Prompt':>8}  {'TTFT delta':>12}  {'Decode delta':>14}")
    print("-"*40)
    for plen in prompt_lens:
        r_api = available["1-api"]["results"].get(str(plen))
        r_ray = available["1-ray"]["results"].get(str(plen))
        if r_api and r_ray:
            dt = r_ray["prefill_mean"] - r_api["prefill_mean"]
            dd = r_ray["decode_per_token_mean"] - r_api["decode_per_token_mean"]
            print(f"{plen:>8}  {dt:>+11.2f}ms  {dd:>+13.2f}ms/tok")

# ── Table 5: SDPA overhead isolation ─────────────────────────────────────────
if "1-ray" in available and "2" in available:
    print("\n[6] SDPA Overhead on 2 layers (Config 2 minus Config 1-ray)")
    print(f"{'Prompt':>8}  {'TTFT delta':>12}  {'Decode delta':>14}")
    print("-"*40)
    for plen in prompt_lens:
        r1 = available["1-ray"]["results"].get(str(plen))
        r2 = available["2"]["results"].get(str(plen))
        if r1 and r2:
            dt = r2["prefill_mean"] - r1["prefill_mean"]
            dd = r2["decode_per_token_mean"] - r1["decode_per_token_mean"]
            print(f"{plen:>8}  {dt:>+11.2f}ms  {dd:>+13.2f}ms/tok")

if "2" in available and "3" in available:
    print("\n[7] SDPA Overhead on 14 more layers (Config 3 minus Config 2)")
    print(f"{'Prompt':>8}  {'TTFT delta':>12}  {'Decode delta':>14}")
    print("-"*40)
    for plen in prompt_lens:
        r2 = available["2"]["results"].get(str(plen))
        r3 = available["3"]["results"].get(str(plen))
        if r2 and r3:
            dt = r3["prefill_mean"] - r2["prefill_mean"]
            dd = r3["decode_per_token_mean"] - r2["decode_per_token_mean"]
            print(f"{plen:>8}  {dt:>+11.2f}ms  {dd:>+13.2f}ms/tok")

print("\n" + "="*75)
print("Config legend:")
for k, v in SHORT_NAMES.items():
    if k in available:
        print(f"  {k:>6}: {v}")
print("="*75)