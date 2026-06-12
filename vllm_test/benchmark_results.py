"""
PP Benchmark Results Aggregator
================================
Reads the three benchmark JSON files and prints a comparison table.

Usage:
    python3 benchmark_results.py
"""

import json
import os

configs = {
    "1": "/tmp/benchmark_config1.json",
    "2": "/tmp/benchmark_config2.json",
    "3": "/tmp/benchmark_config3.json",
}

results = {}
for k, path in configs.items():
    if os.path.exists(path):
        with open(path) as f:
            results[k] = json.load(f)
    else:
        print(f"  Missing: {path} (run benchmark_pp_configs.py with PP_BENCH_CONFIG={k})")

if not results:
    print("No results found. Run benchmark first.")
    exit(1)

print("\n" + "="*80)
print("PP CONFIGURATION BENCHMARK RESULTS")
print("="*80)

# Header
print(f"\n{'Config':<45} {'Mean ms':>8} {'P50 ms':>8} {'P90 ms':>8} {'Mem MB':>8} {'Tok/s':>8}")
print("-"*80)

for k in ["1", "2", "3"]:
    if k not in results:
        continue
    r = results[k]
    name = r["config_name"][:44]
    lat = r["latency_ms"]
    mem = r["memory_mb"]
    tput = r["throughput_tokens_per_sec"]
    print(f"{name:<45} {lat['mean']:>8.2f} {lat['p50']:>8.2f} {lat['p90']:>8.2f} {mem['total']:>8.1f} {tput:>8.2f}")

print("-"*80)

# Speedup vs Config 1
if "1" in results and "2" in results:
    s2 = results["1"]["latency_ms"]["mean"] / results["2"]["latency_ms"]["mean"]
    print(f"\n  Config 2 vs Config 1 speedup: {s2:.3f}x ({'faster' if s2 > 1 else 'slower'})")

if "1" in results and "3" in results:
    s3 = results["1"]["latency_ms"]["mean"] / results["3"]["latency_ms"]["mean"]
    print(f"  Config 3 vs Config 1 speedup: {s3:.3f}x ({'faster' if s3 > 1 else 'slower'})")

if "3" in results and "2" in results:
    s23 = results["3"]["latency_ms"]["mean"] / results["2"]["latency_ms"]["mean"]
    print(f"  Config 2 vs Config 3 speedup: {s23:.3f}x (IREE vs PyTorch baseline)")

# Memory comparison
print("\n  Memory per rank:")
for k in ["1", "2", "3"]:
    if k not in results:
        continue
    r = results[k]
    mem = r["memory_mb"]
    print(f"    Config {k}: rank0={mem['rank0']:.1f}MB  rank1={mem['rank1']:.1f}MB  total={mem['total']:.1f}MB")

print("\n" + "="*80)