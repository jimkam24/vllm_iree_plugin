"""
Diagnostic — does HybridExecutor work under the full vLLM LLM API?

This is a YES/NO gate before investing in concurrent-workload measurement.
Your manual harness (hybrid_executor_test.py) drives execute_model directly.
The LLM API drives the FULL engine: real scheduler, KV cache manager,
continuous batching, sampling loop. The engine calls executor methods the
manual path never exercised — this script finds out whether they all work.

Three escalating checks:
  1. Can LLM() even construct with HybridExecutor? (init path)
  2. Can it generate 1 token from 1 prompt? (basic forward through engine)
  3. Can it generate a longer completion? (decode loop through engine)

Run with the verified env:
    IREE_GPU_ASSIGNMENT=0,1 IREE_WORKER_RANKS=1 VLLM_PP_LAYER_PARTITION=14,2 \
        python3 llm_api_diagnostic.py

Expected failure mode if it walls: /dev/shm too small (the Config-1 problem),
or a missing executor method the full engine requires. Either way the error
tells you exactly what the engine needs that the manual harness sidestepped.
"""

import os
import sys
import traceback
from vllm_plugin.executor.hybrid_executor import HybridExecutor

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("VLLM_PLUGINS", "iree")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("IREE_CUDA_ARCH", "sm_86")
os.environ.setdefault("IREE_WORKER_RANKS", "1")
os.environ.setdefault("VLLM_PP_LAYER_PARTITION", "14,2")
os.environ.setdefault("IREE_GPU_ASSIGNMENT", "0,1")

def main():
    print("=" * 64)
    print("  LLM API diagnostic — HybridExecutor under the full engine")
    print("=" * 64)
    print(f"  /dev/shm check:")
    os.system("df -h /dev/shm | tail -1")
    print()

    # ── Check 1: construction ─────────────────────────────────────────────────────
    print("── CHECK 1: can LLM() construct with HybridExecutor? ──")
    llm = None
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model="meta-llama/Llama-3.2-1B",
            dtype="float32",
            max_model_len=512,
            max_num_seqs=4,
            enforce_eager=True,
            gpu_memory_utilization=0.4,
            pipeline_parallel_size=2,
            distributed_executor_backend=HybridExecutor,
        )
        print("  ✓ CHECK 1 PASSED — LLM() constructed with HybridExecutor\n")
    except Exception:
        print("  ✗ CHECK 1 FAILED — could not construct LLM()")
        print("  ── traceback ──")
        traceback.print_exc()
        print("\n  This is the gate. The error above tells you what the full")
        print("  engine needs that the manual harness sidestepped. Common causes:")
        print("    - /dev/shm too small (NCCL shared memory)")
        print("    - executor missing a method the engine calls during init")
        print("    - KV cache manager lifecycle differs from manual init")
        sys.exit(1)

    # ── Check 2: single-token generation ──────────────────────────────────────────
    print("── CHECK 2: generate 1 token from 1 prompt? ──")
    try:
        out = llm.generate(
            ["The capital of France is"],
            SamplingParams(max_tokens=1, temperature=0.0),
        )
        text = out[0].outputs[0].text
        print(f"  ✓ CHECK 2 PASSED — generated: {text!r}\n")
    except Exception:
        print("  ✗ CHECK 2 FAILED — construction worked but forward pass didn't")
        print("  ── traceback ──")
        traceback.print_exc()
        print("\n  Engine init is fine; the issue is in execute_model/sample under")
        print("  the real scheduler. Compare against your manual decode_test path.")
        sys.exit(1)

    # ── Check 3: longer completion (exercises the decode loop) ────────────────────
    print("── CHECK 3: generate a longer completion (decode loop)? ──")
    try:
        out = llm.generate(
            ["The capital of France is"],
            SamplingParams(max_tokens=20, temperature=0.0),
        )
        text = out[0].outputs[0].text
        print(f"  ✓ CHECK 3 PASSED — generated: {text!r}\n")
    except Exception:
        print("  ✗ CHECK 3 FAILED — single token worked but multi-step decode didn't")
        print("  ── traceback ──")
        traceback.print_exc()
        sys.exit(1)

    print("=" * 64)
    print("  ALL CHECKS PASSED")
    print("  HybridExecutor runs under the full vLLM LLM API.")
    print("  Next: concurrent-request workload to test whether continuous")
    print("  batching hides the FFN-offload latency (the pipelining question).")
    print("=" * 64)
    
    
if __name__ == "__main__":
    main()