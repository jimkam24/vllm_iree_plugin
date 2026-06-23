"""
Per-rank latency profiling for HybridExecutor.

Measures the three components of every forward pass:
  t_rank0  — rank 0 forward time (CUDA-synchronized, inside worker process)
  t_rank1  — rank 1 forward time (CUDA-synchronized, inside worker process)
  t_nccl   — derived: wall_execute - t_rank0 - t_rank1
  t_sample — sample_tokens wall time (executor side)
  t_total  — full execute_model wall time (executor side)

HOW IT WORKS
────────────
Each worker stores its last-step forward time via StepTimingMixin.
After sample_tokens(), the executor fires a cheap get_last_step_latency_ms()
RPC to every rank — off the critical path, no added latency to inference.

INTEGRATION
───────────
1. Add StepTimingMixin to NativeWorkerWithSend and IREEWorker (see patches below).
2. Replace HybridExecutor.execute_model with the version in HybridExecutorWithProfiling.
3. Optionally replace HybridExecutor itself with HybridExecutorWithProfiling.

OUTPUT
──────
Stdout: one line per step with all five timings.
CSV:    appended to IREE_LATENCY_CSV (default: /tmp/rank_latency.csv).
        Columns: step, phase, t_rank0_ms, t_rank1_ms, t_nccl_ms,
                 t_sample_ms, t_total_ms, num_tokens, is_prefill
"""

from __future__ import annotations

import csv
import os
import time
from typing import Optional

import torch

from vllm.logger import init_logger
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

PROFILING_ENABLED = os.environ.get("IREE_PROFILING", "0") == "1"

# ── CSV path ──────────────────────────────────────────────────────────────────
_CSV_PATH = os.environ.get("IREE_LATENCY_CSV", "/tmp/rank_latency.csv")
_CSV_HEADER = [
    "step", "phase",
    "t_rank0_ms", "t_rank1_ms", "t_nccl_ms",
    "t_sample_ms", "t_total_ms",
    "num_tokens", "is_prefill",
]


def _ensure_csv_header(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(_CSV_HEADER)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. WORKER MIXIN — add to NativeWorkerWithSend and IREEWorker
# ═══════════════════════════════════════════════════════════════════════════════

class StepTimingMixin:
    """
    Mixin for vLLM workers (gpu_worker subclasses).

    Wraps execute_model to record CUDA-synchronized forward-pass time.
    The executor reads it after the step via get_last_step_latency_ms().

    Usage — add to your worker class:

        class NativeWorkerWithSend(StepTimingMixin, _NativeWorker):
            ...

        class IREEWorker(StepTimingMixin, _NativeWorker):
            ...

    MRO ensures StepTimingMixin.execute_model runs first, then calls
    super().execute_model() which hits the actual worker implementation.
    """

    _last_step_latency_ms: float = 0.0

    def execute_model(self, scheduler_output: SchedulerOutput):
        # CUDA sync before start so we measure only this step, not queued work
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        result = super().execute_model(scheduler_output)

        # Sync again so the elapsed time includes all GPU work dispatched
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._last_step_latency_ms = (time.perf_counter() - t0) * 1e3

        return result

    def get_last_step_latency_ms(self) -> float:
        """Called by the executor after the step. Cheap — just returns a float."""
        return self._last_step_latency_ms


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EXECUTOR — drop-in replacement for HybridExecutor.execute_model
# ═══════════════════════════════════════════════════════════════════════════════

class HybridExecutorWithProfiling:
    """
    Mixin/subclass that adds per-rank latency profiling to HybridExecutor.

    Either:
      (a) Subclass: class HybridExecutor(HybridExecutorWithProfiling, RayDistributedExecutor)
      (b) Mixin: just replace execute_model on your existing HybridExecutor

    Adds one attribute: _step_counter (int), incremented every execute_model call.
    """

    _step_counter: int = 0

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        non_block: bool = False,
    ) -> ModelRunnerOutput | None:
        """
        Pipeline-parallel forward pass with per-rank timing.

        Timeline (all on executor side, wall clock):
        ┌──────────────────────────────────────────────────────┐
        │  t_wall_start                                        │
        │    collective_rpc("execute_model")   ← rank0+rank1   │
        │      [rank 0: forward layers 0-13 + NCCL send]       │
        │      [rank 1: recv + forward layers 14-15]           │
        │  t_wall_execute_done                                 │
        │    collective_rpc("sample_tokens")                   │
        │  t_wall_sample_done                                  │
        │    collective_rpc("get_last_step_latency_ms") ×2     │
        │  t_wall_end  (off critical path)                     │
        └──────────────────────────────────────────────────────┘
        """
        self._step_counter = getattr(self, "_step_counter", 0) + 1
        step = self._step_counter

        # ── Phase detection ──────────────────────────────────────────────────
        # Prefill: scheduler has new requests with prompt tokens.
        # Decode:  only running requests, no new prompts.
        is_prefill = bool(scheduler_output.scheduled_new_reqs)
        num_tokens = sum(
            len(r.prompt_token_ids)
            for r in scheduler_output.scheduled_new_reqs
        ) if is_prefill else sum(scheduler_output.num_scheduled_tokens.values())
        phase = "prefill" if is_prefill else "decode"

        # ── execute_model across all ranks ───────────────────────────────────
        t_wall_start = time.perf_counter()

        outputs = self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
        )

        t_wall_execute_done = time.perf_counter()

        # ── sample_tokens across all ranks ───────────────────────────────────
        sample_outputs = self.collective_rpc(
            "sample_tokens",
            args=(None,),
        )

        t_wall_sample_done = time.perf_counter()

        # ── collect per-rank timings (off critical path) ─────────────────────
        try:
            if PROFILING_ENABLED:
                rank_latencies = self.collective_rpc(
                    "get_last_step_latency_ms",
                    args=(),
                )
                # rank_latencies is a list ordered by rank
                # rank 0 = NativeWorkerWithSend, rank 1 = IREEWorker
                t_rank0_ms = float(rank_latencies[0]) if rank_latencies else 0.0
                t_rank1_ms = float(rank_latencies[1]) if len(rank_latencies) > 1 else 0.0
        except Exception as exc:
            logger.warning("HybridExecutor: could not collect rank latencies: %s", exc)
            t_rank0_ms = 0.0
            t_rank1_ms = 0.0

        # ── derive timings ───────────────────────────────────────────────────
        t_wall_execute_ms = (t_wall_execute_done - t_wall_start) * 1e3
        t_sample_ms       = (t_wall_sample_done - t_wall_execute_done) * 1e3
        t_total_ms        = (t_wall_sample_done - t_wall_start) * 1e3

        # NCCL = what the executor waited for, minus what the two ranks did
        # Negative values indicate overlap or measurement noise — clamp to 0
        t_nccl_ms = max(0.0, t_wall_execute_ms - t_rank0_ms - t_rank1_ms)

        # ── log to stdout ────────────────────────────────────────────────────
        logger.info(
            "RANK_LATENCY step=%d phase=%s | "
            "rank0=%.2fms rank1=%.2fms nccl=%.2fms | "
            "sample=%.2fms total=%.2fms | tokens=%d",
            step, phase,
            t_rank0_ms, t_rank1_ms, t_nccl_ms,
            t_sample_ms, t_total_ms,
            num_tokens,
        )

        # ── append to CSV ────────────────────────────────────────────────────
        try:
            _ensure_csv_header(_CSV_PATH)
            with open(_CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow([
                    step, phase,
                    f"{t_rank0_ms:.4f}",
                    f"{t_rank1_ms:.4f}",
                    f"{t_nccl_ms:.4f}",
                    f"{t_sample_ms:.4f}",
                    f"{t_total_ms:.4f}",
                    num_tokens,
                    int(is_prefill),
                ])
        except Exception as exc:
            logger.warning("HybridExecutor: CSV write failed: %s", exc)

        # ── return output ────────────────────────────────────────────────────
        for output in sample_outputs:
            if output is not None and output.req_ids:
                return output
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYSIS HELPER — run after benchmarks to print a summary
# ═══════════════════════════════════════════════════════════════════════════════

def summarise_csv(path: str = _CSV_PATH) -> None:
    """
    Print a per-phase breakdown from a rank_latency CSV.

    Call from a notebook or a quick analysis script:
        from rank_latency_profiling import summarise_csv
        summarise_csv("/tmp/rank_latency.csv")
    """
    import statistics

    rows: dict[str, list[dict]] = {"prefill": [], "decode": []}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            phase = row["phase"]
            if phase in rows:
                rows[phase].append({k: float(v) for k, v in row.items()
                                    if k not in ("phase",)})

    for phase, data in rows.items():
        if not data:
            continue
        print(f"\n{'─'*56}")
        print(f"  {phase.upper()}  (n={len(data)})")
        print(f"{'─'*56}")
        for col in ("t_rank0_ms", "t_rank1_ms", "t_nccl_ms",
                    "t_sample_ms", "t_total_ms"):
            vals = [r[col] for r in data]
            print(
                f"  {col:<14}  "
                f"mean={statistics.mean(vals):7.2f}ms  "
                f"p50={statistics.median(vals):7.2f}ms  "
                f"min={min(vals):7.2f}ms  "
                f"max={max(vals):7.2f}ms"
            )