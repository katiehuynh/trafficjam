"""Metrics: latency distributions, SLO tracking, Prometheus exposition.

Design notes:
- Latency observations live in bounded ring buffers; percentiles are
  computed on demand. At demo scale this is exact and simple; in production
  you'd use real histograms (Prometheus client / HDR).
- A 1-second history loop keeps recent per-second snapshots so the
  dashboard can draw live charts without a TSDB.
"""
from __future__ import annotations

import asyncio
import time
from collections import Counter, defaultdict, deque


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


class Metrics:
    def __init__(self) -> None:
        self._bufs: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=5000))
        self.counters: Counter[str] = Counter()
        self.events: deque[str] = deque(maxlen=60)
        self.history: deque[dict] = deque(maxlen=240)  # ~4 min at 1s cadence
        self._started = time.time()
        # externally-updated gauges
        self.gauges: dict[str, float] = {"queue_depth": 0, "workers_large": 0, "workers_small": 0}

    # -- observations -----------------------------------------------------
    def observe(self, name: str, value_ms: float) -> None:
        self._bufs[name].append(value_ms)

    def inc(self, name: str, n: int = 1) -> None:
        self.counters[name] += n

    def event(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.events.appendleft(f"[{ts}] {msg}")

    # -- reads ------------------------------------------------------------
    def pct(self, name: str, p: float) -> float:
        return percentile(sorted(self._bufs[name]), p)

    def count(self, name: str) -> int:
        return len(self._bufs[name])

    def snapshot(self, slo_ttft_p99_ms: float, slo_e2e_p99_ms: float) -> dict:
        ttft_p99 = self.pct("ttft_ms", 99)
        e2e_p99 = self.pct("e2e_ms", 99)
        completed = self.counters["requests_completed"]
        shed = self.counters["requests_shed"]
        availability = 1.0 - (shed / max(1, completed + shed))
        return {
            "uptime_s": round(time.time() - self._started, 1),
            "latency": {
                "ttft_ms": {"p50": round(self.pct("ttft_ms", 50), 1), "p99": round(ttft_p99, 1), "n": self.count("ttft_ms")},
                "tpot_ms": {"p50": round(self.pct("tpot_ms", 50), 1), "p99": round(self.pct("tpot_ms", 99), 1)},
                "e2e_ms": {"p50": round(self.pct("e2e_ms", 50), 1), "p99": round(e2e_p99, 1), "n": self.count("e2e_ms")},
                "queue_wait_ms": {"p50": round(self.pct("queue_wait_ms", 50), 1), "p99": round(self.pct("queue_wait_ms", 99), 1)},
            },
            "slos": {
                "ttft_p99_ms": {"target": slo_ttft_p99_ms, "actual": round(ttft_p99, 1), "met": ttft_p99 <= slo_ttft_p99_ms},
                "e2e_p99_ms": {"target": slo_e2e_p99_ms, "actual": round(e2e_p99, 1), "met": e2e_p99 <= slo_e2e_p99_ms},
                "availability": {"target": 0.999, "actual": round(availability, 5), "met": availability >= 0.999},
            },
            "counts": dict(self.counters),
            "gauges": dict(self.gauges),
            "events": list(self.events)[:12],
            "history": list(self.history)[-120:],
        }

    def prometheus(self) -> str:
        lines = []
        for name, buf in self._bufs.items():
            vals = sorted(buf)
            if not vals:
                continue
            lines.append(f'gw_{name}_p50 {percentile(vals, 50):.3f}')
            lines.append(f'gw_{name}_p99 {percentile(vals, 99):.3f}')
            lines.append(f'gw_{name}_count {len(vals)}')
        for name, val in self.counters.items():
            lines.append(f'gw_{name} {val}')
        for name, val in self.gauges.items():
            lines.append(f'gw_{name} {val}')
        return "\n".join(lines) + "\n"

    async def history_loop(self) -> None:
        """Record a per-second snapshot for live charts."""
        while True:
            await asyncio.sleep(1.0)
            completed = self.counters["requests_completed"]
            prev = self.history[-1]["completed"] if self.history else 0
            self.history.append(
                {
                    "t": time.time(),
                    "rps": completed - prev,
                    "ttft_p99": round(self.pct("ttft_ms", 99), 1),
                    "e2e_p99": round(self.pct("e2e_ms", 99), 1),
                    "queue": self.gauges["queue_depth"],
                    "completed": completed,
                }
            )
