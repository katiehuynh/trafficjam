"""Autoscaler: queue-wait-driven replica control, per model profile.

Scale-up signal is head-of-line queue wait (a latency signal, not just
queue length -- a long queue of tiny requests may still meet SLOs).
Scale-down requires *all* replicas of a profile to be idle past the idle
timeout, with a cooldown to avoid flapping. This mirrors how production
inference autoscalers (e.g. KServe/KEDA latency-based scalers) behave.
"""
from __future__ import annotations

import asyncio
import time

from .config import Settings
from .metrics import Metrics
from .scheduler import Scheduler
from .workers import WorkerPool


class Autoscaler:
    def __init__(self, settings: Settings, scheduler: Scheduler, pool: WorkerPool, metrics: Metrics) -> None:
        self.settings = settings
        self.scheduler = scheduler
        self.pool = pool
        self.metrics = metrics
        self._last_scale: dict[str, float] = {"large": 0.0, "small": 0.0}
        self.enabled = True

    async def run(self) -> None:
        while True:
            await asyncio.sleep(2.0)
            if not self.enabled:
                continue
            for profile in ("large", "small"):
                self._reconcile(profile)

    def _reconcile(self, profile: str) -> None:
        now = time.monotonic()
        if now - self._last_scale[profile] < self.settings.scale_cooldown_s:
            return
        active = [w for w in self.pool.workers_of(profile) if not w.draining]
        max_w = getattr(self.settings, f"max_workers_{profile}")
        min_w = getattr(self.settings, f"min_workers_{profile}")
        wait = self.scheduler.oldest_wait_s(profile)

        if wait > self.settings.scale_up_queue_wait_s and len(active) < max_w:
            self.pool.add_worker(profile)
            self._last_scale[profile] = now
            self.metrics.event(f"autoscale UP: +1 {profile} worker (head-of-line wait {wait:.1f}s)")
        elif (
            len(active) > min_w
            and wait == 0.0
            and all(now - w.last_busy > self.settings.scale_down_idle_s for w in active)
        ):
            self.pool.remove_worker(active[-1])
            self._last_scale[profile] = now
            self.metrics.event(f"autoscale DOWN: -1 {profile} worker (idle)")
