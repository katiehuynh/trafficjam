"""Simulated GPU workers and the worker pool.

Each worker behaves like one GPU replica behind a continuous-batching
inference server (vLLM/TGI style): it holds a batch of requests, does a
batched prefill for newcomers, then emits one decode token per request per
tick. Batching efficiency degrades slightly with batch size, which is what
makes the scheduler's packing decisions matter.

Swap `GPUWorker.run` for a real backend (vLLM engine, Triton, TensorRT-LLM)
and the rest of the gateway -- scheduling, tenancy, autoscaling, SLOs --
works unchanged. That seam is intentional.
"""
from __future__ import annotations

import asyncio
import time
import zlib
from dataclasses import dataclass
from typing import Callable

from .config import ModelProfile, Settings
from .metrics import Metrics
from .scheduler import QueuedRequest

# Deterministic pseudo-vocabulary so completions look like text without a model.
_VOCAB = (
    " the a of and to in is that it was for on are as with his they I at be this have from or one had by word "
    "but not what all were we when your can said there use an each which she do how their if will up other about "
    "out many then them these so some her would make like him into time has look two more write go see number no "
    "way could people my than first water been call who oil its now find long down day did get come made may part "
    "over new sound take only little work know place year live me back give most very after thing our just name "
    "good sentence man think say great where help through much before line right too mean old any same tell boy "
    "follow came want show also around form three small set put end does another well large must big even such "
    "because turn here why ask went men read need land different home us move try kind hand picture again change "
    "off play spell air away animal house point page letter mother answer grow study still learn should code data "
    "model token batch queue scale serve fast slow smart bright clear deep learn train test eval prompt"
).split()


def next_token(req: QueuedRequest) -> str:
    idx = (zlib.crc32(req.prompt.encode()) + req.tokens_generated * 7) % len(_VOCAB)
    piece = _VOCAB[idx]
    return piece if req.tokens_generated == 0 else " " + piece


@dataclass
class CompletedRequest:
    id: str
    tenant_id: str
    model: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    e2e_ms: float
    tpot_ms: float
    preemptions: int
    fell_back: bool


class GPUWorker:
    def __init__(
        self,
        worker_id: str,
        profile: ModelProfile,
        metrics: Metrics,
        on_change: Callable[[], None],
        on_death: Callable[[list[QueuedRequest]], None],
    ) -> None:
        self.id = worker_id
        self.profile = profile
        self.profile_name = profile.name
        self.metrics = metrics
        self._on_change = on_change
        self._on_death = on_death
        self.batch: list[QueuedRequest] = []
        self._new = asyncio.Event()
        self.alive = True
        self.draining = False  # scale-down: finish batch, take no new work
        self.last_busy = time.monotonic()

    @property
    def free_slots(self) -> int:
        return self.profile.max_batch - len(self.batch)

    def assign(self, req: QueuedRequest) -> None:
        req.started_at = time.monotonic()
        self.batch.append(req)
        self._new.set()

    def evict(self, reqs: list[QueuedRequest]) -> None:
        for r in reqs:
            if r in self.batch:
                self.batch.remove(r)

    def kill(self) -> None:
        """Chaos: die now. In-flight requests are requeued by the pool."""
        self.alive = False
        self._new.set()

    # -- main loop ------------------------------------------------------
    async def run(self) -> None:
        try:
            while self.alive:
                if not self.batch:
                    self._new.clear()
                    await self._new.wait()
                    continue
                unpref = [r for r in self.batch if not r.prefilled]
                if unpref:
                    await self._prefill(unpref)
                else:
                    await asyncio.sleep(self._tick_s())
                    self._decode_tick()
        finally:
            # Transparent recovery: whatever was in flight goes back to the
            # scheduler head. Clients see added latency, not errors.
            if self.batch:
                self._on_death(list(self.batch))
                self.batch.clear()

    async def _prefill(self, reqs: list[QueuedRequest]) -> None:
        ms = self.profile.prefill_ms_per_token * max(r.prompt_tokens for r in reqs)
        await asyncio.sleep(ms / 1000.0)
        now = time.monotonic()
        self.last_busy = now
        for r in reqs:
            r.prefilled = True
            self.metrics.observe("queue_wait_ms", (now - r.enqueued_at) * 1000.0)

    def _tick_s(self) -> float:
        # Batching isn't free: per-token latency grows with batch size.
        return self.profile.decode_ms_per_token * (1 + 0.12 * max(0, len(self.batch) - 1)) / 1000.0

    def _decode_tick(self) -> None:
        now = time.monotonic()
        self.last_busy = now
        finished: list[QueuedRequest] = []
        for r in self.batch:
            if r.cancelled:
                finished.append(r)  # collected below, completed as cancelled
                continue
            tok = next_token(r)
            r.tokens_generated += 1
            if r.first_token_at is None:
                r.first_token_at = now
                self.metrics.observe("ttft_ms", (now - r.enqueued_at) * 1000.0)
            if r.stream and not r.token_queue.full():
                r.token_queue.put_nowait(tok)
            self.metrics.inc("tokens_generated")
            if r.tokens_generated >= r.max_tokens:
                finished.append(r)
        for r in finished:
            self.batch.remove(r)
            if r.cancelled:
                if not r.done.done():
                    r.done.cancel()
                continue
            self._finish(r, now)
        if finished:
            self._on_change()

    def _finish(self, r: QueuedRequest, now: float) -> None:
        e2e_ms = (now - r.enqueued_at) * 1000.0
        ttft_ms = (r.first_token_at - r.enqueued_at) * 1000.0 if r.first_token_at else e2e_ms
        tpot_ms = (e2e_ms - ttft_ms) / max(1, r.tokens_generated - 1)
        text = "".join(
            _VOCAB[(zlib.crc32(r.prompt.encode()) + i * 7) % len(_VOCAB)]
            if i == 0 else " " + _VOCAB[(zlib.crc32(r.prompt.encode()) + i * 7) % len(_VOCAB)]
            for i in range(r.tokens_generated)
        )
        r.tenant.record_usage(r.prompt_tokens, r.tokens_generated, self.profile.price_per_1k_tokens)
        self.metrics.observe("e2e_ms", e2e_ms)
        self.metrics.observe("tpot_ms", tpot_ms)
        self.metrics.inc("requests_completed")
        if r.stream:
            r.token_queue.put_nowait(None)  # end-of-stream sentinel
        if not r.done.done():
            r.done.set_result(
                CompletedRequest(
                    id=r.id, tenant_id=r.tenant_id, model=self.profile_name,
                    text=text, prompt_tokens=r.prompt_tokens,
                    completion_tokens=r.tokens_generated, ttft_ms=ttft_ms,
                    e2e_ms=e2e_ms, tpot_ms=tpot_ms,
                    preemptions=r.preemptions, fell_back=r.fell_back,
                )
            )


class WorkerPool:
    def __init__(self, settings: Settings, metrics: Metrics, scheduler) -> None:
        self.settings = settings
        self.metrics = metrics
        self.scheduler = scheduler
        self.workers: list[GPUWorker] = []
        self._seq = 0

    def add_worker(self, profile_name: str) -> GPUWorker:
        profile = self.settings.profiles[profile_name]
        w = GPUWorker(
            worker_id=f"{profile_name}-{self._seq}",
            profile=profile, metrics=self.metrics,
            on_change=self.scheduler.wakeup,
            on_death=self._on_worker_death,
        )
        self._seq += 1
        w.task = asyncio.create_task(w.run(), name=f"worker-{w.id}")
        self.workers.append(w)
        self._sync_gauges()
        self.metrics.event(f"worker {w.id} added (batch={profile.max_batch})")
        self.scheduler.wakeup()
        return w

    def remove_worker(self, w: GPUWorker) -> None:
        """Graceful scale-down: drain, then exit when the batch is empty."""
        w.draining = True
        self.metrics.event(f"worker {w.id} draining for scale-down")

    def kill(self, count: int, profile: str | None = None) -> int:
        """Chaos: hard-kill up to `count` workers. Their in-flight requests
        are transparently requeued (see GPUWorker.run finally-block)."""
        cands = [w for w in self.workers if w.alive and not w.draining
                 and (profile is None or w.profile_name == profile)]
        killed = 0
        for w in cands[:count]:
            w.kill()
            killed += 1
        if killed:
            self.metrics.inc("worker_kills", killed)
            self.metrics.event(f"CHAOS: killed {killed} worker(s)")
        self._sync_gauges()
        return killed

    def workers_of(self, profile: str) -> list[GPUWorker]:
        return [w for w in self.workers if w.profile_name == profile and w.alive]

    def set_size(self, profile: str, count: int) -> None:
        """Manual scaling (admin API): grow/shrink to exactly `count`."""
        active = [w for w in self.workers_of(profile) if not w.draining]
        while len(active) < count:
            active.append(self.add_worker(profile))
        for w in active[count:]:
            self.remove_worker(w)

    def _on_worker_death(self, reqs: list[QueuedRequest]) -> None:
        for r in reqs:
            r.preemptions += 1
            self.scheduler.requeue_front(r)
        self.metrics.inc("requests_requeued_after_death", len(reqs))
        self.metrics.event(f"requeued {len(reqs)} in-flight request(s) after worker death")
        self._sync_gauges()

    def _sync_gauges(self) -> None:
        for profile in ("large", "small"):
            self.metrics.gauges[f"workers_{profile}"] = len(self.workers_of(profile))

    async def reap_loop(self) -> None:
        """Finish draining workers whose batches emptied."""
        while True:
            await asyncio.sleep(1.0)
            for w in list(self.workers):
                if w.draining and not w.batch and w.alive:
                    w.kill()
                    self.workers.remove(w)
                    self._sync_gauges()
                    self.metrics.event(f"worker {w.id} scaled down")
                    self.scheduler.wakeup()
