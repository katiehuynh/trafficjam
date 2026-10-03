"""The scheduler: priority queues, tenant-fair dispatch, preemption, fallback.

This is the component interviewers will ask about. Key ideas:

- Three strict priority classes (0 highest). Within a class, tenants are
  served round-robin so one noisy tenant cannot starve the others.
- Continuous-batching-style dispatch: requests are packed onto workers up
  to each worker's max batch size; a freed slot is refilled immediately.
- Preemption: a higher-priority arrival can evict a batch that contains
  *only* lower-priority work (like vLLM's preemption). Evicted requests
  are requeued at the head and recompute -- their `preemptions` counter
  makes the cost visible in metrics.
- Graceful degradation: a "large"-model request waiting too long spills
  over to the "small" model instead of timing out.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .config import Settings
from .metrics import Metrics
from .tenants import Tenant


@dataclass
class QueuedRequest:
    id: str
    tenant: Tenant
    tenant_id: str
    priority: int  # 0 highest
    prompt: str
    prompt_tokens: int
    max_tokens: int
    model: str  # "large" | "small" (resolved; never "auto")
    allow_fallback: bool
    enqueued_at: float
    stream: bool
    token_queue: asyncio.Queue  # token events for SSE; None = end of stream
    done: asyncio.Future  # set to CompletedRequest (workers) or an exception
    # mutable worker state
    tokens_generated: int = 0
    first_token_at: float | None = None
    started_at: float | None = None
    prefilled: bool = False
    preemptions: int = 0
    fell_back: bool = False
    cancelled: bool = False


class WorkerLike(Protocol):
    id: str
    profile_name: str
    draining: bool
    alive: bool
    batch: list[QueuedRequest]

    @property
    def free_slots(self) -> int: ...
    def assign(self, req: QueuedRequest) -> None: ...
    def evict(self, reqs: list[QueuedRequest]) -> None: ...


class Scheduler:
    def __init__(
        self,
        settings: Settings,
        metrics: Metrics,
        get_workers: Callable[[], list[WorkerLike]],
    ) -> None:
        self.settings = settings
        self.metrics = metrics
        self._get_workers = get_workers
        self.queues: dict[int, deque[QueuedRequest]] = {0: deque(), 1: deque(), 2: deque()}
        self._rr: dict[int, deque[str]] = {0: deque(), 1: deque(), 2: deque()}  # tenant round-robin per priority
        self._wakeup = asyncio.Event()

    # -- external API -----------------------------------------------------
    def wakeup(self) -> None:
        self._wakeup.set()

    def enqueue(self, req: QueuedRequest) -> None:
        self.queues[req.priority].append(req)
        rr = self._rr[req.priority]
        if req.tenant_id not in rr:
            rr.append(req.tenant_id)
        self.wakeup()

    def requeue_front(self, req: QueuedRequest) -> None:
        """Put a request back at the head of its priority queue (preemption /
        worker death). Progress is discarded -- it will recompute."""
        req.tokens_generated = 0
        req.first_token_at = None
        req.started_at = None
        req.prefilled = False
        self.queues[req.priority].appendleft(req)
        rr = self._rr[req.priority]
        if req.tenant_id in rr:
            rr.remove(req.tenant_id)
        rr.appendleft(req.tenant_id)
        self.wakeup()

    def total_queued(self) -> int:
        return sum(len(q) for q in self.queues.values())

    def oldest_wait_s(self, profile: str) -> float:
        now = time.monotonic()
        oldest = None
        for q in self.queues.values():
            for r in q:
                if r.model == profile and (oldest is None or r.enqueued_at < oldest):
                    oldest = r.enqueued_at
        return (now - oldest) if oldest is not None else 0.0

    # -- main loop --------------------------------------------------------
    async def run(self) -> None:
        while True:
            self._wakeup.clear()
            self._reap_expired()
            self._apply_fallbacks()
            self._dispatch()
            self.metrics.gauges["queue_depth"] = self.total_queued()
            await self._wakeup.wait()

    # -- internals --------------------------------------------------------
    def _next_request(self, profile: str) -> QueuedRequest | None:
        """Strict priority across classes; round-robin across tenants inside
        a class (noisy-neighbor protection)."""
        for p in (0, 1, 2):
            q = self.queues[p]
            rr = self._rr[p]
            for _ in range(len(rr)):
                tid = rr[0]
                cand = next((r for r in q if r.tenant_id == tid and r.model == profile), None)
                if cand is not None:
                    q.remove(cand)
                    rr.rotate(-1)
                    if not any(r.tenant_id == tid for r in q):
                        rr.remove(tid)
                    return cand
                rr.rotate(-1)
        return None

    def _pick_worker(self, profile: str) -> WorkerLike | None:
        cands = [
            w for w in self._get_workers()
            if w.alive and not w.draining and w.profile_name == profile and w.free_slots > 0
        ]
        if not cands:
            return None
        return min(cands, key=lambda w: (len(w.batch), w.id))

    def _dispatch(self) -> None:
        for profile in ("large", "small"):
            while True:
                req = self._next_request(profile)
                if req is None:
                    break
                worker = self._pick_worker(profile)
                if worker is None:
                    self._requeue_front_quiet(req)
                    if self.settings.preemption_enabled and self._try_preempt(req, profile):
                        continue
                    break
                worker.assign(req)
                self.metrics.inc("requests_dispatched")

    def _requeue_front_quiet(self, req: QueuedRequest) -> None:
        """Requeue without resetting progress (dispatch couldn't place it)."""
        self.queues[req.priority].appendleft(req)
        rr = self._rr[req.priority]
        if req.tenant_id in rr:
            rr.remove(req.tenant_id)
        rr.appendleft(req.tenant_id)

    def _try_preempt(self, req: QueuedRequest, profile: str) -> bool:
        """Evict the least-important fully-preemptible batch to make room."""
        victims = [
            w for w in self._get_workers()
            if w.alive and not w.draining and w.profile_name == profile
            and w.batch and all(r.priority > req.priority for r in w.batch)
        ]
        if not victims:
            return False
        victim = max(victims, key=lambda w: min(r.priority for r in w.batch))
        evicted = list(victim.batch)
        victim.evict(evicted)
        for r in evicted:
            r.preemptions += 1
            self.requeue_front(r)
        self.metrics.inc("preemptions", len(evicted))
        self.metrics.event(f"preempted {len(evicted)} req(s) on {victim.id} for {req.id} (prio {req.priority})")
        victim.assign(req)
        self.metrics.inc("requests_dispatched")
        return True

    def _apply_fallbacks(self) -> None:
        """Requests waiting too long for 'large' spill to 'small'."""
        now = time.monotonic()
        for q in self.queues.values():
            for r in list(q):
                if r.model == "large" and r.allow_fallback and (now - r.enqueued_at) > self.settings.fallback_after_s:
                    r.model = "small"
                    r.fell_back = True
                    self.metrics.inc("fallbacks")
                    self.metrics.event(f"{r.id} fell back large->small after {now - r.enqueued_at:.1f}s wait")

    def _reap_expired(self) -> None:
        now = time.monotonic()
        for q in self.queues.values():
            for r in list(q):
                if (now - r.enqueued_at) > self.settings.request_timeout_s and not r.done.done():
                    q.remove(r)
                    r.done.set_exception(TimeoutError("request expired waiting in queue"))
                    self.metrics.inc("requests_expired")
