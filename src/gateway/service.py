"""GatewayService: wires tenants, scheduler, workers, autoscaler together.

Owns admission control (auth already done by the API layer):
  1. rate limits (token buckets per tenant)
  2. per-tenant concurrency caps
  3. global queue capacity -> load shedding (429) instead of unbounded queueing

All rejection paths are counted so the availability SLO is honest.
"""
from __future__ import annotations

import asyncio
import time
import uuid

from .autoscaler import Autoscaler
from .config import Settings
from .metrics import Metrics
from .models import GenerateRequest
from .scheduler import QueuedRequest, Scheduler
from .tenants import Tenant, TenantRegistry, demo_registry
from .workers import WorkerPool


class GatewayError(Exception):
    status_code: int = 500
    def __init__(self, detail: str = ""):
        super().__init__(detail)
        self.detail = detail


class RateLimited(GatewayError):
    status_code = 429


class ConcurrencyExceeded(GatewayError):
    status_code = 429


class QueueFull(GatewayError):
    status_code = 429


class GatewayService:
    def __init__(self, settings: Settings | None = None, tenants: TenantRegistry | None = None) -> None:
        self.settings = settings or Settings()
        self.metrics = Metrics()
        self.tenants = tenants or demo_registry()
        self.scheduler = Scheduler(self.settings, self.metrics, get_workers=lambda: self.pool.workers)
        self.pool = WorkerPool(self.settings, self.metrics, self.scheduler)
        self.autoscaler = Autoscaler(self.settings, self.scheduler, self.pool, self.metrics)
        self._tasks: list[asyncio.Task] = []

    # -- lifecycle ------------------------------------------------------
    async def start(self) -> None:
        for _ in range(self.settings.min_workers_large):
            self.pool.add_worker("large")
        for _ in range(self.settings.min_workers_small):
            self.pool.add_worker("small")
        self._tasks = [
            asyncio.create_task(self.scheduler.run(), name="scheduler"),
            asyncio.create_task(self.autoscaler.run(), name="autoscaler"),
            asyncio.create_task(self.metrics.history_loop(), name="metrics-history"),
            asyncio.create_task(self.pool.reap_loop(), name="pool-reaper"),
        ]
        self.metrics.event("gateway started")

    async def stop(self) -> None:
        for w in list(self.pool.workers):
            w.kill()
        for t in self._tasks:
            t.cancel()

    # -- admission + dispatch -------------------------------------------
    async def submit(self, tenant: Tenant, req: GenerateRequest) -> QueuedRequest:
        priority = req.priority if req.priority is not None else tenant.tier.default_priority
        priority = max(0, min(2, priority))
        prompt_tokens = max(1, len(req.prompt) // 4)
        est_tokens = prompt_tokens + req.max_tokens

        hit = tenant.check_rate_limit(est_tokens)
        if hit:
            tenant.requests_rejected += 1
            self.metrics.inc("rate_limited")
            raise RateLimited(f"tenant '{tenant.id}' exceeded {hit} quota")

        if tenant.active >= tenant.tier.max_concurrency:
            tenant.requests_rejected += 1
            self.metrics.inc("rate_limited")
            raise ConcurrencyExceeded(f"tenant '{tenant.id}' at concurrency cap ({tenant.tier.max_concurrency})")

        if self.scheduler.total_queued() >= self.settings.max_queue:
            tenant.requests_rejected += 1
            self.metrics.inc("requests_shed")
            self.metrics.event(f"load shed: queue full ({self.settings.max_queue}), rejecting {tenant.id}")
            raise QueueFull("server overloaded, retry shortly")

        model = req.model if req.model in self.settings.profiles else "auto"
        if model == "auto":
            model = self._least_loaded_profile()

        loop = asyncio.get_running_loop()
        qreq = QueuedRequest(
            id=uuid.uuid4().hex[:12],
            tenant=tenant, tenant_id=tenant.id, priority=priority,
            prompt=req.prompt, prompt_tokens=prompt_tokens,
            max_tokens=req.max_tokens, model=model,
            allow_fallback=req.allow_fallback,
            enqueued_at=time.monotonic(), stream=req.stream,
            token_queue=asyncio.Queue(maxsize=4096),
            done=loop.create_future(),
        )
        tenant.active += 1
        self.scheduler.enqueue(qreq)
        self.metrics.inc("requests_accepted")
        return qreq

    def release(self, qreq: QueuedRequest) -> None:
        qreq.tenant.active = max(0, qreq.tenant.active - 1)

    def cancel(self, qreq: QueuedRequest) -> None:
        """Client went away: stop generating for this request."""
        qreq.cancelled = True

    def _least_loaded_profile(self) -> str:
        def load(profile: str) -> float:
            workers = self.pool.workers_of(profile)
            slots = sum(w.free_slots for w in workers) or 1
            pending = sum(1 for q in self.scheduler.queues.values() for r in q if r.model == profile)
            return pending / slots
        return min(("large", "small"), key=load)

    # -- introspection ----------------------------------------------------
    def stats(self) -> dict:
        snap = self.metrics.snapshot(self.settings.slo_ttft_p99_ms, self.settings.slo_e2e_p99_ms)
        snap["tenants"] = [
            {
                "id": t.id, "tier": t.tier.name, "active": t.active,
                "prompt_tokens": t.prompt_tokens, "completion_tokens": t.completion_tokens,
                "cost_usd": round(t.cost_usd, 4), "requests_ok": t.requests_ok,
                "requests_rejected": t.requests_rejected,
            }
            for t in self.tenants.all()
        ]
        snap["workers"] = [
            {"id": w.id, "profile": w.profile_name, "batch": len(w.batch),
             "draining": w.draining, "alive": w.alive}
            for w in self.pool.workers
        ]
        return snap
