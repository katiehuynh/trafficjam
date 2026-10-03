import asyncio
import time

from gateway.config import TIERS, Settings
from gateway.metrics import Metrics
from gateway.scheduler import QueuedRequest, Scheduler
from gateway.tenants import Tenant


class StubWorker:
    def __init__(self, wid, profile_name="large", max_batch=8):
        self.id = wid
        self.profile_name = profile_name
        self.draining = False
        self.alive = True
        self.batch = []
        self._max = max_batch

    @property
    def free_slots(self):
        return self._max - len(self.batch)

    def assign(self, req):
        self.batch.append(req)

    def evict(self, reqs):
        for r in reqs:
            if r in self.batch:
                self.batch.remove(r)


async def _mk(priority, tenant_id="t1", model="large", allow_fallback=True, age_s=0.0):
    loop = asyncio.get_running_loop()
    tenant = Tenant(id=tenant_id, name=tenant_id, api_key="k-" + tenant_id, tier=TIERS["pro"])
    return QueuedRequest(
        id=f"r{priority}-{tenant_id}-{time.monotonic_ns() % 100000}",
        tenant=tenant, tenant_id=tenant_id, priority=priority,
        prompt="hello", prompt_tokens=2, max_tokens=4, model=model,
        allow_fallback=allow_fallback, enqueued_at=time.monotonic() - age_s,
        stream=False, token_queue=asyncio.Queue(), done=loop.create_future(),
    )


def _sched(workers, **kw):
    settings = Settings(**kw)
    m = Metrics()
    return Scheduler(settings, m, get_workers=lambda: workers), m


def test_strict_priority_order():
    async def go():
        s, _ = _sched([])
        for p in (2, 1, 0):
            s.enqueue(await _mk(p))
        assert [s._next_request("large").priority for _ in range(3)] == [0, 1, 2]
    asyncio.run(go())


def test_tenant_round_robin_within_priority():
    async def go():
        s, _ = _sched([])
        s.enqueue(await _mk(1, "t1"))
        s.enqueue(await _mk(1, "t1"))
        s.enqueue(await _mk(1, "t2"))
        got = [s._next_request("large").tenant_id for _ in range(3)]
        assert got == ["t1", "t2", "t1"], got  # t2 not starved by t1's burst
    asyncio.run(go())


def test_preemption_evicts_lower_priority_batch():
    async def go():
        w = StubWorker("large-0", max_batch=2)
        s, m = _sched([w])
        for _ in range(2):
            r = await _mk(2)
            w.assign(r)
        urgent = await _mk(0)
        assert s._try_preempt(urgent, "large") is True
        assert w.batch == [urgent]
        assert s.total_queued() == 2  # evicted requests requeued at head
        assert m.counters["preemptions"] == 2
    asyncio.run(go())


def test_no_preemption_against_equal_or_higher_priority():
    async def go():
        w = StubWorker("large-0", max_batch=2)
        s, _ = _sched([w])
        w.assign(await _mk(0))
        assert s._try_preempt(await _mk(1), "large") is False
        assert len(w.batch) == 1
    asyncio.run(go())


def test_fallback_after_long_wait():
    async def go():
        s, m = _sched([], fallback_after_s=0.01)
        r = await _mk(1, model="large", age_s=5.0)
        s.enqueue(r)
        s._apply_fallbacks()
        assert r.model == "small" and r.fell_back
        assert m.counters["fallbacks"] == 1
    asyncio.run(go())


def test_reap_expired_requests():
    async def go():
        s, m = _sched([], request_timeout_s=0.01)
        r = await _mk(1, age_s=5.0)
        s.enqueue(r)
        s._reap_expired()
        assert s.total_queued() == 0
        assert isinstance(r.done.exception(), TimeoutError)
        assert m.counters["requests_expired"] == 1
    asyncio.run(go())


def test_dispatch_packs_workers_and_wakes_on_free_slot():
    async def go():
        w = StubWorker("large-0", max_batch=2)
        s, m = _sched([w])
        for _ in range(3):
            s.enqueue(await _mk(1))
        s._dispatch()
        assert len(w.batch) == 2 and s.total_queued() == 1
        # free a slot -> scheduler refills
        w.batch.pop(0)
        s.wakeup()
        s._dispatch()
        assert len(w.batch) == 2 and s.total_queued() == 0
    asyncio.run(go())
