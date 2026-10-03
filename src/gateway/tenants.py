"""Multi-tenancy: identity, rate limiting, quotas, and metering.

Each tenant gets token-bucket rate limiters (requests/min and tokens/min),
a concurrency cap, and a usage ledger so the platform can do per-tenant
cost accounting -- the same primitives behind real multi-tenant serving
platforms (per-customer quotas, noisy-neighbor protection, chargeback).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import TIERS, TenantTier


class TokenBucket:
    """Classic token bucket. `take(n)` returns False instead of blocking,
    so the API layer can translate it into a 429 with Retry-After."""

    def __init__(self, rate_per_sec: float, capacity: float):
        self.rate_per_sec = rate_per_sec
        self.capacity = capacity
        self._tokens = capacity
        self._last = time.monotonic()

    def take(self, n: float = 1.0) -> bool:
        now = time.monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate_per_sec)
        self._last = now
        if self._tokens >= n:
            self._tokens -= n
            return True
        return False


@dataclass
class Tenant:
    id: str
    name: str
    api_key: str
    tier: TenantTier
    req_bucket: TokenBucket = field(init=False)
    token_bucket: TokenBucket = field(init=False)
    active: int = 0  # in-flight requests (concurrency cap)
    # Metering ledger
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    requests_ok: int = 0
    requests_rejected: int = 0

    def __post_init__(self) -> None:
        self.req_bucket = TokenBucket(self.tier.req_per_min / 60.0, float(self.tier.req_per_min))
        self.token_bucket = TokenBucket(self.tier.tokens_per_min / 60.0, float(self.tier.tokens_per_min))

    def check_rate_limit(self, est_tokens: int) -> str | None:
        """Returns None if allowed, else the limit that was hit."""
        if not self.req_bucket.take(1):
            return "requests_per_min"
        if not self.token_bucket.take(est_tokens):
            return "tokens_per_min"
        return None

    def record_usage(self, prompt_tokens: int, completion_tokens: int, price_per_1k: float) -> None:
        total = prompt_tokens + completion_tokens
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost_usd += total / 1000.0 * price_per_1k
        self.requests_ok += 1


class TenantRegistry:
    def __init__(self) -> None:
        self._by_key: dict[str, Tenant] = {}
        self._by_id: dict[str, Tenant] = {}

    def add(self, tenant: Tenant) -> None:
        self._by_key[tenant.api_key] = tenant
        self._by_id[tenant.id] = tenant

    def auth(self, api_key: str | None) -> Tenant | None:
        if not api_key:
            return None
        return self._by_key.get(api_key)

    def all(self) -> list[Tenant]:
        return list(self._by_id.values())


def demo_registry() -> TenantRegistry:
    """Three demo tenants, one per tier, to show noisy-neighbor isolation."""
    reg = TenantRegistry()
    reg.add(Tenant(id="acme", name="Acme Corp", api_key="sk-acme-enterprise-001", tier=TIERS["enterprise"]))
    reg.add(Tenant(id="globex", name="Globex", api_key="sk-globex-pro-002", tier=TIERS["pro"]))
    reg.add(Tenant(id="initech", name="Initech", api_key="sk-initech-free-003", tier=TIERS["free"]))
    return reg
