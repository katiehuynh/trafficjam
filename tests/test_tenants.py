from gateway.config import TIERS, TenantTier
from gateway.tenants import Tenant, TokenBucket, demo_registry


def test_token_bucket_allows_burst_then_denies():
    b = TokenBucket(rate_per_sec=10.0, capacity=10.0)
    assert all(b.take() for _ in range(10))
    assert not b.take()
    b._last -= 1.0  # simulate 1s passing
    assert b.take()


def test_tier_defaults_encode_priority():
    assert TIERS["enterprise"].default_priority == 0
    assert TIERS["pro"].default_priority == 1
    assert TIERS["free"].default_priority == 2
    assert TIERS["free"].max_concurrency < TIERS["enterprise"].max_concurrency


def test_registry_auth():
    reg = demo_registry()
    assert reg.auth("sk-acme-enterprise-001").id == "acme"
    assert reg.auth(None) is None
    assert reg.auth("bogus") is None


def test_rate_limit_eventually_trips():
    tier = TenantTier("tiny", req_per_min=2, tokens_per_min=10_000, max_concurrency=10, default_priority=1)
    t = Tenant(id="t", name="t", api_key="k", tier=tier)
    assert t.check_rate_limit(10) is None
    assert t.check_rate_limit(10) is None
    assert t.check_rate_limit(10) == "requests_per_min"


def test_usage_ledger_accumulates_cost():
    t = Tenant(id="t", name="t", api_key="k", tier=TIERS["pro"])
    t.record_usage(prompt_tokens=100, completion_tokens=100, price_per_1k=2.0)
    assert t.prompt_tokens == 100 and t.completion_tokens == 100
    assert t.cost_usd == 0.4
    assert t.requests_ok == 1
