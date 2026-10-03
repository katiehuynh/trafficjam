"""Central configuration for the inference gateway.

Everything is deliberately explicit (no magic env-var soup) so the
architecture is easy to follow in a code review.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelProfile:
    """A servable model variant. In production this would describe a real
    deployment (e.g. Llama-70B on 8xH100 vs Llama-8B on 1xA10G); here the
    timings drive the GPU worker simulation."""

    name: str
    prefill_ms_per_token: float = 2.0
    decode_ms_per_token: float = 35.0
    max_batch: int = 8
    price_per_1k_tokens: float = 2.00


@dataclass
class TenantTier:
    name: str
    req_per_min: int
    tokens_per_min: int
    max_concurrency: int
    default_priority: int  # 0 = highest


TIERS: dict[str, TenantTier] = {
    "enterprise": TenantTier("enterprise", req_per_min=600, tokens_per_min=200_000, max_concurrency=32, default_priority=0),
    "pro": TenantTier("pro", req_per_min=120, tokens_per_min=40_000, max_concurrency=8, default_priority=1),
    "free": TenantTier("free", req_per_min=20, tokens_per_min=5_000, max_concurrency=2, default_priority=2),
}


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000

    # Admission control
    max_queue: int = 500  # beyond this we shed load with 429

    # Worker pools, per model profile
    min_workers_large: int = 1
    max_workers_large: int = 4
    min_workers_small: int = 1
    max_workers_small: int = 4

    # Autoscaling
    scale_up_queue_wait_s: float = 3.0
    scale_down_idle_s: float = 20.0
    scale_cooldown_s: float = 10.0

    # Reliability
    fallback_after_s: float = 8.0  # waiting this long -> spill to small model
    request_timeout_s: float = 120.0
    preemption_enabled: bool = True

    # SLOs (the product contract the dashboard holds us to)
    slo_ttft_p99_ms: float = 4000.0
    slo_e2e_p99_ms: float = 30000.0
    slo_availability: float = 0.999  # 1 - shed_rate

    profiles: dict[str, ModelProfile] = field(
        default_factory=lambda: {
            "large": ModelProfile("large", prefill_ms_per_token=2.0, decode_ms_per_token=35.0, max_batch=8, price_per_1k_tokens=2.00),
            "small": ModelProfile("small", prefill_ms_per_token=1.0, decode_ms_per_token=12.0, max_batch=16, price_per_1k_tokens=0.20),
        }
    )
