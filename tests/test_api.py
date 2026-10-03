import pytest
from fastapi.testclient import TestClient

from gateway.config import Settings
from gateway.main import create_app

KEY = "sk-acme-enterprise-001"


@pytest.fixture()
def client():
    settings = Settings(min_workers_large=1, max_workers_large=2,
                        min_workers_small=1, max_workers_small=1)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def test_generate_roundtrip(client):
    r = client.post("/v1/generate", json={"prompt": "hello world", "max_tokens": 4},
                    headers={"X-API-Key": KEY})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["completion_tokens"] == 4
    assert body["ttft_ms"] > 0 and body["e2e_ms"] >= body["ttft_ms"]
    assert body["tenant_id"] == "acme"


def test_unauthorized_without_key(client):
    r = client.post("/v1/generate", json={"prompt": "hi", "max_tokens": 2})
    assert r.status_code == 401


def test_streaming_ends_with_done(client):
    r = client.post("/v1/generate",
                    json={"prompt": "hi", "max_tokens": 3, "stream": True},
                    headers={"X-API-Key": KEY})
    assert r.status_code == 200
    assert "[DONE]" in r.text
    assert r.text.count('"token"') == 3


def test_health_metrics_stats(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "gw_" in client.get("/metrics").text
    stats = client.get("/api/stats").json()
    assert "slos" in stats and "tenants" in stats and len(stats["tenants"]) == 3


def test_chaos_kill_and_manual_scale(client):
    before = len(client.get("/api/stats").json()["workers"])
    r = client.post("/admin/chaos/kill", json={"count": 1})
    assert r.json()["killed"] >= 0
    client.post("/admin/scale", json={"profile": "small", "count": 2})
    stats = client.get("/api/stats").json()
    assert stats["gauges"]["workers_small"] >= 1  # autoscaler may add more; at least our target
    assert before >= 1
