# Inference Gateway

A **multi-tenant LLM inference gateway**: the serving layer that sits in front of GPU
workers and decides who gets compute, when, and at what cost. Think a miniature
version of what Together AI, Databricks, or Meta's inference platform teams run:
request routing, GPU scheduling with preemption, per-tenant quotas and metering,
latency-driven autoscaling, SLO dashboards, and chaos-tested reliability.

GPU workers are **simulated** (no GPUs needed to run this), but the simulation is
load-bearing: batched prefill, continuous-batching decode, and batch-size-dependent
token latency, so scheduling decisions have real consequences. The worker is behind
a clean seam — point it at vLLM/Triton and everything above it works unchanged.

## What this demonstrates

- **Scheduling under contention**: strict priorities, tenant-fair round-robin, preemption, graceful degradation
- **Multi-tenancy**: per-tenant rate limits, concurrency caps, cost metering, noisy-neighbor isolation
- **Reliability**: load shedding, transparent recovery from worker death, chaos endpoint, timeout reaping
- **Product metrics**: TTFT / TPOT / E2E latency, SLO tracking (p99 + availability), Prometheus exposition, live dashboard
- **Systems judgment**: every tradeoff below is documented, not accidental

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │                  API (FastAPI)               │
                    │  /v1/generate (JSON + SSE)  /dashboard      │
                    │  /metrics  /api/stats  /admin/*             │
                    └──────────────┬──────────────────────────────┘
                                   │ admission: rate limit → concurrency → queue cap
                    ┌──────────────▼──────────────────────────────┐
                    │                 SCHEDULER                   │
                    │  priority classes → tenant round-robin     │
                    │  → pack workers → preempt → fallback       │
                    └──────┬───────────────┬──────────────────────┘
                           │               │
              ┌────────────▼─────┐  ┌──────▼──────────┐   ┌──────────────┐
              │  GPU workers     │  │  GPU workers    │   │  AUTOSCALER  │
              │  "large" pool    │  │  "small" pool   │   │ queue-wait   │
              │  (simulated)     │  │  (simulated)    │   │ driven       │
              └──────────────────┘  └─────────────────┘   └──────────────┘
```

Deep dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn gateway.main:app --port 8000
# dashboard → http://127.0.0.1:8000/dashboard
```

```bash
# a request (demo keys: sk-acme-enterprise-001 / sk-globex-pro-002 / sk-initech-free-003)
curl -X POST localhost:8000/v1/generate -H "X-API-Key: sk-acme-enterprise-001" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain caching in one sentence.","max_tokens":24,"stream":true}' -N
```

```bash
pytest -q   # 20 tests: scheduler, tenancy, metrics, API
```

## Demo scenarios (all verified live)

**1. Baseline + dashboard** — traffic, then watch TTFT/E2E p99, queue depth, and per-tenant cost:
```bash
python scripts/load_gen.py --duration 60 --rps 5 --concurrency 6
# open http://127.0.0.1:8000/dashboard
```

**2. Noisy-neighbor isolation** — the free-tier tenant gets throttled (429s) under
flood while enterprise/pro sail through; per-tenant round-robin keeps classes fair.

**3. Priority preemption** — freeze autoscaling, fill one worker with low-priority
batch work, then send an urgent request:
```bash
curl -X POST localhost:8000/admin/autoscale -d '{"enabled":false}' -H "Content-Type: application/json"
curl -X POST localhost:8000/admin/scale -d '{"profile":"large","count":1}' -H "Content-Type: application/json"
# ... fill batch with priority:2 requests, then:
curl -X POST localhost:8000/v1/generate -H "X-API-Key: sk-acme-enterprise-001" \
  -H "Content-Type: application/json" -d '{"prompt":"urgent","max_tokens":4,"priority":0}'
# → 8 lower-priority requests evicted and requeued; urgent completes in ~124ms
```

**4. Chaos** — kill workers mid-load; in-flight requests are transparently requeued,
the autoscaler replaces capacity, the dashboard shows the dip and recovery:
```bash
python scripts/load_gen.py --duration 60 --chaos-at 30
# or: curl -X POST localhost:8000/admin/chaos/kill -d '{"count":1}' -H "Content-Type: application/json"
```

## Key design decisions

| Decision | Rationale |
|---|---|
| Scale on head-of-line **queue wait**, not queue length | A deep queue of tiny requests can still meet SLOs; wait time is the latency signal that matters |
| Shed load with 429 instead of unbounded queueing | Bounded queues keep tail latency honest; `availability = 1 − shed_rate` makes the tradeoff visible |
| Preempt only fully-lower-priority batches | Avoids thrash: a batch with one urgent request is never evicted for another |
| Tenant round-robin *within* each priority class | Quotas stop floods at admission; RR stops starvation among admitted tenants |
| Simulated workers behind a backend seam | Schedulers/quotas/SLOs are testable without GPUs; swap `GPUWorker.run` for vLLM later |
| Token-bucket (not fixed-window) rate limits | Allows legitimate bursts, still caps sustained rate; returns 429, never silently queues |

## API

| Endpoint | Description |
|---|---|
| `POST /v1/generate` | Generate; `stream:true` for SSE tokens; `X-API-Key` auth |
| `GET /dashboard` | Live ops dashboard (KPIs, latency charts, tenants, workers, events) |
| `GET /metrics` | Prometheus exposition |
| `GET /api/stats` | JSON snapshot: latencies, SLO attainment, tenants, workers |
| `GET /admin/billing` | Per-tenant token usage and cost |
| `POST /admin/chaos/kill` | Kill N workers (reliability testing) |
| `POST /admin/scale` | Set exact worker count per profile |
| `POST /admin/autoscale` | Enable/disable the autoscaler |

## Project structure

```
src/gateway/
  config.py      # tiers, model profiles, SLOs, scaling policy
  tenants.py     # identity, token-bucket quotas, metering ledger
  scheduler.py   # priority queues, tenant-fair dispatch, preemption, fallback
  workers.py     # simulated GPU workers + pool (the backend seam lives here)
  autoscaler.py  # queue-wait-driven replica control
  metrics.py     # latency distributions, SLOs, Prometheus, history
  service.py     # admission control + lifecycle wiring
  main.py        # FastAPI routes (JSON + SSE streaming)
  dashboard.py   # single-file live ops dashboard
scripts/load_gen.py  # mixed-tenant load generator with chaos mode
tests/               # scheduler, tenancy, metrics, API (20 tests)
```

## Production gaps (honest)

- Workers are simulated; real backend would be vLLM/TensorRT-LLM via the existing seam
- Single-process: scheduler state should move to Redis Streams for multi-replica gateway
- Auth is demo API keys; needs JWT/OIDC + per-tenant secret rotation
- Metrics use ring buffers; production wants real histograms (Prometheus client)
- No request hedging, no KV-cache-aware routing, no prefix caching — the next three features I'd build

## Tech

Python, FastAPI, asyncio, Pydantic, pytest, Docker. No GPU required.
