# Architecture

## Request lifecycle

```
client --POST /v1/generate--> API (auth, validation)
                                 |
                          admission control (service.submit)
                           1. token-bucket rate limits (per tenant)
                           2. per-tenant concurrency cap
                           3. global queue capacity -> 429 load shed
                                 |
                          scheduler.enqueue
                                 |
                     dispatch loop (priority -> tenant RR -> worker)
                      /            |              \
              assign to free   preempt lower-   degrade large->small
              worker slot      priority batch   after fallback_after_s
                                 |
                          GPU worker (per replica)
                           prefill newcomers (batched)
                           decode ticks (continuous batching)
                           TTFT / TPOT / E2E observed per request
                                 |
                          response (JSON or SSE stream) + tenant metering
```

Background loops: `scheduler.run` (dispatch), `autoscaler.run` (replica
control every 2s), `metrics.history_loop` (1s snapshots for charts),
`pool.reap_loop` (finish draining workers).

## Scheduling

- **Strict priority classes** (0 > 1 > 2). A class is only served when all
  higher classes are empty.
- **Tenant round-robin within a class**: each class keeps a rotation of
  tenant IDs, so a tenant flooding the queue cannot starve others in the
  same class. This is the noisy-neighbor protection.
- **Continuous-batching dispatch**: workers expose `free_slots`; the
  scheduler packs requests up to `max_batch` and refills freed slots
  immediately (workers wake the scheduler on every completion).
- **Preemption** (vLLM-style): if a high-priority request finds no free
  slot, the scheduler evicts the least-important batch whose requests are
  *all* lower priority. Evicted requests requeue at the head and recompute;
  `preemptions` is counted per request and globally.
- **Degradation**: a `large`-model request waiting longer than
  `fallback_after_s` is flipped to `small` (when `allow_fallback`), trading
  quality for latency instead of timing out.

## Workers

`GPUWorker` simulates one replica of a continuous-batching inference
server: batched prefill for new arrivals, then one decode tick per batch
where per-token latency grows slightly with batch size (batching isn't
free -- this is what makes packing decisions matter). The seam is
deliberate: replace `GPUWorker.run` with a real backend (vLLM engine,
Triton, TensorRT-LLM) and scheduling, tenancy, autoscaling, and SLOs work
unchanged.

Worker death (chaos or scale-down drain) requeues in-flight requests at
the scheduler head transparently -- clients see latency, not errors.

## Autoscaling

Signal is **head-of-line queue wait** (a latency signal), not queue length:
a deep queue of tiny requests may still meet SLOs and shouldn't scale.
Scale-down needs *all* replicas of a profile idle past `scale_down_idle_s`,
plus a cooldown against flapping. Manual override via `/admin/scale` and
`/admin/autoscale`.

## Multi-tenancy

- Identity: `X-API-Key` -> tenant (demo-grade; prod = JWT / API gateway).
- Quotas: token buckets for req/min and tokens/min, plus a concurrency cap.
- Metering: per-tenant prompt/completion tokens and USD cost, priced per
  model profile (`/admin/billing`).
- Isolation: quotas + per-class tenant round-robin + priority classes.

## Metrics & SLOs

Ring-buffer latency observations (TTFT, TPOT, E2E, queue wait) with
p50/p99; counters for shed, rate-limited, preemptions, fallbacks,
worker kills; per-second history for charts. SLOs: TTFT p99, E2E p99,
availability = 1 - shed_rate. Exposed as Prometheus text (`/metrics`),
JSON (`/api/stats`), and the live dashboard (`/dashboard`).
