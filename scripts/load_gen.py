"""Load generator: hammers the gateway with mixed tenants/priorities and reports SLOs.

Usage:
    python scripts/load_gen.py --duration 60 --rps 8 --chaos-at 30

Open http://127.0.0.1:8000/dashboard in another tab while it runs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import httpx

KEYS = {
    "acme": "sk-acme-enterprise-001",    # enterprise, priority 0
    "globex": "sk-globex-pro-002",       # pro, priority 1
    "initech": "sk-initech-free-003",    # free, priority 2
}
TENANTS = list(KEYS)

PROMPTS = [
    "Explain why the sky is blue in one paragraph.",
    "Write a haiku about distributed systems.",
    "What is the capital of France and why does it matter?",
    "Summarize the plot of Hamlet in three sentences.",
    "Give me a recipe for pancakes.",
    "Explain TCP congestion control simply.",
]


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


async def one_request(client: httpx.AsyncClient, base: str, stream_ratio: float, results: dict) -> None:
    tenant = random.choice(TENANTS)
    headers = {"X-API-Key": KEYS[tenant]}
    body = {
        "prompt": random.choice(PROMPTS),
        "max_tokens": random.randint(8, 48),
        "model": random.choices(["large", "small", "auto"], weights=[0.6, 0.3, 0.1])[0],
        "stream": random.random() < stream_ratio,
    }
    t0 = time.monotonic()
    try:
        if body["stream"]:
            ttft = None
            async with client.stream("POST", f"{base}/v1/generate", json=body,
                                     headers=headers, timeout=130) as r:
                if r.status_code != 200:
                    results["throttled" if r.status_code == 429 else "errors"].append(r.status_code)
                    return
                async for line in r.aiter_lines():
                    if line.startswith("data:") and ttft is None and "[DONE]" not in line:
                        ttft = (time.monotonic() - t0) * 1000
                    if "[DONE]" in line:
                        break
            e2e = (time.monotonic() - t0) * 1000
            results["ttft"].append(ttft or e2e)
            results["e2e"].append(e2e)
        else:
            r = await client.post(f"{base}/v1/generate", json=body, headers=headers, timeout=130)
            if r.status_code != 200:
                results["throttled" if r.status_code == 429 else "errors"].append(r.status_code)
                return
            data = r.json()
            results["ttft"].append(data["ttft_ms"])
            results["e2e"].append(data["e2e_ms"])
        results["ok"] += 1
        results["by_tenant"][tenant] += 1
    except Exception:  # noqa: BLE001 - count and keep going
        results["errors"].append("exception")


async def worker_loop(client: httpx.AsyncClient, base: str, stop_at: float,
                      stream_ratio: float, rps: float, results: dict) -> None:
    while time.monotonic() < stop_at:
        await one_request(client, base, stream_ratio, results)
        await asyncio.sleep(random.expovariate(rps))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--rps", type=float, default=6.0, help="mean requests/sec per worker task")
    ap.add_argument("--concurrency", type=int, default=4, help="parallel worker tasks")
    ap.add_argument("--stream-ratio", type=float, default=0.3)
    ap.add_argument("--chaos-at", type=float, default=0, help="second at which to kill a worker (0=off)")
    args = ap.parse_args()

    results = {"ok": 0, "ttft": [], "e2e": [], "errors": [], "throttled": [],
               "by_tenant": {t: 0 for t in TENANTS}}
    stop_at = time.monotonic() + args.duration
    async with httpx.AsyncClient(trust_env=False) as client:
        tasks = [asyncio.create_task(worker_loop(client, args.base, stop_at,
                                                 args.stream_ratio, args.rps, results))
                 for _ in range(args.concurrency)]
        if args.chaos_at > 0:
            await asyncio.sleep(args.chaos_at)
            r = await client.post(f"{args.base}/admin/chaos/kill", json={"count": 1})
            print(f"--- chaos: killed workers -> {r.json()}")
        await asyncio.gather(*tasks)
        stats = (await client.get(f"{args.base}/api/stats")).json()

    n = len(results["e2e"])
    print(f"\ncompleted={results['ok']} throttled_429={len(results['throttled'])} errors={len(results['errors'])}")
    print(f"TTFT ms  p50={pct(results['ttft'],50):.0f} p99={pct(results['ttft'],99):.0f}")
    print(f"E2E  ms  p50={pct(results['e2e'],50):.0f} p99={pct(results['e2e'],99):.0f}")
    print(f"per-tenant: {results['by_tenant']}")
    print(f"server: shed={stats['counts'].get('requests_shed',0)} "
          f"rate_limited={stats['counts'].get('rate_limited',0)} "
          f"preemptions={stats['counts'].get('preemptions',0)} "
          f"fallbacks={stats['counts'].get('fallbacks',0)} "
          f"worker_kills={stats['counts'].get('worker_kills',0)}")


if __name__ == "__main__":
    asyncio.run(main())
