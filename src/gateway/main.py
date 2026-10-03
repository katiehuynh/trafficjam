"""FastAPI front door: OpenAI-style generate endpoint + ops APIs.

- POST /v1/generate  (sync JSON or SSE streaming)
- GET  /health, /metrics (Prometheus), /api/stats, /dashboard
- Admin: /admin/tenants, /admin/billing, /admin/chaos/kill, /admin/scale

Auth is a simple per-tenant API key in `X-API-Key` (demo-grade; in
production this would be a signed JWT / API gateway integration).
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from .config import Settings
from .dashboard import dashboard_html
from .models import ChaosKillRequest, ErrorResponse, GenerateRequest, GenerateResponse, ScaleRequest
from .service import ConcurrencyExceeded, GatewayError, GatewayService, QueueFull, RateLimited
from .workers import CompletedRequest


def create_app(settings: Settings | None = None) -> FastAPI:
    service = GatewayService(settings or Settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await service.start()
        yield
        await service.stop()

    app = FastAPI(title="Inference Gateway", lifespan=lifespan)
    app.state.service = service

    def require_tenant(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        tenant = service.tenants.auth(x_api_key)
        if tenant is None:
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")
        return tenant

    # -- inference ------------------------------------------------------
    @app.post("/v1/generate", response_model=GenerateResponse,
              responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse},
                         502: {"model": ErrorResponse}, 504: {"model": ErrorResponse}})
    async def generate(req: GenerateRequest, request: Request,
                       x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        tenant = require_tenant(x_api_key)
        try:
            qreq = await service.submit(tenant, req)
        except (RateLimited, ConcurrencyExceeded, QueueFull) as e:
            raise HTTPException(status_code=e.status_code, detail=e.detail,
                                headers={"Retry-After": "2"})
        if req.stream:
            return StreamingResponse(_stream_tokens(service, qreq), media_type="text/event-stream")
        try:
            result: CompletedRequest = await asyncio.wait_for(
                qreq.done, timeout=service.settings.request_timeout_s + 10)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="request timed out")
        except asyncio.CancelledError:
            raise HTTPException(status_code=499, detail="client closed request")
        except TimeoutError as e:
            raise HTTPException(status_code=504, detail=str(e))
        except Exception as e:  # noqa: BLE001 - translate worker failures to 502
            raise HTTPException(status_code=502, detail=f"inference failed: {e}")
        finally:
            service.release(qreq)
        return _to_response(result)

    # -- ops ------------------------------------------------------------
    @app.get("/health")
    async def health():
        return {"status": "ok", "queue_depth": service.scheduler.total_queued(),
                "workers": len([w for w in service.pool.workers if w.alive])}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        return service.metrics.prometheus()

    @app.get("/api/stats")
    async def stats():
        return JSONResponse(service.stats())

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        return dashboard_html()

    @app.get("/")
    async def root():
        return {"service": "inference-gateway",
                "dashboard": "/dashboard", "metrics": "/metrics", "stats": "/api/stats"}

    # -- admin ----------------------------------------------------------
    @app.get("/admin/tenants")
    async def tenants():
        return [{"id": t.id, "name": t.name, "tier": t.tier.name,
                 "active": t.active, "requests_ok": t.requests_ok,
                 "requests_rejected": t.requests_rejected} for t in service.tenants.all()]

    @app.get("/admin/billing")
    async def billing():
        return [{"tenant_id": t.id, "tier": t.tier.name,
                 "prompt_tokens": t.prompt_tokens, "completion_tokens": t.completion_tokens,
                 "cost_usd": round(t.cost_usd, 4)} for t in service.tenants.all()]

    @app.post("/admin/chaos/kill")
    async def chaos_kill(req: ChaosKillRequest):
        if req.profile is not None and req.profile not in service.settings.profiles:
            raise HTTPException(status_code=400, detail="unknown profile")
        killed = service.pool.kill(req.count, req.profile)
        return {"killed": killed}

    @app.post("/admin/scale")
    async def scale(req: ScaleRequest):
        if req.profile not in service.settings.profiles:
            raise HTTPException(status_code=400, detail="unknown profile")
        service.pool.set_size(req.profile, req.count)
        return {"profile": req.profile, "target": req.count}

    @app.post("/admin/autoscale")
    async def autoscale(body: dict):
        enabled = bool(body.get("enabled", True))
        service.autoscaler.enabled = enabled
        return {"autoscale_enabled": enabled}

    return app


async def _stream_tokens(service: GatewayService, qreq):
    """SSE token stream. On client disconnect we cancel the request so the
    GPU doesn't keep burning tokens nobody reads."""
    waiter = asyncio.ensure_future(_await_done(service, qreq))
    try:
        while True:
            tok = await qreq.token_queue.get()
            if tok is None:
                break
            yield f"data: {json.dumps({'id': qreq.id, 'token': tok})}\n\n"
        try:
            await waiter  # surface any terminal error before [DONE]
        except HTTPException as e:
            yield f"data: {json.dumps({'error': e.detail})}\n\n"
            return
        yield "data: [DONE]\n\n"
    except (asyncio.CancelledError, GeneratorExit):
        service.cancel(qreq)
        waiter.cancel()
        raise
    finally:
        service.release(qreq)


async def _await_done(service: GatewayService, qreq):
    try:
        return await asyncio.wait_for(qreq.done, timeout=service.settings.request_timeout_s + 10)
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))


def _to_response(r: CompletedRequest) -> GenerateResponse:
    return GenerateResponse(
        id=r.id, tenant_id=r.tenant_id, model=r.model, text=r.text,
        prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
        ttft_ms=round(r.ttft_ms, 1), e2e_ms=round(r.e2e_ms, 1),
        preemptions=r.preemptions, fell_back=r.fell_back,
    )


app = create_app()
