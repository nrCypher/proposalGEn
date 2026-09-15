"""FastAPI BFF (blueprint §7.1): REST for CRUD/read models, WebSocket for live pushes,
HMAC-verified ingest for edge agents and POS/booking webhooks.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from wte.api.auth import Principal, issue_token, principal_from_request, principal_from_ws, verify_edge_signature
from wte.api.state import AppState, CameraConfig
from wte.connectors.base import SignatureError
from wte.core.schemas import Forecast, FrameMetadata, WaitEstimate
from wte.core.settings import DEV_JWT_SECRET, Settings, get_settings
from wte.forecast.service import aggregate_to_buckets
from wte.vision.zones import zones_from_config

log = logging.getLogger("wte.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state: AppState = app.state.wte
        if settings.database_url:
            from wte.db.repo import Repository

            repo = Repository(settings.database_url)
            await repo.start()
            state.repo = repo
        yield
        if state.repo is not None:
            await state.repo.stop()

    app = FastAPI(title="Restaurant Wait-Time Estimator", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.settings = settings
    app.state.wte = AppState(settings=settings)
    _register_routes(app)
    return app


def _state(request: Request) -> AppState:
    return request.app.state.wte


def _now(request: Request | WebSocket) -> datetime:
    """Injectable clock (``app.state.clock``) so tests and replays can freeze time."""
    clock = getattr(request.app.state, "clock", None)
    return clock() if clock else datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #

class CameraIn(BaseModel):
    location_id: str
    camera_id: str
    zones: dict[str, list[list[float]]] = Field(
        description='{"queue": [[x,y],...], "service": [...], "fulfil": [...]}')


class WaitOut(BaseModel):
    estimate: WaitEstimate
    label: str
    why: dict[str, Any] | None = None  # latest VLM assessment, if any


class ObservationsIn(BaseModel):
    """Historical (ts, wait_minutes) points used to (re)train the forecast (dev/testing)."""

    points: list[tuple[datetime, float]]


class DevTokenIn(BaseModel):
    tenant_id: UUID
    subject: str = "dev"
    role: str = "operator"


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

def _register_routes(app: FastAPI) -> None:

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # ---- dev auth (disable in prod: real IdP issues tokens) ----------------
    @app.post("/v1/dev/token")
    async def dev_token(body: DevTokenIn, request: Request) -> dict[str, str]:
        settings: Settings = request.app.state.settings
        if settings.jwt_secret != DEV_JWT_SECRET:
            raise HTTPException(404)
        return {"token": issue_token(settings, body.tenant_id, body.subject, body.role)}

    # ---- cameras -----------------------------------------------------------
    @app.post("/v1/cameras", status_code=201)
    async def register_camera(body: CameraIn, request: Request,
                              p: Principal = Depends(principal_from_request)) -> dict[str, str]:
        st = _state(request)
        st.register_camera(CameraConfig(tenant_id=p.tenant_id, location_id=body.location_id,
                                        camera_id=body.camera_id,
                                        zones=zones_from_config({"zones": body.zones})))
        return {"camera_id": body.camera_id, "status": "registered"}

    @app.get("/v1/cameras")
    async def list_cameras(request: Request,
                           p: Principal = Depends(principal_from_request)) -> list[dict[str, Any]]:
        st = _state(request)
        return [{"location_id": c.location_id, "camera_id": c.camera_id,
                 "zones": [z.kind.value for z in c.zones]}
                for (tid, _), c in st.cameras.items() if tid == p.tenant_id]

    # ---- edge ingest (HMAC, not JWT) ----------------------------------------
    @app.post("/v1/edge/frames", status_code=202)
    async def edge_frames(request: Request) -> dict[str, int]:
        settings: Settings = request.app.state.settings
        body = await request.body()
        verify_edge_signature(settings, body, request.headers.get("x-wte-signature", ""))
        meta = FrameMetadata.model_validate_json(body)
        if not meta.faces_blurred:
            raise HTTPException(422, "edge must blur faces before upload (§5.2)")
        try:
            events = await _state(request).ingest_frame(meta)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        return {"events": len(events)}

    # ---- POS / booking webhooks ----------------------------------------------
    @app.post("/v1/webhooks/{provider}/{tenant_id}/{location_id}", status_code=202)
    async def webhook(provider: str, tenant_id: UUID, location_id: str,
                      request: Request) -> dict[str, int]:
        st = _state(request)
        if provider not in st.connectors:
            raise HTTPException(404, f"unknown provider {provider}")
        body = await request.body()
        try:
            n = await st.ingest_webhook(provider, body, dict(request.headers), tenant_id, location_id)
        except SignatureError as e:
            raise HTTPException(401, str(e)) from e
        return {"accepted": n}

    # ---- read models -----------------------------------------------------------
    @app.get("/v1/locations/{location_id}/wait", response_model=WaitOut)
    async def get_wait(location_id: str, request: Request,
                       p: Principal = Depends(principal_from_request)) -> WaitOut:
        st = _state(request)
        est = st.estimate(p.tenant_id, location_id, _now(request))
        why = st.vlm_assessments.get((p.tenant_id, location_id))
        return WaitOut(estimate=est, label=est.label,
                       why=why.model_dump() if why else None)

    @app.get("/v1/locations/{location_id}/forecast", response_model=Forecast)
    async def get_forecast(location_id: str, request: Request,
                           hours: int = Query(24, ge=1, le=24 * 7),
                           p: Principal = Depends(principal_from_request)) -> Forecast:
        st = _state(request)
        fc = st.forecast.latest(location_id)
        if fc is None:
            raise HTTPException(404, "no forecast yet — POST observations or wait for nightly run")
        cutoff = _now(request) + timedelta(hours=hours)
        return fc.model_copy(update={"points": [pt for pt in fc.points if pt.ts <= cutoff]})

    @app.post("/v1/locations/{location_id}/forecast/train", response_model=Forecast)
    async def train_forecast(location_id: str, body: ObservationsIn, request: Request,
                             p: Principal = Depends(principal_from_request)) -> Forecast:
        st = _state(request)
        series = aggregate_to_buckets(body.points)
        fc = st.forecast.run(location_id, series, _now(request))
        if st.repo is not None:
            await st.repo.insert_forecast(p.tenant_id, fc)
        return fc

    @app.get("/v1/locations/{location_id}/best-times")
    async def best_times(location_id: str, request: Request, n: int = Query(3, ge=1, le=10),
                         p: Principal = Depends(principal_from_request)) -> list[dict[str, Any]]:
        st = _state(request)
        fc = st.forecast.latest(location_id)
        if fc is None:
            raise HTTPException(404, "no forecast yet")
        horizon = [pt for pt in fc.points if pt.ts <= _now(request) + timedelta(hours=24)]
        window = fc.model_copy(update={"points": horizon})
        return [{"ts": pt.ts.isoformat(), "q10": pt.q10, "q50": pt.q50, "q90": pt.q90}
                for pt in window.best_windows(n=n)]

    @app.get("/v1/ops/vlm-triggers")
    async def vlm_triggers(request: Request, limit: int = 50,
                           p: Principal = Depends(principal_from_request)) -> list[dict[str, Any]]:
        st = _state(request)
        cams = {c.camera_id for (tid, _), c in st.cameras.items() if tid == p.tenant_id}
        return [t.model_dump(mode="json") for t in st.vlm_triggers if t.camera_id in cams][-limit:]

    # ---- live push -------------------------------------------------------------
    @app.websocket("/v1/ws/locations/{location_id}")
    async def ws_location(ws: WebSocket, location_id: str) -> None:
        p = await principal_from_ws(ws)
        if p is None:
            await ws.close(code=4401)
            return
        st: AppState = ws.app.state.wte
        await ws.accept()
        st.hub.add(p.tenant_id, location_id, ws)
        try:
            est = st.estimate(p.tenant_id, location_id, _now(ws))
            await ws.send_json({"type": "wait_estimate", **est.model_dump(mode="json"),
                                "label": est.label})
            while True:
                await ws.receive_text()  # keep-alive / client pings
        except WebSocketDisconnect:
            pass
        finally:
            st.hub.remove(p.tenant_id, location_id, ws)


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("wte.api.main:app", host="0.0.0.0", port=8080, reload=False)
