"""Application state: wires the services together over the event bus.

In-memory by default (dev / tests). With ``WTE_DATABASE_URL`` set, events are
also persisted through the RLS-scoped repository.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import WebSocket
from pydantic import BaseModel

from wte.connectors.base import Connector, MemoryIdempotencyStore
from wte.connectors.matcher import VisionPosMatcher
from wte.connectors.sevenrooms import SevenRoomsConnector
from wte.connectors.square import SquareConnector
from wte.connectors.toast import ToastConnector
from wte.core import bus as topics
from wte.core.bus import InMemoryBus
from wte.core.schemas import (
    BookingEvent,
    FrameMetadata,
    PosEvent,
    VlmAssessment,
    VlmTrigger,
    WaitEstimate,
    ZoneEvent,
    ZoneEventType,
    ZoneKind,
)
from wte.core.settings import Settings
from wte.forecast.service import ForecastService
from wte.vision.zones import Zone, ZoneEngine
from wte.vlm.triggers import CameraSnapshot, TriggerPolicy
from wte.waittime.engine import WaitTimeEngine


class WsHub:
    """Fan-out of live estimates to dashboard / widget sockets, keyed by (tenant, location).

    Production: back this with Redis pub/sub so it scales across pods (§8.1).
    """

    def __init__(self) -> None:
        self._subs: dict[tuple[UUID, str], set[WebSocket]] = defaultdict(set)

    def add(self, tenant_id: UUID, location_id: str, ws: WebSocket) -> None:
        self._subs[(tenant_id, location_id)].add(ws)

    def remove(self, tenant_id: UUID, location_id: str, ws: WebSocket) -> None:
        self._subs[(tenant_id, location_id)].discard(ws)

    async def broadcast(self, tenant_id: UUID, location_id: str, message: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._subs.get((tenant_id, location_id), ())):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.remove(tenant_id, location_id, ws)


@dataclass
class CameraConfig:
    tenant_id: UUID
    location_id: str
    camera_id: str
    zones: list[Zone]


@dataclass
class AppState:
    settings: Settings
    bus: InMemoryBus = field(default_factory=InMemoryBus)
    wait: WaitTimeEngine = field(init=False)
    matcher: VisionPosMatcher = field(default_factory=VisionPosMatcher)
    triggers: TriggerPolicy = field(init=False)
    forecast: ForecastService = field(default_factory=ForecastService)
    idempotency: MemoryIdempotencyStore = field(default_factory=MemoryIdempotencyStore)
    hub: WsHub = field(default_factory=WsHub)
    connectors: dict[str, Connector] = field(default_factory=dict)
    cameras: dict[tuple[UUID, str], CameraConfig] = field(default_factory=dict)
    zone_engines: dict[tuple[UUID, str], ZoneEngine] = field(default_factory=dict)
    repo: Any = None
    # observability / debugging
    last_estimates: dict[tuple[UUID, str], WaitEstimate] = field(default_factory=dict)
    vlm_triggers: list[VlmTrigger] = field(default_factory=list)
    vlm_assessments: dict[tuple[UUID, str], VlmAssessment] = field(default_factory=dict)
    pos_orders_5m: dict[tuple[UUID, str], list[datetime]] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        s = self.settings
        self.wait = WaitTimeEngine(min_samples=s.wait_min_samples)
        self.triggers = TriggerPolicy(cadence_s=s.vlm_cadence_seconds, min_gap_s=s.vlm_min_gap_seconds)
        self.connectors = {
            "toast": ToastConnector(s.toast_webhook_secret),
            "square": SquareConnector(s.square_webhook_signature_key, s.square_notification_url),
            "sevenrooms": SevenRoomsConnector(s.sevenrooms_webhook_secret),
        }
        self.bus.subscribe(topics.TOPIC_ZONE_EVENTS, self._on_zone_event)
        self.bus.subscribe(topics.TOPIC_POS_EVENTS, self._on_pos_event)
        self.bus.subscribe(topics.TOPIC_BOOKING_EVENTS, self._on_booking_event)

    # ---- camera registry ---------------------------------------------------

    def register_camera(self, cfg: CameraConfig) -> None:
        key = (cfg.tenant_id, cfg.camera_id)
        self.cameras[key] = cfg
        self.zone_engines[key] = ZoneEngine(tenant_id=cfg.tenant_id, location_id=cfg.location_id,
                                            camera_id=cfg.camera_id, zones=cfg.zones)

    # ---- ingest ------------------------------------------------------------

    async def ingest_frame(self, meta: FrameMetadata) -> list[ZoneEvent]:
        key = (meta.tenant_id, meta.camera_id)
        engine = self.zone_engines.get(key)
        if engine is None:
            raise KeyError(f"unknown camera {meta.camera_id}")
        async with self._lock:
            events = engine.update(meta.detections, meta.ts)
            self.wait.on_heartbeat(meta.location_id, meta.ts)
        for ev in events:
            await self.bus.publish(topics.TOPIC_ZONE_EVENTS, ev)
        if self.repo is not None and events:
            await self.repo.insert_zone_events(events)

        snap = CameraSnapshot(
            ts=meta.ts, queue_len=engine.occupancy(ZoneKind.QUEUE),
            service_len=engine.occupancy(ZoneKind.SERVICE),
            avg_dwell_s=self._avg_recent_dwell(events),
            pos_orders_last_5m=self._orders_5m(meta.tenant_id, meta.location_id, meta.ts))
        trig = self.triggers.evaluate(meta.location_id, meta.camera_id, snap)
        if trig:
            self.vlm_triggers.append(trig)
            await self.bus.publish(topics.TOPIC_VLM_TRIGGERS, trig)

        await self._publish_estimate(meta.tenant_id, meta.location_id, meta.ts)
        return events

    async def ingest_webhook(self, provider: str, body: bytes, headers: dict[str, str],
                             tenant_id: UUID, location_id: str) -> int:
        import json

        connector = self.connectors[provider]
        connector.verify(body, headers)
        payload = json.loads(body or b"{}")
        events = connector.normalize(payload, tenant_id, location_id)
        n = 0
        for ev in events:
            if self.idempotency.seen(provider, ev.event_id):
                continue
            n += 1
            topic = topics.TOPIC_POS_EVENTS if isinstance(ev, PosEvent) else topics.TOPIC_BOOKING_EVENTS
            await self.bus.publish(topic, ev)
        return n

    # ---- bus handlers -------------------------------------------------------

    async def _on_zone_event(self, _: str, ev: BaseModel) -> None:
        assert isinstance(ev, ZoneEvent)
        self.wait.on_zone_event(ev)
        self.matcher.on_zone_event(ev)
        if ev.event_type == ZoneEventType.EXIT and ev.zone == ZoneKind.QUEUE and ev.dwell_seconds:
            self.triggers.record_dwell(ev.camera_id, ev.dwell_seconds)

    async def _on_pos_event(self, _: str, ev: BaseModel) -> None:
        assert isinstance(ev, PosEvent)
        self.wait.on_pos_event(ev)
        self.matcher.on_pos_event(ev)
        if ev.event_type in ("order_created", "check_open"):
            self.pos_orders_5m.setdefault((ev.tenant_id, ev.location_id), []).append(ev.received_at)
        if self.repo is not None:
            await self.repo.insert_pos_event(ev)
        await self._publish_estimate(ev.tenant_id, ev.location_id, ev.received_at)

    async def _on_booking_event(self, _: str, ev: BaseModel) -> None:
        assert isinstance(ev, BookingEvent)
        # MVP: a seated reservation behaves like a check_open for capacity purposes.
        if ev.event_type == "reservation_seated":
            await self._publish_estimate(ev.tenant_id, ev.location_id, ev.received_at)

    # ---- helpers -----------------------------------------------------------

    async def _publish_estimate(self, tenant_id: UUID, location_id: str, now: datetime) -> None:
        est = self.wait.estimate(location_id, now)
        self.last_estimates[(tenant_id, location_id)] = est
        await self.bus.publish(topics.TOPIC_WAIT_ESTIMATES, est)
        await self.hub.broadcast(tenant_id, location_id,
                                 {"type": "wait_estimate", **est.model_dump(mode="json"),
                                  "label": est.label})

    def estimate(self, tenant_id: UUID, location_id: str, now: datetime) -> WaitEstimate:
        est = self.wait.estimate(location_id, now)
        self.last_estimates[(tenant_id, location_id)] = est
        return est

    def _orders_5m(self, tenant_id: UUID, location_id: str, now: datetime) -> int:
        lst = self.pos_orders_5m.get((tenant_id, location_id), [])
        lst[:] = [t for t in lst if (now - t).total_seconds() <= 300]
        return len(lst)

    @staticmethod
    def _avg_recent_dwell(events: list[ZoneEvent]) -> float | None:
        d = [e.dwell_seconds for e in events
             if e.event_type == ZoneEventType.EXIT and e.zone == ZoneKind.QUEUE and e.dwell_seconds]
        return sum(d) / len(d) if d else None
