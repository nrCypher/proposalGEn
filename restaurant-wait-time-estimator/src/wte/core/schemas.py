"""Shared data contracts.

Everything that crosses a service boundary is defined here so every service
(edge agent, zone engine, wait-time engine, connectors, API) speaks one schema.

Privacy note (blueprint §5.3): persisted events carry *only* timestamp,
anonymous tracker_id, zone, location and dwell. No image data, no identity.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Vision (tier-1)
# --------------------------------------------------------------------------- #

class BBox(BaseModel):
    """Axis-aligned box in pixel coordinates (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def anchor(self) -> tuple[float, float]:
        """Bottom-centre point — the foot position for top-down / oblique cameras."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class Detection(BaseModel):
    bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0)
    tracker_id: int | None = None


class ZoneKind(StrEnum):
    QUEUE = "queue"
    SERVICE = "service"
    FULFILMENT = "fulfil"


class ZoneEventType(StrEnum):
    ENTER = "enter"
    EXIT = "exit"


class ZoneEvent(BaseModel):
    """Emitted by the zone engine when an anonymous track crosses a zone boundary."""

    event_id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID
    location_id: str
    camera_id: str
    zone: ZoneKind
    event_type: ZoneEventType
    tracker_id: int
    ts: datetime
    dwell_seconds: float | None = None  # set on EXIT only


class FrameMetadata(BaseModel):
    """What the edge agent uploads per sampled frame (~1 KB). Never the pixels."""

    tenant_id: UUID
    location_id: str
    camera_id: str
    ts: datetime
    detections: list[Detection]
    frame_w: int
    frame_h: int
    faces_blurred: bool = True


# --------------------------------------------------------------------------- #
# POS / booking (connector hub)
# --------------------------------------------------------------------------- #

PosEventType = Literal[
    "order_created",
    "order_completed",
    "order_cancelled",
    "check_open",
    "check_closed",
]
Channel = Literal["dine_in", "takeaway", "delivery"]


class PosEvent(BaseModel):
    """Normalised POS event (blueprint §6.1)."""

    tenant_id: UUID
    location_id: str
    provider: str
    event_id: str
    event_type: PosEventType
    occurred_at: datetime
    received_at: datetime = Field(default_factory=utcnow)
    order_id: str
    party_size: int | None = None
    channel: Channel = "dine_in"
    raw: dict = Field(default_factory=dict)


BookingEventType = Literal["reservation_created", "reservation_seated", "reservation_cancelled"]


class BookingEvent(BaseModel):
    tenant_id: UUID
    location_id: str
    provider: str
    event_id: str
    event_type: BookingEventType
    occurred_at: datetime
    received_at: datetime = Field(default_factory=utcnow)
    reservation_id: str
    party_size: int | None = None
    slot_at: datetime | None = None
    raw: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Wait-time (live) and forecast
# --------------------------------------------------------------------------- #

class WaitSource(StrEnum):
    VISION = "vision"
    POS_ONLY = "pos_only"  # graceful degradation when video is unavailable
    NO_DATA = "no_data"


class WaitEstimate(BaseModel):
    """Customer-facing estimate. Percentiles, never a single precise mean (pitfalls #3, #6)."""

    location_id: str
    ts: datetime
    queue_length: int
    p50_minutes: float | None
    p90_minutes: float | None
    band_low_minutes: int | None
    band_high_minutes: int | None
    source: WaitSource
    sample_size: int

    @property
    def label(self) -> str:
        if self.band_low_minutes is None or self.band_high_minutes is None:
            return "n/a"
        if self.band_low_minutes == 0 and self.band_high_minutes <= 2:
            return "no wait"
        return f"{self.band_low_minutes}–{self.band_high_minutes} min"


class ForecastPoint(BaseModel):
    ts: datetime
    q10: float
    q50: float
    q90: float


class Forecast(BaseModel):
    location_id: str
    generated_at: datetime
    model: str  # "seasonal_naive" | "prophet" | "chronos2" | "tft"
    tier: Literal["cold", "warm", "mature"]
    step_minutes: int = 15
    points: list[ForecastPoint]

    def best_windows(self, n: int = 3, min_gap_steps: int = 4) -> list[ForecastPoint]:
        """Lowest-q50 points, spaced at least ``min_gap_steps`` apart."""
        ranked = sorted(self.points, key=lambda p: p.q50)
        chosen: list[ForecastPoint] = []
        for p in ranked:
            if all(abs((p.ts - c.ts).total_seconds()) >= min_gap_steps * self.step_minutes * 60
                   for c in chosen):
                chosen.append(p)
            if len(chosen) >= n:
                break
        return sorted(chosen, key=lambda p: p.ts)


# --------------------------------------------------------------------------- #
# VLM (tier-2)
# --------------------------------------------------------------------------- #

class Bottleneck(BaseModel):
    location: Literal["counter", "kitchen", "fulfillment", "none"]
    evidence: str


class VlmAssessment(BaseModel):
    """Matches the JSON schema in blueprint §3.4."""

    queue_state: Literal["short", "normal", "long", "gridlocked"]
    estimated_wait_minutes: float
    bottleneck: Bottleneck
    operational_recommendation: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class VlmTrigger(BaseModel):
    reason: str
    location_id: str
    camera_id: str
    ts: datetime
    metadata_summary: str
