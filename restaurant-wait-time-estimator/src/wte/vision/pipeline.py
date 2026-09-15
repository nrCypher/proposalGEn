"""Tier-1 per-camera pipeline: detect → track → zone → events (blueprint §2.6).

Adds the frame-rate adaptation from §1.6: when nothing is detected the caller is
told to sample at the idle rate; when people appear it snaps back to active.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from wte.core.schemas import Detection, ZoneEvent, ZoneKind
from wte.vision.detector import Detector
from wte.vision.tracker import Tracker
from wte.vision.zones import ZoneEngine


@dataclass
class FrameResult:
    ts: datetime
    detections: list[Detection]
    events: list[ZoneEvent]
    queue_occupancy: int
    service_occupancy: int
    suggested_fps: float


@dataclass
class CameraPipeline:
    detector: Detector
    tracker: Tracker
    zone_engine: ZoneEngine
    fps_active: float = 10.0
    fps_idle: float = 3.0
    idle_after_empty_frames: int = 30
    _empty_streak: int = field(default=0, init=False)

    def process(self, frame: Any, ts: datetime) -> FrameResult:
        raw = self.detector.detect(frame)
        tracked = self.tracker.update(raw)
        events = self.zone_engine.update(tracked, ts)

        self._empty_streak = self._empty_streak + 1 if not tracked else 0
        fps = self.fps_idle if self._empty_streak >= self.idle_after_empty_frames else self.fps_active

        return FrameResult(
            ts=ts,
            detections=tracked,
            events=events,
            queue_occupancy=self.zone_engine.occupancy(ZoneKind.QUEUE),
            service_occupancy=self.zone_engine.occupancy(ZoneKind.SERVICE),
            suggested_fps=fps,
        )

    def process_tracked(self, tracked: list[Detection], ts: datetime) -> FrameResult:
        """Cloud-side entry point when the edge already ran detection + tracking."""
        events = self.zone_engine.update(tracked, ts)
        return FrameResult(
            ts=ts, detections=tracked, events=events,
            queue_occupancy=self.zone_engine.occupancy(ZoneKind.QUEUE),
            service_occupancy=self.zone_engine.occupancy(ZoneKind.SERVICE),
            suggested_fps=self.fps_active,
        )
