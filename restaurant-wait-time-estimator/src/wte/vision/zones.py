"""Zone engine: polygon hit-testing + dwell bookkeeping → ZoneEvents.

Dependency-free (pure Python point-in-polygon) so the same logic runs on the
edge, in the cloud and in tests. ``supervision.PolygonZone`` can be swapped in
for the hot path; the event contract is identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from wte.core.schemas import Detection, ZoneEvent, ZoneEventType, ZoneKind

Point = tuple[float, float]


def point_in_polygon(pt: Point, polygon: list[Point]) -> bool:
    """Ray-casting test. Works for concave polygons; vertices in any winding."""
    x, y = pt
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y)
        if intersects:
            x_at_y = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


@dataclass
class Zone:
    kind: ZoneKind
    polygon: list[Point]

    def contains(self, det: Detection) -> bool:
        return point_in_polygon(det.bbox.anchor, self.polygon)


@dataclass
class ZoneEngine:
    """Stateful per-camera engine. Feed it tracked detections; it emits enter/exit events.

    Tracks that vanish (tracker lost them) are closed lazily by ``expire`` — a track
    that hasn't been seen for ``max_silence_s`` is treated as having exited.
    """

    tenant_id: UUID
    location_id: str
    camera_id: str
    zones: list[Zone]
    max_silence_s: float = 15.0
    _enter_ts: dict[tuple[int, ZoneKind], datetime] = field(default_factory=dict)
    _last_seen: dict[int, datetime] = field(default_factory=dict)

    def occupancy(self, kind: ZoneKind) -> int:
        return sum(1 for (_, k) in self._enter_ts if k == kind)

    def update(self, detections: list[Detection], ts: datetime) -> list[ZoneEvent]:
        events: list[ZoneEvent] = []
        present: set[tuple[int, ZoneKind]] = set()

        for det in detections:
            if det.tracker_id is None:
                continue
            tid = det.tracker_id
            self._last_seen[tid] = ts
            for zone in self.zones:
                key = (tid, zone.kind)
                inside = zone.contains(det)
                if inside:
                    present.add(key)
                    if key not in self._enter_ts:
                        self._enter_ts[key] = ts
                        events.append(self._event(tid, zone.kind, ZoneEventType.ENTER, ts))

        # Exits: keys we were tracking that are no longer inside (track still visible)
        for key in list(self._enter_ts):
            tid, kind = key
            if key not in present and self._last_seen.get(tid) == ts:
                events.append(self._close(key, ts))

        events.extend(self.expire(ts))
        return events

    def expire(self, ts: datetime) -> list[ZoneEvent]:
        """Close zones for tracks not seen for ``max_silence_s`` (tracker lost them)."""
        events: list[ZoneEvent] = []
        for key in list(self._enter_ts):
            tid, _ = key
            last = self._last_seen.get(tid)
            if last is not None and (ts - last).total_seconds() > self.max_silence_s:
                events.append(self._close(key, last))
        for tid, last in list(self._last_seen.items()):
            if (ts - last).total_seconds() > self.max_silence_s:
                del self._last_seen[tid]
        return events

    def _close(self, key: tuple[int, ZoneKind], ts: datetime) -> ZoneEvent:
        tid, kind = key
        enter = self._enter_ts.pop(key)
        dwell = max(0.0, (ts - enter).total_seconds())
        return self._event(tid, kind, ZoneEventType.EXIT, ts, dwell)

    def _event(self, tid: int, kind: ZoneKind, et: ZoneEventType, ts: datetime,
               dwell: float | None = None) -> ZoneEvent:
        return ZoneEvent(
            tenant_id=self.tenant_id, location_id=self.location_id, camera_id=self.camera_id,
            zone=kind, event_type=et, tracker_id=tid, ts=ts, dwell_seconds=dwell,
        )


def zones_from_config(cfg: dict) -> list[Zone]:
    """Build zones from a per-camera YAML/JSON dict::

        zones:
          queue:   [[100, 400], [600, 400], [600, 700], [100, 700]]
          service: [[600, 300], [900, 300], [900, 700], [600, 700]]
    """
    out: list[Zone] = []
    for name, poly in cfg.get("zones", {}).items():
        out.append(Zone(kind=ZoneKind(name), polygon=[(float(x), float(y)) for x, y in poly]))
    return out
