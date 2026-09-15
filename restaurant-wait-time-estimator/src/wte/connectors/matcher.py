"""Vision ↔ POS reconciliation (blueprint §6.3 step 1, pitfall #5).

A ``check_open`` means a party was seated / an order was taken. We pair it with a
queue EXIT whose exit time is within ±``tolerance`` of the POS *server-arrival*
time (never the POS clock), preferring the closest exit. Pairing lets us stop the
queue timer precisely and label dwell samples as "served" vs "abandoned".
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from wte.core.schemas import PosEvent, ZoneEvent, ZoneEventType, ZoneKind


@dataclass
class Match:
    pos_event_id: str
    tracker_id: int
    camera_id: str
    delta_seconds: float
    dwell_seconds: float | None


@dataclass
class VisionPosMatcher:
    tolerance: timedelta = timedelta(seconds=60)
    keep: timedelta = timedelta(minutes=10)
    _exits: dict[str, deque[ZoneEvent]] = field(default_factory=dict)
    _pending_pos: dict[str, deque[PosEvent]] = field(default_factory=dict)

    def on_zone_event(self, ev: ZoneEvent) -> Match | None:
        if ev.zone != ZoneKind.QUEUE or ev.event_type != ZoneEventType.EXIT:
            return None
        q = self._exits.setdefault(ev.location_id, deque())
        q.append(ev)
        self._trim(ev.location_id, ev.ts)
        # A POS event may have arrived slightly *before* the exit was emitted.
        for pos in list(self._pending_pos.get(ev.location_id, [])):
            m = self._try(pos, ev)
            if m:
                self._pending_pos[ev.location_id].remove(pos)
                q.remove(ev)
                return m
        return None

    def on_pos_event(self, ev: PosEvent) -> Match | None:
        if ev.event_type != "check_open":
            return None
        self._trim(ev.location_id, ev.received_at)
        best: tuple[float, ZoneEvent] | None = None
        for ex in self._exits.get(ev.location_id, []):
            d = abs((ex.ts - ev.received_at).total_seconds())
            if d <= self.tolerance.total_seconds() and (best is None or d < best[0]):
                best = (d, ex)
        if best is None:
            self._pending_pos.setdefault(ev.location_id, deque()).append(ev)
            return None
        self._exits[ev.location_id].remove(best[1])
        return Match(pos_event_id=ev.event_id, tracker_id=best[1].tracker_id,
                     camera_id=best[1].camera_id, delta_seconds=best[0],
                     dwell_seconds=best[1].dwell_seconds)

    def _try(self, pos: PosEvent, ex: ZoneEvent) -> Match | None:
        d = abs((ex.ts - pos.received_at).total_seconds())
        if d > self.tolerance.total_seconds():
            return None
        return Match(pos_event_id=pos.event_id, tracker_id=ex.tracker_id, camera_id=ex.camera_id,
                     delta_seconds=d, dwell_seconds=ex.dwell_seconds)

    def _trim(self, location_id: str, now: datetime) -> None:
        for store in (self._exits, self._pending_pos):
            q = store.get(location_id)
            if not q:
                continue
            while q and (now - _ts(q[0])) > self.keep:
                q.popleft()


def _ts(ev: ZoneEvent | PosEvent) -> datetime:
    return ev.ts if isinstance(ev, ZoneEvent) else ev.received_at
