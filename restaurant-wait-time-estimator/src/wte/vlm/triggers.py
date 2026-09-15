"""Tier-2 trigger logic (blueprint §3.2): decide *when* to spend a VLM call.

Cadence: at most one call per ``cadence_s`` per camera on a timer, plus
immediate calls on anomalies (debounced by ``min_gap_s``):

1. queue length jumped > 50 % in 60 s
2. average dwell > p95 of the historical baseline
3. service zone empty while the queue is long (staffing)
4. customer count high but POS order rate flat (bottleneck forming)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from wte.core.schemas import VlmTrigger


@dataclass
class CameraSnapshot:
    ts: datetime
    queue_len: int
    service_len: int
    avg_dwell_s: float | None
    pos_orders_last_5m: int


@dataclass
class TriggerPolicy:
    cadence_s: int = 60
    min_gap_s: int = 30
    long_queue: int = 8
    high_count: int = 12
    baseline_max: int = 2000
    _last_fire: dict[str, datetime] = field(default_factory=dict)
    _history: dict[str, deque[CameraSnapshot]] = field(default_factory=dict)
    _dwell_baseline: dict[str, deque[float]] = field(default_factory=dict)

    def record_dwell(self, camera_id: str, dwell_s: float) -> None:
        q = self._dwell_baseline.setdefault(camera_id, deque(maxlen=self.baseline_max))
        q.append(dwell_s)

    def evaluate(self, location_id: str, camera_id: str, snap: CameraSnapshot) -> VlmTrigger | None:
        hist = self._history.setdefault(camera_id, deque(maxlen=120))
        reasons = self._anomalies(camera_id, snap, hist)
        hist.append(snap)

        last = self._last_fire.get(camera_id)
        gap = (snap.ts - last).total_seconds() if last else float("inf")

        reason: str | None = None
        if reasons and gap >= self.min_gap_s:
            reason = "anomaly:" + ",".join(reasons)
        elif gap >= self.cadence_s:
            reason = "cadence"
        if reason is None:
            return None

        self._last_fire[camera_id] = snap.ts
        return VlmTrigger(reason=reason, location_id=location_id, camera_id=camera_id,
                          ts=snap.ts, metadata_summary=self._summary(snap, hist))

    def _anomalies(self, camera_id: str, snap: CameraSnapshot,
                   hist: deque[CameraSnapshot]) -> list[str]:
        out: list[str] = []
        minute_ago = [h for h in hist if (snap.ts - h.ts) <= timedelta(seconds=60)]
        if minute_ago:
            base = minute_ago[0].queue_len
            if base >= 2 and snap.queue_len > base * 1.5:
                out.append("queue_jump")
        baseline = self._dwell_baseline.get(camera_id)
        if baseline and len(baseline) >= 30 and snap.avg_dwell_s is not None:
            if snap.avg_dwell_s > float(np.percentile(np.array(baseline), 95)):
                out.append("dwell_p95")
        if snap.queue_len >= self.long_queue and snap.service_len == 0:
            out.append("service_empty")
        if snap.queue_len >= self.high_count and minute_ago:
            prior = [h.pos_orders_last_5m for h in hist if (snap.ts - h.ts) <= timedelta(minutes=5)]
            if prior and snap.pos_orders_last_5m <= min(prior):
                out.append("pos_flat")
        return out

    @staticmethod
    def _summary(snap: CameraSnapshot, hist: deque[CameraSnapshot]) -> str:
        recent = list(hist)[-6:]
        trend = " → ".join(str(h.queue_len) for h in recent) or "n/a"
        dwell = f"{snap.avg_dwell_s / 60:.1f} min" if snap.avg_dwell_s else "n/a"
        return (f"Queue now {snap.queue_len} (last minute: {trend}); service zone "
                f"{snap.service_len} staff/customers; avg dwell {dwell}; "
                f"{snap.pos_orders_last_5m} POS orders in last 5 min.")
