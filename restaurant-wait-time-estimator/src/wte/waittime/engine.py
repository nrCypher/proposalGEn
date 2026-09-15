"""Live wait-time engine (blueprint §4.5 windowing, pitfalls #3, #6, #7).

Per location it keeps:
- a sliding window of completed *queue* dwell times (from ZoneEvent EXIT),
- the current queue occupancy (ENTER − EXIT),
- the recent POS order rate, used for graceful degradation when video is stale.

Output is a ``WaitEstimate`` with p50 / p90 and a rounded band ("10–15 min"),
never a single precise number.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from wte.core.schemas import (
    PosEvent,
    WaitEstimate,
    WaitSource,
    ZoneEvent,
    ZoneEventType,
    ZoneKind,
)


def _band(p50: float, p90: float) -> tuple[int, int]:
    """Round to a customer-friendly band: 5-min steps above 10 min, 2-min steps below."""
    step = 5 if p90 >= 10 else 2
    low = int(np.floor(p50 / step) * step)
    high = int(np.ceil(p90 / step) * step)
    if high <= low:
        high = low + step
    return low, high


@dataclass
class _LocationState:
    dwell: deque[tuple[datetime, float]] = field(default_factory=deque)  # (exit_ts, seconds)
    in_queue: set[tuple[str, int]] = field(default_factory=set)  # (camera_id, tracker_id)
    orders: deque[datetime] = field(default_factory=deque)
    last_video_ts: datetime | None = None


@dataclass
class WaitTimeEngine:
    window: timedelta = timedelta(minutes=30)
    min_samples: int = 5
    video_stale_after: timedelta = timedelta(seconds=120)
    pos_window: timedelta = timedelta(minutes=15)
    _state: dict[str, _LocationState] = field(default_factory=dict)

    def _loc(self, location_id: str) -> _LocationState:
        return self._state.setdefault(location_id, _LocationState())

    # ---- inputs ----------------------------------------------------------

    def on_zone_event(self, ev: ZoneEvent) -> None:
        st = self._loc(ev.location_id)
        st.last_video_ts = max(ev.ts, st.last_video_ts) if st.last_video_ts else ev.ts
        if ev.zone != ZoneKind.QUEUE:
            return
        key = (ev.camera_id, ev.tracker_id)
        if ev.event_type == ZoneEventType.ENTER:
            st.in_queue.add(key)
        else:
            st.in_queue.discard(key)
            if ev.dwell_seconds is not None and ev.dwell_seconds >= 3.0:  # drop pass-throughs
                st.dwell.append((ev.ts, ev.dwell_seconds))
        self._trim(st, ev.ts)

    def on_pos_event(self, ev: PosEvent) -> None:
        st = self._loc(ev.location_id)
        if ev.event_type in ("order_created", "check_open"):
            st.orders.append(ev.received_at)
        self._trim(st, ev.received_at)

    def on_heartbeat(self, location_id: str, ts: datetime) -> None:
        """Edge agent liveness — keeps ``source=vision`` even when the queue is empty."""
        st = self._loc(location_id)
        st.last_video_ts = max(ts, st.last_video_ts) if st.last_video_ts else ts

    # ---- output ----------------------------------------------------------

    def estimate(self, location_id: str, now: datetime) -> WaitEstimate:
        st = self._loc(location_id)
        self._trim(st, now)
        video_ok = st.last_video_ts is not None and (now - st.last_video_ts) <= self.video_stale_after

        if video_ok:
            queue_len = len(st.in_queue)
            samples = np.array([d for _, d in st.dwell], dtype=float) / 60.0
            if len(samples) >= self.min_samples:
                p50, p90 = float(np.percentile(samples, 50)), float(np.percentile(samples, 90))
                return self._build(location_id, now, queue_len, p50, p90, WaitSource.VISION,
                                   len(samples))
            if queue_len == 0:
                return self._build(location_id, now, 0, 0.0, 0.0, WaitSource.VISION, len(samples))
            # Few completed dwells yet — fall through to POS throughput if we have it.
            est = self._pos_estimate(st, now, queue_len)
            if est is not None:
                return self._build(location_id, now, queue_len, est, est * 1.5,
                                   WaitSource.VISION, len(samples))
            return WaitEstimate(location_id=location_id, ts=now, queue_length=queue_len,
                                p50_minutes=None, p90_minutes=None, band_low_minutes=None,
                                band_high_minutes=None, source=WaitSource.VISION,
                                sample_size=len(samples))

        # Video stale → POS-only (pitfall #7: deliberate fallback, not a panic mode)
        est = self._pos_estimate(st, now, queue_len=None)
        if est is None:
            return WaitEstimate(location_id=location_id, ts=now, queue_length=0,
                                p50_minutes=None, p90_minutes=None, band_low_minutes=None,
                                band_high_minutes=None, source=WaitSource.NO_DATA, sample_size=0)
        return self._build(location_id, now, 0, est, est * 1.5, WaitSource.POS_ONLY,
                           len(st.orders))

    # ---- helpers ---------------------------------------------------------

    def _pos_estimate(self, st: _LocationState, now: datetime, queue_len: int | None) -> float | None:
        """Little's law: wait ≈ queue_length / throughput. With no video, assume the
        queue is proportional to the last-5-min order burst vs the 15-min average."""
        if len(st.orders) < 3:
            return None
        minutes = self.pos_window.total_seconds() / 60.0
        rate_per_min = len(st.orders) / minutes
        if rate_per_min <= 0:
            return None
        if queue_len is None:
            recent = sum(1 for t in st.orders if (now - t) <= timedelta(minutes=5))
            queue_len = max(0, int(round(recent - rate_per_min * 5)))
        return queue_len / rate_per_min

    def _build(self, location_id: str, now: datetime, queue_len: int, p50: float, p90: float,
               source: WaitSource, n: int) -> WaitEstimate:
        low, high = _band(p50, p90)
        return WaitEstimate(location_id=location_id, ts=now, queue_length=queue_len,
                            p50_minutes=round(p50, 1), p90_minutes=round(p90, 1),
                            band_low_minutes=low, band_high_minutes=high, source=source,
                            sample_size=n)

    def _trim(self, st: _LocationState, now: datetime) -> None:
        while st.dwell and (now - st.dwell[0][0]) > self.window:
            st.dwell.popleft()
        while st.orders and (now - st.orders[0]) > self.pos_window:
            st.orders.popleft()
