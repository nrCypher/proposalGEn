from datetime import timedelta

from wte.core.schemas import PosEvent, WaitSource, ZoneEvent, ZoneEventType, ZoneKind
from wte.waittime.engine import WaitTimeEngine, _band

from tests.conftest import CAMERA, LOCATION, TENANT_A, at


def zev(tid: int, et: ZoneEventType, t: float, dwell: float | None = None,
        zone: ZoneKind = ZoneKind.QUEUE) -> ZoneEvent:
    return ZoneEvent(tenant_id=TENANT_A, location_id=LOCATION, camera_id=CAMERA, zone=zone,
                     event_type=et, tracker_id=tid, ts=at(t), dwell_seconds=dwell)


def pos(t: float, et: str = "order_created") -> PosEvent:
    return PosEvent(tenant_id=TENANT_A, location_id=LOCATION, provider="toast",
                    event_id=f"e{t}", event_type=et, occurred_at=at(t), received_at=at(t),
                    order_id=f"o{t}")


def test_band_rounding():
    assert _band(6.2, 9.8) == (6, 10)      # < 10 min → 2-min steps
    assert _band(12.0, 17.5) == (10, 20)   # ≥ 10 min → 5-min steps
    assert _band(0.0, 0.0) == (0, 2)       # never an empty band


def test_percentiles_from_completed_dwells():
    eng = WaitTimeEngine(min_samples=3)
    for i, d in enumerate([240, 300, 360, 420, 600]):  # seconds
        eng.on_zone_event(zev(i, ZoneEventType.ENTER, 0))
        eng.on_zone_event(zev(i, ZoneEventType.EXIT, d, dwell=d))
    eng.on_zone_event(zev(99, ZoneEventType.ENTER, 610))  # one person still waiting
    est = eng.estimate(LOCATION, at(620))
    assert est.source == WaitSource.VISION
    assert est.queue_length == 1
    assert est.sample_size == 5
    assert est.p50_minutes == 6.0
    assert est.p90_minutes > est.p50_minutes
    assert est.band_low_minutes <= est.p50_minutes <= est.p90_minutes <= est.band_high_minutes
    assert "min" in est.label


def test_pass_throughs_are_dropped():
    eng = WaitTimeEngine(min_samples=1)
    eng.on_zone_event(zev(1, ZoneEventType.ENTER, 0))
    eng.on_zone_event(zev(1, ZoneEventType.EXIT, 1, dwell=1.0))  # walked through, < 3 s
    est = eng.estimate(LOCATION, at(2))
    assert est.sample_size == 0 and est.queue_length == 0


def test_empty_queue_with_live_video_is_no_wait():
    eng = WaitTimeEngine(min_samples=3)
    eng.on_heartbeat(LOCATION, at(0))
    est = eng.estimate(LOCATION, at(5))
    assert est.source == WaitSource.VISION
    assert est.label == "no wait"


def test_degrades_to_pos_only_when_video_stale():
    eng = WaitTimeEngine(min_samples=3, video_stale_after=timedelta(seconds=120))
    eng.on_heartbeat(LOCATION, at(0))
    for t in range(0, 600, 60):  # 1 order/min for 10 min
        eng.on_pos_event(pos(t))
    for t in (601, 602, 603, 604, 605, 606):  # burst in the last 5 min
        eng.on_pos_event(pos(t))
    est = eng.estimate(LOCATION, at(700))  # video stale for 700 s
    assert est.source == WaitSource.POS_ONLY
    assert est.p50_minutes is not None and est.p50_minutes > 0


def test_no_data_at_all():
    eng = WaitTimeEngine()
    est = eng.estimate(LOCATION, at(0))
    assert est.source == WaitSource.NO_DATA
    assert est.label == "n/a"


def test_window_trims_old_samples():
    eng = WaitTimeEngine(min_samples=1, window=timedelta(minutes=30))
    eng.on_zone_event(zev(1, ZoneEventType.ENTER, 0))
    eng.on_zone_event(zev(1, ZoneEventType.EXIT, 600, dwell=600))
    eng.on_heartbeat(LOCATION, at(3000))
    est = eng.estimate(LOCATION, at(3000))  # 40 min later
    assert est.sample_size == 0
