from wte.vlm.triggers import CameraSnapshot, TriggerPolicy

from tests.conftest import CAMERA, LOCATION, at


def snap(t: float, q: int, s: int = 1, dwell: float | None = None, orders: int = 5) -> CameraSnapshot:
    return CameraSnapshot(ts=at(t), queue_len=q, service_len=s, avg_dwell_s=dwell,
                          pos_orders_last_5m=orders)


def test_cadence_fires_at_most_every_n_seconds():
    p = TriggerPolicy(cadence_s=60, min_gap_s=30)
    assert p.evaluate(LOCATION, CAMERA, snap(0, 3)).reason == "cadence"
    assert p.evaluate(LOCATION, CAMERA, snap(30, 3)) is None
    assert p.evaluate(LOCATION, CAMERA, snap(59, 3)) is None
    assert p.evaluate(LOCATION, CAMERA, snap(60, 3)).reason == "cadence"


def test_queue_jump_anomaly_fires_early():
    p = TriggerPolicy(cadence_s=600, min_gap_s=10)
    p.evaluate(LOCATION, CAMERA, snap(0, 4))  # cadence (first ever)
    assert p.evaluate(LOCATION, CAMERA, snap(20, 5)) is None
    trig = p.evaluate(LOCATION, CAMERA, snap(40, 9))  # 4 → 9 within 60 s (> +50 %)
    assert trig is not None and "queue_jump" in trig.reason


def test_service_empty_with_long_queue():
    p = TriggerPolicy(cadence_s=600, min_gap_s=0, long_queue=8)
    p.evaluate(LOCATION, CAMERA, snap(0, 2))
    trig = p.evaluate(LOCATION, CAMERA, snap(5, 8, s=0))
    assert trig is not None and "service_empty" in trig.reason


def test_dwell_p95_anomaly_needs_baseline():
    p = TriggerPolicy(cadence_s=600, min_gap_s=0)
    p.evaluate(LOCATION, CAMERA, snap(0, 2))
    assert p.evaluate(LOCATION, CAMERA, snap(1, 2, dwell=9999)) is None  # no baseline yet
    for d in range(100, 400, 10):  # 30 samples
        p.record_dwell(CAMERA, d)
    trig = p.evaluate(LOCATION, CAMERA, snap(2, 2, dwell=900))
    assert trig is not None and "dwell_p95" in trig.reason


def test_min_gap_debounces_anomalies():
    p = TriggerPolicy(cadence_s=600, min_gap_s=30, long_queue=8)
    p.evaluate(LOCATION, CAMERA, snap(0, 2))
    assert p.evaluate(LOCATION, CAMERA, snap(5, 8, s=0)) is None  # 5 s after last fire
    assert p.evaluate(LOCATION, CAMERA, snap(31, 8, s=0)) is not None


def test_summary_mentions_metadata_not_pixels():
    p = TriggerPolicy(cadence_s=1)
    trig = p.evaluate(LOCATION, CAMERA, snap(0, 6, s=2, dwell=300, orders=7))
    assert "Queue now 6" in trig.metadata_summary
    assert "5.0 min" in trig.metadata_summary
    assert "7 POS orders" in trig.metadata_summary
