from wte.core.schemas import ZoneEventType, ZoneKind
from wte.vision.zones import ZoneEngine, point_in_polygon, zones_from_config

from tests.conftest import CAMERA, LOCATION, TENANT_A, ZONES, at, person

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
CONCAVE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (5.0, 5.0), (0.0, 10.0)]  # notch at top


def test_point_in_polygon_basic():
    assert point_in_polygon((5, 5), SQUARE)
    assert not point_in_polygon((15, 5), SQUARE)
    assert not point_in_polygon((-1, 5), SQUARE)


def test_point_in_polygon_concave():
    assert point_in_polygon((1, 8), CONCAVE)
    assert point_in_polygon((9, 8), CONCAVE)
    assert not point_in_polygon((5, 8), CONCAVE)  # inside the notch


def _engine() -> ZoneEngine:
    return ZoneEngine(tenant_id=TENANT_A, location_id=LOCATION, camera_id=CAMERA,
                      zones=zones_from_config({"zones": ZONES}))


def test_enter_then_exit_emits_dwell():
    eng = _engine()
    ev = eng.update([person(1, x=400)], at(0))
    assert [e.event_type for e in ev] == [ZoneEventType.ENTER]
    assert ev[0].zone == ZoneKind.QUEUE and ev[0].tracker_id == 1
    assert eng.occupancy(ZoneKind.QUEUE) == 1

    # Moves to service zone at t=90s → queue EXIT (dwell 90) + service ENTER
    ev = eng.update([person(1, x=900)], at(90))
    kinds = {(e.zone, e.event_type) for e in ev}
    assert (ZoneKind.QUEUE, ZoneEventType.EXIT) in kinds
    assert (ZoneKind.SERVICE, ZoneEventType.ENTER) in kinds
    exit_ev = next(e for e in ev if e.event_type == ZoneEventType.EXIT)
    assert exit_ev.dwell_seconds == 90.0
    assert eng.occupancy(ZoneKind.QUEUE) == 0
    assert eng.occupancy(ZoneKind.SERVICE) == 1


def test_lost_track_is_expired():
    eng = _engine()
    eng.update([person(7, x=400)], at(0))
    # Track disappears (tracker lost it); next frames have nobody for 20 s
    ev = eng.update([], at(20))
    assert len(ev) == 1
    assert ev[0].event_type == ZoneEventType.EXIT and ev[0].tracker_id == 7
    assert ev[0].dwell_seconds == 0.0  # closed at last-seen ts, not at expiry time
    assert eng.occupancy(ZoneKind.QUEUE) == 0


def test_untracked_detections_are_ignored():
    eng = _engine()
    d = person(1, x=400)
    d.tracker_id = None
    assert eng.update([d], at(0)) == []


def test_two_people_independent():
    eng = _engine()
    eng.update([person(1, x=400), person(2, x=450)], at(0))
    assert eng.occupancy(ZoneKind.QUEUE) == 2
    ev = eng.update([person(1, x=900), person(2, x=450)], at(60))
    exits = [e for e in ev if e.event_type == ZoneEventType.EXIT]
    assert len(exits) == 1 and exits[0].tracker_id == 1
    assert eng.occupancy(ZoneKind.QUEUE) == 1
