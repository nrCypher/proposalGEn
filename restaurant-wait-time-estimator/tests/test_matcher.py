from datetime import timedelta

from wte.connectors.matcher import VisionPosMatcher
from wte.core.schemas import PosEvent, ZoneEvent, ZoneEventType, ZoneKind

from tests.conftest import CAMERA, LOCATION, TENANT_A, at


def exit_ev(tid: int, t: float, dwell: float = 200.0) -> ZoneEvent:
    return ZoneEvent(tenant_id=TENANT_A, location_id=LOCATION, camera_id=CAMERA, zone=ZoneKind.QUEUE,
                     event_type=ZoneEventType.EXIT, tracker_id=tid, ts=at(t), dwell_seconds=dwell)


def check_open(eid: str, t: float) -> PosEvent:
    return PosEvent(tenant_id=TENANT_A, location_id=LOCATION, provider="toast", event_id=eid,
                    event_type="check_open", occurred_at=at(t - 300),  # POS clock is wrong on purpose
                    received_at=at(t), order_id="o")


def test_matches_closest_exit_within_tolerance():
    m = VisionPosMatcher(tolerance=timedelta(seconds=60))
    assert m.on_zone_event(exit_ev(1, 0)) is None
    assert m.on_zone_event(exit_ev(2, 40)) is None
    match = m.on_pos_event(check_open("p1", 45))
    assert match is not None and match.tracker_id == 2 and match.delta_seconds == 5
    assert match.dwell_seconds == 200.0
    # The matched exit is consumed; the other one is still available
    match2 = m.on_pos_event(check_open("p2", 50))
    assert match2 is not None and match2.tracker_id == 1


def test_uses_server_arrival_not_pos_clock():
    m = VisionPosMatcher(tolerance=timedelta(seconds=60))
    m.on_zone_event(exit_ev(1, 0))
    assert m.on_pos_event(check_open("p", 400)) is None  # received 400 s later → no match


def test_pos_before_exit_is_held_and_matched_later():
    m = VisionPosMatcher(tolerance=timedelta(seconds=60))
    assert m.on_pos_event(check_open("p", 100)) is None
    match = m.on_zone_event(exit_ev(3, 130))
    assert match is not None and match.pos_event_id == "p" and match.tracker_id == 3


def test_only_queue_exits_are_candidates():
    m = VisionPosMatcher()
    enter = exit_ev(1, 0).model_copy(update={"event_type": ZoneEventType.ENTER})
    assert m.on_zone_event(enter) is None
    assert m.on_pos_event(check_open("p", 1)) is None
