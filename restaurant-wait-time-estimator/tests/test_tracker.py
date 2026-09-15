from wte.core.schemas import BBox, Detection
from wte.vision.tracker import IouTracker, iou


def box(x: float, y: float = 600) -> Detection:
    return Detection(bbox=BBox(x1=x - 30, y1=y - 90, x2=x + 30, y2=y), confidence=0.9)


def test_iou_symmetry_and_range():
    a, b = box(100).bbox, box(110).bbox
    assert 0 < iou(a, b) < 1
    assert iou(a, b) == iou(b, a)
    assert iou(a, a) == 1.0
    assert iou(a, box(500).bbox) == 0.0


def test_ids_persist_across_small_motion():
    t = IouTracker()
    f1 = t.update([box(100), box(300)])
    f2 = t.update([box(108), box(305)])
    assert {d.tracker_id for d in f1} == {d.tracker_id for d in f2}
    assert f1[0].tracker_id != f1[1].tracker_id


def test_new_object_gets_new_id_and_lost_buffer():
    t = IouTracker(lost_track_buffer=2)
    f1 = t.update([box(100)])
    tid = f1[0].tracker_id
    t.update([])  # miss 1
    t.update([])  # miss 2 (still buffered)
    f4 = t.update([box(104)])
    assert f4[0].tracker_id == tid  # recovered
    t.update([]); t.update([]); t.update([])  # exceeds buffer → dropped
    f8 = t.update([box(104)])
    assert f8[0].tracker_id != tid
