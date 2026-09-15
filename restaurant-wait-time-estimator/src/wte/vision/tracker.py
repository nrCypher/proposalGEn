"""Multi-object trackers behind one interface.

- ``ByteTrackTracker`` / ``BoTSortTracker`` — production (via ``supervision`` / ``boxmot``).
- ``IouTracker`` — dependency-free greedy IoU matcher with a lost-track buffer.
  Good enough for CPU-only edge fallback and for deterministic tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from wte.core.schemas import BBox, Detection


class Tracker(Protocol):
    def update(self, detections: list[Detection]) -> list[Detection]: ...


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    tid: int
    bbox: BBox
    misses: int = 0


@dataclass
class IouTracker:
    """Greedy IoU association. ``lost_track_buffer`` frames of grace before a track dies."""

    iou_threshold: float = 0.3
    lost_track_buffer: int = 30
    _tracks: list[_Track] = field(default_factory=list)
    _next_id: int = 1

    def update(self, detections: list[Detection]) -> list[Detection]:
        pairs = sorted(
            ((iou(t.bbox, d.bbox), ti, di)
             for ti, t in enumerate(self._tracks)
             for di, d in enumerate(detections)),
            reverse=True,
        )
        matched_t: set[int] = set()
        matched_d: set[int] = set()
        out: list[Detection] = [d.model_copy() for d in detections]

        for score, ti, di in pairs:
            if score < self.iou_threshold:
                break
            if ti in matched_t or di in matched_d:
                continue
            matched_t.add(ti)
            matched_d.add(di)
            track = self._tracks[ti]
            track.bbox = detections[di].bbox
            track.misses = 0
            out[di].tracker_id = track.tid

        # Age the existing tracks first, so brand-new tracks don't start with a miss.
        survivors: list[_Track] = []
        for ti, t in enumerate(self._tracks):
            if ti not in matched_t:
                t.misses += 1
            if t.misses <= self.lost_track_buffer:
                survivors.append(t)
        self._tracks = survivors

        for di, d in enumerate(detections):
            if di not in matched_d:
                track = _Track(tid=self._next_id, bbox=d.bbox)
                self._next_id += 1
                self._tracks.append(track)
                out[di].tracker_id = track.tid
        return out


class ByteTrackTracker:
    """ByteTrack via roboflow/supervision (blueprint §2.2 default)."""

    def __init__(self, activation_threshold: float = 0.25, lost_track_buffer: int = 60,
                 frame_rate: int = 10) -> None:
        import supervision as sv  # type: ignore

        self._sv = sv
        self._bt = sv.ByteTrack(track_activation_threshold=activation_threshold,
                                lost_track_buffer=lost_track_buffer, frame_rate=frame_rate)

    def update(self, detections: list[Detection]) -> list[Detection]:
        import numpy as np

        if not detections:
            self._bt.update_with_detections(self._sv.Detections.empty())
            return []
        xyxy = np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections],
                        dtype=np.float32)
        conf = np.array([d.confidence for d in detections], dtype=np.float32)
        cls = np.zeros(len(detections), dtype=int)
        sv_det = self._sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
        tracked = self._bt.update_with_detections(sv_det)
        return [
            Detection(bbox=BBox(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3])),
                      confidence=float(c), tracker_id=int(t) if t is not None else None)
            for b, c, t in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id)
        ]


class BoTSortTracker:
    """BoT-SORT via boxmot — premium tier / pan-tilt cameras (blueprint §2.2)."""

    def __init__(self, reid_weights: str | None = None, device: str = "cpu") -> None:
        from pathlib import Path

        from boxmot import BotSort  # type: ignore

        self._t = BotSort(reid_weights=Path(reid_weights) if reid_weights else None,
                          device=device, half=False, with_reid=bool(reid_weights))

    def update(self, detections: list[Detection], frame=None) -> list[Detection]:
        import numpy as np

        arr = np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, d.confidence, 0]
                        for d in detections], dtype=np.float32).reshape(-1, 6)
        out = self._t.update(arr, frame)
        return [
            Detection(bbox=BBox(x1=float(r[0]), y1=float(r[1]), x2=float(r[2]), y2=float(r[3])),
                      confidence=float(r[5]), tracker_id=int(r[4]))
            for r in out
        ]


def make_tracker(name: str, lost_track_buffer: int = 60) -> Tracker:
    name = name.lower()
    if name == "bytetrack":
        try:
            return ByteTrackTracker(lost_track_buffer=lost_track_buffer)
        except ImportError:
            return IouTracker(lost_track_buffer=lost_track_buffer)
    if name == "botsort":
        return BoTSortTracker()
    return IouTracker(lost_track_buffer=lost_track_buffer)
