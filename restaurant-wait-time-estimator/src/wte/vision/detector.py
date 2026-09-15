"""Person detectors behind one interface.

``YoloDetector`` wraps Ultralytics (YOLO26 by default — NMS-free, quantisation
friendly; blueprint §2.1). ``StaticDetector`` replays canned detections for
tests and for driving the pipeline from pre-computed metadata.
"""

from __future__ import annotations

from typing import Any, Protocol

from wte.core.schemas import BBox, Detection

PERSON_CLASS = 0


class Detector(Protocol):
    def detect(self, frame: Any) -> list[Detection]: ...


class YoloDetector:
    def __init__(self, weights: str = "yolo26n.pt", conf: float = 0.35, iou: float = 0.6,
                 imgsz: int = 640, device: str | None = None) -> None:
        from ultralytics import YOLO  # type: ignore

        self._model = YOLO(weights)
        self._conf, self._iou, self._imgsz, self._device = conf, iou, imgsz, device

    def detect(self, frame: Any) -> list[Detection]:
        res = self._model(frame, classes=[PERSON_CLASS], conf=self._conf, iou=self._iou,
                          imgsz=self._imgsz, device=self._device, verbose=False)[0]
        out: list[Detection] = []
        if res.boxes is None:
            return out
        for xyxy, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
            out.append(Detection(bbox=BBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]),
                                 confidence=float(c)))
        return out


class StaticDetector:
    """Returns pre-baked detections per call (round-robin). For tests and replays."""

    def __init__(self, frames: list[list[Detection]]) -> None:
        self._frames = frames
        self._i = 0

    def detect(self, frame: Any) -> list[Detection]:
        out = self._frames[self._i % len(self._frames)]
        self._i += 1
        return [d.model_copy() for d in out]


def make_detector(weights: str, conf: float, iou: float) -> Detector:
    return YoloDetector(weights=weights, conf=conf, iou=iou)
