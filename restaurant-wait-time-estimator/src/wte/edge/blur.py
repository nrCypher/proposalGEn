"""Irreversible face anonymisation at the edge (blueprint §5.2).

Pipeline: detect faces → expand each box by ``expand`` → Gaussian blur (σ) or
pixelate → return the frame. Nothing leaves the site before this runs.

Face detector back-ends (first available wins):
1. RetinaFace ONNX via ``onnxruntime`` if ``WTE_FACE_ONNX`` points at a model.
2. OpenCV's bundled Haar cascade — zero extra deps, ~5–10 ms/frame on an N100.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

Box = tuple[int, int, int, int]  # x, y, w, h


class HaarFaceDetector:
    def __init__(self) -> None:
        import cv2

        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self._c = cv2.CascadeClassifier(path)
        self._cv2 = cv2

    def detect(self, frame: np.ndarray) -> list[Box]:
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        faces = self._c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(16, 16))
        return [tuple(int(v) for v in f) for f in faces]  # type: ignore[misc]


class OnnxFaceDetector:
    """Placeholder for RetinaFace / YOLO-face ONNX. Same ``detect`` contract."""

    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort  # type: ignore

        self._sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def detect(self, frame: np.ndarray) -> list[Box]:  # pragma: no cover - model-specific
        raise NotImplementedError("Wire your RetinaFace/YOLO-face post-processing here")


def make_face_detector() -> Any:
    onnx = os.environ.get("WTE_FACE_ONNX")
    if onnx:
        return OnnxFaceDetector(onnx)
    return HaarFaceDetector()


@dataclass
class FaceBlurrer:
    sigma: float = 20.0
    expand: float = 0.30
    mode: str = "gaussian"  # gaussian | pixelate
    detector: Any = None
    last_face_count: int = 0

    def __post_init__(self) -> None:
        if self.detector is None:
            self.detector = make_face_detector()

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.blur(frame)

    def blur(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        out = frame.copy()
        h, w = out.shape[:2]
        faces = self.detector.detect(frame)
        self.last_face_count = len(faces)
        for (x, y, fw, fh) in faces:
            dx, dy = int(fw * self.expand), int(fh * self.expand)
            x1, y1 = max(0, x - dx), max(0, y - dy)
            x2, y2 = min(w, x + fw + dx), min(h, y + fh + dy)
            roi = out[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            if self.mode == "pixelate":
                small = cv2.resize(roi, (max(1, (x2 - x1) // 12), max(1, (y2 - y1) // 12)),
                                   interpolation=cv2.INTER_LINEAR)
                roi[:] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            else:
                k = int(self.sigma * 3) | 1  # odd kernel
                roi[:] = cv2.GaussianBlur(roi, (k, k), self.sigma)
        return out

    def verify(self, frame: np.ndarray) -> bool:
        """Spot-audit helper (§5.2 step 4): True if no face is still detectable."""
        return len(self.detector.detect(frame)) == 0
