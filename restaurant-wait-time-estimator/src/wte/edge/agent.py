"""Edge agent (blueprint §1.3 / §1.4 "pragmatic hybrid").

Runs on a mini-PC / Jetson at the restaurant:

    RTSP → sample fps → **face blur** → YOLO26-n detect → track → zone events
         → POST metadata (JSON, ~1 KB/s) to the cloud, HMAC-signed.

Raw video never leaves the site. If the uplink is down, metadata is buffered
on disk and replayed (the cloud degrades to POS-only meanwhile — pitfall #7).
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import httpx

from wte.core.schemas import FrameMetadata
from wte.core.settings import get_settings
from wte.edge.blur import FaceBlurrer
from wte.vision.detector import make_detector
from wte.vision.pipeline import CameraPipeline
from wte.vision.tracker import make_tracker
from wte.vision.zones import ZoneEngine, zones_from_config

log = logging.getLogger("wte.edge")


def sign(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class Uplink:
    """HTTP uploader with an on-disk spool for outages."""

    def __init__(self, base_url: str, secret: str, spool_dir: Path, max_spool: int = 50_000) -> None:
        self._url = base_url.rstrip("/") + "/v1/edge/frames"
        self._secret = secret
        self._spool = spool_dir
        self._spool.mkdir(parents=True, exist_ok=True)
        self._buf: deque[bytes] = deque(maxlen=max_spool)
        self._client = httpx.Client(timeout=5.0)

    def send(self, meta: FrameMetadata) -> None:
        body = meta.model_dump_json().encode()
        if not self._post(body):
            self._buf.append(body)
            (self._spool / f"{int(time.time() * 1000)}.json").write_bytes(body)
        else:
            self._drain()

    def _post(self, body: bytes) -> bool:
        try:
            r = self._client.post(self._url, content=body, headers={
                "content-type": "application/json", "x-wte-signature": sign(self._secret, body)})
            return r.status_code < 300
        except httpx.HTTPError as e:  # network blip — spool and move on
            log.warning("uplink failed: %s", e)
            return False

    def _drain(self) -> None:
        for p in sorted(self._spool.glob("*.json"))[:200]:
            if self._post(p.read_bytes()):
                p.unlink()
            else:
                break


def run(camera_cfg: dict, cloud_url: str, once: bool = False) -> None:
    import cv2

    s = get_settings()
    tenant_id = UUID(camera_cfg["tenant_id"])
    zone_engine = ZoneEngine(tenant_id=tenant_id, location_id=camera_cfg["location_id"],
                             camera_id=camera_cfg["camera_id"], zones=zones_from_config(camera_cfg))
    pipeline = CameraPipeline(
        detector=make_detector(s.detector_weights, s.detector_conf, s.detector_iou),
        tracker=make_tracker(s.tracker, s.lost_track_buffer_frames),
        zone_engine=zone_engine, fps_active=s.edge_fps_active, fps_idle=s.edge_fps_idle)
    blur = FaceBlurrer(sigma=s.blur_sigma, expand=s.blur_expand)
    uplink = Uplink(cloud_url, s.edge_shared_secret,
                    Path(os.environ.get("WTE_SPOOL_DIR", "/var/lib/wte/spool")))

    cap = cv2.VideoCapture(camera_cfg["rtsp_url"])
    target_fps = s.edge_fps_active
    last = 0.0
    log.info("edge agent up: camera=%s", camera_cfg["camera_id"])
    while True:
        ok, frame = cap.read()
        if not ok:
            log.warning("stream read failed; reconnecting")
            time.sleep(2)
            cap = cv2.VideoCapture(camera_cfg["rtsp_url"])
            continue
        now = time.time()
        if now - last < 1.0 / target_fps:
            continue
        last = now

        frame = blur(frame)  # irreversible, before anything else touches the frame
        ts = datetime.now(timezone.utc)
        res = pipeline.process(frame, ts)
        target_fps = res.suggested_fps
        h, w = frame.shape[:2]
        uplink.send(FrameMetadata(tenant_id=tenant_id, location_id=camera_cfg["location_id"],
                                  camera_id=camera_cfg["camera_id"], ts=ts,
                                  detections=res.detections, frame_w=w, frame_h=h,
                                  faces_blurred=True))
        if once:
            break


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="WTE edge agent")
    ap.add_argument("--config", required=True, help="camera JSON/YAML config")
    ap.add_argument("--cloud", default=os.environ.get("WTE_CLOUD_URL", "http://localhost:8080"))
    a = ap.parse_args()
    text = Path(a.config).read_text()
    if a.config.endswith((".yml", ".yaml")):
        import yaml  # type: ignore

        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    run(cfg, a.cloud)


if __name__ == "__main__":
    main()
