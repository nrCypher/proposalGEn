"""Simulate an edge agent: synthetic tracked people moving queue → service → fulfil.

Posts HMAC-signed FrameMetadata to the API so you can exercise the whole cloud
path (zone engine → wait-time → WebSocket → widget) without cameras.

    python scripts/simulate_edge.py --api http://localhost:8080 --minutes 20 --arrivals-per-min 2
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import random
import time
from datetime import datetime, timedelta, timezone
from uuid import UUID

import httpx

from wte.core.schemas import BBox, Detection, FrameMetadata

QUEUE_X = (100, 700)
SERVICE_X = (700, 1100)
FULFIL_X = (1100, 1500)
Y = 600


class Person:
    def __init__(self, pid: int, t0: datetime, queue_s: float, service_s: float, fulfil_s: float):
        self.pid, self.t0 = pid, t0
        self.stages = [(QUEUE_X, queue_s), (SERVICE_X, service_s), (FULFIL_X, fulfil_s)]

    def bbox_at(self, t: datetime) -> BBox | None:
        elapsed = (t - self.t0).total_seconds()
        if elapsed < 0:
            return None
        for (x0, x1), dur in self.stages:
            if elapsed < dur:
                x = x0 + 50 + (x1 - x0 - 100) * min(1.0, elapsed / max(dur, 1))
                return BBox(x1=x - 30, y1=Y - 90, x2=x + 30, y2=Y)
            elapsed -= dur
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--tenant", default="11111111-1111-1111-1111-111111111111")
    ap.add_argument("--location", default="lisboa-baixa")
    ap.add_argument("--camera", default="cam-entrance-1")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--arrivals-per-min", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--speed", type=float, default=20.0, help="simulated seconds per wall second")
    a = ap.parse_args()

    secret = os.environ.get("WTE_EDGE_SHARED_SECRET", "dev-edge-secret")
    client = httpx.Client(base_url=a.api, timeout=5)
    tenant = UUID(a.tenant)

    people: list[Person] = []
    t = datetime.now(timezone.utc)
    end = t + timedelta(minutes=a.minutes)
    next_arrival = t
    pid = 1
    step = timedelta(seconds=1.0 / a.fps)

    while t < end:
        while next_arrival <= t:
            people.append(Person(pid, next_arrival, random.uniform(120, 480),
                                 random.uniform(40, 90), random.uniform(60, 180)))
            pid += 1
            next_arrival += timedelta(seconds=random.expovariate(a.arrivals_per_min / 60.0))
        dets = []
        for p in people:
            b = p.bbox_at(t)
            if b:
                dets.append(Detection(bbox=b, confidence=0.9, tracker_id=p.pid))
        people = [p for p in people if p.bbox_at(t) or p.t0 > t]
        meta = FrameMetadata(tenant_id=tenant, location_id=a.location, camera_id=a.camera, ts=t,
                             detections=dets, frame_w=1600, frame_h=900, faces_blurred=True)
        body = meta.model_dump_json().encode()
        sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        r = client.post("/v1/edge/frames", content=body,
                        headers={"content-type": "application/json", "x-wte-signature": sig})
        if r.status_code >= 300:
            print("upload failed:", r.status_code, r.text)
        print(f"{t:%H:%M:%S} people={len(dets):2d} events={r.json().get('events', '?')}")
        t += step
        time.sleep(step.total_seconds() / a.speed)


if __name__ == "__main__":
    main()
