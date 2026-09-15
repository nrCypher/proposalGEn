import hashlib
import hmac
import json
from datetime import timedelta

from wte.core.schemas import FrameMetadata

from tests.conftest import (
    CAMERA,
    LOCATION,
    T0,
    TENANT_A,
    TENANT_B,
    ZONES,
    at,
    auth,
    edge_sign,
    person,
)


def _register(client, token):
    r = client.post("/v1/cameras", json={"location_id": LOCATION, "camera_id": CAMERA, "zones": ZONES},
                    headers=auth(token))
    assert r.status_code == 201, r.text


def _frame(settings, detections, t):
    meta = FrameMetadata(tenant_id=TENANT_A, location_id=LOCATION, camera_id=CAMERA, ts=at(t),
                         detections=detections, frame_w=1600, frame_h=900, faces_blurred=True)
    body = meta.model_dump_json().encode()
    return body, {"content-type": "application/json", "x-wte-signature": edge_sign(settings, body)}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_auth_required(client):
    assert client.get(f"/v1/locations/{LOCATION}/wait").status_code == 401
    assert client.get(f"/v1/locations/{LOCATION}/wait",
                      headers=auth("not-a-jwt")).status_code == 401


def test_dev_token_roundtrip(client):
    r = client.post("/v1/dev/token", json={"tenant_id": str(TENANT_A)})
    assert r.status_code == 200
    tok = r.json()["token"]
    assert client.get("/v1/cameras", headers=auth(tok)).status_code == 200


def test_edge_ingest_requires_valid_signature_and_blur(client, settings, token_a):
    _register(client, token_a)
    body, headers = _frame(settings, [person(1, 400)], 0)
    bad = dict(headers, **{"x-wte-signature": "00"})
    assert client.post("/v1/edge/frames", content=body, headers=bad).status_code == 401

    unblurred = json.loads(body); unblurred["faces_blurred"] = False
    ub = json.dumps(unblurred).encode()
    h = {"content-type": "application/json", "x-wte-signature": edge_sign(settings, ub)}
    assert client.post("/v1/edge/frames", content=ub, headers=h).status_code == 422

    assert client.post("/v1/edge/frames", content=body, headers=headers).status_code == 202


def test_unknown_camera_is_404(client, settings):
    body, headers = _frame(settings, [person(1, 400)], 0)
    assert client.post("/v1/edge/frames", content=body, headers=headers).status_code == 404


def test_end_to_end_wait_estimate_from_frames(client, settings, token_a):
    _register(client, token_a)
    # 4 people enter the queue at t=0 and are served (move to service zone) after 4–7 min
    dwell = {1: 240, 2: 300, 3: 360, 4: 420}
    body, h = _frame(settings, [person(i, 400 + i * 20) for i in dwell], 0)
    assert client.post("/v1/edge/frames", content=body, headers=h).json()["events"] == 4
    for tid, d in dwell.items():
        still = [person(j, 400 + j * 20) for j in dwell if dwell[j] > d] + [person(tid, 900)]
        body, h = _frame(settings, still, d)
        client.post("/v1/edge/frames", content=body, headers=h)
    # one new arrival, then read the estimate
    body, h = _frame(settings, [person(9, 400)], 430)
    client.post("/v1/edge/frames", content=body, headers=h)

    client.app.state.clock = lambda: at(440)  # freeze "now" for the read model
    r = client.get(f"/v1/locations/{LOCATION}/wait", headers=auth(token_a))
    assert r.status_code == 200, r.text
    est = r.json()["estimate"]
    assert est["source"] == "vision"
    assert est["queue_length"] == 1
    assert est["sample_size"] == 4
    assert est["p50_minutes"] == 5.5
    assert est["p90_minutes"] == 6.7  # p90 of {4,5,6,7} min
    assert r.json()["label"] == "4–8 min"  # < 10 min → 2-min band steps


def test_tenant_isolation_on_cameras_and_wait(client, settings, token_a, token_b):
    _register(client, token_a)
    assert len(client.get("/v1/cameras", headers=auth(token_a)).json()) == 1
    assert client.get("/v1/cameras", headers=auth(token_b)).json() == []
    # Tenant B sees no data for A's location (no cross-tenant leak of estimates)
    r = client.get(f"/v1/locations/{LOCATION}/wait", headers=auth(token_b))
    assert r.status_code == 200
    assert r.json()["estimate"]["source"] in ("no_data", "vision")
    assert client.get("/v1/ops/vlm-triggers", headers=auth(token_b)).json() == []


def test_toast_webhook_dedupes_and_feeds_engine(client, settings, token_a):
    payload = {"guid": "evt-1", "eventType": "CHECK_OPENED", "timestamp": "2026-09-15T12:00:00Z",
               "order": {"guid": "o1", "numberOfGuests": 2}}
    body = json.dumps(payload).encode()
    sig = hmac.new(settings.toast_webhook_secret.encode(), body, hashlib.sha256).hexdigest()
    url = f"/v1/webhooks/toast/{TENANT_A}/{LOCATION}"
    assert client.post(url, content=body, headers={"Toast-Signature": "bad"}).status_code == 401
    r = client.post(url, content=body, headers={"Toast-Signature": sig})
    assert r.status_code == 202 and r.json()["accepted"] == 1
    r = client.post(url, content=body, headers={"Toast-Signature": sig})  # provider retry
    assert r.json()["accepted"] == 0
    assert client.post(f"/v1/webhooks/nope/{TENANT_A}/{LOCATION}", content=b"{}").status_code == 404


def test_forecast_train_and_best_times(client, token_a):
    pts = []
    t = T0 - timedelta(days=3)
    while t < T0:
        m = t.hour * 60 + t.minute
        pts.append([t.isoformat(), 15.0 if 690 <= m < 840 or 1080 <= m < 1320 else 2.0])
        t += timedelta(minutes=15)
    r = client.post(f"/v1/locations/{LOCATION}/forecast/train", json={"points": pts}, headers=auth(token_a))
    assert r.status_code == 200, r.text
    fc = r.json()
    assert fc["tier"] == "cold" and len(fc["points"]) == 7 * 96
    r = client.get(f"/v1/locations/{LOCATION}/forecast?hours=6", headers=auth(token_a))
    assert r.status_code == 200 and 0 < len(r.json()["points"]) <= 24
    r = client.get(f"/v1/locations/{LOCATION}/best-times?n=2", headers=auth(token_a))
    assert r.status_code == 200 and len(r.json()) == 2
    assert all(b["q50"] <= 2.0 for b in r.json())


def test_forecast_404_before_training(client, token_a):
    assert client.get(f"/v1/locations/{LOCATION}/forecast", headers=auth(token_a)).status_code == 404


def test_websocket_pushes_live_estimate(client, settings, token_a):
    _register(client, token_a)
    with client.websocket_connect(f"/v1/ws/locations/{LOCATION}?token={token_a}") as ws:
        first = ws.receive_json()
        assert first["type"] == "wait_estimate"
        body, h = _frame(settings, [person(1, 400)], 0)
        client.post("/v1/edge/frames", content=body, headers=h)
        pushed = ws.receive_json()
        assert pushed["type"] == "wait_estimate" and pushed["queue_length"] == 1


def test_websocket_rejects_missing_token(client):
    from starlette.websockets import WebSocketDisconnect
    import pytest

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/v1/ws/locations/{LOCATION}"):
            pass
