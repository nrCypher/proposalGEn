import base64
import hashlib
import hmac
import json

import pytest

from wte.connectors.base import MemoryIdempotencyStore, SignatureError, verify_hmac_sha256
from wte.connectors.sevenrooms import SevenRoomsConnector
from wte.connectors.square import SquareConnector
from wte.connectors.toast import ToastConnector

from tests.conftest import LOCATION, TENANT_A


def _hex(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_verify_hmac_hex_and_base64():
    body = b'{"a":1}'
    assert verify_hmac_sha256("s", body, _hex("s", body))
    assert not verify_hmac_sha256("s", body, _hex("wrong", body))
    b64 = base64.b64encode(hmac.new(b"s", body, hashlib.sha256).digest()).decode()
    assert verify_hmac_sha256("s", body, b64, encoding="base64")
    assert not verify_hmac_sha256("s", body, "")


def test_idempotency_store():
    s = MemoryIdempotencyStore(max_size=2)
    assert not s.seen("toast", "1")
    assert s.seen("toast", "1")
    assert not s.seen("square", "1")  # keyed per provider
    assert not s.seen("toast", "2")
    assert not s.seen("toast", "3")   # evicts "1"
    assert not s.seen("toast", "1")


TOAST_PAYLOAD = {
    "guid": "evt-123", "eventType": "CHECK_OPENED", "timestamp": "2026-09-15T12:00:00.000+0000",
    "order": {"guid": "ord-9", "numberOfGuests": 3, "diningOption": {"behavior": "DINE_IN"}},
}


def test_toast_verify_and_normalize():
    c = ToastConnector("toast-secret")
    body = json.dumps(TOAST_PAYLOAD).encode()
    c.verify(body, {"Toast-Signature": _hex("toast-secret", body)})
    with pytest.raises(SignatureError):
        c.verify(body, {"toast-signature": "deadbeef"})
    events = c.normalize(TOAST_PAYLOAD, TENANT_A, LOCATION)
    assert len(events) == 1
    e = events[0]
    assert e.event_type == "check_open" and e.order_id == "ord-9" and e.party_size == 3
    assert e.channel == "dine_in" and e.provider == "toast" and e.event_id == "evt-123"
    assert e.occurred_at.hour == 12


def test_toast_ignores_irrelevant_events():
    c = ToastConnector("x")
    assert c.normalize({"eventType": "ORDER_MODIFIED", "guid": "1", "order": {}}, TENANT_A, LOCATION) == []
    assert c.normalize({"eventType": "MENU_UPDATED", "guid": "2"}, TENANT_A, LOCATION) == []


def test_square_signature_includes_notification_url():
    url = "https://api.example.com/v1/webhooks/square"
    c = SquareConnector("sq-key", url)
    body = json.dumps({"type": "order.created", "event_id": "sq-1", "created_at": "2026-09-15T12:00:00Z",
                       "data": {"object": {"order": {"id": "o1"}}}}).encode()
    sig = base64.b64encode(hmac.new(b"sq-key", url.encode() + body, hashlib.sha256).digest()).decode()
    c.verify(body, {"x-square-hmacsha256-signature": sig})
    with pytest.raises(SignatureError):
        c.verify(body, {"x-square-hmacsha256-signature": sig[:-2] + "AA"})
    ev = c.normalize(json.loads(body), TENANT_A, LOCATION)
    assert ev[0].event_type == "order_created" and ev[0].order_id == "o1"


def test_square_fulfillment_only_completed_state():
    c = SquareConnector("k", "u")
    p = {"type": "order.fulfillment.updated", "event_id": "1", "data": {"object": {
        "order_fulfillment_updated": {"order_id": "o2", "fulfillment_update": [{"new_state": "PREPARED"}]}}}}
    assert c.normalize(p, TENANT_A, LOCATION) == []
    p["data"]["object"]["order_fulfillment_updated"]["fulfillment_update"][0]["new_state"] = "COMPLETED"
    assert c.normalize(p, TENANT_A, LOCATION)[0].event_type == "order_completed"


def test_sevenrooms_normalize():
    c = SevenRoomsConnector("sr")
    p = {"id": "sr-1", "event": "reservation.seated", "timestamp": "2026-09-15T19:05:00Z",
         "reservation": {"id": "r1", "max_guests": 4, "date_time": "2026-09-15T19:00:00Z"}}
    body = json.dumps(p).encode()
    c.verify(body, {"X-SevenRooms-Signature": _hex("sr", body)})
    ev = c.normalize(p, TENANT_A, LOCATION)
    assert ev[0].event_type == "reservation_seated" and ev[0].party_size == 4
    assert ev[0].slot_at.hour == 19
