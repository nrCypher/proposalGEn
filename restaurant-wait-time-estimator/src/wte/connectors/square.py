"""Square POS connector (blueprint §6.1).

Square signs ``notification_url + body`` with HMAC-SHA256 and sends it base64 in
``x-square-hmacsha256-signature``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from wte.connectors.base import SignatureError, header, verify_hmac_sha256
from wte.core.schemas import PosEvent, utcnow

_EVENT_MAP = {
    "order.created": "order_created",
    "order.fulfillment.updated": "order_completed",
    "order.updated": None,
    "payment.completed": "check_closed",
}


class SquareConnector:
    provider = "square"

    def __init__(self, signature_key: str, notification_url: str) -> None:
        self._key, self._url = signature_key, notification_url

    def verify(self, body: bytes, headers: dict[str, str]) -> None:
        sig = header(headers, "x-square-hmacsha256-signature")
        if not verify_hmac_sha256(self._key, self._url.encode() + body, sig, encoding="base64"):
            raise SignatureError("invalid Square signature")

    def normalize(self, payload: dict[str, Any], tenant_id: UUID,
                  location_id: str) -> list[PosEvent]:
        et = _EVENT_MAP.get(payload.get("type", ""))
        if et is None:
            return []
        obj = payload.get("data", {}).get("object", {})
        order = obj.get("order") or obj.get("payment") or obj.get("order_fulfillment_updated") or {}
        if et == "order_completed":
            states = {u.get("new_state") for u in order.get("fulfillment_update", [])}
            if "COMPLETED" not in states:
                return []
        created = payload.get("created_at")
        return [PosEvent(
            tenant_id=tenant_id, location_id=location_id, provider=self.provider,
            event_id=str(payload.get("event_id")), event_type=et,
            occurred_at=datetime.fromisoformat(created.replace("Z", "+00:00")) if created else utcnow(),
            order_id=str(order.get("id") or order.get("order_id") or ""),
            party_size=None, channel="dine_in", raw=payload,
        )]
