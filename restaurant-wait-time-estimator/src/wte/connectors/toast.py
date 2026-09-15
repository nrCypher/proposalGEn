"""Toast POS connector (blueprint §6.1): REST + webhooks, HMAC-SHA256 signed.

Toast signs the raw body with the webhook secret and sends the hex digest in
``Toast-Signature``. Event types we care about map to the internal PosEvent set.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from wte.connectors.base import SignatureError, header, verify_hmac_sha256
from wte.core.schemas import PosEvent

_EVENT_MAP = {
    "ORDER_CREATED": "order_created",
    "ORDER_MODIFIED": None,  # ignored unless it closes the check
    "ORDER_COMPLETED": "order_completed",
    "ORDER_VOIDED": "order_cancelled",
    "CHECK_OPENED": "check_open",
    "CHECK_CLOSED": "check_closed",
}

_CHANNEL_MAP = {"DINE_IN": "dine_in", "TAKE_OUT": "takeaway", "DELIVERY": "delivery"}


class ToastConnector:
    provider = "toast"

    def __init__(self, webhook_secret: str) -> None:
        self._secret = webhook_secret

    def verify(self, body: bytes, headers: dict[str, str]) -> None:
        sig = header(headers, "Toast-Signature")
        if not verify_hmac_sha256(self._secret, body, sig):
            raise SignatureError("invalid Toast-Signature")

    def normalize(self, payload: dict[str, Any], tenant_id: UUID,
                  location_id: str) -> list[PosEvent]:
        et = _EVENT_MAP.get(payload.get("eventType", ""))
        if et is None:
            return []
        order = payload.get("order", payload.get("data", {}))
        occurred = payload.get("timestamp") or order.get("openedDate")
        party = order.get("numberOfGuests")
        return [PosEvent(
            tenant_id=tenant_id, location_id=location_id, provider=self.provider,
            event_id=str(payload.get("guid") or payload.get("eventId")),
            event_type=et,
            occurred_at=_parse_ts(occurred),
            order_id=str(order.get("guid") or order.get("orderGuid") or ""),
            party_size=int(party) if party is not None else None,
            channel=_CHANNEL_MAP.get(order.get("diningOption", {}).get("behavior", ""), "dine_in"),
            raw=payload,
        )]


def _parse_ts(v: str | None) -> datetime:
    from wte.core.schemas import utcnow

    if not v:
        return utcnow()
    return datetime.fromisoformat(v.replace("Z", "+00:00"))
