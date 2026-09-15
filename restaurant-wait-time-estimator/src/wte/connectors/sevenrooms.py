"""SevenRooms booking connector (blueprint §6.2 — the most open booking API).

Webhooks are HMAC-SHA256 signed (hex) in ``X-SevenRooms-Signature``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from wte.connectors.base import SignatureError, header, verify_hmac_sha256
from wte.core.schemas import BookingEvent, utcnow

_EVENT_MAP = {
    "reservation.created": "reservation_created",
    "reservation.arrived": "reservation_seated",
    "reservation.seated": "reservation_seated",
    "reservation.cancelled": "reservation_cancelled",
    "reservation.no_show": "reservation_cancelled",
}


class SevenRoomsConnector:
    provider = "sevenrooms"

    def __init__(self, webhook_secret: str) -> None:
        self._secret = webhook_secret

    def verify(self, body: bytes, headers: dict[str, str]) -> None:
        sig = header(headers, "X-SevenRooms-Signature")
        if not verify_hmac_sha256(self._secret, body, sig):
            raise SignatureError("invalid SevenRooms signature")

    def normalize(self, payload: dict[str, Any], tenant_id: UUID,
                  location_id: str) -> list[BookingEvent]:
        et = _EVENT_MAP.get(payload.get("event", ""))
        if et is None:
            return []
        r = payload.get("reservation", payload.get("data", {}))
        slot = r.get("date_time") or r.get("arrival_time")
        return [BookingEvent(
            tenant_id=tenant_id, location_id=location_id, provider=self.provider,
            event_id=str(payload.get("id") or payload.get("event_id")), event_type=et,
            occurred_at=_ts(payload.get("timestamp")),
            reservation_id=str(r.get("id") or r.get("reservation_id") or ""),
            party_size=int(r["max_guests"]) if r.get("max_guests") is not None else None,
            slot_at=_ts(slot) if slot else None, raw=payload,
        )]


def _ts(v: str | None) -> datetime:
    if not v:
        return utcnow()
    return datetime.fromisoformat(v.replace("Z", "+00:00"))
