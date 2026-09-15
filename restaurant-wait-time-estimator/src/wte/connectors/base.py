"""Connector hub primitives (blueprint §6.3–6.4).

- ``verify_hmac_sha256`` — constant-time webhook signature check.
- ``IdempotencyStore`` — dedupe by ``(provider, event_id)``; POS providers retry.
- ``Connector`` — protocol every POS / booking adapter implements.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import OrderedDict
from typing import Any, Protocol
from uuid import UUID

from wte.core.schemas import BookingEvent, PosEvent


def verify_hmac_sha256(secret: str, body: bytes, signature: str, *, encoding: str = "hex",
                       prefix: str = "") -> bool:
    mac = hmac.new(secret.encode(), body, hashlib.sha256)
    if encoding == "base64":
        import base64

        expected = base64.b64encode(mac.digest()).decode()
    else:
        expected = mac.hexdigest()
    return hmac.compare_digest(prefix + expected, signature or "")


class IdempotencyStore(Protocol):
    def seen(self, provider: str, event_id: str) -> bool: ...


class MemoryIdempotencyStore:
    """LRU with TTL. Production: Redis ``SET NX EX``."""

    def __init__(self, max_size: int = 100_000, ttl_s: int = 7 * 24 * 3600) -> None:
        self._d: OrderedDict[str, float] = OrderedDict()
        self._max, self._ttl = max_size, ttl_s

    def seen(self, provider: str, event_id: str) -> bool:
        now = time.time()
        key = f"{provider}:{event_id}"
        exp = self._d.get(key)
        if exp is not None and exp > now:
            self._d.move_to_end(key)
            return True
        self._d[key] = now + self._ttl
        self._d.move_to_end(key)
        while len(self._d) > self._max:
            self._d.popitem(last=False)
        return False


class SignatureError(Exception):
    pass


class Connector(Protocol):
    provider: str

    def verify(self, body: bytes, headers: dict[str, str]) -> None:
        """Raise ``SignatureError`` if the payload is not authentic."""

    def normalize(self, payload: dict[str, Any], tenant_id: UUID,
                  location_id: str) -> list[PosEvent | BookingEvent]: ...


def header(headers: dict[str, str], name: str) -> str:
    """Case-insensitive header lookup."""
    lname = name.lower()
    for k, v in headers.items():
        if k.lower() == lname:
            return v
    return ""
