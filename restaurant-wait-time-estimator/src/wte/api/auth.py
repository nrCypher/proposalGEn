"""Tenant context from JWT + edge HMAC verification (blueprint §7.4, §6.4).

The API never trusts a tenant_id from the body — it comes from the validated
JWT ``tid`` claim (or, for edge uploads, from the HMAC-authenticated agent's
config, cross-checked against the payload).
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from uuid import UUID

import jwt
from fastapi import HTTPException, Request, WebSocket

from wte.core.settings import Settings


@dataclass(frozen=True)
class Principal:
    tenant_id: UUID
    subject: str
    role: str  # owner | operator | viewer


def issue_token(settings: Settings, tenant_id: UUID, subject: str, role: str = "operator",
                ttl_s: int = 3600) -> str:
    import time

    now = int(time.time())
    return jwt.encode({"tid": str(tenant_id), "sub": subject, "role": role,
                       "iat": now, "exp": now + ttl_s},
                      settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _decode(settings: Settings, token: str) -> Principal:
    try:
        claims = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as e:
        raise HTTPException(401, f"invalid token: {e}") from e
    try:
        return Principal(tenant_id=UUID(claims["tid"]), subject=claims.get("sub", ""),
                         role=claims.get("role", "viewer"))
    except (KeyError, ValueError) as e:
        raise HTTPException(401, "token missing tenant claim") from e


def principal_from_request(request: Request) -> Principal:
    settings: Settings = request.app.state.settings
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token")
    return _decode(settings, auth[7:].strip())


async def principal_from_ws(ws: WebSocket) -> Principal | None:
    settings: Settings = ws.app.state.settings
    token = ws.query_params.get("token") or ws.headers.get("authorization", "")[7:]
    if not token:
        return None
    try:
        return _decode(settings, token)
    except HTTPException:
        return None


def verify_edge_signature(settings: Settings, body: bytes, signature: str) -> None:
    expected = hmac.new(settings.edge_shared_secret.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature or ""):
        raise HTTPException(401, "invalid edge signature")
