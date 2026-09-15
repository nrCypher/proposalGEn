from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from wte.api.auth import issue_token
from wte.api.main import create_app
from wte.core.schemas import BBox, Detection
from wte.core.settings import DEV_JWT_SECRET, Settings

TENANT_A = UUID("11111111-1111-1111-1111-111111111111")
TENANT_B = UUID("22222222-2222-2222-2222-222222222222")
LOCATION = "lisboa-baixa"
CAMERA = "cam-1"

ZONES = {
    "queue": [[100, 300], [700, 300], [700, 900], [100, 900]],
    "service": [[700, 300], [1100, 300], [1100, 900], [700, 900]],
    "fulfil": [[1100, 300], [1500, 300], [1500, 900], [1100, 900]],
}

T0 = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def person(tid: int, x: float, y: float = 600.0) -> Detection:
    """A 60×90 px person whose foot anchor is at (x, y)."""
    return Detection(bbox=BBox(x1=x - 30, y1=y - 90, x2=x + 30, y2=y), confidence=0.9,
                     tracker_id=tid)


@pytest.fixture
def settings() -> Settings:
    return Settings(jwt_secret=DEV_JWT_SECRET, edge_shared_secret="edge-secret",
                    toast_webhook_secret="toast-secret", wait_min_samples=3,
                    _env_file=None)


@pytest.fixture
def app(settings: Settings):
    return create_app(settings)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def token_a(settings: Settings) -> str:
    return issue_token(settings, TENANT_A, "alice", "operator")


@pytest.fixture
def token_b(settings: Settings) -> str:
    return issue_token(settings, TENANT_B, "bob", "operator")


def auth(token: str) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


def edge_sign(settings: Settings, body: bytes) -> str:
    return hmac.new(settings.edge_shared_secret.encode(), body, hashlib.sha256).hexdigest()
