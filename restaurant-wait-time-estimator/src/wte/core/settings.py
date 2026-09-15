"""Runtime settings (12-factor, env-driven). Prefix: ``WTE_``."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ≥32 bytes so PyJWT doesn't warn; the dev-token endpoint is only enabled with this value.
DEV_JWT_SECRET = "dev-secret-change-me-before-any-deployment-0123"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WTE_", env_file=".env", extra="ignore")

    # --- API / auth -------------------------------------------------------
    jwt_secret: str = Field(default=DEV_JWT_SECRET, description="HS256 secret; dev default")
    jwt_algorithm: str = "HS256"
    edge_shared_secret: str = "dev-edge-secret"  # HMAC key for edge → cloud metadata uploads

    # --- Persistence (all optional: in-memory fallbacks keep dev/test light) ---
    database_url: str | None = None  # postgresql://wte_app:...@host/wte
    redis_url: str | None = None
    kafka_bootstrap: str | None = None

    # --- Vision -----------------------------------------------------------
    detector_weights: str = "yolo26n.pt"
    detector_conf: float = 0.35
    detector_iou: float = 0.6
    tracker: str = "bytetrack"  # bytetrack | botsort | iou (dependency-free fallback)
    lost_track_buffer_frames: int = 60  # tune per site: fast-casual ~30, fine-dining ~120

    # --- Edge agent -------------------------------------------------------
    edge_fps_active: float = 10.0
    edge_fps_idle: float = 3.0
    blur_sigma: float = 20.0
    blur_expand: float = 0.30

    # --- Wait-time engine -------------------------------------------------
    wait_window_minutes: int = 30
    wait_min_samples: int = 5
    video_stale_after_seconds: int = 120  # after this, degrade to POS-only

    # --- VLM tier-2 -------------------------------------------------------
    vlm_base_url: str = "http://localhost:8000/v1"  # vLLM OpenAI-compatible endpoint
    vlm_api_key: str = "EMPTY"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct-FP8"
    vlm_cadence_seconds: int = 60
    vlm_min_gap_seconds: int = 30

    # --- POS webhooks -----------------------------------------------------
    toast_webhook_secret: str = "dev-toast-secret"
    square_webhook_signature_key: str = "dev-square-key"
    square_notification_url: str = "https://example.com/v1/webhooks/pos/square"
    sevenrooms_webhook_secret: str = "dev-sevenrooms-secret"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
