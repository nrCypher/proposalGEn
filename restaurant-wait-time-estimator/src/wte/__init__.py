"""Restaurant Wait-Time Estimator (WTE).

Package layout mirrors the service breakdown in blueprint §10.1:

- ``wte.core``        shared schemas, settings, event bus
- ``wte.edge``        edge agent: RTSP → face-blur → sampling → metadata upload
- ``wte.vision``      detector / tracker / zone engine (tier-1, per frame)
- ``wte.waittime``    live wait-time engine (p50/p90 bands, POS-only fallback)
- ``wte.vlm``         tier-2 VLM reasoning: anomaly triggers + structured client
- ``wte.forecast``    "Best Time to Visit" forecasting (cold / warm / mature tiers)
- ``wte.connectors``  POS & booking connector hub (webhook verify, normalize, dedupe)
- ``wte.api``         FastAPI BFF: REST + WebSocket, tenant isolation middleware
- ``wte.db``          SQL migrations (RLS + TimescaleDB) and repository
"""

__version__ = "0.1.0"
