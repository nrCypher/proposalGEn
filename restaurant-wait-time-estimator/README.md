# Restaurant Wait-Time Estimator (WTE)

Privacy-first, camera-based wait-time estimation for restaurants — MVP implementation of the
[technical blueprint](./blueprint.md) (see also the [requirements map](./requirements-diagram.md)).

```
câmara → edge agent (blur + YOLO26 + track) → metadata JSON → cloud
                                                        ├─ zone engine   → ZoneEvents
                                                        ├─ wait-time     → p50/p90 bands, POS-only fallback
                                                        ├─ VLM triggers  → "why is the queue slow?"
                                                        ├─ forecast      → "Best Time to Visit" (q10/q50/q90)
                                                        └─ POS/booking   → Toast · Square · SevenRooms
```

## What's implemented (MVP, blueprint §10.2)

| Area | Module | Status |
|---|---|---|
| Shared contracts (ZoneEvent, PosEvent, WaitEstimate, Forecast, VLM schema) | `wte.core.schemas` | ✅ |
| Event bus (in-memory; Kafka adapter) | `wte.core.bus` | ✅ / adapter |
| Face blur at the edge (Haar fallback, ONNX hook) | `wte.edge.blur` | ✅ |
| Edge agent: RTSP → blur → detect → track → HMAC-signed upload, disk spool | `wte.edge.agent` | ✅ |
| Detector (YOLO26 via Ultralytics) + trackers (ByteTrack / BoT-SORT / IoU fallback) | `wte.vision` | ✅ |
| Zone engine (polygon hit-test, dwell, enter/exit events) | `wte.vision.zones` | ✅ |
| Frame-rate adaptation (10 → 3 fps when idle) | `wte.vision.pipeline` | ✅ |
| Wait-time engine: p50/p90, bands, POS-only degradation | `wte.waittime.engine` | ✅ |
| VLM two-tier triggers (cadence + 4 anomaly rules) + structured client | `wte.vlm` | ✅ |
| Forecast tiers: cold (seasonal-naive ≈ Chronos stand-in) / warm (Prophet) | `wte.forecast` | ✅ |
| POS connectors: Toast, Square · Booking: SevenRooms (HMAC verify, normalise, dedupe) | `wte.connectors` | ✅ |
| Vision ↔ POS matcher (±60 s on server-arrival time) | `wte.connectors.matcher` | ✅ |
| FastAPI BFF: REST + WebSocket + JWT tenant context | `wte.api` | ✅ |
| Postgres/TimescaleDB schema with RLS, hypertables, retention, continuous aggregate | `wte.db` | ✅ |
| Embeddable widget + "Best Time" chart (Web Component + ECharts) | `web/widget` | ✅ |
| Docker Compose (TimescaleDB, Redis, API) | `deploy/` | ✅ |

Out of MVP scope (V1/V2 in the blueprint): TFT, multi-camera fusion, crowd-density mode,
React operator dashboard, Stripe billing, OpenTable/TheFork/Lightspeed/Clover/Micros.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + tests (no GPU / CV deps)
pytest                           # 40+ tests, all dependency-free
uvicorn wte.api.main:app --port 8080
```

Optional extras: `pip install -e ".[vision]"` (Ultralytics + supervision + OpenCV),
`".[forecast]"` (Prophet), `".[vlm]"` (OpenAI-compatible client), `".[db]"` (asyncpg).

### End-to-end in 60 seconds (no cameras needed)

```bash
API=http://localhost:8080
TENANT=11111111-1111-1111-1111-111111111111
TOKEN=$(curl -s -XPOST $API/v1/dev/token -H 'content-type: application/json' \
        -d "{\"tenant_id\":\"$TENANT\"}" | python -c 'import sys,json;print(json.load(sys.stdin)["token"])')

# 1. register a camera with three zones (pixel polygons)
curl -s -XPOST $API/v1/cameras -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' \
     -d @config/camera.example.json

# 2. simulate the edge agent (signed metadata frames)
python scripts/simulate_edge.py --api $API --tenant $TENANT --minutes 20

# 3. read the live estimate
curl -s $API/v1/locations/lisboa-baixa/wait -H "authorization: Bearer $TOKEN"
# {"estimate":{...,"p50_minutes":6.2,"p90_minutes":9.8,...},"label":"5–10 min"}

# 4. open the widget
open web/widget/index.html      # set API/token in the page
```

### Edge agent (on the mini-PC / Jetson)

```bash
pip install -e ".[vision,edge]"
WTE_EDGE_SHARED_SECRET=... WTE_DETECTOR_WEIGHTS=yolo26n.pt \
wte-edge --config /etc/wte/camera.json --cloud https://api.example.com
```

Faces are blurred *before* detection; only bounding boxes + tracker ids leave the site.

### Database

```bash
docker compose -f deploy/docker-compose.yml up -d timescaledb
psql postgresql://wte_migrator:wte_migrator@localhost:5432/wte -f src/wte/db/migrations/001_init.sql
export WTE_DATABASE_URL=postgresql://wte_app:wte_app@localhost:5432/wte
```

The runtime role `wte_app` has `NOBYPASSRLS`; every query runs inside a transaction that sets
`app.current_tenant_id` (transaction-local), so cross-tenant reads return zero rows.

## API surface

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/v1/dev/token` | – | Dev-only JWT minting (disabled when `WTE_JWT_SECRET` is set) |
| `POST` | `/v1/cameras` | JWT | Register camera + zone polygons |
| `POST` | `/v1/edge/frames` | HMAC | Edge metadata ingest |
| `POST` | `/v1/webhooks/{provider}/{tenant}/{location}` | provider HMAC | Toast / Square / SevenRooms |
| `GET`  | `/v1/locations/{id}/wait` | JWT | Live estimate (p50/p90, band, source, VLM "why") |
| `GET`  | `/v1/locations/{id}/forecast?hours=24` | JWT | q10/q50/q90 per 15-min bucket |
| `GET`  | `/v1/locations/{id}/best-times?n=3` | JWT | Lowest-wait windows in the next 24 h |
| `POST` | `/v1/locations/{id}/forecast/train` | JWT | (Re)train from observations |
| `WS`   | `/v1/ws/locations/{id}?token=…` | JWT | Live pushes |

## Privacy posture (blueprint §5) — what the code enforces

- No face recognition, age, gender or emotion anywhere; person = anonymous bbox.
- `FaceBlurrer` runs before anything else touches the frame; `VlmClient.assess` refuses
  un-blurred keyframes; `/v1/edge/frames` rejects `faces_blurred=false`.
- Persisted event = `ts, tracker_id, zone, location, dwell` (~40 B). No pixels in the cloud.
- Retention policies in SQL: events 90 d, aggregates 24 m, audit 12 m, raw video 0.
- Re-ID (if enabled) is clothing/silhouette only, RAM-resident, discarded with the track.

## Layout

```
src/wte/
  core/        schemas · settings · bus · registry
  edge/        blur · agent
  vision/      detector · tracker · zones · pipeline
  waittime/    engine
  vlm/         triggers · client
  forecast/    features · models · service
  connectors/  base · toast · square · sevenrooms · matcher
  api/         auth · state · main
  db/          migrations/001_init.sql · repo
web/widget/    wait-widget.js · index.html
deploy/        docker-compose.yml · Dockerfile.api · Dockerfile.edge
scripts/       simulate_edge.py
tests/
```
