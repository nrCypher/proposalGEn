# Restaurant Wait-Time Estimator: A Technical Implementation Blueprint for a Cloud-Based, Privacy-Compliant Commercial Product (2026)

This document is a hands-on technical blueprint for engineering teams building a commercial, cloud-based Restaurant Wait-Time Estimator powered by video analytics, time-series forecasting, and POS/booking integrations. It assumes the four-stage pipeline already proposed (detect → track → zone → time-estimate) and extends it with a "Best Time to Visit" forecasting layer, a VLM reasoning layer, a multi-tenant SaaS backplane, and the privacy controls necessary for EU/US deployment.

> See also: [Mapa de Requisitos](./requirements-diagram.md) — a diagram-first companion to this document.

The system must:

- **Scale elastically** from a single café to multi-thousand-location chains with peak lunch/dinner traffic.
- **Preserve privacy by design** (GDPR, CCPA, EDPB Guidelines 3/2019 on video devices) — never falling into the "biometric / special-category data" classification.
- **Integrate** with Toast, Square, Lightspeed, Clover, Oracle Micros, Revel POS systems and OpenTable, Resy, SevenRooms, TheFork booking platforms.
- **Mix fast CV** (YOLO26 + ByteTrack/BoT-SORT) with **slower contextual VLMs** (Qwen2.5-VL / GPT-4o / Gemini 2.5) at controlled cadence.
- **Forecast** wait-times with Prophet / Chronos-2 / Temporal Fusion Transformer, conditioned on weather, events, and POS signals.

-----

## 1. Cloud Architecture & Deployment Patterns

### 1.1 Reference architecture (recommended)

A pragmatic 2026 deployment for a multi-tenant SaaS wait-time estimator looks like this (cloud-native, hub-and-spoke, with a thin edge agent):

```
[Restaurant site]                              [Cloud control plane (per-region)]
 ┌─────────────────────┐    TLS/RTSP/SRT      ┌────────────────────────────────────┐
 │ IP cameras (ONVIF)  │ ───────────────────► │ Ingestion gateway (Fargate/GKE)    │
 │ PoE switch          │   or WebRTC          │  └─ Kinesis Video Streams / GCP    │
 │ Tiny edge agent     │   (low-latency)      │     Pub/Sub Live or Mediasoup SFU  │
 │  (NVIDIA Jetson /   │                      ├────────────────────────────────────┤
 │   Intel N100, opt.) │  HTTPS metadata      │ Frame router (gRPC) →              │
 │  - face blur        │ ───────────────────► │   GPU inference pool (YOLO26 +     │
 │  - frame sampler    │                      │   ByteTrack / BoT-SORT)            │
 └─────────────────────┘                      │   Triton Inference Server          │
                                              ├────────────────────────────────────┤
                                              │ VLM reasoning service (async)      │
                                              │  - Qwen2.5-VL-7B on vLLM (self)    │
                                              │  - GPT-4o / Gemini 2.5 (fallback)  │
                                              ├────────────────────────────────────┤
                                              │ Event bus: Kafka / Kinesis / PubSub│
                                              │   ↓                                │
                                              │ Stream processor: Flink / Beam     │
                                              │   ↓                                │
                                              │ State: Redis (live), TimescaleDB   │
                                              │ (events), Postgres (tenants),      │
                                              │ S3/GCS (cold blobs)                │
                                              ├────────────────────────────────────┤
                                              │ Forecasting service (Prophet /     │
                                              │ Chronos-2 / TFT) on SageMaker /    │
                                              │ Vertex AI Pipelines                │
                                              ├────────────────────────────────────┤
                                              │ API gateway → FastAPI BFF + WS hub │
                                              │ POS/booking connector hub          │
                                              └────────────────────────────────────┘
```

### 1.2 AWS vs GCP vs Azure — choosing the substrate

|Capability               |AWS                                                                                              |GCP                           |Azure                               |
|-------------------------|-------------------------------------------------------------------------------------------------|------------------------------|------------------------------------|
|Managed video ingestion  |**Kinesis Video Streams** (HLS/WebRTC, KMS-encrypted, hot+warm tiers)                            |Live Stream API + Pub/Sub     |Azure Video Indexer / Media Services|
|Managed CV               |Rekognition Video / SageMaker Async/RT endpoints                                                 |Vertex AI + Video Intelligence|Cognitive Services + ACI/AKS        |
|Time-series store        |**Timestream for InfluxDB** (Timestream LiveAnalytics is closed to new customers as of June 2025)|BigQuery + Bigtable           |Data Explorer                       |
|Stream processing        |Kinesis Data Streams, MSK, Flink on KDA                                                          |Dataflow (Beam), Pub/Sub      |Event Hubs, Stream Analytics        |
|Pretrained forecast      |Forecast (legacy), Chronos-2 on SageMaker                                                        |Vertex AI Forecast / TimesFM  |AutoML Forecast                     |
|K8s                      |EKS + Karpenter                                                                                  |GKE Autopilot                 |AKS                                 |
|GPU                      |g5/g6 (L4), p4d/p5 (A100/H100), Inferentia2                                                      |A2/A3 (A100/H100), L4         |NC/ND series                        |
|EU data-residency posture|strong (Frankfurt/Ireland/Stockholm)                                                             |strong (Belgium, Frankfurt)   |strong (multiple)                   |

**Recommendation.** Start on **AWS** for V1 because Kinesis Video Streams is the most production-hardened path for ingesting RTSP at scale (it auto-scales to millions of devices, encrypts at rest with KMS, supports HLS playback, on-demand image extraction APIs, fragment-level access for ML, and managed WebRTC for sub-second viewing). Use **Timestream for InfluxDB** for time-series (AWS now steers new customers there since Timestream LiveAnalytics closed to new customers in June 2025). Keep Terraform/Pulumi modules portable to GCP for European customers with hard residency mandates.

### 1.3 Video ingestion: which protocol when

- **RTSP from camera → cloud gateway**: works on every IP camera, simple to operationalize. AWS publishes a reference design that runs a GStreamer + `kvssink` pipeline inside an AWS Fargate container, pushing fragments into Kinesis Video Streams. This is the recommended ingestion path for cameras you don't control. Latency is 3–6 s — fine for queue analytics.
- **WebRTC / KVS-WebRTC**: choose when you need sub-second latency for an in-app live view (e.g., the host stand sees a live overlay). Glass-to-glass latency ~140–160 ms in benchmarks; supports 1:1 and 1:N but not for storage/analytics ingestion.
- **SRT or low-latency HLS (LL-HLS)**: practical alternative for self-hosted ingestion; works well over the public internet with packet loss recovery.
- **Edge agent (recommended for production)**: a lightweight Docker container on a $200–400 mini-PC (Intel N100 or Jetson Orin Nano) at each site that (a) terminates RTSP locally, (b) face-blurs at source for GDPR posture, (c) samples to 5–10 fps, and (d) pushes only blurred frames + lightweight metadata to the cloud. This is the **single most important architectural decision** for European deployments — it converts the cloud service from a "video processor" into a "metadata processor," radically reducing the DPIA surface area.

### 1.4 Edge-cloud hybrid vs fully cloud — recommended pattern

|Pattern                        |Pros                                                                                                         |Cons                                                                                     |When to use                     |
|-------------------------------|-------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|--------------------------------|
|Fully cloud (raw RTSP to cloud)|Single code path; easier ML iteration; no on-site hardware                                                   |Bandwidth cost (1–4 Mbps × cameras × 24×7); higher GDPR exposure; cloud egress for review|US-only chains, demo deployments|
|**Hybrid (recommended)**       |Privacy-by-design (face blur at edge, never leaves site); 10–100× less bandwidth; resilient to internet blips|Edge fleet to manage; OTA updates required                                               |EU customers, enterprise chains |
|Fully on-prem                  |Maximal privacy, no internet dependency                                                                      |Each site needs GPU; no cross-site forecasting; defeats SaaS economics                   |Casinos / luxury hotels only    |

A "pragmatic hybrid" is: edge runs detection+blur+frame-sampling; cloud runs tracking, zone reasoning, VLMs, forecasting, and dashboards. Detection metadata (bbox + tracker_id + zone events) is JSON, ~1 KB/s/camera vs ~500 KB/s for video. This pattern is what AWS, Realtek, and Plumerai showcased in their reference design — "AI detection directly on the device, video data stays local until needed."

### 1.5 Containerization and serverless GPU

- **Containerize** every stage with Docker; deploy with **Kubernetes (EKS/GKE)**. Use Karpenter or GKE Autopilot to bin-pack GPU nodes.
- **GPU inference** options for the YOLO26 + tracker hot path:
  - **NVIDIA Triton Inference Server** on dedicated g5.xlarge / L4 nodes (best for steady-state, cheapest per-frame at >40% utilization).
  - **AWS SageMaker Async Endpoints** for VLM batches and forecasting jobs.
  - **Modal** for bursty workloads — sub-second cold starts via container caching, per-second billing, Python-native SDK; ~$3.50/hr for A100-40GB.
  - **RunPod Serverless** — cheapest in benchmarks ($0.40/hr T4, $2.17/hr A100 80GB; 48% of cold starts <200 ms with FlashBoot).
  - **Replicate** — best when you also want a public model marketplace face; per-token billing for LLMs.
  - **Baseten (Truss)** — good DX; A10G/A100/H100 with config-as-code.
- **Decision rule**: dedicated nodes when utilization >40–50%, serverless below that. For wait-time analytics, the detection path is steady (cameras are always streaming) → dedicated GPUs; the VLM path is spiky → serverless GPU is a perfect fit.

### 1.6 Auto-scaling for restaurant peak hours

Lunch (11:30–14:00) and dinner (18:00–22:00) generate the load; queue length per camera also spikes 2–5×. Three layers of scaling:

1. **Frame-rate adaptation**: drop input fps from 10→3 when no detections, and back to 10 when bbox count rises. Saves 50–70 % GPU.
2. **Horizontal scaling**: HPA on `gpu_utilization` and `frames_in_flight`, with KEDA scalers tied to Kafka queue depth.
3. **Predictive pre-warming**: a cron job pre-scales the GPU pool 10 min before the known peak per region, avoiding cold-start scramble. Use the forecasting service's own predictions as the scaling signal — the same model that powers "Best Time to Visit" tells the cluster when to expand.

### 1.7 Cost envelope (illustrative, mid-2026 list prices)

For a single restaurant with 2 cameras, 10 fps after sampling, 16 hours/day:

- KVS ingestion: ~$25/mo (data ingested + stored 7 days hot).
- GPU inference (shared L4, ~5 % of one card): ~$15–25/mo amortized.
- TimescaleDB / Timestream-InfluxDB: ~$5/mo for events.
- Egress + dashboard: ~$5/mo.
- **Total compute COGS ~$50–60/restaurant/month**, leaving healthy gross margin if you price at $99–149/site/month.

For a 500-site chain with edge agents (only metadata to cloud): COGS drops to ~$15–25/site/month because GPU usage shifts to the (one-time) edge hardware (~$300 capex amortized over 3 yrs ≈ $8/mo). VLM API calls are the variable cost driver — ration them aggressively (Section 3).

-----

## 2. Object Detection & Tracking Stack

### 2.1 Detector selection: YOLO26 is the new default

The Ultralytics **YOLO26** family was unveiled at YOLO Vision 2025 (London, Sept 2025) and reached general availability in January 2026. Compared to YOLO11 / YOLOv12 / YOLOv13, YOLO26 introduces:

- **Removal of Distribution Focal Loss (DFL)** — simplifies inference, broadens export targets (TFLite, CoreML, OpenVINO, TensorRT, ONNX).
- **Native NMS-free end-to-end inference**, eliminating the sequential post-processing latency variability that plagued previous YOLO versions in dense scenes (critical for queue counting where occlusion is common).
- **ProgLoss** (progressive loss balancing) and **STAL** (Small-Target-Aware Label assignment) — measurably better small-object detection, helpful for top-down camera angles where heads are small.
- **MuSGD optimizer** — more stable convergence during fine-tuning on your queue dataset.
- Quantization-aware design — INT8/FP16 export remains accurate; YOLO26 outperformed YOLO11/v12 at identical quantization in published benchmarks.

For practical use:

- **YOLO26-n / YOLO26-s** on the edge agent (Jetson Orin Nano, 30+ FPS at 640×640 INT8).
- **YOLO26-m / YOLO26-l** on cloud GPUs for higher-resolution streams (1080p).
- Compare against **RT-DETR** and **RF-DETR** (transformer-based) only if you have heavily occluded scenes; they slightly outperform on crowded benchmarks but cost ~2× the latency.

Train with the standard Ultralytics workflow:

```python
from ultralytics import YOLO

model = YOLO("yolo26m.pt")
model.train(
    data="queue_dataset.yaml",   # only 'person' class
    epochs=100, imgsz=960,
    batch=16, device=[0],
    optimizer="MuSGD",
    cos_lr=True, mixup=0.1,
    project="wait_time", name="yolo26m_v1"
)
model.export(format="engine", half=True)  # TensorRT FP16 for cloud
```

Fine-tune on **top-down restaurant queue imagery** — the COCO-pretrained `person` class underperforms when cameras are mounted on ceilings. Roboflow Universe has several public top-down person datasets to bootstrap; combine with 2–5 hours of customer footage per chain (post-blur).

### 2.2 Tracker selection for queue scenarios

Empirical guidance from published benchmarks (MOT17/20, DanceTrack, and traffic/retail studies):

|Tracker                    |Strengths                                                                                                               |Weaknesses                                    |Verdict                                                                                                                          |
|---------------------------|------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
|**ByteTrack**              |Fastest, recovers via low-confidence boxes, no appearance net needed                                                    |Pure motion → IDswitches under heavy occlusion|**Default choice.** Two-stage matching keeps tracks alive during brief occlusions, which is exactly what queue stalls look like. |
|**BoT-SORT**               |More accurate; ECC camera-motion compensation; ReID supported                                                           |~2× compute of ByteTrack                      |Use when cameras pan/tilt or for premium tier.                                                                                   |
|**OC-SORT**                |Excellent on non-linear motion; CPU-fast                                                                                |Slightly weaker than BoT-SORT in dense scenes |Good fallback for CPU-only edge.                                                                                                 |
|**StrongSORT / DeepOCSORT**|Best ID stability with deep ReID; lowest ADE in benchmarks (DeepOCSORT 31 px vs ByteTrack 114 px on a tiny-object test) |Requires GPU and a ReID model                 |Use only if ID swaps hurt KPI accuracy materially.                                                                               |
|**DeepSORT**               |Classic, well-documented                                                                                                |Older; superseded                             |Avoid for new builds.                                                                                                            |

**Recommendation**: ByteTrack as default; expose BoT-SORT as a tenant-level setting for premium tier. Both are first-class in `roboflow/supervision`.

### 2.3 Re-identification (Re-ID) without biometrics

To preserve a "non-biometric" classification for GDPR purposes (Section 5), do **not** train Re-ID on faces. Instead:

- Use a **clothing/silhouette embedding** model (e.g., `osnet_x0_25` from `torchreid`, or a custom CLIP-image encoder fine-tuned on body crops with faces masked).
- Embeddings live only in RAM during a tracking session and are discarded when the track ends. Never write to disk.
- Document this in your DPIA: "the system processes appearance features that are non-unique, transient, and incapable of identifying a specific individual across visits."

### 2.4 Occlusion handling

- **Top-down ("bird's-eye") cameras** are the gold standard for queue counting. Mount at 3–4 m, lens fov 90–110°. Top-down virtually eliminates the worst occlusion class (people behind people).
- **Multi-camera fusion**: when a single FoV cannot cover the queue, use 2–3 cameras with overlapping zones; reconcile tracks via timestamp + zone-handoff (a track exits zone A in cam1 within 2 s of entering zone A' in cam2 → same trajectory).
- **Crowd density mode** activates when in-zone bbox count exceeds ~25: switch from detection+tracking to a density-map model (CSRNet for sparse, **P2PNet / P2PNeXt** for dense, point-based). The density map gives count without identities; you fall back to tracking only on the queue boundary.

### 2.5 Recommended Python stack

- **ultralytics** (training, inference, exports).
- **supervision** (Roboflow) — the production-grade framework for zones, line counters, annotators, ByteTrack glue, occupancy analytics. Examples for retail queue dwell-time exist in Roboflow's blog.
- **mmtracking** / **boxmot** — tracker zoo if you want BoT-SORT / StrongSORT / DeepOCSORT side-by-side.
- **torchreid** — for the clothing/silhouette embedding model.
- **opencv-python-headless** + **gstreamer** — RTSP demux on the edge.
- **NVIDIA DeepStream** (optional) — best-in-class GPU pipeline if you go all-in on Jetson.

### 2.6 Reference snippet — the core pipeline (cloud-side, single camera)

```python
import cv2, numpy as np, supervision as sv
from ultralytics import YOLO

model = YOLO("yolo26m.engine")          # TensorRT FP16
tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=60)

# Zones loaded from per-camera config
QUEUE   = sv.PolygonZone(polygon=cfg["queue_polygon"])
SERVICE = sv.PolygonZone(polygon=cfg["service_polygon"])
FULFIL  = sv.PolygonZone(polygon=cfg["fulfillment_polygon"])

dwell = {}  # tracker_id -> {zone: enter_ts}
events = []

def process(frame, ts):
    res  = model(frame, classes=[0], conf=0.35, iou=0.6, verbose=False)[0]
    det  = sv.Detections.from_ultralytics(res)
    det  = tracker.update_with_detections(det)
    for zone, name in [(QUEUE,"queue"),(SERVICE,"service"),(FULFIL,"fulfil")]:
        in_zone = zone.trigger(det)
        for tid, inside in zip(det.tracker_id, in_zone):
            if tid is None: continue
            key = (int(tid), name)
            if inside and key not in dwell:
                dwell[key] = ts
            if (not inside) and key in dwell:
                events.append({
                    "tracker_id": int(tid), "zone": name,
                    "enter": dwell.pop(key), "exit": ts,
                })
    return det, events
```

This emits zone-transition events to Kafka; the wait-time service consumes them.

-----

## 3. Vision-Language Model Integration

### 3.1 Why blend VLMs at all?

YOLO/ByteTrack will tell you "12 people in the queue zone, average dwell 4 m 12 s." A VLM tells you *why*: "the queue is held up because the espresso machine is being cleaned" or "two large parties just walked in for pickup." That semantic context is what lets the product upgrade from a counter to an *advisor*.

### 3.2 Two-tier inference pattern (cost discipline)

Run VLMs on a **debounced, event-driven** schedule, never per-frame:

1. **Tier-1 (every frame, fast, deterministic)**: YOLO26 + ByteTrack + zone logic. Latency budget ≤ 100 ms.
2. **Tier-2 (every 30–60 s, or on anomalies)**: VLM call with one keyframe + the last minute of metadata.

Anomalies that should trigger an immediate VLM call:

- Queue length jumped >50 % in 60 s.
- Average dwell exceeded the 95th percentile of historical baseline.
- Service zone is empty but queue is long (likely staffing issue).
- Customer count is high but POS order rate is flat (forming bottleneck).

This pattern keeps the VLM bill predictable. At 30 s cadence × 16 h × 30 days × 2 cameras × $0.005/call (GPT-4o Vision input cost) ≈ **$58/restaurant/month** worst case; ~$8/month with anomaly-only triggering.

### 3.3 Model choices and where to host them

|VLM                         |Hosting                  |Strengths                                   |Use                                          |
|----------------------------|-------------------------|--------------------------------------------|---------------------------------------------|
|**GPT-4o / GPT-4.1**        |OpenAI API               |Best reasoning, JSON mode, vision quality   |Premium tier, US tenants                     |
|**Gemini 2.5 Pro / Flash**  |Google API               |Long video context, cheapest at scale       |Default cloud tier                           |
|**Claude Sonnet 4 (vision)**|Anthropic API            |Best instruction-following, careful refusals|Customer-facing summaries                    |
|**Qwen2.5-VL-7B / 72B**     |Self-hosted via vLLM     |Apache-2.0 (7B), accurate bbox + JSON       |**EU privacy tier** (no data leaves your VPC)|
|**InternVL2.5 / 3**         |Self-hosted              |Strong on documents/charts, MIT/Apache      |Alternative to Qwen                          |
|**MiniCPM-V 2.6**           |Self-hosted, edge-capable|Runs on a single 4090 / Jetson Orin         |Air-gapped or edge VLM                       |
|**LLaVA-NeXT / Video-LLaVA**|Self-hosted              |Mature OSS                                  |Research/baseline                            |

**Recommendation for an EU-friendly default**: deploy Qwen2.5-VL-7B-Instruct on **vLLM** in your own AWS Frankfurt VPC, with structured output via `guided_json`. The 7B model fits on a single L4 (24 GB) with FP8 quantization and produces reliable bounding boxes and JSON. Reserve GPT-4o/Gemini for premium-tier accounts where the customer accepted a US sub-processor.

### 3.4 Prompting for structured output

Use JSON-schema-constrained generation. Example (Qwen2.5-VL via vLLM):

```python
SCHEMA = {
  "type":"object",
  "properties":{
    "queue_state":{"type":"string","enum":["short","normal","long","gridlocked"]},
    "estimated_wait_minutes":{"type":"number"},
    "bottleneck":{
      "type":"object",
      "properties":{
        "location":{"type":"string","enum":["counter","kitchen","fulfillment","none"]},
        "evidence":{"type":"string"}
      },
      "required":["location","evidence"]
    },
    "operational_recommendation":{"type":"string"},
    "confidence":{"type":"number","minimum":0,"maximum":1}
  },
  "required":["queue_state","estimated_wait_minutes","bottleneck","confidence"]
}

system = """You are a restaurant operations analyst. The image shows a queue area in
a restaurant. Faces have been blurred for privacy. Use posture, group size, distance to
counter, and fulfillment activity to assess flow. Be concise. Never speculate about
identity. Output strictly the JSON schema."""

# vLLM call
client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct-FP8",
    messages=[{"role":"system","content":system},
              {"role":"user","content":[{"type":"image_url","image_url":{"url":frame_b64}},
                                        {"type":"text","text":metadata_summary}]}],
    extra_body={"guided_json": SCHEMA},
    temperature=0.1, max_tokens=400
)
```

Always send a **face-blurred** keyframe (Section 5) and a textual summary of the last minute of CV metadata so the VLM doesn't need to count people itself — it should *reason*, not *perceive*. This halves token cost and removes the model's main failure mode.

### 3.5 Latency budgets

- GPT-4o vision: 1.5–3 s typical.
- Gemini 2.5 Flash: 0.8–1.5 s.
- Qwen2.5-VL-7B on vLLM (single L4): ~700 ms for 1 keyframe + 200 output tokens.

These are **never** in the customer-facing critical path. The wait-time number is computed from CV alone; VLM output flows asynchronously into the dashboard ("Why is this queue slow?" panel) and into POS alerts ("notify manager: counter understaffed").

-----

## 4. Time-Series Forecasting for "Best Time to Visit"

### 4.1 Model selection

There is no single best model. The recommended stack is layered:

|Model                                     |Role                                      |Why                                                                                                                                                                                                                 |
|------------------------------------------|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|**Prophet**                               |Baseline; per-restaurant; explainable     |Mature, fast to train, native holiday/event regressors, interpretable component plots — excellent customer-facing chart material.                                                                                   |
|**NeuralProphet**                         |Drop-in successor to Prophet              |Adds AR component and PyTorch backend; 50–90 % accuracy gains for short-medium horizons in published benchmarks; same API.                                                                                          |
|**Temporal Fusion Transformer (TFT)**     |Premium, multi-horizon, multivariate      |Native handling of static covariates (restaurant ID, cuisine), known-future inputs (weather forecast, holidays), and unknown-future (real-time observations). Quantile output gives confidence bands for free.      |
|**Chronos-2 (AWS)** / **TimesFM (Google)**|Zero-shot foundation model for new tenants|New tenants have <1 month of data; Chronos-2 (released Oct 2025) provides usable forecasts day 1 from minimal context. ~300 forecasts/sec on 1 GPU; verified to beat Auto-ARIMA by 15–25 % on retail visit datasets.|
|**TimeGPT (Nixtla)**                      |Hosted alternative                        |Convenience over control; benchmarks vary.                                                                                                                                                                          |

**Recommended production strategy:**

1. **Cold-start (0–14 days of data)**: Chronos-2 zero-shot, computed nightly per tenant.
2. **Warm (14–60 days)**: Prophet/NeuralProphet per restaurant per zone, retrained nightly.
3. **Mature (>60 days)** *and* >5 restaurants: TFT trained globally across the chain with per-restaurant static embeddings; per-tenant fine-tune.

Forecast at **15-minute granularity** for a 7-day horizon; aggregate up for the customer-facing graph.

### 4.2 Feature engineering

Predictors (all should be aligned to the 15-min bucket):

- **Calendar**: minute-of-day, day-of-week, week-of-year, is_holiday (country-specific, `python-holidays`), is_school_holiday, days-since-payday, is_month_end.
- **Weather** (OpenWeather API or Open-Meteo, free tier): temperature, precipitation probability, wind, "feels-like." Use the *forecast* values as known-future inputs in TFT/Prophet.
- **Local events**: Ticketmaster Discovery API and Google Events / SeatGeek to detect concerts / sports matches within a 2-km radius. Encode as `event_within_2km_in_3h: bool`, `expected_attendance_log: float`.
- **POS-derived**: lagged `orders_per_hour`, `avg_check_size`, `staff_on_clock` (from POS labor module if available).
- **Marketing**: active promotions, ads, app-coupon redemptions.
- **Self-features**: wait-time same hour last week, last 4 same-day-of-week averages.

### 4.3 Confidence intervals

Customers want ranges, not points. Two equivalent options:

- Prophet: `interval_width=0.8`, render the 80% band.
- TFT: train with quantile loss for q ∈ {0.1, 0.5, 0.9}; render q10–q90 as the band, q50 as the line.

Display the band as a translucent fill; *never* show a single number for "Best Time" because the choice should be visibly probabilistic — this is also a UX safeguard for trust.

### 4.4 Database choice

After June 2025, AWS Timestream LiveAnalytics is closed to new customers; AWS now points to Timestream for InfluxDB. Pragmatic 2026 picks:

|DB                                |Use it when…                                                                                                                                                                                                    |
|----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|**TimescaleDB**                   |You want one Postgres for tenants AND time-series. Hyperfunctions (`time_bucket`, `first`, `last`, interpolation) shine. v2.25 hit 289× faster MIN/MAX on compressed data. **Default choice for this product.** |
|**InfluxDB 3 (Apache 2.0/MIT)**   |Pure metrics workload, IoT-style ingestion, Telegraf agents. Best storage compression (10–20×).                                                                                                                 |
|**ClickHouse**                    |Analytics on billions of rows with sub-second p95 queries; plug-in alongside TimescaleDB if you need a separate OLAP layer.                                                                                     |
|**Amazon Timestream for InfluxDB**|Want managed InfluxDB without ops. Good for AWS-native shops.                                                                                                                                                   |
|**QuestDB / VictoriaMetrics**     |Highest ingest throughput / Prometheus-compat metrics.                                                                                                                                                          |

**Recommendation**: TimescaleDB for events/wait-times; Postgres for tenants/users; Redis for live state; ClickHouse only if dashboard queries become slow at >10 B rows.

### 4.5 Streaming analytics

- **Kafka (or Amazon MSK)** as the event bus — every zone enter/exit and POS webhook becomes a topic message.
- **Apache Flink** (KDA / GKE-Flink) for windowed aggregations: 1-minute occupancy, 5-minute moving wait-time, hourly visits. Feed results into TimescaleDB and Redis.
- For smaller installs, **Materialize** or **RisingWave** (Postgres-wire-compatible streaming SQL) is a lower-ops alternative.

-----

## 5. GDPR/CCPA Privacy Compliance

This is the section that decides whether a European launch is even legal. Read carefully.

### 5.1 Legal regime for video analytics in EU restaurants

The controlling document is the **EDPB Guidelines 3/2019 on processing of personal data through video devices** (final version adopted Jan 2020; reaffirmed 2024–2026 communications; April 2026 EDPB summary "Video devices & data protection: When to act and what to do"):

- **Lawful basis (Art. 6 GDPR)**: Consent is unworkable for cameras (cannot be "freely given" by people who simply enter the premises). Use **Art. 6(1)(f) legitimate interests** with a documented Legitimate Interests Assessment (LIA), which involves a balancing test against data-subject rights.
- The EDPB explicitly states: "individuals would usually expect not to be monitored in leisure areas such as gyms and **restaurants**" — meaning the balancing test is *against* you by default, and the system must be narrowly purposed.
- **Special-category / biometric data (Art. 9)**: triggered if you process data "for the purpose of uniquely identifying a natural person." If you ever attach an identity, run face recognition, age/gender estimation that could uniquely identify, you cross into Art. 9 territory and need either explicit consent or another Art. 9 exception — which is essentially *impossible* in a queue analytics product. Stay out of Art. 9.
- **DPIA** (Art. 35) is mandatory because video analytics is "systematic monitoring of a publicly accessible area on a large scale." Do it once per product, then maintain it as a living document and supplement per-customer where the deployment differs materially.

### 5.2 Engineering for "non-biometric, no special-category"

- **Never** run face recognition, age, gender, or emotion classification.
- **Detect persons as anonymous bounding boxes** only.
- **Re-ID** with clothing/silhouette only, with embeddings ephemeral (RAM only, TTL = session).
- **Blur every face at the edge before any frame leaves the camera/edge box.** The reference pipeline:
  1. Run a face detector (RetinaFace ONNX or YOLO-face, ~5 ms on CPU).
  2. Apply a Gaussian blur (σ=20) or pixelization (12×12 mosaic) to each face bbox, expanded by 30 %.
  3. Optionally use a generative anonymizer (DeepPrivacy2) for marketing demos, but do not need it for compliance — irreversibility is what matters.
  4. Verify: no recognizable face in transmitted frames; spot-audit weekly.
- The **EuroPriSe-certified face-pixelation paradigm** is the precedent here: the European Privacy Seal has been issued to surveillance products that do exactly this, recognizing it as GDPR-compatible.

### 5.3 Data minimization in storage

- **Persist** only: timestamp, anonymous tracker_id (rotates daily), zone_id, restaurant_id, dwell_seconds. ~40 bytes/event.
- **Never** persist raw frames in the cloud. If the customer requests a "live view," stream it via KVS-WebRTC peer-to-peer (face-blurred); no fragments are stored.
- VLM keyframes: stored only if the operator pinned a specific incident; default retention 0.
- Aggregated analytics retained for 24 months (Prophet/TFT need ≥1 yr to capture annual seasonality).

### 5.4 Consent, signage, transparency (the "layered" approach mandated by EDPB)

- **Layer 1 (sign at entrance)**: pictogram + short text: *"Anonymous queue-time analytics in operation. No faces or personal data are recorded."* Include controller name, purpose, contact for rights.
- **Layer 2 (QR / URL)**: full privacy notice with legal basis, retention, sub-processors, DPIA summary, data-subject rights.
- **Layer 3 (DPO email)** for SAR and objection requests.
- The signage does not change the legal-basis analysis (EDPB is explicit) but it does fulfill Art. 13 transparency. Provide signage templates as part of the customer-onboarding kit.

### 5.5 Retention policy (template)

|Data class                            |Retention                            |Justification          |
|--------------------------------------|-------------------------------------|-----------------------|
|Raw video                             |0 (never leaves edge or KVS hot tier)|Minimization           |
|Face-blurred frames pinned by operator|7 days                               |Operational review     |
|Tracker metadata (events)             |90 days                              |Operational reporting  |
|Aggregated wait-time series (15-min)  |24 months                            |Forecasting seasonality|
|Audit logs                            |12 months                            |Security, Art. 32      |

### 5.6 Sub-processor and cross-border transfers

- Maintain a public sub-processor list (AWS, OpenAI, Google, Anthropic, etc.).
- AWS, GCP, Azure all offer SCCs + Data Processing Addenda. Use EU regions exclusively for EU tenants and disable cross-region replication to non-EU.
- For the **VLM tier specifically**: GPT-4o and Gemini API calls send images to US sub-processors. Either (a) require the customer to opt in and update SCCs, or (b) default EU tenants to the **self-hosted Qwen2.5-VL** path inside your Frankfurt VPC. Post-Schrems II, even with SCCs the supplementary measures must include encryption-in-transit and minimization (face blur) — which you already do.

### 5.7 CCPA / US state laws

CCPA/CPRA, Texas TDPSA, and Colorado CPA all have softer thresholds, but all share these obligations:

- Disclose categories of personal information collected (none, if you do non-biometric only — but the law treats *any* image of a person as personal info, so disclose "anonymized video metadata").
- Honor **opt-out** ("Do Not Sell or Share") even though you don't sell — wire a public opt-out endpoint.
- Restaurant operators in IL must additionally consider **BIPA** — strictly avoid biometrics (you already do). Texas CUBI similarly.

### 5.8 DPIA outline (template)

A defensible DPIA covers: (1) systematic description of processing, (2) necessity & proportionality vs. alternatives (manual headcount, infrared sensors), (3) risks to rights (re-identification, function-creep, child data), (4) mitigations (edge blur, no biometrics, retention caps, access control, encryption), (5) residual risk. Tools like OneTrust, Vigilant Software, TrustArc, or open-source CNIL PIA software accelerate this.

-----

## 6. POS & Booking System Integrations

### 6.1 POS landscape — what to build

|POS                                            |API style                                  |Auth                         |Notes                                                                                                                                |
|-----------------------------------------------|-------------------------------------------|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
|**Toast**                                      |REST + Webhooks (orders, menu, employees)  |OAuth 2.0 partner integration|Webhook events for `order.created`, `order.modified`, status changes; HMAC-SHA256 signed payloads. Most US fast-casual market share. |
|**Square**                                     |REST + Webhooks; very developer-friendly   |OAuth 2.0                    |Excellent docs; covers small operators.                                                                                              |
|**Lightspeed** (Restaurant K-Series / X-Series)|REST + Webhooks, GraphQL on X-Series       |OAuth 2.0                    |Strong in Europe.                                                                                                                    |
|**Clover**                                     |REST + Webhooks; merchant app marketplace  |OAuth 2.0                    |Tightly tied to hardware.                                                                                                            |
|**Oracle Micros (Simphony)**                   |OHIP REST API, legacy SOAP for older sites |API key + JWT                |Enterprise / fine dining; longer integration cycle.                                                                                  |
|**Revel**                                      |REST                                       |API key                      |iPad-based; mid-market.                                                                                                              |

Build a **connector hub** as a separate microservice: each connector is a Python adapter that normalizes the POS data into a single internal schema:

```python
class PosEvent(BaseModel):
    tenant_id: UUID
    location_id: str
    event_id: str
    event_type: Literal["order_created","order_completed","order_cancelled","check_open","check_closed"]
    occurred_at: datetime
    order_id: str
    party_size: int | None
    channel: Literal["dine_in","takeaway","delivery"]
    raw: dict   # original payload for audit
```

### 6.2 Booking platforms

|Platform                                  |API access                                                                               |Notes                                           |
|------------------------------------------|-----------------------------------------------------------------------------------------|------------------------------------------------|
|**OpenTable**                             |Partner API (`platform.opentable.com/documentation`); native Meta Pixel, no public TikTok|Historically gated; partner agreement needed.   |
|**Resy**                                  |No third-party pixel; partner API on case-by-case                                        |More gated; Amex-owned.                         |
|**SevenRooms**                            |100+ integrations + flexible REST API; widget JS events (`successfulCheckout`)           |Most open; 2-way reservation sync, CRM webhooks.|
|**TheFork** (LaFourchette)                |Partner API; the European leader                                                         |Required for EU coverage.                       |
|**Yelp Reservations / Yelp Guest Manager**|API for partners                                                                         |Smaller share now.                              |
|**Tock / Eat App / ResDiary**             |Partner APIs                                                                             |Niche but growing.                              |

### 6.3 Webhook → host-stand sync

The integration loop:

1. **POS `check_open` webhook** → mark party as "seated" → stop their queue timer if matched by tracker_id (heuristic: party size + zone exit time within ±60 s of POS timestamp).
2. **POS `check_closed` webhook** → table free; update available capacity → push new wait estimate to host stand and to consumer-facing widget.
3. **Booking webhook (reservation walk-in)** → reduce predicted wait for that party; surface "your party is next" notification via Twilio SMS / browser push.

Two-way sync challenges:

- **Idempotency**: every webhook handler must dedupe by `event_id`. POS retries on transient HTTP errors.
- **Clock skew**: don't rely on client timestamps; use server arrival time and reconcile.
- **Replay**: persist raw payloads; expose an admin endpoint to replay missed windows after an outage.
- **Rate limits**: Toast and Square enforce per-app rate limits — implement token-bucket on outbound calls and exponential backoff with jitter.
- **Data-model drift**: modifiers, taxes, voided items, splits-by-seat differ across POS. Keep `raw` JSON for forensics.

### 6.4 Authentication patterns

- **OAuth 2.0** for Toast, Square, Lightspeed, Clover, OpenTable partner — store refresh tokens encrypted with KMS, rotate.
- **HMAC signing** for webhooks — verify every payload (Toast uses HMAC-SHA256).
- **mTLS** for enterprise (Oracle Micros) where the customer requires it.
- **Per-tenant scoped tokens**; never share an API key across tenants.

-----

## 7. Backend & API Architecture

### 7.1 API design

- **REST (FastAPI)** for CRUD: tenants, locations, cameras, zones, users, billing.
- **GraphQL (Strawberry / Hasura)** is **optional** — use only if your dashboard is highly nested; not needed for V1.
- **gRPC** for internal service-to-service (frame-router → inference, inference → tracker, tracker → wait-time service). Lower overhead, schema discipline.
- **WebSocket** for live dashboard pushes (FastAPI's `@app.websocket` + a Redis-pub/sub fan-out to scale across pods).
- **Server-Sent Events (SSE)** is a simpler one-way alternative — reach for it for the customer-facing widget where you only push and never receive.

### 7.2 Recommended frameworks

- **FastAPI + Uvicorn** (Python) for the BFF — enormous Python ecosystem alignment with the ML stack, async-first, native OpenAPI, native WebSocket.
- **Go (Fiber / chi)** for the connector hub — lower memory per webhook handler, easier static binary deploys.
- **Node.js** if your team is JS-heavy; Fastify is a good pick.

### 7.3 Database architecture

```
┌──────────────────┐  ┌────────────────────────────┐  ┌──────────────────┐
│ Postgres (RDS)   │  │ TimescaleDB (RDS Postgres + │ │ Redis (Elasticache│
│ tenants, users,  │  │  timescaledb extension)     │ │  / Memorystore)  │
│ locations,       │  │ - zone_events               │ │ - live_state per │
│ cameras, zones,  │  │ - wait_time_buckets (15m)   │ │   location       │
│ rbac, billing    │  │ - pos_events                │ │ - rate-limit     │
│ (RLS for tenant) │  │ - forecasts                 │ │   counters       │
└──────────────────┘  └────────────────────────────┘  └──────────────────┘
       │                            │                          │
       └────────────── single VPC, private subnets ─────────────┘
```

S3/GCS for cold archives (aggregated daily exports for tenants requesting their data per Art. 20 GDPR portability).

### 7.4 Multi-tenancy: row-level security as the spine

The 2025–2026 best-practice for SaaS multi-tenancy on Postgres is **PostgreSQL Row-Level Security (RLS) with a session GUC variable**, as documented in the AWS Prescriptive Guidance and reproduced by Nile, Permit.io, and many others. The pattern:

```sql
-- one tenant_id column on every tenant-scoped table
ALTER TABLE zone_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE zone_events FORCE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON zone_events
  USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE INDEX idx_zone_events_tenant_ts ON zone_events (tenant_id, ts DESC);
```

In the FastAPI middleware, on every request:

```python
@app.middleware("http")
async def set_tenant_context(request, call_next):
    tenant_id = extract_from_jwt(request)  # validate signature first
    async with db.acquire() as conn:
        await conn.execute("SELECT set_config('app.current_tenant_id', $1, true)", str(tenant_id))
        request.state.conn = conn
        return await call_next(request)
```

Crucial details:

- Use `SET LOCAL` (transaction-scoped) so a leaked connection cannot leak data after release back to the pgbouncer pool.
- Have a separate `app_migrator` role with `BYPASSRLS` for migrations only; the runtime app role must NOT have BYPASSRLS.
- Add an integration-test suite that, for every endpoint, asserts cross-tenant access returns 0 rows. Nile's blog recommends Postgres Testcontainers for this.

When to escalate from RLS-pooled to **schema-per-tenant** or **DB-per-tenant**:

- HIPAA-style customers requesting hard isolation (rare for restaurants).
- Tenants exceeding ~10 % of total data volume (noisy-neighbor performance).
- Explicit data-residency contract (then DB-per-region).

### 7.5 API gateway, rate-limiting, observability

- **AWS API Gateway / GCP API Gateway / Kong / Tyk** in front of FastAPI; usage plans per tenant.
- **Per-tenant rate limit** (e.g., 100 r/s, 10 000 r/min) implemented in Redis with leaky-bucket; surface 429 with `Retry-After`.
- **OpenTelemetry** SDK in FastAPI; export to **Tempo / Jaeger** and **Prometheus**.
- **mTLS** between internal services (Linkerd / Istio).

-----

## 8. Frontend Dashboard & Customer-Facing UI

### 8.1 Operator dashboard (web)

- **React 19 + TypeScript** with **TanStack Query** (data) and **Zustand** (UI state).
- **Charts**: **Apache ECharts** is the strongest all-rounder for live dashboards (rich animations, candlestick-quality polish, supports >100 k points). **Recharts** is simpler but limited; **Plotly** is good when scientific interactivity is needed; **D3.js** only when you need custom visualisations beyond chart libraries.
- **Layout**: shadcn/ui + Tailwind for speed.
- **Live updates**: a single WebSocket per tab subscribing to `tenant:{id}:location:{id}` Redis channel; backend hub fans out via Redis pub/sub. SSE is an acceptable simpler alternative for read-only widgets.

### 8.2 Customer-facing wait-time widget

- Embeddable **iframe** + **Web Component** so restaurants can drop it on any site without a build step.
- **Mobile-first**: a single number ("~12 min") + a colour band + a "Best time today" mini-chart.
- "Notify me when wait < 5 min" via web push (VAPID) and SMS (Twilio).
- Render with React 19 + Vite, ~30 KB gzipped.

### 8.3 "Best Time to Visit" interactive graph

Implementation outline (ECharts):

- X axis: hour of day (00–23) or 7-day calendar.
- Series 1: forecast median (q50), thick line.
- Series 2: confidence band (q10–q90), translucent fill.
- Series 3: live observation today, dotted line.
- Series 4: historical same-day-of-week median, dashed.
- Brush/scrubbing: dataZoom component (`dataZoom: [{type:'inside'}, {type:'slider'}]`).
- Annotations: vertical markers for live events ("Concert at 19:00 — expected +20 min").

### 8.4 Live video overlay

- For the operator, render the **face-blurred** stream via **HLS** (3–5 s latency, easy) or **KVS-WebRTC** for sub-second.
- Bounding boxes drawn client-side from the metadata WebSocket — never burned into the video, so the operator can toggle privacy view.
- Use `<video>` + `hls.js` or AWS amplify-video; for WebRTC, the AWS KVS WebRTC web SDK.

-----

## 9. MLOps & Monitoring

### 9.1 Model versioning and A/B testing

- **MLflow** or **Weights & Biases** for experiment tracking.
- Models packaged as Triton ensembles or Truss (Baseten) artifacts; checked into a model registry (MLflow / S3 / GCS).
- Deploy with **shadow mode** first: new YOLO26 candidate runs alongside production for 7 days; compare detection counts and zone events. Promote only when delta < 2 % and recall on a held-out queue test set is non-decreasing.
- For forecasting, run **online A/B** at the location level (each restaurant randomly assigned to model v1 or v2) and compare 1-week forecast MAE.

### 9.2 Drift detection

- **Detection drift signals**: rolling 7-day average detections per camera, per hour-of-day. Alert when |z| > 3 vs. 28-day baseline. Common cause: lens dirt, lighting change, camera moved.
- **Feature drift**: PSI on POS feature distribution.
- **Forecast drift**: rolling MAPE; auto-trigger retrain when MAPE drifts > 25 % above baseline for 3 consecutive days.
- Tools: **EvidentlyAI**, **WhyLabs**, or homegrown Prometheus rules.

### 9.3 Observability

- **Prometheus + Grafana** for infra (or Datadog / New Relic if you want a managed pane of glass).
- **Loki / OpenSearch** for logs.
- **Tempo / Jaeger** for traces.
- Dashboard SLOs to enforce: 95th-percentile detection latency <100 ms; webhook delivery success >99.9 %; forecast service p99 <500 ms.

### 9.4 Continuous retraining pipeline

- **Kubeflow Pipelines** / **SageMaker Pipelines** / **Vertex AI Pipelines** orchestrate nightly:
  1. Pull last 24 h face-blurred keyframes flagged by anomaly detector.
  2. Auto-label with the current best model, send a sample to a human-in-the-loop tool (**Roboflow**, **Label Studio**) for verification.
  3. Fine-tune a candidate; evaluate on golden set.
  4. Promote if metrics pass.
- Weekly forecasting retrain by tenant; daily for top-tier customers.

-----

## 10. Recommended Project Structure & Roadmap

### 10.1 Microservices breakdown

```
services/
├── ingest-gateway       # RTSP→KVS / WebRTC; Fargate
├── edge-agent           # Docker for Jetson/N100; face-blur, sampling
├── inference-detector   # YOLO26 on Triton; gRPC
├── inference-tracker    # ByteTrack/BoT-SORT; stateful per-camera
├── zone-engine          # polygon hits → events to Kafka
├── vlm-reasoner         # Qwen2.5-VL or vendor APIs; async
├── wait-time-engine     # Flink jobs computing live wait-times
├── forecast-service     # Prophet/Chronos/TFT; nightly + on-demand
├── pos-connector-hub    # Toast/Square/Lightspeed/Clover/Micros
├── booking-connector    # OpenTable/Resy/SevenRooms/TheFork
├── api-bff              # FastAPI; REST + WebSocket
├── auth-service         # Auth0/Keycloak/Cognito wrapper
├── billing-service      # Stripe; metered usage
├── admin-portal         # internal ops
└── dashboard-web        # React app
```

### 10.2 MVP feature set (90-day)

- 1 POS integration (Toast, the largest US share).
- 1 booking integration (SevenRooms, the most open API).
- Detection + tracking + 3 zones, hard-coded per camera in YAML.
- Live wait-time number on a hosted dashboard + an embeddable widget.
- Prophet baseline forecast; "Best Time today" graph.
- Edge agent with face-blur, RTSP-to-KVS, OTA via balena/AWS IoT Greengrass.
- Stripe metered billing.
- DPIA + privacy notice templates.

### 10.3 V1 scope (months 4–9)

- YOLO26 fine-tuned on top-down customer footage; ByteTrack + BoT-SORT toggle.
- Multi-camera fusion within a location.
- VLM reasoning (Qwen2.5-VL self-hosted).
- TFT forecasting with weather + events.
- All major POS connectors; OpenTable + TheFork.
- SOC 2 Type 1; ISO 27001 prep.

### 10.4 V2 (year 2)

- Crowd-density mode (P2PNet).
- Multi-language UI (FR, DE, ES, IT, JA).
- Self-serve onboarding (camera registration via QR + browser ONVIF discovery).
- White-label embeddable booking + wait-list.
- Notify-me push + SMS + WhatsApp Business integration.
- Enterprise SSO (Okta, Azure AD).
- Pricing intelligence: feed wait predictions into POS-side dynamic offers.

### 10.5 Team composition (lean commercial launch)

|Role                      |FTEs at MVP |At V1        |
|--------------------------|------------|-------------|
|Tech lead / staff eng     |1           |1            |
|ML / CV engineer          |1           |2            |
|Data / MLOps              |0.5         |1            |
|Backend (Python/Go)       |1           |2            |
|Frontend (React)          |1           |1            |
|Devops / SRE              |0.5         |1            |
|Designer (PT)             |0.5         |0.5          |
|Privacy/legal counsel     |external    |0.25         |
|Product / customer success|1           |2            |
|**Total**                 |**~6.5 FTE**|**~10.5 FTE**|

Realistic timeline: 4–5 months to a paying pilot customer; 9–12 months to a credible V1 with 20–50 sites.

### 10.6 Open-source projects to fork or learn from

- **`roboflow/supervision`** — production utilities for zones, ByteTrack glue, annotators. The official tutorial *"Monitor and Analyze Retail Queues Using Computer Vision"* on the Roboflow blog is the closest existing analogue; the `count_people_in_zone` and `occupancy_analytics` notebooks are excellent starting templates.
- **`ultralytics/ultralytics`** — YOLO26 training/export.
- **`mikel-brostrom/boxmot`** (formerly yolo_tracking) — every modern tracker behind one CLI; ideal for benchmarking ByteTrack vs BoT-SORT vs StrongSORT vs DeepOCSORT.
- **`facebookresearch/sam2`** — for one-shot zone definition (segment a queue area by clicking once).
- **`vllm-project/vllm`** — VLM serving (Qwen2.5-VL recipes published).
- **`Ultralytics solutions`** (within ultralytics) — heatmap, region counter, distance estimation, all of which can be customer-visible features.
- **`facebook/prophet`**, **`unit8co/darts`** (Prophet, NeuralProphet, TFT, N-BEATS in one API), **`amazon-science/chronos-forecasting`**.
- **`aws-samples/amazon-kinesis-video-streams-fargate-rtsp`** — exact reference for the cloud RTSP gateway.
- **`roboflow/supervision` Discussions #580 / #184 / #564** — community-grade examples of multi-camera RTSP zone counting (worth reading for the pitfalls — single-process Python + multiple RTSPs deadlocks; use the threaded-`VideoCapture` pattern shown there or move to GStreamer).

-----

## Pitfalls and lessons learned (from production deployments of similar systems)

1. **Lighting is the #1 cause of detection drift.** Re-evaluate weekly; have an automated "is this camera blind?" check (mean luminance, edge-density) that pages on-call before customers complain.
2. **Tracker hyperparameters matter more than detector accuracy** for queue dwell times. Tune `lost_track_buffer` (ByteTrack) per site — fast-casual ≈ 30 frames, fine-dining ≈ 120 frames.
3. **"Average wait" is the wrong KPI.** Customers care about p90 ("most people wait less than X"). Compute and display percentiles, not means.
4. **Calendar effects dominate weather.** Local school holidays move wait-times more than a 5°C temperature swing — get the school-calendar regressor right early.
5. **POS clocks drift.** Always use server-arrival timestamps for joins; never trust the POS event's `occurred_at` field for matching with vision events tighter than ±60 s.
6. **Don't show a single "wait time" number with high precision.** "12 min" looks scientific but invites complaints when a customer waits 18. Show "10–15 min" bands; users tolerate the wider range and trust grows.
7. **Edge agents WILL go offline.** Build the cloud to gracefully degrade to "live wait estimated from POS only" when video is unavailable — and make this a deliberate fallback, not a panic mode.
8. **GDPR DPIA must be done before pilot, not before launch.** Customers' DPOs ask for it on day 1 of evaluation. Have it ready.
9. **Avoid "demographic analytics" feature creep.** The instant a sales rep suggests "let's also estimate age and gender for marketing," push back hard — that one feature will move you from Art. 6 to Art. 9 territory and torpedo EU sales.
10. **Cold-start the forecast.** The first 14 days at a new restaurant will look bad if you only train per-restaurant. Use a global Chronos-2 or TFT model with a static restaurant embedding so a new tenant gets reasonable forecasts on day 1.

-----

### A note on sources and confidence

- YOLO26 details and benchmarks come from the official Ultralytics releases (Sept 2025 / Jan 2026), the arXiv preprints `2509.25164` and `2601.12882`, and Roboflow's January-2026 launch coverage. These are recent and consistent across sources.
- AWS Timestream LiveAnalytics being closed to new customers (June 2025) and the pivot to Timestream for InfluxDB is documented by AWS and reflected in 2025–2026 third-party comparisons.
- EDPB Guidelines 3/2019 on video devices (final Jan 2020, with 2026 EDPB summary still in circulation) remain the authoritative interpretive document for EU video analytics; national DPAs (CNIL, ICO, Garante) reaffirm it in 2024–2026 enforcement actions.
- VLM benchmarks (Qwen2.5-VL, GPT-4o, Gemini 2.5) shift weekly; the architectural choice (self-host for EU, vendor API for US premium) is more durable than any specific model name.
- Pricing figures for Modal/RunPod/Replicate/Baseten reflect public pricing pages as of late 2025/early 2026 and should be re-checked at procurement time.

This blueprint is intended to be read end-to-end before any code is written; the most consequential choices are made in **Sections 1.4 (edge-cloud hybrid), 5 (privacy posture), and 7.4 (RLS multi-tenancy)** — get those right and the rest is execution.
