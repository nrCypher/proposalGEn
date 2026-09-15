-- Restaurant Wait-Time Estimator — initial schema
-- Postgres 16 + TimescaleDB. Multi-tenancy via Row-Level Security (blueprint §7.4).
-- Privacy: zone_events store ONLY ts / anonymous tracker_id / zone / dwell (§5.3).

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------------
-- Roles: runtime role can NOT bypass RLS; migrator can (migrations only).
-- ---------------------------------------------------------------------------
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'wte_app') THEN
    CREATE ROLE wte_app LOGIN PASSWORD 'wte_app' NOBYPASSRLS;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'wte_migrator') THEN
    CREATE ROLE wte_migrator LOGIN PASSWORD 'wte_migrator' BYPASSRLS;
  END IF;
END $$;

-- ---------------------------------------------------------------------------
-- Control-plane tables (Postgres)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tenants (
  tenant_id   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name        text NOT NULL,
  region      text NOT NULL DEFAULT 'eu-central-1',   -- data residency (§5.6)
  vlm_tier    text NOT NULL DEFAULT 'self_hosted',     -- self_hosted | vendor_api
  tracker     text NOT NULL DEFAULT 'bytetrack',       -- bytetrack | botsort (premium)
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS locations (
  tenant_id    uuid NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
  location_id  text NOT NULL,
  name         text NOT NULL,
  timezone     text NOT NULL DEFAULT 'Europe/Lisbon',
  country      text NOT NULL DEFAULT 'PT',
  lat          double precision,
  lon          double precision,
  PRIMARY KEY (tenant_id, location_id)
);

CREATE TABLE IF NOT EXISTS cameras (
  tenant_id    uuid NOT NULL,
  location_id  text NOT NULL,
  camera_id    text NOT NULL,
  mount        text NOT NULL DEFAULT 'top_down',
  zones        jsonb NOT NULL DEFAULT '{}'::jsonb,   -- {"queue": [[x,y],...], ...}
  PRIMARY KEY (tenant_id, location_id, camera_id),
  FOREIGN KEY (tenant_id, location_id) REFERENCES locations(tenant_id, location_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pos_connections (
  tenant_id     uuid NOT NULL,
  location_id   text NOT NULL,
  provider      text NOT NULL,                         -- toast | square | ...
  credentials   bytea NOT NULL,                        -- KMS-encrypted OAuth refresh token (§6.4)
  PRIMARY KEY (tenant_id, location_id, provider)
);

-- ---------------------------------------------------------------------------
-- Time-series tables (TimescaleDB hypertables)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS zone_events (
  ts            timestamptz NOT NULL,
  tenant_id     uuid NOT NULL,
  location_id   text NOT NULL,
  camera_id     text NOT NULL,
  zone          text NOT NULL,                         -- queue | service | fulfil
  event_type    text NOT NULL,                         -- enter | exit
  tracker_id    int  NOT NULL,                         -- anonymous, rotates daily
  dwell_seconds real
);
SELECT create_hypertable('zone_events', 'ts', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_zone_events_tenant_ts ON zone_events (tenant_id, location_id, ts DESC);

CREATE TABLE IF NOT EXISTS pos_events (
  received_at   timestamptz NOT NULL,                  -- server arrival, used for joins (pitfall #5)
  occurred_at   timestamptz NOT NULL,
  tenant_id     uuid NOT NULL,
  location_id   text NOT NULL,
  provider      text NOT NULL,
  event_id      text NOT NULL,
  event_type    text NOT NULL,
  order_id      text NOT NULL,
  party_size    int,
  channel       text NOT NULL,
  raw           jsonb NOT NULL,
  UNIQUE (provider, event_id, received_at)             -- idempotency (§6.3)
);
SELECT create_hypertable('pos_events', 'received_at', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_pos_events_tenant_ts ON pos_events (tenant_id, location_id, received_at DESC);

CREATE TABLE IF NOT EXISTS wait_time_buckets (
  bucket        timestamptz NOT NULL,                  -- 15-min
  tenant_id     uuid NOT NULL,
  location_id   text NOT NULL,
  p50_minutes   real,
  p90_minutes   real,
  queue_len_avg real,
  samples       int NOT NULL DEFAULT 0,
  source        text NOT NULL DEFAULT 'vision',
  PRIMARY KEY (tenant_id, location_id, bucket)
);
SELECT create_hypertable('wait_time_buckets', 'bucket', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS forecasts (
  generated_at  timestamptz NOT NULL,
  tenant_id     uuid NOT NULL,
  location_id   text NOT NULL,
  ts            timestamptz NOT NULL,
  model         text NOT NULL,
  tier          text NOT NULL,
  q10           real NOT NULL,
  q50           real NOT NULL,
  q90           real NOT NULL
);
SELECT create_hypertable('forecasts', 'generated_at', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_forecasts_lookup ON forecasts (tenant_id, location_id, generated_at DESC, ts);

CREATE TABLE IF NOT EXISTS audit_log (
  ts         timestamptz NOT NULL DEFAULT now(),
  tenant_id  uuid,
  actor      text NOT NULL,
  action     text NOT NULL,
  detail     jsonb
);
SELECT create_hypertable('audit_log', 'ts', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- Continuous aggregate: 15-min wait-time buckets straight from zone_events
-- ---------------------------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS wait_time_15m
WITH (timescaledb.continuous) AS
SELECT time_bucket('15 minutes', ts)             AS bucket,
       tenant_id, location_id,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY dwell_seconds) / 60.0 AS p50_minutes,
       percentile_cont(0.9) WITHIN GROUP (ORDER BY dwell_seconds) / 60.0 AS p90_minutes,
       count(*)                                  AS samples
FROM zone_events
WHERE zone = 'queue' AND event_type = 'exit' AND dwell_seconds >= 3
GROUP BY bucket, tenant_id, location_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('wait_time_15m',
  start_offset => INTERVAL '1 day', end_offset => INTERVAL '15 minutes',
  schedule_interval => INTERVAL '15 minutes', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- Retention (blueprint §5.5)
-- ---------------------------------------------------------------------------
SELECT add_retention_policy('zone_events',  INTERVAL '90 days',  if_not_exists => TRUE);
SELECT add_retention_policy('pos_events',   INTERVAL '90 days',  if_not_exists => TRUE);
SELECT add_retention_policy('forecasts',    INTERVAL '90 days',  if_not_exists => TRUE);
SELECT add_retention_policy('audit_log',    INTERVAL '365 days', if_not_exists => TRUE);
-- wait_time_buckets / wait_time_15m: 24 months (forecast seasonality)
SELECT add_retention_policy('wait_time_buckets', INTERVAL '24 months', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- Row-Level Security — the multi-tenancy spine (§7.4)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION current_tenant() RETURNS uuid
LANGUAGE sql STABLE AS $$
  SELECT NULLIF(current_setting('app.current_tenant_id', true), '')::uuid
$$;

DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['tenants','locations','cameras','pos_connections','zone_events',
                           'pos_events','wait_time_buckets','forecasts','audit_log'] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format('DROP POLICY IF EXISTS tenant_isolation ON %I', t);
    EXECUTE format(
      'CREATE POLICY tenant_isolation ON %I USING (tenant_id = current_tenant()) '
      'WITH CHECK (tenant_id = current_tenant())', t);
    EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON %I TO wte_app', t);
  END LOOP;
END $$;

GRANT SELECT ON wait_time_15m TO wte_app;
