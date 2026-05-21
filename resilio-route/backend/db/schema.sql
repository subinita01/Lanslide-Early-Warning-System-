-- backend/db/schema.sql
-- ──────────────────────
-- TimescaleDB schema for Resilio-Route.
-- Run once to initialise the database.
--
-- Usage:
--   psql $DATABASE_URL -f backend/db/schema.sql

-- ── Extensions ────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Raw sensor readings ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sensor_readings (
    id              BIGSERIAL,
    node_id         VARCHAR(16)   NOT NULL,
    highway         VARCHAR(16)   NOT NULL DEFAULT 'UNKNOWN',
    km_marker       DECIMAL(6,2),
    timestamp       TIMESTAMPTZ   NOT NULL,

    -- Tilt
    tilt_deg        DECIMAL(7,4),
    tilt_rate       DECIMAL(8,5),

    -- Moisture (6 depths: 10/20/30/40/60/80 cm)
    moisture_d1     DECIMAL(5,2),
    moisture_d2     DECIMAL(5,2),
    moisture_d3     DECIMAL(5,2),
    moisture_d4     DECIMAL(5,2),
    moisture_d5     DECIMAL(5,2),
    moisture_d6     DECIMAL(5,2),

    -- Pressure & rainfall
    pore_pressure   DECIMAL(6,3),
    rainfall_1hr    DECIMAL(6,2),

    -- Node health
    battery_pct     SMALLINT,
    rssi_dbm        SMALLINT,
    sequence_num    SMALLINT,

    created_at      TIMESTAMPTZ   DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
);

-- Convert to hypertable (partitioned by time, 1-week chunks)
SELECT create_hypertable(
    'sensor_readings', 'timestamp',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_sensor_node_time
    ON sensor_readings (node_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_sensor_highway_time
    ON sensor_readings (highway, timestamp DESC);

-- ── Navigator predictions ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id              UUID          DEFAULT uuid_generate_v4(),
    node_id         VARCHAR(16)   NOT NULL,
    segment         VARCHAR(32),
    timestamp       TIMESTAMPTZ   NOT NULL,

    -- Classification
    label           VARCHAR(8)    NOT NULL CHECK (label IN ('GREEN','YELLOW','RED')),
    label_int       SMALLINT      NOT NULL CHECK (label_int IN (0,1,2)),
    p_green         DECIMAL(6,4),
    p_yellow        DECIMAL(6,4),
    p_red           DECIMAL(6,4),
    confidence      DECIMAL(6,4),

    -- Model details
    rf_p_green      DECIMAL(6,4),
    rf_p_yellow     DECIMAL(6,4),
    rf_p_red        DECIMAL(6,4),
    cnn_p_green     DECIMAL(6,4),
    cnn_p_yellow    DECIMAL(6,4),
    cnn_p_red       DECIMAL(6,4),
    model_version   VARCHAR(16),
    inference_ms    INTEGER,

    created_at      TIMESTAMPTZ   DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable(
    'predictions', 'timestamp',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_pred_node_time
    ON predictions (node_id, timestamp DESC);

-- ── Alert events ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alerts (
    id              UUID          DEFAULT uuid_generate_v4() PRIMARY KEY,
    node_id         VARCHAR(16)   NOT NULL,
    segment         VARCHAR(32),
    highway         VARCHAR(16),
    km_from         DECIMAL(6,2),
    km_to           DECIMAL(6,2),
    label           VARCHAR(8)    NOT NULL,
    p_red           DECIMAL(6,4),
    lead_time_min   INTEGER,
    issued_at       TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    cleared_at      TIMESTAMPTZ,
    acknowledged_by VARCHAR(64),
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_alerts_node
    ON alerts (node_id, issued_at DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_highway
    ON alerts (highway, issued_at DESC);

-- ── Labeled events (ground truth) ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS labeled_events (
    id              UUID          DEFAULT uuid_generate_v4() PRIMARY KEY,
    node_id         VARCHAR(16)   NOT NULL,
    event_time      TIMESTAMPTZ   NOT NULL,
    label           VARCHAR(8)    NOT NULL,
    label_int       SMALLINT      NOT NULL,
    label_source    VARCHAR(32)   NOT NULL,  -- 'physics','historical','expert','active_learning'
    factor_of_safety DECIMAL(6,4),
    notes           TEXT,
    labeled_by      VARCHAR(64),
    created_at      TIMESTAMPTZ   DEFAULT NOW()
);

-- ── Registered webhooks ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS webhooks (
    id              UUID          DEFAULT uuid_generate_v4() PRIMARY KEY,
    url             TEXT          NOT NULL UNIQUE,
    secret          VARCHAR(128),
    events          TEXT[]        NOT NULL,
    highways        TEXT[]        NOT NULL,
    active          BOOLEAN       DEFAULT TRUE,
    created_at      TIMESTAMPTZ   DEFAULT NOW(),
    last_triggered  TIMESTAMPTZ
);

-- ── Useful views ──────────────────────────────────────────────────────────

-- Latest prediction per node
CREATE OR REPLACE VIEW latest_predictions AS
SELECT DISTINCT ON (node_id)
    node_id, label, p_red, confidence, timestamp, model_version
FROM predictions
ORDER BY node_id, timestamp DESC;

-- Active RED alerts (not yet cleared)
CREATE OR REPLACE VIEW active_red_alerts AS
SELECT *
FROM alerts
WHERE label = 'RED'
  AND cleared_at IS NULL
ORDER BY issued_at DESC;

-- Node health summary
CREATE OR REPLACE VIEW node_health AS
SELECT
    node_id,
    MAX(timestamp)                          AS last_seen,
    NOW() - MAX(timestamp)                  AS time_since_last,
    AVG(battery_pct)                        AS avg_battery_pct,
    COUNT(*)                                AS total_readings_7d
FROM sensor_readings
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY node_id
ORDER BY last_seen DESC;

COMMENT ON TABLE sensor_readings  IS 'Raw 5-minute sensor readings from all field nodes';
COMMENT ON TABLE predictions      IS 'Navigator ML predictions per node per inference cycle';
COMMENT ON TABLE alerts           IS 'Alert events issued when prediction crosses RED/YELLOW threshold';
COMMENT ON TABLE labeled_events   IS 'Ground truth labels for model training (all sources)';
