"""
ml/data_schema.py
─────────────────
Single source of truth for every column name used across the pipeline.
Import these lists instead of hard-coding column names.

Usage:
    from ml.data_schema import RAW_COLS, FEATURE_COLS, LABEL_COL
"""

# ── Raw sensor columns (as they arrive from gateway) ─────────────────────
RAW_COLS = [
    "node_id",
    "timestamp",
    "tilt_deg",
    "moisture_d1",   # % saturation, depth  10 cm
    "moisture_d2",   # % saturation, depth  20 cm
    "moisture_d3",   # % saturation, depth  30 cm
    "moisture_d4",   # % saturation, depth  40 cm
    "moisture_d5",   # % saturation, depth  60 cm
    "moisture_d6",   # % saturation, depth  80 cm
    "pore_pressure", # kPa
    "rainfall_1hr",  # mm/hr
    "battery_pct",
    "rssi_dbm",
]

MOISTURE_COLS = [f"moisture_d{i}" for i in range(1, 7)]

# ── Engineered feature columns (output of feature_engineering.py) ─────────
FEATURE_COLS = [
    # Raw tilt
    "tilt_deg",
    # Derived tilt
    "tilt_rate",             # deg/min — (T_t - T_{t-1}) / interval
    "tilt_rolling_mean_3hr", # deg    — smoothed baseline
    "tilt_variance_1hr",     # deg²   — instability signal
    # Moisture
    "saturation_ratio",      # 0–1    — mean(M1..M6) / 100
    "moisture_gradient",     # %      — M1 - M6 (top vs bottom)
    *MOISTURE_COLS,          # raw depth readings kept as features
    # Pressure
    "pore_pressure",
    "pore_spike",            # kPa change over last 30 min
    # Rainfall
    "rainfall_1hr",
    "rainfall_6hr",          # rolling 6-hr cumulative
    "rainfall_24hr",         # rolling 24-hr cumulative
    # Interaction
    "rain_tilt_product",     # rainfall_1hr × tilt_deg
]

# ── Sequence channels for CNN ─────────────────────────────────────────────
SEQUENCE_CHANNELS = [
    "tilt_deg",
    "saturation_ratio",
    "pore_pressure",
    "rainfall_1hr",
    "tilt_rate",
    "moisture_gradient",
]

# ── Labels ────────────────────────────────────────────────────────────────
LABEL_COL      = "label"        # string: GREEN / YELLOW / RED
LABEL_INT_COL  = "label_int"    # int:    0 / 1 / 2
FS_COL         = "factor_of_safety"

LABEL_MAP     = {"GREEN": 0, "YELLOW": 1, "RED": 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

# ── Window parameters ─────────────────────────────────────────────────────
WINDOW_STEPS  = 36   # 36 readings × 5 min = 3 hours of history
HORIZON_STEPS = 6    # predict worst class in next 30 min
