"""
ml/feature_engineering.py
─────────────────────────
Transforms raw sensor readings into the 18-feature ML input.
Handles temporal features (rolling windows, rates of change).

Usage:
    python ml/feature_engineering.py \
        --input  ml/data/synthetic_labeled.parquet \
        --output ml/data/features.parquet
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from ml.data_schema import (
    FEATURE_COLS, MOISTURE_COLS,
    LABEL_COL, LABEL_INT_COL, FS_COL,
)


# ── Feature engineering ───────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input : raw sensor DataFrame (from simulate_data or gateway ingest)
    Output: DataFrame with all 18 engineered features added

    Must be called per node group (sorted by timestamp).
    """
    df = df.copy().sort_values(["node_id", "timestamp"]).reset_index(drop=True)

    grp = df.groupby("node_id", group_keys=False)

    # ── Tilt-derived features ─────────────────────────────────────────────
    # Rate of change: degrees per minute
    df["tilt_rate"] = grp["tilt_deg"].transform(
        lambda x: x.diff() / 5.0   # 5-minute interval
    )

    # Rolling mean over 3 hours (36 readings)
    df["tilt_rolling_mean_3hr"] = grp["tilt_deg"].transform(
        lambda x: x.rolling(36, min_periods=6).mean()
    )

    # Rolling variance over 1 hour (12 readings) — instability signal
    df["tilt_variance_1hr"] = grp["tilt_deg"].transform(
        lambda x: x.rolling(12, min_periods=4).var()
    )

    # ── Moisture-derived features ─────────────────────────────────────────
    # Saturation ratio: mean across all 6 depths, normalised to 0–1
    df["saturation_ratio"] = df[MOISTURE_COLS].mean(axis=1) / 100.0

    # Moisture gradient: top layer vs deepest
    df["moisture_gradient"] = df["moisture_d1"] - df["moisture_d6"]

    # ── Pore pressure features ────────────────────────────────────────────
    # Spike: change over last 30 min (6 readings)
    df["pore_spike"] = grp["pore_pressure"].transform(
        lambda x: x.diff(6)
    )

    # ── Rainfall cumulative windows ───────────────────────────────────────
    df["rainfall_6hr"] = grp["rainfall_1hr"].transform(
        lambda x: x.rolling(6, min_periods=1).sum()
    )
    df["rainfall_24hr"] = grp["rainfall_1hr"].transform(
        lambda x: x.rolling(24, min_periods=1).sum()
    )

    # ── Interaction features ──────────────────────────────────────────────
    df["rain_tilt_product"] = df["rainfall_1hr"] * df["tilt_deg"]

    # ── Drop rows where rolling windows haven't filled yet ────────────────
    df = df.dropna(subset=["tilt_rate", "tilt_rolling_mean_3hr", "pore_spike"])
    df = df.reset_index(drop=True)

    logger.info(f"Features engineered: {len(df):,} rows, {len(FEATURE_COLS)} features")
    return df


# ── Sequence builder for CNN ───────────────────────────────────────────────

def build_sequences(
    df: pd.DataFrame,
    window: int = 36,
    horizon: int = 6,
    channels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X_seq, y_seq) for CNN training.

    X_seq shape: (n_samples, n_channels, window)
    y_seq shape: (n_samples,)  — int label (0/1/2)

    The label for each window is the WORST class in the next `horizon` steps
    (safety-first: if RED appears in the next 30 min, label this window RED).
    """
    if channels is None:
        from ml.data_schema import SEQUENCE_CHANNELS
        channels = SEQUENCE_CHANNELS

    X_parts, y_parts = [], []

    for node_id, node_df in df.groupby("node_id"):
        node_df = node_df.reset_index(drop=True)
        n = len(node_df)
        n_windows = n - horizon - window
        if n_windows <= 0:
            continue

        # Sliding windows built with numpy strides instead of a per-row
        # pandas .iloc loop — the loop version peaks at multiple GB of RAM
        # for even a single node's worth of readings (pandas caches each
        # sliced sub-frame), which reliably OOM-kills CNN training.
        values = node_df[channels].to_numpy(dtype=np.float32)          # (n, n_channels)
        labels = node_df[LABEL_INT_COL].to_numpy(dtype=np.int64)       # (n,)

        windows = np.lib.stride_tricks.sliding_window_view(
            values, window, axis=0
        )                                                              # (n-window+1, n_channels, window)
        x_windows = windows[:n_windows]

        # Worst label in next `horizon` steps for each window
        label_windows = np.lib.stride_tricks.sliding_window_view(labels, horizon)  # (n-horizon+1, horizon)
        y_future = label_windows[window : window + n_windows].max(axis=1)

        X_parts.append(x_windows)
        y_parts.append(y_future)

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.int64)
    logger.info(f"Sequences built: X={X.shape}, y={y.shape}")
    return X, y


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engineer features from raw sensor data")
    parser.add_argument("--input",  type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ml/data/features.parquet"))
    args = parser.parse_args()

    logger.info(f"Loading {args.input}")
    raw = pd.read_parquet(args.input)

    featured = engineer_features(raw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(args.output, index=False)
    logger.info(f"Saved → {args.output}")
