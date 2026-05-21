"""
ml/simulate_data.py
───────────────────
Generates synthetic sensor readings for development and testing.
Run this FIRST — before you have real hardware — to get a working dataset.

Usage:
    python ml/simulate_data.py
    python ml/simulate_data.py --nodes 5 --days 30 --output ml/data/synthetic.parquet
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

from ml.data_schema import RAW_COLS, MOISTURE_COLS


# ── Scenario generators ───────────────────────────────────────────────────

def _green_readings(n: int, rng: np.random.Generator) -> dict:
    """Stable dry-season day — gentle baseline values."""
    return {
        "tilt_deg":    rng.normal(0.08, 0.015, n).clip(0.01, 0.40),
        "moisture_d1": rng.normal(42,   4,     n).clip(20, 70),
        "moisture_d2": rng.normal(44,   4,     n).clip(20, 72),
        "moisture_d3": rng.normal(46,   4,     n).clip(22, 74),
        "moisture_d4": rng.normal(47,   4,     n).clip(22, 75),
        "moisture_d5": rng.normal(48,   4,     n).clip(22, 76),
        "moisture_d6": rng.normal(50,   4,     n).clip(24, 78),
        "pore_pressure": rng.normal(0.7, 0.1,  n).clip(0.1, 2.0),
        "rainfall_1hr":  rng.exponential(1.5,  n).clip(0, 8),
    }


def _yellow_readings(n: int, rng: np.random.Generator) -> dict:
    """Monsoon build-up — elevated but not yet critical."""
    return {
        "tilt_deg":    rng.normal(0.22, 0.04, n).clip(0.10, 0.55),
        "moisture_d1": rng.normal(68,   6,    n).clip(50, 88),
        "moisture_d2": rng.normal(70,   6,    n).clip(52, 90),
        "moisture_d3": rng.normal(71,   5,    n).clip(54, 90),
        "moisture_d4": rng.normal(72,   5,    n).clip(55, 92),
        "moisture_d5": rng.normal(73,   5,    n).clip(56, 92),
        "moisture_d6": rng.normal(74,   5,    n).clip(57, 94),
        "pore_pressure": rng.normal(2.1, 0.4,  n).clip(0.8, 4.0),
        "rainfall_1hr":  rng.exponential(8,    n).clip(0, 25),
    }


def _red_readings(n: int, rng: np.random.Generator) -> dict:
    """Pre-failure signature — tilt accelerating, soil saturated."""
    # Tilt shows acceleration: starts moderate, ramps steeply
    t = np.linspace(0, 1, n)
    tilt_base = 0.35 + 0.65 * (t ** 2.2)   # convex ramp
    return {
        "tilt_deg":    (tilt_base + rng.normal(0, 0.02, n)).clip(0.25, 1.20),
        "moisture_d1": rng.normal(87,   4,    n).clip(75, 100),
        "moisture_d2": rng.normal(88,   4,    n).clip(76, 100),
        "moisture_d3": rng.normal(89,   4,    n).clip(77, 100),
        "moisture_d4": rng.normal(90,   3,    n).clip(78, 100),
        "moisture_d5": rng.normal(91,   3,    n).clip(79, 100),
        "moisture_d6": rng.normal(92,   3,    n).clip(80, 100),
        "pore_pressure": rng.normal(3.8, 0.5,  n).clip(2.0, 6.5),
        "rainfall_1hr":  rng.exponential(14,   n).clip(4, 35),
    }


# ── Main generator ────────────────────────────────────────────────────────

def generate(
    n_nodes: int = 10,
    n_days:  int = 60,
    seed:    int = 42,
    output:  Path = Path("ml/data/synthetic.parquet"),
) -> pd.DataFrame:
    """
    Generate synthetic sensor readings for n_nodes over n_days.
    Class distribution mirrors real monsoon season:
        GREEN  ≈ 80%,  YELLOW ≈ 14%,  RED ≈ 6%
    """
    rng = np.random.default_rng(seed)
    interval_sec = 300                      # 5-minute readings
    readings_per_day = 86400 // interval_sec
    total_readings = n_days * readings_per_day

    records = []

    for node_idx in range(n_nodes):
        node_id = f"NM-{node_idx+1:02d}"
        t0 = datetime(2024, 6, 1)           # monsoon start

        # Assign class probabilities for this node's timeline
        labels_raw = rng.choice(
            ["GREEN", "YELLOW", "RED"],
            size=total_readings,
            p=[0.80, 0.14, 0.06],
        )

        timestamps = [
            t0 + timedelta(seconds=i * interval_sec)
            for i in range(total_readings)
        ]

        # Generate readings per label group
        rows = {"timestamp": timestamps, "node_id": [node_id] * total_readings}
        rows["label"] = list(labels_raw)

        # Build sensor columns block by block
        sensor_cols = ["tilt_deg"] + MOISTURE_COLS + ["pore_pressure", "rainfall_1hr"]
        for col in sensor_cols:
            rows[col] = np.zeros(total_readings)

        for label in ["GREEN", "YELLOW", "RED"]:
            mask = np.array(labels_raw) == label
            n = mask.sum()
            if n == 0:
                continue
            gen_fn = {"GREEN": _green_readings, "YELLOW": _yellow_readings, "RED": _red_readings}[label]
            reading_dict = gen_fn(n, rng)
            for col, vals in reading_dict.items():
                rows[col][mask] = vals

        rows["battery_pct"] = rng.integers(55, 100, size=total_readings)
        rows["rssi_dbm"]    = rng.integers(-115, -75, size=total_readings)

        records.append(pd.DataFrame(rows))

    df = pd.concat(records, ignore_index=True)
    df = df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)

    logger.info(f"Generated {len(df):,} readings | {n_nodes} nodes | {n_days} days")
    logger.info(f"Class distribution:\n{df['label'].value_counts(normalize=True).round(3)}")
    logger.info(f"Saved → {output}")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic sensor data")
    parser.add_argument("--nodes",  type=int,  default=10)
    parser.add_argument("--days",   type=int,  default=60)
    parser.add_argument("--seed",   type=int,  default=42)
    parser.add_argument("--output", type=Path, default=Path("ml/data/synthetic.parquet"))
    args = parser.parse_args()

    generate(n_nodes=args.nodes, n_days=args.days, seed=args.seed, output=args.output)
