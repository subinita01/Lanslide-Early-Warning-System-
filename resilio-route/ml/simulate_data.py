"""
ml/simulate_data.py
───────────────────
Generates synthetic sensor readings for development and testing.
Run this FIRST — before you have real hardware — to get a working dataset.

Every node's readings are driven by a persistent hazard process that rises
during simulated monsoon storms and recedes afterward, rather than an
independent GREEN/YELLOW/RED draw at every 5-minute step. Persistence
matters: the CNN forecasts the worst class in the *next* 30 minutes from
the *past* 3-hour window, so if consecutive readings carry no correlation
there is no precursor signal for any model to learn from the sensor history.

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


# ── Hazard process ────────────────────────────────────────────────────────

def _simulate_hazard(
    n:                int,
    rng:              np.random.Generator,
    storm_start_prob: float = 1 / 1440,   # ~one storm onset every 5 days
    ramp_rate:        float = 0.03,       # how fast H approaches the storm's target
    decay_rate:       float = 0.01,       # how fast H recedes once the storm ends
    noise_std:        float = 0.01,
) -> np.ndarray:
    """
    A mean-reverting process, 0 (dry baseline) to 1 (failure), that ramps
    toward a randomly drawn target while a storm is active and decays back
    toward 0 once it ends. Sensor channels are smooth functions of H, so
    consecutive readings are autocorrelated — real precursor trends rather
    than i.i.d. noise.
    """
    H = np.empty(n, dtype=np.float64)
    h = 0.0
    in_storm = False
    storm_target = 0.0
    storm_steps_left = 0

    for t in range(n):
        if in_storm:
            h += (storm_target - h) * ramp_rate + rng.normal(0, noise_std)
            storm_steps_left -= 1
            if storm_steps_left <= 0:
                in_storm = False
        else:
            h += -decay_rate * h + rng.normal(0, noise_std)
            if rng.random() < storm_start_prob:
                in_storm = True
                # Most storms stay mild; a minority build into real hazard.
                storm_target = rng.beta(1.5, 3.0)
                storm_steps_left = int(rng.integers(48, 288))  # 4h – 24h buildup

        h = float(np.clip(h, -0.05, 1.05))
        H[t] = h

    return np.clip(H, 0.0, 1.0)


# ── Sensor synthesis from the hazard process ──────────────────────────────

def _readings_from_hazard(H: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(H)

    # Convex ramp — tilt accelerates as hazard approaches failure.
    tilt_deg = 0.08 + 1.12 * H ** 1.8 + rng.normal(0, 0.015, n)

    moisture_base  = [42, 44, 46, 47, 48, 50]
    moisture_range = [45, 44, 43, 43, 43, 42]
    moisture = {
        f"moisture_d{i+1}": moisture_base[i] + moisture_range[i] * H + rng.normal(0, 3, n)
        for i in range(6)
    }

    # Calibrated so pore_pressure crosses the Factor-of-Safety YELLOW/RED
    # thresholds (~11.75 / ~17.92 kPa under the default soil parameters)
    # right around H ≈ 0.19 / 0.37.
    pore_pressure = 5.0 + 35.0 * H + rng.normal(0, 0.8, n)
    rainfall_1hr  = rng.exponential(2.0 + 28.0 * H, n)

    readings = {
        "tilt_deg":      tilt_deg.clip(0.01, 1.20),
        "pore_pressure": pore_pressure.clip(0.1, 45.0),
        "rainfall_1hr":  rainfall_1hr.clip(0, 40),
    }
    for col, vals in moisture.items():
        readings[col] = vals.clip(15, 100)

    return readings


# ── Main generator ────────────────────────────────────────────────────────

def generate(
    n_nodes: int = 10,
    n_days:  int = 60,
    seed:    int = 42,
    output:  Path = Path("ml/data/synthetic.parquet"),
) -> pd.DataFrame:
    """
    Generate synthetic sensor readings for n_nodes over n_days.
    Each node follows its own persistent hazard process (storms build up
    and recede over hours), so class distribution varies by node rather
    than being a fixed per-reading split.
    """
    rng = np.random.default_rng(seed)
    interval_sec = 300                      # 5-minute readings
    readings_per_day = 86400 // interval_sec
    total_readings = n_days * readings_per_day

    records = []

    for node_idx in range(n_nodes):
        node_id = f"NM-{node_idx+1:02d}"
        t0 = datetime(2024, 6, 1)           # monsoon start

        timestamps = [
            t0 + timedelta(seconds=i * interval_sec)
            for i in range(total_readings)
        ]

        H = _simulate_hazard(total_readings, rng)
        readings = _readings_from_hazard(H, rng)

        # Scenario bucket, for reporting only — the real ground-truth label
        # is computed later by physics_labels.py from actual pore_pressure.
        label = np.where(H < 0.193, "GREEN", np.where(H < 0.369, "YELLOW", "RED"))

        rows = {
            "timestamp": timestamps,
            "node_id":   [node_id] * total_readings,
            "label":     label,
            **readings,
            "battery_pct": rng.integers(55, 100, size=total_readings),
            "rssi_dbm":    rng.integers(-115, -75, size=total_readings),
        }
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
