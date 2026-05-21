"""
ml/physics_labels.py
────────────────────
Cold-start Strategy 1: generate labels from physics (Factor of Safety).
Works from day 0 — no real landslide events needed.

Usage:
    python ml/physics_labels.py --input ml/data/synthetic.parquet
    python ml/physics_labels.py --input ml/data/raw.parquet \
        --cohesion 9.0 --friction 30.0
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from ml.data_schema import LABEL_COL, LABEL_INT_COL, LABEL_MAP, FS_COL
from config.settings import settings


# ── Core physics ──────────────────────────────────────────────────────────

def factor_of_safety(
    pore_pressure: np.ndarray,
    slope_angle_deg: float,
    c_prime: float   = settings.SOIL_COHESION_KPA,
    gamma: float     = settings.SOIL_UNIT_WEIGHT,
    z: float         = settings.SOIL_FAILURE_DEPTH_M,
    phi_deg: float   = settings.SOIL_FRICTION_DEG,
) -> np.ndarray:
    """
    Infinite slope model (most widely used for shallow translational slides).

    Parameters
    ----------
    pore_pressure    : kPa — measured by piezometer (your sensor)
    slope_angle_deg  : degrees — from DEM or tilt sensor baseline
    c_prime          : kPa — soil cohesion (from geotechnical survey)
    gamma            : kN/m³ — unit weight of soil
    z                : m — depth of failure plane
    phi_deg          : degrees — internal friction angle

    Returns
    -------
    FS : array — Factor of Safety per reading
         FS < 1.0  → failure (RED)
         FS < 1.3  → marginal (YELLOW)
         FS ≥ 1.3  → stable  (GREEN)
    """
    beta = np.radians(slope_angle_deg)
    phi  = np.radians(phi_deg)

    numerator   = c_prime + (gamma * z * np.cos(beta)**2 - pore_pressure) * np.tan(phi)
    denominator = gamma * z * np.sin(beta) * np.cos(beta)

    # Avoid division by zero on flat slopes
    denominator = np.where(denominator < 1e-6, 1e-6, denominator)

    fs = numerator / denominator
    return fs.clip(0.0, 5.0)   # cap at 5 for readability


def fs_to_label(fs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert FS array → string labels and int labels."""
    labels_str = np.where(fs < 1.0, "RED", np.where(fs < 1.3, "YELLOW", "GREEN"))
    labels_int = np.array([LABEL_MAP[l] for l in labels_str])
    return labels_str, labels_int


# ── Pipeline ──────────────────────────────────────────────────────────────

def label_dataset(
    input_path:       Path,
    output_path:      Path | None = None,
    slope_angle_deg:  float = 25.0,   # default NE India highway slope
    c_prime:          float = settings.SOIL_COHESION_KPA,
    gamma:            float = settings.SOIL_UNIT_WEIGHT,
    z:                float = settings.SOIL_FAILURE_DEPTH_M,
    phi_deg:          float = settings.SOIL_FRICTION_DEG,
) -> pd.DataFrame:

    logger.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)

    logger.info("Computing Factor of Safety for all readings...")
    fs = factor_of_safety(
        pore_pressure   = df["pore_pressure"].values,
        slope_angle_deg = slope_angle_deg,
        c_prime         = c_prime,
        gamma           = gamma,
        z               = z,
        phi_deg         = phi_deg,
    )

    df[FS_COL]       = fs
    labels_str, labels_int = fs_to_label(fs)
    df[LABEL_COL]    = labels_str
    df[LABEL_INT_COL] = labels_int

    dist = df[LABEL_COL].value_counts()
    logger.info(f"Label distribution:\n{dist}")
    logger.info(f"  RED   count: {(dist.get('RED', 0) / len(df) * 100):.1f}%")
    logger.info(f"  Mean FS: {fs.mean():.3f}  |  Min FS: {fs.min():.3f}")

    if output_path is None:
        output_path = input_path.parent / (input_path.stem + "_labeled.parquet")

    df.to_parquet(output_path, index=False)
    logger.info(f"Labeled data saved → {output_path}")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate physics-based labels (Factor of Safety)")
    parser.add_argument("--input",    type=Path,  required=True)
    parser.add_argument("--output",   type=Path,  default=None)
    parser.add_argument("--slope",    type=float, default=25.0,  help="Slope angle in degrees")
    parser.add_argument("--cohesion", type=float, default=settings.SOIL_COHESION_KPA)
    parser.add_argument("--friction", type=float, default=settings.SOIL_FRICTION_DEG)
    parser.add_argument("--depth",    type=float, default=settings.SOIL_FAILURE_DEPTH_M)
    parser.add_argument("--weight",   type=float, default=settings.SOIL_UNIT_WEIGHT)
    args = parser.parse_args()

    label_dataset(
        input_path      = args.input,
        output_path     = args.output,
        slope_angle_deg = args.slope,
        c_prime         = args.cohesion,
        phi_deg         = args.friction,
        z               = args.depth,
        gamma           = args.weight,
    )
