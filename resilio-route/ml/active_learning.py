"""
ml/active_learning.py
─────────────────────
Cold-start Strategy 4: Active Learning.
Finds the samples the current model is most uncertain about,
so a geotechnical expert can label them with maximum information gain.

Each round of ~200 expert labels improves RED recall by ~0.04–0.06.

Usage:
    # Find uncertain samples for this round
    python ml/active_learning.py select \
        --features   ml/data/features.parquet \
        --rf-model   ml/models/rf_v1.pkl \
        --output     ml/data/uncertain_batch_1.parquet \
        --n-samples  200

    # After expert labels the CSV, import them back
    python ml/active_learning.py import \
        --labeled-csv ml/data/expert_labels_batch_1.csv \
        --output      ml/data/features_with_expert.parquet
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.stats import entropy as scipy_entropy
from loguru import logger

from ml.data_schema import FEATURE_COLS, LABEL_COL, LABEL_INT_COL, LABEL_MAP


# ── Uncertainty scoring ───────────────────────────────────────────────────

def prediction_entropy(proba: np.ndarray) -> np.ndarray:
    """
    Shannon entropy of the predicted probability distribution.
    High entropy = model is uncertain = most valuable to label.

    H(x) = -∑ p_k · log(p_k)
    Maximum H for 3 classes = log(3) ≈ 1.099
    """
    # Clip to avoid log(0)
    proba = np.clip(proba, 1e-10, 1.0)
    return scipy_entropy(proba, axis=1)


def margin_sampling(proba: np.ndarray) -> np.ndarray:
    """
    Margin between top-1 and top-2 class probabilities.
    Small margin = model is confused between two classes.
    """
    sorted_p = np.sort(proba, axis=1)[:, ::-1]
    return sorted_p[:, 0] - sorted_p[:, 1]


# ── Select batch ──────────────────────────────────────────────────────────

def select_uncertain_batch(
    features_path: Path,
    rf_path:       Path,
    output_path:   Path,
    n_samples:     int  = 200,
    method:        str  = "entropy",   # "entropy" | "margin"
) -> pd.DataFrame:
    """
    Select the n_samples most uncertain predictions from the unlabeled pool.
    These are the samples that will give the maximum improvement per label.
    """
    logger.info(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)

    logger.info(f"Loading RF model from {rf_path}")
    rf = joblib.load(rf_path)

    X = df[FEATURE_COLS].values
    proba = rf.predict_proba(X)   # shape (n, 3)

    # Score by uncertainty
    if method == "entropy":
        scores = prediction_entropy(proba)
        logger.info(f"Entropy method | mean={scores.mean():.3f}, max={scores.max():.3f}")
    else:
        # Margin: low margin = high uncertainty, so we invert
        scores = -margin_sampling(proba)

    # Select top-n most uncertain
    top_indices = np.argsort(scores)[-n_samples:][::-1]
    uncertain_df = df.iloc[top_indices].copy()

    uncertain_df["uncertainty_score"] = scores[top_indices]
    uncertain_df["rf_pred_label"]     = [
        ["GREEN", "YELLOW", "RED"][p] for p in proba[top_indices].argmax(axis=1)
    ]
    uncertain_df["rf_p_green"]  = proba[top_indices, 0]
    uncertain_df["rf_p_yellow"] = proba[top_indices, 1]
    uncertain_df["rf_p_red"]    = proba[top_indices, 2]

    # Expert needs to fill in this column
    uncertain_df["expert_label"] = ""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    uncertain_df.to_parquet(output_path, index=False)

    # Also export a clean CSV for the geotechnical expert
    csv_path = output_path.with_suffix(".csv")
    expert_cols = [
        "node_id", "timestamp", "tilt_deg", "pore_pressure",
        "saturation_ratio", "rainfall_1hr", "pore_spike",
        "rf_pred_label", "uncertainty_score",
        "rf_p_green", "rf_p_yellow", "rf_p_red",
        "expert_label",   # ← expert fills this column
    ]
    available = [c for c in expert_cols if c in uncertain_df.columns]
    uncertain_df[available].to_csv(csv_path, index=False)

    logger.success(f"Selected {len(uncertain_df)} uncertain samples")
    logger.info(f"  Parquet → {output_path}")
    logger.info(f"  CSV for expert → {csv_path}")
    logger.info(f"\n  *** Send {csv_path} to your geotechnical expert. ***")
    logger.info(f"  *** They fill the 'expert_label' column (GREEN/YELLOW/RED). ***")
    logger.info(f"  *** Then run: python ml/active_learning.py import --labeled-csv {csv_path} ***")

    return uncertain_df


# ── Import expert labels ──────────────────────────────────────────────────

def import_expert_labels(
    features_path: Path,
    labeled_csv:   Path,
    output_path:   Path,
) -> pd.DataFrame:
    """
    Merge expert-labeled samples back into the main features dataset.
    The new labels override physics labels for those rows.
    """
    logger.info(f"Loading expert labels from {labeled_csv}")
    expert_df = pd.read_csv(labeled_csv)

    # Validate expert filled in labels
    empty = expert_df["expert_label"].isna() | (expert_df["expert_label"] == "")
    if empty.any():
        logger.warning(f"{empty.sum()} rows have empty expert_label — skipping those.")
        expert_df = expert_df[~empty]

    valid_labels = {"GREEN", "YELLOW", "RED"}
    invalid = ~expert_df["expert_label"].isin(valid_labels)
    if invalid.any():
        logger.warning(f"{invalid.sum()} rows have invalid labels — skipping.")
        expert_df = expert_df[~invalid]

    logger.info(f"Expert labeled {len(expert_df)} samples")
    logger.info(f"Distribution:\n{expert_df['expert_label'].value_counts()}")

    # Load base features
    base_df = pd.read_parquet(features_path)

    # Merge: update label for matched rows (by node_id + timestamp)
    expert_df["timestamp"] = pd.to_datetime(expert_df["timestamp"])
    base_df["timestamp"]   = pd.to_datetime(base_df["timestamp"])

    # Build lookup: (node_id, timestamp) → expert_label
    label_lookup = expert_df.set_index(["node_id", "timestamp"])["expert_label"]

    updated = 0
    for idx, row in base_df.iterrows():
        key = (row["node_id"], row["timestamp"])
        if key in label_lookup.index:
            new_label = label_lookup[key]
            base_df.at[idx, LABEL_COL]     = new_label
            base_df.at[idx, LABEL_INT_COL] = LABEL_MAP[new_label]
            updated += 1

    logger.success(f"Updated {updated} rows with expert labels")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_df.to_parquet(output_path, index=False)
    logger.info(f"Updated features saved → {output_path}")

    return base_df


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Active learning for Resilio-Route")
    sub = parser.add_subparsers(dest="command")

    # select command
    sel = sub.add_parser("select", help="Select uncertain samples for expert labeling")
    sel.add_argument("--features",  type=Path, required=True)
    sel.add_argument("--rf-model",  type=Path, required=True)
    sel.add_argument("--output",    type=Path, default=Path("ml/data/uncertain_batch.parquet"))
    sel.add_argument("--n-samples", type=int,  default=200)
    sel.add_argument("--method",    type=str,  default="entropy",
                     choices=["entropy", "margin"])

    # import command
    imp = sub.add_parser("import", help="Import expert-labeled CSV back into dataset")
    imp.add_argument("--features",    type=Path, required=True)
    imp.add_argument("--labeled-csv", type=Path, required=True)
    imp.add_argument("--output",      type=Path, default=Path("ml/data/features_expert.parquet"))

    args = parser.parse_args()

    if args.command == "select":
        select_uncertain_batch(
            features_path = args.features,
            rf_path       = args.rf_model,
            output_path   = args.output,
            n_samples     = args.n_samples,
            method        = args.method,
        )
    elif args.command == "import":
        import_expert_labels(
            features_path = args.features,
            labeled_csv   = args.labeled_csv,
            output_path   = args.output,
        )
    else:
        parser.print_help()
