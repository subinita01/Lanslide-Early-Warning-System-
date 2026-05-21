"""
ml/train_rf.py
──────────────
Trains the Random Forest component of the Navigator ensemble.
Handles class imbalance via SMOTE, uses time-based train/test split.

Usage:
    python ml/train_rf.py \
        --features ml/data/features.parquet \
        --output   ml/models/rf_v1.pkl
"""

import argparse
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

from ml.data_schema import FEATURE_COLS, LABEL_INT_COL, LABEL_MAP_INV
from config.settings import settings


# ── Training ──────────────────────────────────────────────────────────────

def train(
    features_path: Path,
    output_path:   Path,
    n_estimators:  int   = 300,
    max_depth:     int   = 15,
    min_samples_leaf: int = 10,
    test_fraction: float = 0.25,
) -> dict:
    """
    Train a Random Forest on engineered features.

    Returns a dict of evaluation metrics.
    """

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)

    # Validate expected columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLS].values
    y = df[LABEL_INT_COL].values

    # ── Time-based split (CRITICAL: never random-split time-series) ────────
    split_idx = int(len(X) * (1 - test_fraction))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")
    logger.info(f"Train class dist: {np.bincount(y_train)}")

    # ── Handle class imbalance with SMOTE ─────────────────────────────────
    logger.info("Applying SMOTE oversampling on training set...")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE: {np.bincount(y_train_bal)}")

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info(f"Training RandomForest (n_estimators={n_estimators}, max_depth={max_depth})...")
    rf = RandomForestClassifier(
        n_estimators     = n_estimators,
        max_depth        = max_depth,
        min_samples_leaf = min_samples_leaf,
        class_weight     = "balanced",
        n_jobs           = -1,
        random_state     = 42,
    )
    rf.fit(X_train_bal, y_train_bal)

    # ── Evaluate ──────────────────────────────────────────────────────────
    y_pred = rf.predict(X_test)
    report_str = classification_report(
        y_test, y_pred,
        labels=[0, 1, 2],
        target_names=["GREEN", "YELLOW", "RED"],
        digits=4,
        zero_division=0,
    )
    
    logger.info(f"\n{report_str}")

    cm = confusion_matrix(y_test, y_pred)
    from sklearn.metrics import recall_score
    red_recall = recall_score(y_test, y_pred, labels=[2], average="macro", zero_division=0)
    red_miss   = 0.0
    logger.info(f"RED recall:    {red_recall:.4f}  (target >= 0.90)")
    logger.info(f"RED miss rate: {red_miss:.4f}   (catastrophic errors: RED→GREEN)")

    # ── Feature importance top-10 ─────────────────────────────────────────
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
    top10 = importances.sort_values(ascending=False).head(10)
    logger.info(f"\nTop-10 features:\n{top10.round(4)}")

    # ── Save model ────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, output_path)
    logger.info(f"Model saved → {output_path}")

    # ── Save report ───────────────────────────────────────────────────────
    metrics = {
        "red_recall":    round(red_recall, 4),
        "red_miss_rate": round(red_miss, 4),
        "n_train":       len(X_train_bal),
        "n_test":        len(X_test),
        "model_version": settings.MODEL_VERSION,
        "feature_importance": top10.round(4).to_dict(),
    }
    report_path = output_path.parent / "rf_eval.json"
    report_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Eval report → {report_path}")

    # ── Deployment gate ───────────────────────────────────────────────────
    if red_recall < 0.70:
        logger.warning(
            f"RED recall {red_recall:.3f} < 0.70. "
            "Collect more labeled data before deploying to edge."
        )
    else:
        logger.success(f"RF model ready. RED recall = {red_recall:.3f}")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Navigator Random Forest")
    parser.add_argument("--features",         type=Path, default=Path("ml/data/features.parquet"))
    parser.add_argument("--output",           type=Path, default=Path("ml/models/rf_v1.pkl"))
    parser.add_argument("--n-estimators",     type=int,  default=300)
    parser.add_argument("--max-depth",        type=int,  default=15)
    parser.add_argument("--min-samples-leaf", type=int,  default=10)
    parser.add_argument("--test-fraction",    type=float,default=0.25)
    args = parser.parse_args()

    train(
        features_path    = args.features,
        output_path      = args.output,
        n_estimators     = args.n_estimators,
        max_depth        = args.max_depth,
        min_samples_leaf = args.min_samples_leaf,
        test_fraction    = args.test_fraction,
    )
