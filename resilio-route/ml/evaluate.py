"""
ml/evaluate.py
──────────────
Comprehensive evaluation of the Navigator ensemble.
Computes all safety-critical metrics: RED recall, lead time,
false alarm rate, confusion matrix, and deployment readiness.

Usage:
    python ml/evaluate.py \
        --features  ml/data/features.parquet \
        --rf-model  ml/models/rf_v1.pkl \
        --cnn-model ml/models/cnn_v1.pt
"""

import argparse
import json
import numpy as np
import pandas as pd
import joblib
import torch
from pathlib import Path
from datetime import timedelta
from loguru import logger

from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support,
)

from ml.navigator import Navigator
from ml.feature_engineering import build_sequences
from ml.data_schema import (
    FEATURE_COLS, SEQUENCE_CHANNELS,
    WINDOW_STEPS, HORIZON_STEPS,
    LABEL_INT_COL, LABEL_MAP_INV,
)
from config.settings import settings


# ── Deployment thresholds ─────────────────────────────────────────────────

DEPLOYMENT_CRITERIA = {
    "red_recall":    0.90,   # miss fewer than 1 in 10 slides
    "red_precision": 0.60,   # fewer than 4 in 10 alerts are false
    "lead_time_p10": 25,     # 25-min minimum lead time (10th percentile)
}


# ── Lead time estimation ──────────────────────────────────────────────────

def estimate_lead_times(
    df:         pd.DataFrame,
    navigator:  Navigator,
    n_events:   int = 50,
) -> list[int]:
    """
    For each RED event in the test set, find how many minutes BEFORE
    the true RED label the Navigator first predicted RED.
    Returns a list of lead times in minutes.
    """
    lead_times = []

    for node_id, node_df in df.groupby("node_id"):
        node_df = node_df.reset_index(drop=True)

        # Find indices where TRUE label is RED
        red_indices = node_df.index[node_df[LABEL_INT_COL] == 2].tolist()
        if not red_indices:
            continue

        for red_idx in red_indices[:n_events]:
            if red_idx < WINDOW_STEPS:
                continue

            # Walk backward from red_idx to find when Navigator first went RED
            first_red_pred = None
            for lookback in range(1, min(20, red_idx - WINDOW_STEPS + 1)):
                window_end = red_idx - lookback
                window = node_df.iloc[window_end - WINDOW_STEPS : window_end]
                result = navigator.predict(window)
                if result.label == "RED":
                    first_red_pred = lookback
                else:
                    break   # consecutive non-RED stops the search

            if first_red_pred is not None:
                lead_min = first_red_pred * 5   # 5 min per step
                lead_times.append(lead_min)

    return lead_times


# ── Main evaluation ───────────────────────────────────────────────────────

def evaluate(
    features_path: Path,
    rf_path:       Path,
    cnn_path:      Path,
    test_fraction: float = 0.25,
    report_path:   Path  = None,
) -> dict:

    logger.info("=" * 50)
    logger.info("  Navigator Evaluation")
    logger.info("=" * 50)

    # ── Load ──────────────────────────────────────────────────────────────
    df = pd.read_parquet(features_path)
    nav = Navigator.load(rf_path, cnn_path)

    # Time-based test split
    split = int(len(df) * (1 - test_fraction))
    test_df = df.iloc[split:].reset_index(drop=True)
    logger.info(f"Test set: {len(test_df):,} rows")

    # ── Tabular predictions (RF + ensemble) ──────────────────────────────
    X_tab   = test_df[FEATURE_COLS].values
    y_true  = test_df[LABEL_INT_COL].values

    rf_proba = nav.rf.predict_proba(X_tab)

    # CNN on sequences
    X_seq, y_seq = build_sequences(test_df, window=WINDOW_STEPS, horizon=HORIZON_STEPS)
    cnn_proba_list = []
    nav.cnn.eval()
    batch_size = 256
    for i in range(0, len(X_seq), batch_size):
        batch = torch.tensor(X_seq[i:i+batch_size])
        with torch.no_grad():
            logits = nav.cnn(batch)
            p = torch.softmax(logits, dim=1).numpy()
        cnn_proba_list.append(p)
    cnn_proba = np.vstack(cnn_proba_list)

    # Align lengths (sequences are shorter due to windowing)
    n = min(len(rf_proba), len(cnn_proba), len(y_seq))
    rf_p  = rf_proba[-n:]
    cnn_p = cnn_proba[:n]
    y_t   = y_seq[:n]

    combined = nav.rf_w * rf_p + nav.cnn_w * cnn_p
    y_pred = np.array([
        2 if p[2] >= settings.RED_THRESHOLD
        else (1 if p[2] >= settings.YELLOW_THRESHOLD or p[1] >= 0.50 else 0)
        for p in combined
    ])

    # ── Classification metrics ────────────────────────────────────────────
    report = classification_report(
        y_t, y_pred,
        target_names=["GREEN", "YELLOW", "RED"],
        digits=4,
        output_dict=True,
    )
    logger.info("\n" + classification_report(
        y_t, y_pred, target_names=["GREEN", "YELLOW", "RED"], digits=4
    ))

    cm = confusion_matrix(y_t, y_pred, labels=[0, 1, 2])
    logger.info(f"\nConfusion matrix (rows=true, cols=pred):")
    logger.info(f"          GREEN  YELLOW  RED")
    for i, name in enumerate(["GREEN ", "YELLOW", "RED   "]):
        logger.info(f"  {name}:  {cm[i]}")

    red_total      = cm[2, :].sum()
    red_recall     = cm[2, 2] / red_total if red_total > 0 else 0.0
    red_precision  = report["RED"]["precision"]
    red_miss_rate  = cm[2, 0] / red_total if red_total > 0 else 0.0
    false_alarm    = cm[0, 2] / cm[:, 2].sum() if cm[:, 2].sum() > 0 else 0.0

    logger.info(f"\n── Safety metrics ───────────────────────────")
    logger.info(f"  RED recall:      {red_recall:.4f}  "
                f"({'✅' if red_recall >= DEPLOYMENT_CRITERIA['red_recall'] else '❌'} "
                f"target={DEPLOYMENT_CRITERIA['red_recall']})")
    logger.info(f"  RED precision:   {red_precision:.4f}  "
                f"({'✅' if red_precision >= DEPLOYMENT_CRITERIA['red_precision'] else '❌'} "
                f"target={DEPLOYMENT_CRITERIA['red_precision']})")
    logger.info(f"  RED miss rate:   {red_miss_rate:.4f}  (RED predicted as GREEN)")
    logger.info(f"  False alarm rate:{false_alarm:.4f}  (GREEN predicted as RED)")

    # ── Lead time ─────────────────────────────────────────────────────────
    logger.info("\nEstimating lead times (may take a minute)...")
    lead_times = estimate_lead_times(test_df, nav, n_events=30)

    if lead_times:
        lt_p10 = int(np.percentile(lead_times, 10))
        lt_p50 = int(np.percentile(lead_times, 50))
        lt_p90 = int(np.percentile(lead_times, 90))
        logger.info(f"  Lead time P10: {lt_p10} min  "
                    f"({'✅' if lt_p10 >= DEPLOYMENT_CRITERIA['lead_time_p10'] else '❌'} "
                    f"target>={DEPLOYMENT_CRITERIA['lead_time_p10']})")
        logger.info(f"  Lead time P50: {lt_p50} min")
        logger.info(f"  Lead time P90: {lt_p90} min")
    else:
        lt_p10, lt_p50, lt_p90 = 0, 0, 0
        logger.warning("No RED events found in test set for lead time estimation.")

    # ── Deployment verdict ────────────────────────────────────────────────
    passes = {
        "red_recall":    red_recall    >= DEPLOYMENT_CRITERIA["red_recall"],
        "red_precision": red_precision >= DEPLOYMENT_CRITERIA["red_precision"],
        "lead_time_p10": lt_p10        >= DEPLOYMENT_CRITERIA["lead_time_p10"],
    }
    deploy_ready = all(passes.values())

    logger.info(f"\n── Deployment readiness ─────────────────────")
    for check, ok in passes.items():
        logger.info(f"  {'✅' if ok else '❌'} {check}")
    if deploy_ready:
        logger.success("\n  ✅ NAVIGATOR IS DEPLOYMENT-READY")
    else:
        failed = [k for k, v in passes.items() if not v]
        logger.warning(f"\n  ❌ NOT YET READY — failing: {failed}")
        logger.warning("  Collect more labeled data and retrain.")

    # ── Save report ───────────────────────────────────────────────────────
    metrics = {
        "red_recall":     round(red_recall, 4),
        "red_precision":  round(red_precision, 4),
        "red_miss_rate":  round(red_miss_rate, 4),
        "false_alarm_rate": round(false_alarm, 4),
        "lead_time_p10":  lt_p10,
        "lead_time_p50":  lt_p50,
        "lead_time_p90":  lt_p90,
        "deploy_ready":   deploy_ready,
        "passes":         passes,
        "confusion_matrix": cm.tolist(),
        "n_test_samples": int(n),
        "model_version":  settings.MODEL_VERSION,
    }

    if report_path is None:
        report_path = Path("ml/reports/evaluation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"\nFull report → {report_path}")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Navigator ensemble")
    parser.add_argument("--features",  type=Path, default=Path("ml/data/features.parquet"))
    parser.add_argument("--rf-model",  type=Path, default=Path("ml/models/rf_v1.pkl"))
    parser.add_argument("--cnn-model", type=Path, default=Path("ml/models/cnn_v1.pt"))
    parser.add_argument("--report",    type=Path, default=Path("ml/reports/evaluation.json"))
    parser.add_argument("--test-frac", type=float, default=0.25)
    args = parser.parse_args()

    evaluate(
        features_path = args.features,
        rf_path       = args.rf_model,
        cnn_path      = args.cnn_model,
        test_fraction = args.test_frac,
        report_path   = args.report,
    )
