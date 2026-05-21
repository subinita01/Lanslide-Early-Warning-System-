"""
run_pipeline.py
───────────────
ONE COMMAND to run the entire Resilio-Route ML pipeline from scratch.

What it does, in order:
  1. Generate synthetic sensor data  (if no real data yet)
  2. Generate physics-based labels   (Factor of Safety — cold start)
  3. Engineer all 18 features
  4. Train Random Forest
  5. Train CNN
  6. Run ensemble evaluation
  7. Print deployment readiness summary

Run:
    python run_pipeline.py                 # full pipeline on synthetic data
    python run_pipeline.py --real          # use real data from ml/data/raw.parquet
    python run_pipeline.py --skip-train    # only engineering + evaluation
"""

import argparse
import json
import sys
from pathlib import Path
from loguru import logger

# ── Path setup ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA  = ROOT / "ml" / "data"
MODELS = ROOT / "ml" / "models"
REPORTS = ROOT / "ml" / "reports"

for d in [DATA, MODELS, REPORTS]:
    d.mkdir(parents=True, exist_ok=True)


def run_pipeline(use_real_data: bool = False, skip_train: bool = False):

    logger.info("=" * 60)
    logger.info("  RESILIO-ROUTE — Navigator ML Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Data ──────────────────────────────────────────────────────
    raw_path = DATA / "raw.parquet"

    if use_real_data:
        if not raw_path.exists():
            logger.error(f"Real data not found at {raw_path}. "
                         "Export from your TimescaleDB gateway first.")
            sys.exit(1)
        logger.info(f"[Step 1] Using real sensor data from {raw_path}")
    else:
        logger.info("[Step 1] Generating synthetic sensor data (development mode)...")
        from ml.simulate_data import generate
        generate(n_nodes=10, n_days=60, output=raw_path)

    # ── Step 2: Physics labels ────────────────────────────────────────────
    labeled_path = DATA / "labeled.parquet"
    logger.info("[Step 2] Generating Factor of Safety labels...")
    from ml.physics_labels import label_dataset
    label_dataset(input_path=raw_path, output_path=labeled_path)

    # ── Step 3: Feature engineering ───────────────────────────────────────
    features_path = DATA / "features.parquet"
    logger.info("[Step 3] Engineering features...")
    import pandas as pd
    from ml.feature_engineering import engineer_features
    df = pd.read_parquet(labeled_path)
    featured = engineer_features(df)
    featured.to_parquet(features_path, index=False)
    logger.success(f"  Features saved → {features_path}  ({len(featured):,} rows)")

    if skip_train:
        logger.info("--skip-train flag set. Stopping after feature engineering.")
        return

    # ── Step 4: Train Random Forest ───────────────────────────────────────
    rf_path = MODELS / "rf_v1.pkl"
    logger.info("[Step 4] Training Random Forest...")
    from ml.train_rf import train as train_rf
    rf_metrics = train_rf(features_path=features_path, output_path=rf_path)

    # ── Step 5: Train CNN ─────────────────────────────────────────────────
    cnn_path = MODELS / "cnn_v1.pt"
    logger.info("[Step 5] Training CNN (this takes a few minutes)...")
    from ml.train_cnn import train as train_cnn
    cnn_metrics = train_cnn(features_path=features_path, output_path=cnn_path,
                             epochs=30)   # 30 for quick start; use 50 for production

    # ── Step 6: Ensemble test ─────────────────────────────────────────────
    logger.info("[Step 6] Testing ensemble Navigator...")
    from ml.navigator import Navigator
    from ml.data_schema import WINDOW_STEPS

    nav = Navigator.load(rf_path, cnn_path)
    df_feat = pd.read_parquet(features_path)

    results = nav.predict_batch(df_feat)

    # ── Step 7: Summary ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE — Deployment Readiness Summary")
    logger.info("=" * 60)

    rf_recall  = rf_metrics.get("red_recall", 0)
    cnn_recall = cnn_metrics.get("red_recall", 0)

    print(f"""
  Random Forest
    RED recall:     {rf_recall:.4f}   {'✅ PASS' if rf_recall >= 0.70 else '❌ NEEDS MORE DATA'}
    Model path:     {rf_path}

  CNN
    RED recall:     {cnn_recall:.4f}  {'✅ PASS' if cnn_recall >= 0.70 else '❌ NEEDS MORE DATA'}
    Model path:     {cnn_path}

  Navigator (ensemble)
    Nodes tested:   {len(results)}
    Predictions:    {[r.label for _, r in results]}

  Next steps:
    ✅ Models trained and saved
    👉 Run: uvicorn backend.api.main:app --reload
       Then: curl http://localhost:8000/v1/health
    👉 When real hardware arrives, replace synthetic data with:
       ml/data/raw.parquet (export from TimescaleDB gateway)
    👉 For real-event labels, run:
       python ml/physics_labels.py --input ml/data/raw.parquet
""")

    # Save summary report
    summary = {"rf": rf_metrics, "cnn": cnn_metrics, "nodes_tested": len(results)}
    (REPORTS / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Full report → {REPORTS / 'pipeline_summary.json'}")


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full Resilio-Route ML pipeline")
    parser.add_argument("--real",       action="store_true", help="Use real sensor data")
    parser.add_argument("--skip-train", action="store_true", help="Skip training steps")
    args = parser.parse_args()

    run_pipeline(use_real_data=args.real, skip_train=args.skip_train)
