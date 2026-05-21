"""
tests/test_pipeline.py
──────────────────────
Smoke tests — run these to verify your environment is set up correctly.

Run:
    python -m pytest tests/ -v
    python tests/test_pipeline.py          # without pytest
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def test_physics_labels():
    """FS labeling must produce all three classes for a range of pore pressures."""
    from ml.physics_labels import factor_of_safety, fs_to_label

    pore = np.array([0.5, 2.0, 12.0])   # low, medium, high pressure
    fs   = factor_of_safety(pore, slope_angle_deg=25.0)

    assert fs[0] > 1.3, f"Low pore pressure should give GREEN FS, got {fs[0]:.3f}"
    assert fs[2] < 1.3, f"High pore pressure should give YELLOW/RED FS, got {fs[2]:.3f}"

    labels_str, labels_int = fs_to_label(fs)
    assert set(labels_str).issubset({"GREEN", "YELLOW", "RED"})
    assert labels_int.dtype == int
    print("PASS test_physics_labels")


def test_simulate_data():
    """Synthetic data must have correct shape and all three classes."""
    from ml.simulate_data import generate
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.parquet"
        df = generate(n_nodes=2, n_days=3, output=out)

    assert len(df) > 0
    assert "tilt_deg" in df.columns
    assert "label"    in df.columns
    assert set(df["label"].unique()).issubset({"GREEN", "YELLOW", "RED"})
    print(f"PASS test_simulate_data ({len(df)} rows)")


def test_feature_engineering():
    """Feature engineering must produce all 18 expected features."""
    from ml.simulate_data import generate
    from ml.physics_labels import label_dataset
    from ml.feature_engineering import engineer_features
    from ml.data_schema import FEATURE_COLS
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        raw_path     = Path(tmp) / "raw.parquet"
        labeled_path = Path(tmp) / "labeled.parquet"

        generate(n_nodes=2, n_days=5, output=raw_path)
        label_dataset(input_path=raw_path, output_path=labeled_path)

        df = pd.read_parquet(labeled_path)
        featured = engineer_features(df)

    missing = [c for c in FEATURE_COLS if c not in featured.columns]
    assert not missing, f"Missing features: {missing}"
    assert len(featured) > 0
    print(f"PASS test_feature_engineering ({len(featured)} rows, {len(FEATURE_COLS)} features)")


def test_cnn_forward_pass():
    """CNN must accept correct input shape and return 3-class logits."""
    import torch
    from ml.train_cnn import LandslideCNN
    from ml.data_schema import SEQUENCE_CHANNELS, WINDOW_STEPS

    model = LandslideCNN()
    model.eval()

    batch = torch.randn(4, len(SEQUENCE_CHANNELS), WINDOW_STEPS)
    with torch.no_grad():
        out = model(batch)

    assert out.shape == (4, 3), f"Expected (4,3) output, got {out.shape}"
    print(f"PASS test_cnn_forward_pass (output shape: {tuple(out.shape)})")


def test_navigator_predict():
    """Navigator must return a valid NavigatorResult with all three probabilities."""
    from unittest.mock import MagicMock
    import numpy as np
    import pandas as pd
    import torch
    from ml.navigator import Navigator, NavigatorResult
    from ml.train_cnn import LandslideCNN
    from ml.data_schema import FEATURE_COLS, SEQUENCE_CHANNELS, WINDOW_STEPS

    # Mock RF model
    mock_rf = MagicMock()
    mock_rf.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])

    # Real CNN (randomly initialised)
    cnn = LandslideCNN()

    nav = Navigator(rf_model=mock_rf, cnn_model=cnn)

    # Build a fake window
    rng = np.random.default_rng(42)
    data = {col: rng.random(WINDOW_STEPS) for col in FEATURE_COLS + SEQUENCE_CHANNELS}
    df_window = pd.DataFrame(data)

    result = nav.predict(df_window)

    assert isinstance(result, NavigatorResult)
    assert result.label in {"GREEN", "YELLOW", "RED"}
    assert abs(result.p_green + result.p_yellow + result.p_red - 1.0) < 1e-4
    assert 0.0 <= result.confidence <= 1.0
    print(f"PASS test_navigator_predict (label={result.label}, p_red={result.p_red:.3f})")


def test_api_health():
    """Health handler must remain available even when the DB is unavailable."""
    from sqlalchemy.exc import SQLAlchemyError
    from backend.api.main import health

    class UnavailableDB:
        def query(self, *args, **kwargs):
            raise SQLAlchemyError("db unavailable")

    data = health(db=UnavailableDB())
    assert data.status == "ok"
    assert data.nodes_online == 0
    assert data.model_version
    print(f"PASS test_api_health (version={data.model_version})")


# ── Run all ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_physics_labels,
        test_simulate_data,
        test_feature_engineering,
        test_cnn_forward_pass,
        test_navigator_predict,
        test_api_health,
    ]
    print(f"\nRunning {len(tests)} smoke tests for Resilio-Route...\n")
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAILED {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("  Environment is ready. Run: python run_pipeline.py")
    print(f"{'='*40}\n")
    sys.exit(failed)
