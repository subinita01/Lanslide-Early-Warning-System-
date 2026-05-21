"""
ml/navigator.py
───────────────
The Navigator — real-time ensemble inference engine.
Combines Random Forest (tabular) + CNN (time-series) predictions.
This is what runs on the edge gateway every 5 minutes.

Usage (programmatic):
    from ml.navigator import Navigator
    nav = Navigator.load("ml/models/rf_v1.pkl", "ml/models/cnn_v1.pt")
    result = nav.predict(df_window)
    print(result)   # NavigatorResult(label='RED', p_red=0.83, ...)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
import torch
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

from ml.train_cnn import LandslideCNN
from ml.data_schema import (
    FEATURE_COLS, SEQUENCE_CHANNELS,
    WINDOW_STEPS, LABEL_MAP_INV,
)
from config.settings import settings


# ── Result dataclass ──────────────────────────────────────────────────────

@dataclass
class NavigatorResult:
    label:      str          # "GREEN" / "YELLOW" / "RED"
    label_int:  int          # 0 / 1 / 2
    p_green:    float
    p_yellow:   float
    p_red:      float
    confidence: float
    rf_proba:   list[float]
    cnn_proba:  list[float]

    def to_dict(self) -> dict:
        return {
            "label":      self.label,
            "label_int":  self.label_int,
            "p_green":    round(self.p_green,  4),
            "p_yellow":   round(self.p_yellow, 4),
            "p_red":      round(self.p_red,    4),
            "confidence": round(self.confidence, 4),
        }

    def led_command(self, node_id: str) -> str:
        """Format the serial command for the LED warning board."""
        return f"{node_id}:{self.label}:{self.confidence:.2f}\n"

    def __str__(self) -> str:
        bar = "🟢" if self.label == "GREEN" else "🟡" if self.label == "YELLOW" else "🔴"
        return (f"{bar} {self.label}  "
                f"[G={self.p_green:.2f} Y={self.p_yellow:.2f} R={self.p_red:.2f}]  "
                f"confidence={self.confidence:.2f}")


# ── Navigator class ───────────────────────────────────────────────────────

class Navigator:
    """
    Ensemble: Random Forest (tabular snapshot) + CNN (3-hour time window).

    Parameters
    ----------
    rf_model   : trained RandomForestClassifier
    cnn_model  : trained LandslideCNN
    rf_weight  : weight for RF probabilities (default 0.6)
    cnn_weight : weight for CNN probabilities (default 0.4)
    """

    def __init__(
        self,
        rf_model,
        cnn_model:  LandslideCNN,
        rf_weight:  float = 0.6,
        cnn_weight: float = 0.4,
    ):
        assert abs(rf_weight + cnn_weight - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.rf      = rf_model
        self.cnn     = cnn_model
        self.rf_w    = rf_weight
        self.cnn_w   = cnn_weight
        self.cnn.eval()

    # ── Loading ───────────────────────────────────────────────────────────

    @classmethod
    def load(cls, rf_path: Path, cnn_path: Path, **kwargs) -> "Navigator":
        """Load both models from disk and return a ready Navigator."""
        logger.info(f"Loading RF  from {rf_path}")
        rf = joblib.load(rf_path)

        logger.info(f"Loading CNN from {cnn_path}")
        ckpt = torch.load(cnn_path, map_location="cpu")
        cnn  = LandslideCNN()
        cnn.load_state_dict(ckpt["model_state"])
        cnn.eval()

        logger.success("Navigator loaded and ready.")
        return cls(rf_model=rf, cnn_model=cnn, **kwargs)

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, df_window: pd.DataFrame) -> NavigatorResult:
        """
        Predict risk for a single node window.

        Parameters
        ----------
        df_window : DataFrame with the last WINDOW_STEPS rows for one node.
                    Must contain all FEATURE_COLS and SEQUENCE_CHANNELS.

        Returns
        -------
        NavigatorResult
        """
        if len(df_window) < WINDOW_STEPS:
            logger.warning(
                f"Window has {len(df_window)} rows, expected {WINDOW_STEPS}. "
                "Padding with last row."
            )
            pad = pd.concat(
                [df_window.iloc[[0]]] * (WINDOW_STEPS - len(df_window)) + [df_window],
                ignore_index=True,
            )
            df_window = pad

        # ── RF: uses last (most recent) row as snapshot ───────────────────
        x_tab = df_window[FEATURE_COLS].iloc[-1:].values  # shape (1, n_features)
        rf_proba = self.rf.predict_proba(x_tab)[0]        # shape (3,)

        # ── CNN: uses full window as time-series ──────────────────────────
        x_seq = df_window[SEQUENCE_CHANNELS].values.T     # shape (n_channels, window)
        x_tensor = torch.tensor(x_seq[np.newaxis], dtype=torch.float32)  # (1, C, T)
        with torch.no_grad():
            logits = self.cnn(x_tensor)
            cnn_proba = torch.softmax(logits, dim=1).numpy()[0]           # shape (3,)

        # ── Ensemble ──────────────────────────────────────────────────────
        combined = self.rf_w * rf_proba + self.cnn_w * cnn_proba

        p_green, p_yellow, p_red = combined

        # ── Classify (safety-first thresholds) ───────────────────────────
        if p_red >= settings.RED_THRESHOLD:
            label = "RED"
        elif p_red >= settings.YELLOW_THRESHOLD or p_yellow >= 0.50:
            label = "YELLOW"
        else:
            label = "GREEN"

        label_int = {"GREEN": 0, "YELLOW": 1, "RED": 2}[label]

        return NavigatorResult(
            label      = label,
            label_int  = label_int,
            p_green    = float(p_green),
            p_yellow   = float(p_yellow),
            p_red      = float(p_red),
            confidence = float(combined.max()),
            rf_proba   = rf_proba.tolist(),
            cnn_proba  = cnn_proba.tolist(),
        )

    def predict_batch(self, df: pd.DataFrame) -> list[NavigatorResult]:
        """
        Predict for all nodes in a DataFrame (grouped by node_id).
        Returns one NavigatorResult per node (using that node's last WINDOW_STEPS rows).
        """
        results = []
        for node_id, node_df in df.groupby("node_id"):
            window = node_df.tail(WINDOW_STEPS)
            result = self.predict(window)
            results.append((node_id, result))
            logger.info(f"Node {node_id}: {result}")
        return results


# ── Quick self-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    rf_path  = Path("ml/models/rf_v1.pkl")
    cnn_path = Path("ml/models/cnn_v1.pt")

    if not rf_path.exists() or not cnn_path.exists():
        print("Models not found. Run train_rf.py and train_cnn.py first.")
        sys.exit(1)

    nav = Navigator.load(rf_path, cnn_path)

    # Load features and test on a random window
    df = pd.read_parquet("ml/data/features.parquet")
    sample_node = df["node_id"].iloc[0]
    window = df[df["node_id"] == sample_node].tail(WINDOW_STEPS)

    result = nav.predict(window)
    print(f"\nNavigator test result for node {sample_node}:")
    print(result)
    print(result.to_dict())
    print(f"LED command: {result.led_command(sample_node)!r}")
