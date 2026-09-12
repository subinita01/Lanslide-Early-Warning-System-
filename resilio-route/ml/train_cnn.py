"""
ml/train_cnn.py
───────────────
Trains the CNN component of the Navigator ensemble on 3-hour sensor windows.

Usage:
    python ml/train_cnn.py \
        --features ml/data/features.parquet \
        --output   ml/models/cnn_v1.pt
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from loguru import logger

from sklearn.metrics import classification_report, recall_score

from ml.feature_engineering import build_sequences
from ml.data_schema import SEQUENCE_CHANNELS, WINDOW_STEPS, HORIZON_STEPS, LABEL_MAP_INV
from config.settings import settings


# ── Dataset ───────────────────────────────────────────────────────────────

class LandslideDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):        return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Model ─────────────────────────────────────────────────────────────────

class LandslideCNN(nn.Module):
    """
    1-D CNN operating on sensor time-series windows.

    Input:  (batch, n_channels=6, window=36)
    Output: (batch, 3)  — logits for [GREEN, YELLOW, RED]

    Three convolutional layers detect patterns at increasing timescales:
      Layer 1 (k=3): sudden spikes (15 min)
      Layer 2 (k=7): medium trends (35 min)
      Layer 3 (k=9): pre-failure acceleration (45 min)
    """

    def __init__(
        self,
        n_channels: int = len(SEQUENCE_CHANNELS),
        window:     int = WINDOW_STEPS,
        n_classes:  int = 3,
        dropout:    float = 0.25,
    ):
        super().__init__()

        self.conv_block = nn.Sequential(
            # ── Layer 1: spike detector ─────────────────────────────────
            nn.Conv1d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            # ── Layer 2: trend detector ─────────────────────────────────
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),          # 36 → 18 time steps
            nn.Dropout(dropout),

            # ── Layer 3: acceleration signature ─────────────────────────
            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),  # → 4 time steps
        )

        # 128 filters × 4 time steps = 512-dim feature vector
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
            # Note: no softmax — CrossEntropyLoss includes log-softmax
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block(x))

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Return softmax probabilities as numpy array."""
        self.eval()
        with torch.no_grad():
            logits = self(x)
            return torch.softmax(logits, dim=1).numpy()


# ── Training loop ─────────────────────────────────────────────────────────

def train(
    features_path: Path,
    output_path:   Path,
    epochs:        int   = 50,
    batch_size:    int   = 64,
    lr:            float = 1e-3,
    val_fraction:  float = 0.20,
    window:        int   = WINDOW_STEPS,
    horizon:       int   = HORIZON_STEPS,
) -> dict:

    # ── Build sequences ───────────────────────────────────────────────────
    logger.info(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)

    logger.info("Building time-series sequences...")
    X, y = build_sequences(df, window=window, horizon=horizon)

    # ── Time-based split ──────────────────────────────────────────────────
    split = int(len(X) * (1 - val_fraction))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,}")

    # ── Class weights for imbalanced data ─────────────────────────────────
    counts = np.bincount(y_train, minlength=3).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = torch.tensor(weights / weights.sum(), dtype=torch.float32)
    logger.info(f"Class weights: {weights.numpy().round(4)}")

    # ── Dataloaders ───────────────────────────────────────────────────────
    train_loader = DataLoader(LandslideDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(LandslideDataset(X_val,   y_val),
                              batch_size=batch_size)

    # ── Model, optimizer, scheduler ───────────────────────────────────────
    model     = LandslideCNN()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    best_state    = None

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_loss += criterion(model(X_b), y_b).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d}/{epochs} | "
                        f"train={train_loss:.4f} | val={val_loss:.4f}")

    # ── Restore best checkpoint ───────────────────────────────────────────
    model.load_state_dict(best_state)

    # ── Final evaluation ──────────────────────────────────────────────────
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            preds = model(X_b).argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_true.extend(y_b.tolist())

    report = classification_report(
        all_true, all_preds,
        target_names=["GREEN", "YELLOW", "RED"],
        digits=4,
    )
    logger.info(f"\n{report}")

    red_recall = recall_score(all_true, all_preds, labels=[2], average="macro")
    logger.info(f"RED recall: {red_recall:.4f}  (target >= 0.90)")

    # ── Save ──────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "model_version": settings.MODEL_VERSION,
                "channels": SEQUENCE_CHANNELS, "window": window}, output_path)
    logger.info(f"CNN saved → {output_path}")

    metrics = {"red_recall": round(red_recall, 4), "best_val_loss": round(best_val_loss, 4),
               "epochs_trained": epochs, "model_version": settings.MODEL_VERSION}
    (output_path.parent / "cnn_eval.json").write_text(json.dumps(metrics, indent=2))

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Navigator CNN")
    parser.add_argument("--features",    type=Path,  default=Path("ml/data/features.parquet"))
    parser.add_argument("--output",      type=Path,  default=Path("ml/models/cnn_v1.pt"))
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    args = parser.parse_args()

    train(features_path=args.features, output_path=args.output,
          epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
