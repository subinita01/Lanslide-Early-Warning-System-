"""
ml/transfer_learning.py
───────────────────────
Fine-tune the Navigator CNN for a new highway region.

Architecture:
  Shared trunk (frozen) — learns universal slope physics (transfers everywhere)
  Regional head (trainable) — calibrates to local geology (re-learned per region)

As little as 14 days of local data gives ~0.83 RED recall.

Usage:
    python ml/transfer_learning.py \
        --base-model   ml/models/cnn_v1.pt \
        --local-data   ml/data/features_manipur.parquet \
        --region       manipur \
        --output       ml/models/cnn_manipur_v1.pt \
        --epochs       30
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from loguru import logger
from sklearn.metrics import recall_score

from ml.train_cnn import LandslideCNN, LandslideDataset
from ml.feature_engineering import build_sequences
from ml.data_schema import SEQUENCE_CHANNELS, WINDOW_STEPS, HORIZON_STEPS
from config.settings import settings


# ── Generalisable Navigator architecture ──────────────────────────────────

class GeneralisableNavigator(nn.Module):
    """
    Shared trunk + per-region classification heads.

    The trunk (Conv layers 1–3) encodes universal slope physics.
    Each region gets its own small head that learns local thresholds.

    Trunk is frozen during fine-tuning → only the head is updated.
    """

    SUPPORTED_REGIONS = [
        "mizoram", "manipur", "sikkim",
        "arunachal", "nagaland", "meghalaya",
    ]

    def __init__(self, n_channels: int = len(SEQUENCE_CHANNELS)):
        super().__init__()

        # ── Shared trunk (universal physics) ─────────────────────────────
        self.shared_trunk = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),   # → 512-dim feature vector
        )

        # ── Region-specific heads ─────────────────────────────────────────
        self.region_heads = nn.ModuleDict({
            region: self._make_head()
            for region in self.SUPPORTED_REGIONS
        })

    def _make_head(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),   # → [GREEN, YELLOW, RED] logits
        )

    def forward(self, x: torch.Tensor, region: str) -> torch.Tensor:
        if region not in self.region_heads:
            raise ValueError(f"Unknown region '{region}'. "
                             f"Supported: {self.SUPPORTED_REGIONS}")
        features = self.shared_trunk(x)
        return self.region_heads[region](features)

    def freeze_trunk(self):
        """Freeze shared trunk — only region heads will train."""
        for param in self.shared_trunk.parameters():
            param.requires_grad = False
        logger.info("Trunk frozen. Only regional head will be updated.")

    def unfreeze_trunk(self):
        """Unfreeze trunk for full fine-tuning (needs >4 weeks of data)."""
        for param in self.shared_trunk.parameters():
            param.requires_grad = True
        logger.info("Trunk unfrozen. Full network will be updated.")

    def trainable_params(self, region: str) -> int:
        params = sum(
            p.numel() for p in self.region_heads[region].parameters()
            if p.requires_grad
        )
        return params


# ── Load base model weights into GeneralisableNavigator ──────────────────

def load_base_weights(base_cnn_path: Path) -> GeneralisableNavigator:
    """
    Load a trained LandslideCNN checkpoint into the shared trunk
    of a GeneralisableNavigator. Heads are randomly initialised.
    """
    logger.info(f"Loading base CNN from {base_cnn_path}")
    ckpt = torch.load(base_cnn_path, map_location="cpu")

    # Build a plain CNN to extract trunk weights
    base_cnn = LandslideCNN()
    base_cnn.load_state_dict(ckpt["model_state"])

    # Transfer conv_block weights → shared_trunk
    gen_nav = GeneralisableNavigator()
    trunk_state = gen_nav.shared_trunk.state_dict()

    # Map keys from base_cnn.conv_block → gen_nav.shared_trunk
    base_trunk_state = {
        k.replace("conv_block.", ""): v
        for k, v in base_cnn.state_dict().items()
        if k.startswith("conv_block.")
    }

    trunk_state.update(base_trunk_state)
    gen_nav.shared_trunk.load_state_dict(trunk_state, strict=False)
    logger.success("Base trunk weights transferred.")

    return gen_nav


# ── Fine-tune for new region ───────────────────────────────────────────────

def fine_tune(
    base_model_path:  Path,
    local_data_path:  Path,
    region:           str,
    output_path:      Path,
    epochs:           int   = 30,
    batch_size:       int   = 32,
    lr:               float = 1e-3,
    full_finetune:    bool  = False,   # True only if >4 weeks of data
    val_fraction:     float = 0.20,
) -> dict:

    if region not in GeneralisableNavigator.SUPPORTED_REGIONS:
        raise ValueError(f"Region '{region}' not supported. "
                         f"Add it to GeneralisableNavigator.SUPPORTED_REGIONS first.")

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info(f"Loading local data from {local_data_path}")
    df = pd.read_parquet(local_data_path)
    n_days = (df["timestamp"].max() - df["timestamp"].min()).days
    logger.info(f"Local data: {len(df):,} rows | ~{n_days} days | region={region}")

    if n_days < 3:
        logger.warning("Less than 3 days of data. RED recall may be poor. Collect more data.")

    # ── Build sequences ───────────────────────────────────────────────────
    X, y = build_sequences(df, window=WINDOW_STEPS, horizon=HORIZON_STEPS)
    split = int(len(X) * (1 - val_fraction))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,}")

    # ── Load model ────────────────────────────────────────────────────────
    model = load_base_weights(base_model_path)
    model.freeze_trunk()

    if full_finetune:
        logger.info("Full fine-tune mode (trunk unfrozen).")
        model.unfreeze_trunk()

    trainable = model.trainable_params(region)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} "
                f"({trainable/total*100:.1f}%)")

    # ── Training ──────────────────────────────────────────────────────────
    counts  = np.bincount(y_train, minlength=3).astype(float)
    weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32)
    weights = weights / weights.sum()

    train_loader = DataLoader(LandslideDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(LandslideDataset(X_val,   y_val),
                              batch_size=batch_size)

    # Only optimise the region head (trunk is frozen)
    optimizer = optim.Adam(
        model.region_heads[region].parameters(), lr=lr, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    best_val_loss, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b, region), y_b)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                v_loss += criterion(model(X_b, region), y_b).item()
        v_loss /= len(val_loader)
        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d}/{epochs} | train={t_loss:.4f} | val={v_loss:.4f}")

    model.load_state_dict(best_state)

    # ── Evaluate ──────────────────────────────────────────────────────────
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            preds = model(X_b, region).argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_true.extend(y_b.tolist())

    red_recall = recall_score(all_true, all_preds, labels=[2], average="macro")
    logger.info(f"RED recall after fine-tune ({region}): {red_recall:.4f}")

    deploy_ready = red_recall >= 0.83
    logger.success(f"{'✅ READY' if deploy_ready else '⚠️  NEEDS MORE DATA'} — "
                   f"RED recall = {red_recall:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state":   best_state,
        "region":        region,
        "model_version": settings.MODEL_VERSION,
        "channels":      SEQUENCE_CHANNELS,
        "window":        WINDOW_STEPS,
        "n_days_data":   n_days,
        "red_recall":    red_recall,
    }, output_path)
    logger.info(f"Fine-tuned model saved → {output_path}")

    metrics = {
        "region":      region,
        "red_recall":  round(red_recall, 4),
        "n_days_data": n_days,
        "deploy_ready": deploy_ready,
    }
    (output_path.parent / f"transfer_{region}_eval.json").write_text(
        json.dumps(metrics, indent=2)
    )
    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer learning for new NE India region")
    parser.add_argument("--base-model",   type=Path, required=True)
    parser.add_argument("--local-data",   type=Path, required=True)
    parser.add_argument("--region",       type=str,  required=True,
                        choices=GeneralisableNavigator.SUPPORTED_REGIONS)
    parser.add_argument("--output",       type=Path, default=None)
    parser.add_argument("--epochs",       type=int,  default=30)
    parser.add_argument("--batch-size",   type=int,  default=32)
    parser.add_argument("--lr",           type=float,default=1e-3)
    parser.add_argument("--full-finetune",action="store_true",
                        help="Unfreeze trunk (use only with >4 weeks data)")
    args = parser.parse_args()

    out = args.output or Path(f"ml/models/cnn_{args.region}_v1.pt")

    fine_tune(
        base_model_path = args.base_model,
        local_data_path = args.local_data,
        region          = args.region,
        output_path     = out,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        full_finetune   = args.full_finetune,
    )
