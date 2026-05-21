"""
gateway/run_navigator.py
────────────────────────
Edge inference loop — runs on the Raspberry Pi 4 highway depot gateway.
Reads latest sensor data from the local LoRa buffer, runs Navigator,
then pushes results to the LED board and cloud API.

Run on gateway:
    python gateway/run_navigator.py --interval 300 --led-port /dev/ttyUSB0
"""

import argparse
import json
import time
try:
    import serial
except ImportError:  # Allows non-gateway environments to import the module.
    serial = None
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger

# Allow running standalone or as part of the package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.navigator import Navigator
from ml.data_schema import WINDOW_STEPS
from config.settings import settings


# ── LED board communication ────────────────────────────────────────────────

class LEDBoard:
    """Serial driver for the roadside LED warning sign."""

    def __init__(self, port: str, baud: int = 9600):
        self.port    = port
        self.baud    = baud
        self._serial = None

    def connect(self):
        if serial is None:
            logger.warning("pyserial is not installed. Running in log-only mode.")
            return
        try:
            self._serial = serial.Serial(self.port, self.baud, timeout=2)
            logger.info(f"LED board connected on {self.port}")
        except Exception as e:
            logger.warning(f"LED board not available ({e}). Running in log-only mode.")

    def send(self, node_id: str, label: str, confidence: float):
        cmd = f"{node_id}:{label}:{confidence:.2f}\n"
        if self._serial and self._serial.is_open:
            self._serial.write(cmd.encode())
            logger.info(f"LED → {cmd.strip()}")
        else:
            logger.info(f"[LED-MOCK] → {cmd.strip()}")


# ── Cloud push ─────────────────────────────────────────────────────────────

def push_to_cloud(payload: dict):
    try:
        resp = requests.post(
            f"{settings.CLOUD_API_BASE_URL.rstrip('/')}/v1/ingest",
            json=payload,
            headers={"X-Gateway-Key": settings.CLOUD_API_KEY},
            timeout=10,
        )
        if resp.ok:
            logger.debug(f"Cloud push OK: {resp.status_code}")
        else:
            logger.warning(f"Cloud push failed: {resp.status_code} {resp.text[:80]}")
    except Exception as e:
        logger.warning(f"Cloud push error: {e} — data buffered locally")


# ── Simulated LoRa buffer read (replace with real LoRa library) ────────────

def read_lora_buffer(node_ids: list[str], window: int = WINDOW_STEPS) -> pd.DataFrame:
    """
    STUB: Read the latest `window` readings from the local LoRa buffer.

    In production, replace this with your actual LoRa gateway library
    (e.g. RAK2245 Python SDK, or your custom serial protocol parser).
    This stub returns synthetic data so you can test the inference loop
    before hardware arrives.
    """
    import numpy as np
    from ml.feature_engineering import engineer_features
    from ml.physics_labels import label_dataset
    import tempfile

    rng = np.random.default_rng(int(time.time()))
    records = []
    now = datetime.now(timezone.utc)

    for node_id in node_ids:
        for i in range(window):
            ts = now.timestamp() - (window - i) * 300
            records.append({
                "node_id":       node_id,
                "timestamp":     pd.Timestamp(ts, unit="s", tz="UTC"),
                "tilt_deg":      float(rng.normal(0.15, 0.05)),
                "moisture_d1":   float(rng.normal(65, 8)),
                "moisture_d2":   float(rng.normal(67, 8)),
                "moisture_d3":   float(rng.normal(68, 7)),
                "moisture_d4":   float(rng.normal(69, 7)),
                "moisture_d5":   float(rng.normal(70, 7)),
                "moisture_d6":   float(rng.normal(71, 7)),
                "pore_pressure": float(rng.normal(1.8, 0.4)),
                "rainfall_1hr":  float(rng.exponential(5)),
                "battery_pct":   int(rng.integers(70, 95)),
                "rssi_dbm":      int(rng.integers(-110, -80)),
            })

    df = pd.DataFrame(records)
    # Add physics labels (needed for feature engineering)
    df["factor_of_safety"] = 1.5   # stub FS
    df["label"]            = "GREEN"
    df["label_int"]        = 0

    return engineer_features(df)


# ── Main loop ─────────────────────────────────────────────────────────────

def run(
    rf_path:     Path,
    cnn_path:    Path,
    node_ids:    list[str],
    interval_sec: int  = 300,
    led_port:    str   = None,
):
    logger.info("Starting Resilio-Route edge inference loop...")
    logger.info(f"Nodes: {node_ids}")
    logger.info(f"Interval: {interval_sec}s | LED port: {led_port or 'mock'}")

    nav = Navigator.load(rf_path, cnn_path)
    led = LEDBoard(led_port or "/dev/null")
    if led_port:
        led.connect()

    while True:
        cycle_start = time.time()
        logger.info(f"\n{'─'*40}")
        logger.info(f"Inference cycle: {datetime.now(timezone.utc).isoformat()}")

        try:
            # Read latest window from LoRa buffer
            df = read_lora_buffer(node_ids)

            # Predict per node
            results = nav.predict_batch(df)

            for node_id, result in results:
                logger.info(f"{node_id}: {result}")

                # Update LED board
                led.send(str(node_id), result.label, result.confidence)

                # Push to cloud
                push_to_cloud({
                    "node_id":       str(node_id),
                    "timestamp":     datetime.now(timezone.utc).isoformat(),
                    "classification": result.label,
                    "probabilities": result.to_dict(),
                })

        except Exception as e:
            logger.error(f"Inference cycle failed: {e}")

        elapsed = time.time() - cycle_start
        sleep_time = max(0, interval_sec - elapsed)
        logger.debug(f"Cycle took {elapsed:.1f}s. Sleeping {sleep_time:.0f}s...")
        time.sleep(sleep_time)


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resilio-Route edge inference loop")
    parser.add_argument("--rf-model",  type=Path, default=Path("ml/models/rf_v1.pkl"))
    parser.add_argument("--cnn-model", type=Path, default=Path("ml/models/cnn_v1.pt"))
    parser.add_argument("--nodes",     type=str,  default="NM-01,NM-02,NM-03",
                        help="Comma-separated node IDs")
    parser.add_argument("--interval",  type=int,  default=300, help="Seconds between cycles")
    parser.add_argument("--led-port",  type=str,  default=None, help="Serial port for LED board")
    args = parser.parse_args()

    run(
        rf_path      = args.rf_model,
        cnn_path     = args.cnn_model,
        node_ids     = args.nodes.split(","),
        interval_sec = args.interval,
        led_port     = args.led_port,
    )
