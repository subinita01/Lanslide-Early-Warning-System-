"""
tools/monitor.py
────────────────
Live terminal monitoring dashboard for the deployed Navigator.
Shows node status, recent predictions, alert history, and model health.

Run:
    python tools/monitor.py                     # refresh every 30s
    python tools/monitor.py --interval 10       # refresh every 10s
    python tools/monitor.py --once              # single snapshot
"""

import argparse
import json
import time
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────
API_BASE   = "http://localhost:8000"
API_KEY    = "rr_dev_key"      # reads from .env in production
HEADERS    = {"Authorization": f"Bearer {API_KEY}"}


# ── Formatting helpers ────────────────────────────────────────────────────

def label_icon(label: str) -> str:
    return {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(label, "⚪")

def battery_bar(pct: int) -> str:
    filled = int(pct / 10)
    return "█" * filled + "░" * (10 - filled) + f" {pct}%"

def clear():
    os.system("cls" if os.name == "nt" else "clear")


# ── API calls ─────────────────────────────────────────────────────────────

def fetch_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/v1/health", timeout=3)
        return r.json() if r.ok else {}
    except Exception:
        return {}

def fetch_nodes() -> list:
    try:
        r = requests.get(f"{API_BASE}/v1/nodes/status", headers=HEADERS, timeout=5)
        return r.json().get("nodes", []) if r.ok else []
    except Exception:
        return []

def fetch_segment(highway: str, km_from: float, km_to: float) -> dict:
    try:
        r = requests.get(
            f"{API_BASE}/v1/risk/segment",
            headers=HEADERS,
            params={"highway": highway, "km_from": km_from, "km_to": km_to},
            timeout=5,
        )
        return r.json() if r.ok else {}
    except Exception:
        return {}


# ── Dashboard renderer ────────────────────────────────────────────────────

def render(nodes: list, health: dict):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           RESILIO-ROUTE  —  Navigator Dashboard             ║")
    print(f"║  {now}                                     ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    # System health
    status = health.get("status", "unreachable")
    ver    = health.get("model_version", "—")
    n_on   = health.get("nodes_online", "—")
    health_icon = "✅" if status == "ok" else "❌"
    print(f"║  System: {health_icon} {status.upper():<8} Model v{ver:<8} Nodes online: {n_on:<4}  ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    if not nodes:
        print("║  No nodes available — is the API running?                   ║")
        print("║  Run: uvicorn backend.api.main:app --reload                 ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        return

    # Node table header
    print("║  Node    KM       Status    P(RED)   Last reading              ║")
    print("║  ─────────────────────────────────────────────────────────     ║")

    for node in nodes:
        nid    = node.get("node_id", "—")
        km     = node.get("km", 0)
        label  = node.get("classification", "—")
        p_red  = node.get("p_red", 0.0)
        icon   = label_icon(label)
        ts     = node.get("last_reading", "")[:19].replace("T", " ")

        # Alert if RED
        alert = " ⚠️  ALERT" if label == "RED" else ""
        print(f"║  {nid:<8} {km:<8.1f} {icon} {label:<8} {p_red:.3f}   {ts}{alert:<12}  ║")

    print("╠══════════════════════════════════════════════════════════════╣")

    # Summary counts
    red_count    = sum(1 for n in nodes if n.get("classification") == "RED")
    yellow_count = sum(1 for n in nodes if n.get("classification") == "YELLOW")
    green_count  = sum(1 for n in nodes if n.get("classification") == "GREEN")
    print(f"║  Summary:  🟢 {green_count} safe   🟡 {yellow_count} caution   🔴 {red_count} danger        ║")

    if red_count > 0:
        print("║                                                              ║")
        print("║  ⚠️  RED ALERT ACTIVE — check LED boards and notify fleet!   ║")

    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  Ctrl+C to exit  |  Refreshing automatically...")


# ── Main loop ─────────────────────────────────────────────────────────────

def run(interval: int = 30, once: bool = False):
    while True:
        clear()
        health = fetch_health()
        nodes  = fetch_nodes()
        render(nodes, health)

        if once:
            break
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resilio-Route live monitor")
    parser.add_argument("--interval", type=int,  default=30, help="Refresh interval in seconds")
    parser.add_argument("--once",     action="store_true",   help="Single snapshot then exit")
    args = parser.parse_args()
    run(interval=args.interval, once=args.once)
