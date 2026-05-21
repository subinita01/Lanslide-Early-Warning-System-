"""
tools/deploy_check.py
─────────────────────
Pre-deployment checklist.  Run this before activating the LED boards
and sending real alerts to drivers.  Every check must PASS.

Usage:
    python tools/deploy_check.py \
        --rf-model  ml/models/rf_v1.pkl \
        --cnn-model ml/models/cnn_v1.pt \
        --api-url   http://localhost:8000

Exit codes:
    0  — all checks passed (safe to deploy)
    1  — one or more checks failed (do NOT deploy)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests
import joblib
import numpy as np

from loguru import logger


# ── Individual checks ─────────────────────────────────────────────────────

def check_rf_model(rf_path: Path) -> tuple[bool, str]:
    """RF model must exist and have a valid RED recall in its eval report."""
    if not rf_path.exists():
        return False, f"Model file not found: {rf_path}"

    report_path = rf_path.parent / "rf_eval.json"
    if not report_path.exists():
        return False, "rf_eval.json not found — run ml/train_rf.py first"

    with open(report_path) as f:
        report = json.load(f)

    recall = report.get("red_recall", 0)
    if recall < 0.70:
        return False, f"RED recall {recall:.3f} < 0.70 threshold"

    return True, f"RED recall = {recall:.3f}"


def check_cnn_model(cnn_path: Path) -> tuple[bool, str]:
    """CNN model must exist and have a valid RED recall."""
    if not cnn_path.exists():
        return False, f"Model file not found: {cnn_path}"

    report_path = cnn_path.parent / "cnn_eval.json"
    if not report_path.exists():
        return False, "cnn_eval.json not found — run ml/train_cnn.py first"

    with open(report_path) as f:
        report = json.load(f)

    recall = report.get("red_recall", 0)
    if recall < 0.70:
        return False, f"RED recall {recall:.3f} < 0.70 threshold"

    return True, f"RED recall = {recall:.3f}"


def check_evaluation_report(report_path: Path) -> tuple[bool, str]:
    """Full evaluation report must exist and all deployment criteria must pass."""
    if not report_path.exists():
        return False, f"Evaluation report not found: {report_path} — run ml/evaluate.py"

    with open(report_path) as f:
        report = json.load(f)

    if not report.get("deploy_ready", False):
        failed = [k for k, v in report.get("passes", {}).items() if not v]
        return False, f"Deployment criteria not met: {failed}"

    red_recall  = report.get("red_recall", 0)
    lead_time   = report.get("lead_time_p10", 0)
    return True, f"RED recall={red_recall:.3f}, lead_time_P10={lead_time}min"


def check_api_health(api_url: str) -> tuple[bool, str]:
    """Risk Intelligence API must be reachable and healthy."""
    try:
        r = requests.get(f"{api_url}/v1/health", timeout=5)
        if r.ok:
            data = r.json()
            nodes = data.get("nodes_online", 0)
            return True, f"API healthy | {nodes} nodes online | v{data.get('model_version')}"
        return False, f"API returned {r.status_code}"
    except Exception as e:
        return False, f"API unreachable: {e}"


def check_api_auth(api_url: str, api_key: str) -> tuple[bool, str]:
    """Auth must work and reject bad keys."""
    # Good key
    try:
        r = requests.get(
            f"{api_url}/v1/nodes/status",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        if r.status_code != 200:
            return False, f"Valid key rejected: {r.status_code}"

        # Bad key should be rejected
        r2 = requests.get(
            f"{api_url}/v1/nodes/status",
            headers={"Authorization": "Bearer bad_key_xyz"},
            timeout=5,
        )
        if r2.status_code != 403:
            return False, f"Bad key NOT rejected (got {r2.status_code}) — security issue"

        return True, "Auth working correctly (valid accepted, invalid rejected)"
    except Exception as e:
        return False, f"Auth check failed: {e}"


def check_node_coverage(api_url: str, api_key: str,
                         min_nodes: int = 10) -> tuple[bool, str]:
    """Enough nodes must be online."""
    try:
        r = requests.get(
            f"{api_url}/v1/nodes/status",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        if not r.ok:
            return False, f"Cannot get node status: {r.status_code}"

        nodes = r.json().get("nodes", [])
        online = [n for n in nodes if n.get("status") == "online"]
        if len(online) < min_nodes:
            return False, f"Only {len(online)} nodes online (minimum {min_nodes})"

        return True, f"{len(online)} nodes online"
    except Exception as e:
        return False, f"Node check failed: {e}"


def check_env_file() -> tuple[bool, str]:
    """.env file must exist and not use template placeholder values."""
    env_path = Path(".env")
    if not env_path.exists():
        return False, ".env file missing — run: cp .env.template .env"

    content = env_path.read_text()
    if "xxxxxxxxxxxx" in content:
        return False, ".env still has placeholder values — fill in real credentials"

    return True, ".env file present and customised"


def check_model_version_match(rf_path: Path, cnn_path: Path) -> tuple[bool, str]:
    """RF and CNN must have matching model versions."""
    try:
        rf_report  = json.loads((rf_path.parent / "rf_eval.json").read_text())
        cnn_report = json.loads((cnn_path.parent / "cnn_eval.json").read_text())
        rf_ver  = rf_report.get("model_version", "unknown")
        cnn_ver = cnn_report.get("model_version", "unknown")
        if rf_ver != cnn_ver:
            return False, f"Version mismatch: RF={rf_ver}, CNN={cnn_ver}"
        return True, f"Both models at version {rf_ver}"
    except Exception as e:
        return False, f"Version check failed: {e}"


# ── Run all checks ────────────────────────────────────────────────────────

def run_all_checks(
    rf_path:     Path,
    cnn_path:    Path,
    api_url:     str,
    api_key:     str,
    min_nodes:   int = 10,
) -> bool:
    eval_report = rf_path.parent / "evaluation.json"
    # Also try the reports directory
    if not eval_report.exists():
        eval_report = Path("ml/reports/evaluation.json")

    checks = [
        ("Environment",         check_env_file,
            {}),
        ("RF model",            check_rf_model,
            {"rf_path": rf_path}),
        ("CNN model",           check_cnn_model,
            {"cnn_path": cnn_path}),
        ("Model versions match", check_model_version_match,
            {"rf_path": rf_path, "cnn_path": cnn_path}),
        ("Evaluation report",   check_evaluation_report,
            {"report_path": eval_report}),
        ("API health",          check_api_health,
            {"api_url": api_url}),
        ("API authentication",  check_api_auth,
            {"api_url": api_url, "api_key": api_key}),
        ("Node coverage",       check_node_coverage,
            {"api_url": api_url, "api_key": api_key, "min_nodes": min_nodes}),
    ]

    print()
    print("═" * 62)
    print("  RESILIO-ROUTE  —  Pre-Deployment Checklist")
    print("═" * 62)

    passed, failed = 0, 0

    for name, fn, kwargs in checks:
        try:
            ok, msg = fn(**kwargs)
        except Exception as e:
            ok, msg = False, str(e)

        icon = "✅" if ok else "❌"
        status = "PASS" if ok else "FAIL"
        print(f"  {icon}  {name:<28}  {status}  {msg}")

        if ok:
            passed += 1
        else:
            failed += 1

    print("─" * 62)
    print(f"  Results: {passed} passed, {failed} failed")

    if failed == 0:
        print()
        print("  ✅  ALL CHECKS PASSED")
        print("  Safe to activate LED boards and enable fleet webhooks.")
    else:
        print()
        print("  ❌  DEPLOYMENT BLOCKED — fix the failing checks above.")
        print("  Do NOT activate LED boards until all checks pass.")

    print("═" * 62)
    print()

    return failed == 0


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-deployment checklist")
    parser.add_argument("--rf-model",   type=Path, default=Path("ml/models/rf_v1.pkl"))
    parser.add_argument("--cnn-model",  type=Path, default=Path("ml/models/cnn_v1.pt"))
    parser.add_argument("--api-url",    type=str,  default="http://localhost:8000")
    parser.add_argument("--api-key",    type=str,  default="rr_dev_key")
    parser.add_argument("--min-nodes",  type=int,  default=10)
    args = parser.parse_args()

    all_passed = run_all_checks(
        rf_path   = args.rf_model,
        cnn_path  = args.cnn_model,
        api_url   = args.api_url,
        api_key   = args.api_key,
        min_nodes = args.min_nodes,
    )

    sys.exit(0 if all_passed else 1)
