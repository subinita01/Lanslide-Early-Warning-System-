from datetime import datetime, timezone

from fastapi.testclient import TestClient
import pytest

from backend.api.main import app as api_app
from backend.db import Alert, Prediction, SensorReading, SessionLocal, Webhook
from backend.ingest.main import app as ingest_app

API_HEADERS = {"Authorization": "Bearer rr_dev_key"}
INGEST_HEADERS = {"X-Gateway-Key": "rr_dev_key"}

pytestmark = pytest.mark.integration


def clean_test_rows() -> None:
    with SessionLocal() as db:
        db.query(Alert).delete()
        db.query(Prediction).delete()
        db.query(SensorReading).delete()
        db.query(Webhook).delete()
        db.commit()


def test_ingest_to_public_api_flow():
    clean_test_rows()
    ingest = TestClient(ingest_app)
    api = TestClient(api_app)
    ts = int(datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc).timestamp())

    ingest_resp = ingest.post(
        "/v1/ingest",
        headers=INGEST_HEADERS,
        json={
            "node_id": "NM-01",
            "highway": "NH-306",
            "km_marker": 120.0,
            "timestamp": ts,
            "tilt_deg": 0.4,
            "moisture_d1": 70,
            "moisture_d2": 71,
            "moisture_d3": 72,
            "moisture_d4": 73,
            "moisture_d5": 74,
            "moisture_d6": 75,
            "pore_pressure": 2.0,
            "rainfall_1hr": 8.0,
            "battery_pct": 91,
            "rssi_dbm": -90,
        },
    )
    assert ingest_resp.status_code == 201

    with SessionLocal() as db:
        db.add(
            Prediction(
                node_id="NM-01",
                timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
                label="RED",
                label_int=2,
                p_green=0.05,
                p_yellow=0.12,
                p_red=0.83,
                confidence=0.83,
                model_version="0.1.0",
            )
        )
        db.commit()

    nodes_resp = api.get("/v1/nodes/status", headers=API_HEADERS)
    assert nodes_resp.status_code == 200
    nodes = nodes_resp.json()["nodes"]
    assert nodes[0]["node_id"] == "NM-01"
    assert nodes[0]["classification"] == "RED"

    segment_resp = api.get(
        "/v1/risk/segment",
        headers=API_HEADERS,
        params={"highway": "NH-306", "km_from": 100, "km_to": 130},
    )
    assert segment_resp.status_code == 200
    assert segment_resp.json()["classification"] == "RED"


def test_webhook_registration_persists():
    clean_test_rows()
    api = TestClient(api_app)
    resp = api.post(
        "/v1/webhooks/register",
        headers=API_HEADERS,
        json={
            "url": "https://example.test/hook",
            "events": ["RED_ALERT"],
            "highways": ["NH-306"],
            "secret": "secret",
        },
    )
    assert resp.status_code == 201
    with SessionLocal() as db:
        webhook = db.query(Webhook).filter_by(url="https://example.test/hook").one()
        assert webhook.events == ["RED_ALERT"]
