"""Risk Intelligence API."""

from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.db import Alert, Prediction, SensorReading, Webhook, get_db
from config.settings import settings

app = FastAPI(
    title="Resilio-Route Risk Intelligence API",
    description="Real-time landslide risk scores for NE India highway corridors.",
    version=settings.MODEL_VERSION,
)


def verify_api_key(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    key = authorization.removeprefix("Bearer ").strip()
    if key != settings.CLOUD_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key


class RiskResponse(BaseModel):
    segment: dict
    classification: str
    probabilities: dict
    confidence: float
    updated_at: str
    lead_time_min: int
    nearest_node: str
    active_alerts: list


class HealthResponse(BaseModel):
    status: str
    model_version: str
    nodes_online: int
    timestamp: str


class WebhookRegisterRequest(BaseModel):
    url: str
    events: list[str]
    highways: list[str]
    secret: str


def latest_prediction_rows(db: Session):
    latest_pred_ts = (
        db.query(Prediction.node_id, func.max(Prediction.timestamp).label("timestamp"))
        .group_by(Prediction.node_id)
        .subquery()
    )
    latest_reading_ts = (
        db.query(SensorReading.node_id, func.max(SensorReading.timestamp).label("timestamp"))
        .group_by(SensorReading.node_id)
        .subquery()
    )
    return (
        db.query(Prediction, SensorReading)
        .join(
            latest_pred_ts,
            (Prediction.node_id == latest_pred_ts.c.node_id)
            & (Prediction.timestamp == latest_pred_ts.c.timestamp),
        )
        .join(latest_reading_ts, Prediction.node_id == latest_reading_ts.c.node_id)
        .join(
            SensorReading,
            (SensorReading.node_id == latest_reading_ts.c.node_id)
            & (SensorReading.timestamp == latest_reading_ts.c.timestamp),
        )
    )


@app.get("/v1/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)):
    try:
        nodes_online = latest_prediction_rows(db).count()
    except SQLAlchemyError as exc:
        logger.warning(f"Health DB check failed: {exc}")
        nodes_online = 0
    return HealthResponse(
        status="ok",
        model_version=settings.MODEL_VERSION,
        nodes_online=nodes_online,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/v1/risk/segment", response_model=RiskResponse)
def risk_segment(
    highway: str,
    km_from: float,
    km_to: float,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
):
    nodes_in_segment = (
        latest_prediction_rows(db)
        .filter(
            SensorReading.highway == highway,
            SensorReading.km_marker >= km_from,
            SensorReading.km_marker <= km_to,
        )
        .all()
    )
    if not nodes_in_segment:
        raise HTTPException(status_code=404, detail=f"No nodes found on {highway} km {km_from}-{km_to}")

    worst_pred, _reading = max(nodes_in_segment, key=lambda row: row[0].p_red or 0.0)
    alerts = (
        db.query(Alert)
        .filter(
            Alert.highway == highway,
            Alert.cleared_at.is_(None),
            Alert.km_from <= km_to,
            Alert.km_to >= km_from,
        )
        .order_by(Alert.issued_at.desc())
        .all()
    )
    logger.info(f"Segment query: {highway} km {km_from}-{km_to} -> {worst_pred.label}")
    return RiskResponse(
        segment={"highway": highway, "km_from": km_from, "km_to": km_to},
        classification=worst_pred.label,
        probabilities={
            "GREEN": worst_pred.p_green or 0.0,
            "YELLOW": worst_pred.p_yellow or 0.0,
            "RED": worst_pred.p_red or 0.0,
        },
        confidence=worst_pred.confidence
        or max(worst_pred.p_green or 0.0, worst_pred.p_yellow or 0.0, worst_pred.p_red or 0.0),
        updated_at=worst_pred.timestamp.isoformat(),
        lead_time_min=settings.ALERT_LEAD_TIME_MIN,
        nearest_node=worst_pred.node_id,
        active_alerts=[
            {
                "node_id": alert.node_id,
                "label": alert.label,
                "p_red": alert.p_red,
                "issued_at": alert.issued_at.isoformat() if alert.issued_at else None,
            }
            for alert in alerts
        ],
    )


@app.post("/v1/webhooks/register", status_code=201)
def register_webhook(
    payload: WebhookRegisterRequest,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
):
    webhook = Webhook(url=payload.url, secret=payload.secret, events=payload.events, highways=payload.highways)
    db.add(webhook)
    db.commit()
    db.refresh(webhook)
    return {
        "status": "registered",
        "webhook_id": str(webhook.id),
        "events": payload.events,
        "highways": payload.highways,
        "registered_at": webhook.created_at.isoformat() if webhook.created_at else datetime.now(timezone.utc).isoformat(),
    }


@app.get("/v1/nodes/status")
def nodes_status(db: Session = Depends(get_db), _key: str = Depends(verify_api_key)):
    rows = latest_prediction_rows(db).all()
    return {
        "nodes": [
            {
                "node_id": pred.node_id,
                "classification": pred.label,
                "p_red": pred.p_red,
                "km": reading.km_marker,
                "highway": reading.highway,
                "status": "online",
                "last_reading": reading.timestamp.isoformat(),
            }
            for pred, reading in rows
        ],
        "total_online": len(rows),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
