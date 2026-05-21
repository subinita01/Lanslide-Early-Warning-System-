"""Internal ingest API for gateway sensor readings."""

from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from backend.db import SensorReading, get_db
from config.settings import settings

app = FastAPI(title="Resilio-Route Ingest API", version=settings.MODEL_VERSION)


def verify_gateway_key(x_gateway_key: str = Header(...)):
    if x_gateway_key != settings.CLOUD_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid gateway key")
    return x_gateway_key


class SensorPayload(BaseModel):
    node_id: str = Field(..., examples=["NM-07"])
    highway: str = Field(default="UNKNOWN", max_length=16)
    km_marker: float | None = None
    timestamp: int = Field(..., description="Unix epoch seconds")
    tilt_deg: float = Field(..., ge=0.0, le=5.0)
    moisture_d1: float = Field(..., ge=0.0, le=100.0)
    moisture_d2: float = Field(..., ge=0.0, le=100.0)
    moisture_d3: float = Field(..., ge=0.0, le=100.0)
    moisture_d4: float = Field(..., ge=0.0, le=100.0)
    moisture_d5: float = Field(..., ge=0.0, le=100.0)
    moisture_d6: float = Field(..., ge=0.0, le=100.0)
    pore_pressure: float = Field(..., ge=0.0, le=100.0)
    rainfall_1hr: float = Field(..., ge=0.0, le=200.0)
    battery_pct: int = Field(..., ge=0, le=100)
    rssi_dbm: int = Field(..., ge=-130, le=0)

    @field_validator("node_id")
    @classmethod
    def node_id_format(cls, value: str) -> str:
        if not value.startswith("NM-"):
            raise ValueError("node_id must start with NM-")
        return value


class BatchPayload(BaseModel):
    readings: list[SensorPayload]


def build_reading(payload: SensorPayload) -> SensorReading:
    return SensorReading(
        node_id=payload.node_id,
        highway=payload.highway,
        km_marker=payload.km_marker,
        timestamp=datetime.fromtimestamp(payload.timestamp, tz=timezone.utc),
        tilt_deg=payload.tilt_deg,
        moisture_d1=payload.moisture_d1,
        moisture_d2=payload.moisture_d2,
        moisture_d3=payload.moisture_d3,
        moisture_d4=payload.moisture_d4,
        moisture_d5=payload.moisture_d5,
        moisture_d6=payload.moisture_d6,
        pore_pressure=payload.pore_pressure,
        rainfall_1hr=payload.rainfall_1hr,
        battery_pct=payload.battery_pct,
        rssi_dbm=payload.rssi_dbm,
    )


@app.post("/v1/ingest", status_code=201)
def ingest_single(payload: SensorPayload, db: Session = Depends(get_db), _key: str = Depends(verify_gateway_key)):
    reading = build_reading(payload)
    db.add(reading)
    db.commit()
    logger.debug(f"Ingested: {payload.node_id} @ {reading.timestamp.isoformat()}")
    return {"status": "ok", "node_id": payload.node_id, "timestamp": reading.timestamp.isoformat()}


@app.post("/v1/ingest/batch", status_code=201)
def ingest_batch(payload: BatchPayload, db: Session = Depends(get_db), _key: str = Depends(verify_gateway_key)):
    readings = [build_reading(reading) for reading in payload.readings]
    db.bulk_save_objects(readings)
    db.commit()
    logger.info(f"Batch ingested: {len(readings)} readings")
    return {"status": "ok", "count": len(readings)}


@app.get("/v1/ingest/health")
def health():
    return {"status": "ok", "service": "ingest"}
