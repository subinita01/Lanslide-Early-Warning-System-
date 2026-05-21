"""
Canonical SQLAlchemy models and session helpers for Resilio-Route.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Generator
from uuid import uuid4

from sqlalchemy import ARRAY, Boolean, Column, DateTime, Float, Integer, SmallInteger, String, Text, create_engine
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config.settings import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=(settings.APP_ENV == "development"),
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class SensorReading(Base):
    __tablename__ = "sensor_readings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    node_id = Column(String(16), nullable=False, index=True)
    highway = Column(String(16), nullable=False, default="UNKNOWN", index=True)
    km_marker = Column(Float)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    tilt_deg = Column(Float)
    tilt_rate = Column(Float)
    moisture_d1 = Column(Float)
    moisture_d2 = Column(Float)
    moisture_d3 = Column(Float)
    moisture_d4 = Column(Float)
    moisture_d5 = Column(Float)
    moisture_d6 = Column(Float)
    pore_pressure = Column(Float)
    rainfall_1hr = Column(Float)
    battery_pct = Column(SmallInteger)
    rssi_dbm = Column(SmallInteger)
    sequence_num = Column(SmallInteger)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    node_id = Column(String(16), nullable=False, index=True)
    segment = Column(String(32))
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    label = Column(String(8), nullable=False)
    label_int = Column(SmallInteger, nullable=False)
    p_green = Column(Float)
    p_yellow = Column(Float)
    p_red = Column(Float)
    confidence = Column(Float)
    rf_p_green = Column(Float)
    rf_p_yellow = Column(Float)
    rf_p_red = Column(Float)
    cnn_p_green = Column(Float)
    cnn_p_yellow = Column(Float)
    cnn_p_red = Column(Float)
    model_version = Column(String(16))
    inference_ms = Column(Integer)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    node_id = Column(String(16), nullable=False, index=True)
    segment = Column(String(32))
    highway = Column(String(16), index=True)
    km_from = Column(Float)
    km_to = Column(Float)
    label = Column(String(8), nullable=False)
    p_red = Column(Float)
    lead_time_min = Column(Integer)
    issued_at = Column(DateTime(timezone=True), default=utcnow)
    cleared_at = Column(DateTime(timezone=True))
    acknowledged_by = Column(String(64))
    notes = Column(Text)


class LabeledEvent(Base):
    __tablename__ = "labeled_events"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    node_id = Column(String(16), nullable=False)
    event_time = Column(DateTime(timezone=True), nullable=False)
    label = Column(String(8), nullable=False)
    label_int = Column(SmallInteger, nullable=False)
    label_source = Column(String(32), nullable=False)
    factor_of_safety = Column(Float)
    notes = Column(Text)
    labeled_by = Column(String(64))
    created_at = Column(DateTime(timezone=True), default=utcnow)


class Webhook(Base):
    __tablename__ = "webhooks"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    url = Column(Text, nullable=False, unique=True)
    secret = Column(String(128))
    events = Column(ARRAY(Text), nullable=False)
    highways = Column(ARRAY(Text), nullable=False)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    last_triggered = Column(DateTime(timezone=True))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
