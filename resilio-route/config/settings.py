"""
config/settings.py
──────────────────
Central settings loaded from .env  (or environment variables in production).
Import this everywhere instead of reading os.environ directly.

Usage:
    from config.settings import settings
    print(settings.RED_THRESHOLD)
"""

from pydantic_settings import BaseSettings
from pathlib import Path

ROOT = Path(__file__).parent.parent   # repo root


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://resilio:password@localhost:5432/resilio_route"

    # ── API ───────────────────────────────────────────────────────────────
    CLOUD_API_KEY: str = "rr_dev_key"
    CLOUD_API_BASE_URL: str = "http://localhost:8001"
    NHIDCL_WEBHOOK_SECRET: str = "whsec_dev"

    # ── Model ─────────────────────────────────────────────────────────────
    MODEL_VERSION: str = "0.1.0"
    RED_THRESHOLD: float = 0.70
    YELLOW_THRESHOLD: float = 0.40
    ALERT_LEAD_TIME_MIN: int = 30

    # ── Node network ──────────────────────────────────────────────────────
    LORA_FREQUENCY: float = 865.0
    SAMPLE_INTERVAL_SEC: int = 300
    GATEWAY_UPLINK_INTERVAL_SEC: int = 60

    # ── Soil parameters (defaults for NE India) ───────────────────────────
    SOIL_COHESION_KPA: float = 8.0
    SOIL_FRICTION_DEG: float = 28.0
    SOIL_UNIT_WEIGHT: float = 19.0
    SOIL_FAILURE_DEPTH_M: float = 1.5

    # ── Paths ─────────────────────────────────────────────────────────────
    DATA_DIR:    Path = ROOT / "ml" / "data"
    MODELS_DIR:  Path = ROOT / "ml" / "models"
    REPORTS_DIR: Path = ROOT / "ml" / "reports"

    # ── App ───────────────────────────────────────────────────────────────
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
