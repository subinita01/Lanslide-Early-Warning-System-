"""Initial operational schema.

Revision ID: 20260517_0001
Revises:
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260517_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS timescaledb')
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    op.create_table(
        "sensor_readings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("node_id", sa.String(length=16), nullable=False),
        sa.Column("highway", sa.String(length=16), nullable=False, server_default="UNKNOWN"),
        sa.Column("km_marker", sa.Float(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("tilt_deg", sa.Float(), nullable=True),
        sa.Column("tilt_rate", sa.Float(), nullable=True),
        sa.Column("moisture_d1", sa.Float(), nullable=True),
        sa.Column("moisture_d2", sa.Float(), nullable=True),
        sa.Column("moisture_d3", sa.Float(), nullable=True),
        sa.Column("moisture_d4", sa.Float(), nullable=True),
        sa.Column("moisture_d5", sa.Float(), nullable=True),
        sa.Column("moisture_d6", sa.Float(), nullable=True),
        sa.Column("pore_pressure", sa.Float(), nullable=True),
        sa.Column("rainfall_1hr", sa.Float(), nullable=True),
        sa.Column("battery_pct", sa.SmallInteger(), nullable=True),
        sa.Column("rssi_dbm", sa.SmallInteger(), nullable=True),
        sa.Column("sequence_num", sa.SmallInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sensor_readings_node_id", "sensor_readings", ["node_id"], unique=False)
    op.create_index("ix_sensor_readings_highway", "sensor_readings", ["highway"], unique=False)
    op.create_index("ix_sensor_readings_timestamp", "sensor_readings", ["timestamp"], unique=False)

    op.create_table(
        "predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("node_id", sa.String(length=16), nullable=False),
        sa.Column("segment", sa.String(length=32), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("label", sa.String(length=8), nullable=False),
        sa.Column("label_int", sa.SmallInteger(), nullable=False),
        sa.Column("p_green", sa.Float(), nullable=True),
        sa.Column("p_yellow", sa.Float(), nullable=True),
        sa.Column("p_red", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("rf_p_green", sa.Float(), nullable=True),
        sa.Column("rf_p_yellow", sa.Float(), nullable=True),
        sa.Column("rf_p_red", sa.Float(), nullable=True),
        sa.Column("cnn_p_green", sa.Float(), nullable=True),
        sa.Column("cnn_p_yellow", sa.Float(), nullable=True),
        sa.Column("cnn_p_red", sa.Float(), nullable=True),
        sa.Column("model_version", sa.String(length=16), nullable=True),
        sa.Column("inference_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_predictions_node_id", "predictions", ["node_id"], unique=False)
    op.create_index("ix_predictions_timestamp", "predictions", ["timestamp"], unique=False)

    op.create_table(
        "alerts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("node_id", sa.String(length=16), nullable=False),
        sa.Column("segment", sa.String(length=32), nullable=True),
        sa.Column("highway", sa.String(length=16), nullable=True),
        sa.Column("km_from", sa.Float(), nullable=True),
        sa.Column("km_to", sa.Float(), nullable=True),
        sa.Column("label", sa.String(length=8), nullable=False),
        sa.Column("p_red", sa.Float(), nullable=True),
        sa.Column("lead_time_min", sa.Integer(), nullable=True),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cleared_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acknowledged_by", sa.String(length=64), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_alerts_node_id", "alerts", ["node_id"], unique=False)
    op.create_index("ix_alerts_highway", "alerts", ["highway"], unique=False)

    op.create_table(
        "labeled_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("node_id", sa.String(length=16), nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("label", sa.String(length=8), nullable=False),
        sa.Column("label_int", sa.SmallInteger(), nullable=False),
        sa.Column("label_source", sa.String(length=32), nullable=False),
        sa.Column("factor_of_safety", sa.Float(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("labeled_by", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "webhooks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("secret", sa.String(length=128), nullable=True),
        sa.Column("events", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("highways", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_triggered", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("url"),
    )

    op.execute(
        """
        CREATE OR REPLACE VIEW latest_predictions AS
        SELECT DISTINCT ON (node_id)
            node_id, label, p_red, confidence, timestamp, model_version
        FROM predictions
        ORDER BY node_id, timestamp DESC
        """
    )
    op.execute(
        """
        CREATE OR REPLACE VIEW active_red_alerts AS
        SELECT *
        FROM alerts
        WHERE label = 'RED' AND cleared_at IS NULL
        ORDER BY issued_at DESC
        """
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS active_red_alerts")
    op.execute("DROP VIEW IF EXISTS latest_predictions")
    op.drop_table("webhooks")
    op.drop_table("labeled_events")
    op.drop_index("ix_alerts_highway", table_name="alerts")
    op.drop_index("ix_alerts_node_id", table_name="alerts")
    op.drop_table("alerts")
    op.drop_index("ix_predictions_timestamp", table_name="predictions")
    op.drop_index("ix_predictions_node_id", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_sensor_readings_timestamp", table_name="sensor_readings")
    op.drop_index("ix_sensor_readings_highway", table_name="sensor_readings")
    op.drop_index("ix_sensor_readings_node_id", table_name="sensor_readings")
    op.drop_table("sensor_readings")
