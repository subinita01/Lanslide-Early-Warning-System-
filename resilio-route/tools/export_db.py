"""
tools/export_db.py
──────────────────
Exports raw sensor readings from TimescaleDB to a parquet file
for offline ML retraining.  Run this on the cloud server (or via
SSH tunnel) after each monsoon season.

Usage:
    python tools/export_db.py \
        --start 2024-06-01 \
        --end   2024-10-31 \
        --output ml/data/raw_monsoon_2024.parquet

    python tools/export_db.py --days 90   # last 90 days
"""

import argparse
import pandas as pd
import sqlalchemy as sa
from datetime import datetime, timezone, timedelta
from pathlib import Path
from loguru import logger

from config.settings import settings


def export(
    output_path: Path,
    start:       datetime | None = None,
    end:         datetime | None = None,
    days:        int | None = None,
    node_ids:    list[str] | None = None,
    min_rows:    int = 1000,
) -> pd.DataFrame:
    """
    Export sensor_readings from TimescaleDB to parquet.

    Parameters
    ----------
    output_path : where to write the parquet file
    start       : UTC datetime for start of window
    end         : UTC datetime for end of window (default: now)
    days        : shorthand for "last N days" (overrides start/end)
    node_ids    : filter to specific nodes (default: all)
    min_rows    : fail if fewer rows exported (sanity check)
    """

    # ── Build time window ─────────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    if days is not None:
        end   = now
        start = now - timedelta(days=days)
    else:
        end   = end   or now
        start = start or (now - timedelta(days=30))

    logger.info(f"Export window: {start.date()} → {end.date()}")

    # ── Build query ───────────────────────────────────────────────────────
    base_query = """
        SELECT
            node_id,
            timestamp,
            tilt_deg,
            moisture_d1, moisture_d2, moisture_d3,
            moisture_d4, moisture_d5, moisture_d6,
            pore_pressure,
            rainfall_1hr,
            battery_pct,
            rssi_dbm
        FROM sensor_readings
        WHERE timestamp BETWEEN :start AND :end
        {node_filter}
        ORDER BY node_id, timestamp ASC
    """

    node_filter = ""
    params = {"start": start, "end": end}

    if node_ids:
        placeholders = ", ".join(f":node_{i}" for i in range(len(node_ids)))
        node_filter  = f"AND node_id IN ({placeholders})"
        for i, nid in enumerate(node_ids):
            params[f"node_{i}"] = nid

    query = base_query.format(node_filter=node_filter)

    # ── Execute ───────────────────────────────────────────────────────────
    logger.info("Connecting to TimescaleDB...")
    engine = sa.create_engine(settings.DATABASE_URL)

    logger.info("Running query...")
    df = pd.read_sql(sa.text(query), engine, params=params,
                     parse_dates=["timestamp"])

    if len(df) < min_rows:
        logger.error(f"Only {len(df)} rows exported — expected >= {min_rows}. "
                     "Check DB connection and date range.")
        raise ValueError(f"Too few rows: {len(df)}")

    logger.info(f"Exported {len(df):,} readings | "
                f"{df['node_id'].nunique()} nodes | "
                f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    # Class distribution from battery/rssi proxy (before labeling)
    logger.info(f"Columns: {list(df.columns)}")

    # ── Save ──────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.success(f"Saved → {output_path}  ({output_path.stat().st_size / 1024:.0f} KB)")

    return df


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export sensor data from TimescaleDB")
    parser.add_argument("--output", type=Path, default=Path("ml/data/raw_export.parquet"))
    parser.add_argument("--start",  type=str,  default=None, help="YYYY-MM-DD")
    parser.add_argument("--end",    type=str,  default=None, help="YYYY-MM-DD")
    parser.add_argument("--days",   type=int,  default=None, help="Last N days")
    parser.add_argument("--nodes",  type=str,  default=None,
                        help="Comma-separated node IDs, e.g. NM-01,NM-02")
    parser.add_argument("--min-rows", type=int, default=1000)
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc) \
            if args.start else None
    end   = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) \
            if args.end else None
    nodes = args.nodes.split(",") if args.nodes else None

    export(
        output_path = args.output,
        start       = start,
        end         = end,
        days        = args.days,
        node_ids    = nodes,
        min_rows    = args.min_rows,
    )
