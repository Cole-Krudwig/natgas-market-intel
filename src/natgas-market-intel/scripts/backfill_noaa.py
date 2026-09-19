"""One-time historical NOAA CPC degree-day backfill.

Run from the project root with:
    python scripts/backfill_noaa.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from providers.noaa_provider import NOAAProvider


DATABASE_PATH = PROJECT_ROOT / "data" / "commodities.duckdb"
START_YEAR = 2000


def create_schema(connection: duckdb.DuckDBPyConnection) -> None:
    """Create the tables used for realized degree days, normals, and forecasts."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS noaa_degree_days_daily (
            date DATE NOT NULL,
            region_code VARCHAR NOT NULL,
            region VARCHAR,
            hdd DOUBLE,
            cdd DOUBLE,
            normal_hdd DOUBLE,
            normal_cdd DOUBLE,
            hdd_anomaly DOUBLE,
            cdd_anomaly DOUBLE,
            source VARCHAR NOT NULL,
            ingested_at TIMESTAMP NOT NULL,
            PRIMARY KEY (date, region_code)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS noaa_degree_day_normals (
            month INTEGER NOT NULL,
            day INTEGER NOT NULL,
            region_code VARCHAR NOT NULL,
            region VARCHAR,
            normal_hdd DOUBLE,
            normal_cdd DOUBLE,
            source VARCHAR NOT NULL,
            ingested_at TIMESTAMP NOT NULL,
            PRIMARY KEY (month, day, region_code)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS noaa_degree_day_forecasts (
            forecast_date DATE NOT NULL,
            target_date DATE NOT NULL,
            region_code VARCHAR NOT NULL,
            region VARCHAR,
            forecast_hdd DOUBLE,
            forecast_cdd DOUBLE,
            normal_hdd DOUBLE,
            normal_cdd DOUBLE,
            hdd_anomaly DOUBLE,
            cdd_anomaly DOUBLE,
            source VARCHAR NOT NULL,
            ingested_at TIMESTAMP NOT NULL,
            PRIMARY KEY (forecast_date, target_date, region_code)
        )
        """
    )


def _with_metadata(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    output["source"] = "NOAA CPC"
    output["ingested_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    return output


def prepare_historical_rows(data: pd.DataFrame) -> pd.DataFrame:
    rows = _with_metadata(data)
    rows["date"] = pd.to_datetime(rows["date"]).dt.date
    return rows[
        [
            "date", "region_code", "region", "hdd", "cdd", "normal_hdd",
            "normal_cdd", "hdd_anomaly", "cdd_anomaly", "source", "ingested_at",
        ]
    ]


def prepare_normal_rows(data: pd.DataFrame) -> pd.DataFrame:
    rows = _with_metadata(data)
    return rows[
        [
            "month", "day", "region_code", "region", "normal_hdd", "normal_cdd",
            "source", "ingested_at",
        ]
    ]


def prepare_forecast_rows(data: pd.DataFrame) -> pd.DataFrame:
    rows = _with_metadata(data)
    rows["forecast_date"] = pd.to_datetime(rows["forecast_date"]).dt.date
    rows["target_date"] = pd.to_datetime(rows["target_date"]).dt.date
    return rows[
        [
            "forecast_date", "target_date", "region_code", "region", "forecast_hdd",
            "forecast_cdd", "normal_hdd", "normal_cdd", "hdd_anomaly", "cdd_anomaly",
            "source", "ingested_at",
        ]
    ]


def upsert_rows(
    connection: duckdb.DuckDBPyConnection, table: str, rows: pd.DataFrame
) -> None:
    if rows.empty:
        return
    connection.register("noaa_rows", rows)
    try:
        connection.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM noaa_rows")
    finally:
        connection.unregister("noaa_rows")


def backfill_noaa(start_year: int = START_YEAR) -> None:
    """Load all available daily CPC degree days plus the static normal calendar."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    provider = NOAAProvider()
    end_year = pd.Timestamp.today().year

    with duckdb.connect(str(DATABASE_PATH)) as connection:
        create_schema(connection)
        print(f"Fetching NOAA CPC realized degree days ({start_year}-{end_year})")
        historical = provider.fetch_historical_with_normals(start_year, end_year)
        if historical.empty:
            print("Realized degree days: no rows returned")
        else:
            rows = prepare_historical_rows(historical)
            upsert_rows(connection, "noaa_degree_days_daily", rows)
            print(f"Realized degree days: upserted {len(rows):,} rows")

        normals = provider.fetch_normals()
        normal_rows = prepare_normal_rows(normals)
        upsert_rows(connection, "noaa_degree_day_normals", normal_rows)
        print(f"Normals: upserted {len(normal_rows):,} rows")
        print("NOAA historical backfill complete")


if __name__ == "__main__":
    backfill_noaa()
