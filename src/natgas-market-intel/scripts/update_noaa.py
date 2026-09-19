"""Daily NOAA CPC degree-day update.

Run from the project root with:
    python scripts/update_noaa.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from providers.noaa_provider import NOAAProvider
from models.storage_balance import run_model
from scripts.backfill_noaa import (
    DATABASE_PATH,
    create_schema,
    prepare_forecast_rows,
    prepare_historical_rows,
    prepare_normal_rows,
    upsert_rows,
)


def update_noaa() -> None:
    """Refresh the current year of realized data and save today's forecast vintage."""
    provider = NOAAProvider()
    current_year = pd.Timestamp.today().year

    with duckdb.connect(str(DATABASE_PATH)) as connection:
        create_schema(connection)

        realized = provider.fetch_historical_with_normals(current_year, current_year)
        if realized.empty:
            print("Realized degree days: no rows returned")
        else:
            rows = prepare_historical_rows(realized)
            upsert_rows(connection, "noaa_degree_days_daily", rows)
            print(f"Realized degree days: upserted {len(rows):,} rows")

        # Normals change rarely, but upserting them keeps a new database complete.
        normals = provider.fetch_normals()
        normal_rows = prepare_normal_rows(normals)
        upsert_rows(connection, "noaa_degree_day_normals", normal_rows)
        print(f"Normals: upserted {len(normal_rows):,} rows")

        forecast = provider.fetch_forecast_with_normals()
        forecast_rows = prepare_forecast_rows(forecast)
        upsert_rows(connection, "noaa_degree_day_forecasts", forecast_rows)
        print(f"Forecast: upserted {len(forecast_rows):,} rows")

    # Rebuild after the writer connection closes so the derived signal stays current.
    try:
        run_model()
    except Exception as error:
        print(f"Storage balance model: skipped ({error})")


if __name__ == "__main__":
    update_noaa()
