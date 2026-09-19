"""Daily EIA natural-gas update with a short revision window.

Run from the project root with:
    python scripts/update_eia.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from providers.eia_provider import EIAProvider
from models.storage_balance import run_model
from scripts.backfill_eia import (
    DATABASE_PATH,
    SERIES_CONFIG,
    create_schema,
    prepare_rows,
    upsert_rows,
)


def revision_start(config: dict) -> str:
    """Return a conservative window so EIA revisions are captured on daily runs."""
    today = pd.Timestamp.today()
    if config["frequency"] == "weekly":
        return (today - pd.Timedelta(days=90)).date().isoformat()
    return (today - pd.DateOffset(months=24)).strftime("%Y-%m")


def update_eia() -> None:
    """Upsert recent observations for every existing EIA natural-gas series."""
    provider = EIAProvider()

    with duckdb.connect(str(DATABASE_PATH)) as connection:
        create_schema(connection)
        print("Updating EIA natural-gas series")

        for config in SERIES_CONFIG:
            start = revision_start(config)
            try:
                data = provider.fetch_series(
                    route=config["route"],
                    series_id=config["series_id"],
                    frequency=config["frequency"],
                    start=start,
                )
            except Exception as error:
                print(f"{config['metric']}: skipped ({error})")
                continue

            if data.empty:
                print(f"{config['metric']}: no rows")
                continue

            rows = prepare_rows(data, config)
            upsert_rows(connection, rows)
            print(f"{config['metric']}: upserted {len(rows)} rows since {start}")

    # Rebuild after the writer connection closes so the derived signal stays current.
    try:
        run_model()
    except Exception as error:
        print(f"Storage balance model: skipped ({error})")


if __name__ == "__main__":
    update_eia()
