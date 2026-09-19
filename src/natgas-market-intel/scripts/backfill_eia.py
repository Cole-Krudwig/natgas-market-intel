"""
One-time historical EIA natural gas backfill.

Fetches:
    - Weekly Lower-48 working gas in storage
    - Monthly U.S. dry natural gas production
    - Monthly electric-power natural gas consumption
    - Monthly total U.S. natural gas exports

Stores everything in:
    data/commodities.duckdb

Run:
    python scripts/backfill_eia.py
"""

from __future__ import annotations

import sys

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from providers.eia_provider import EIAProvider


# ============================================================
# CONFIG
# ============================================================

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "commodities.duckdb"
)


# These are the exact EIA series we currently care about.
SERIES_CONFIG = [
    {
        "metric": "working_gas_storage_lower48",
        "route": "natural-gas/stor/wkly",
        "series_id": "NW2_EPG0_SWO_R48_BCF",
        "frequency": "weekly",
        "start": "2000-01-01",
    },
    {
        "metric": "dry_natural_gas_production",
        "route": "natural-gas/prod/sum",
        "series_id": "N9070US2",
        "frequency": "monthly",
        "start": "2000-01",
    },
    {
        "metric": "electric_power_consumption",
        "route": "natural-gas/cons/sum",
        "series_id": "N3045US2",
        "frequency": "monthly",
        "start": "2000-01",
    },
    {
        "metric": "natural_gas_exports",
        "route": "natural-gas/move/expc",
        "series_id": "N9130US2",
        "frequency": "monthly",
        "start": "2000-01",
    },
]


# ============================================================
# DATABASE
# ============================================================

def create_schema(
    connection: duckdb.DuckDBPyConnection,
) -> None:

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS eia_natural_gas (
            period DATE NOT NULL,
            raw_period VARCHAR NOT NULL,

            metric VARCHAR NOT NULL,
            series_id VARCHAR NOT NULL,

            frequency VARCHAR NOT NULL,
            value DOUBLE,
            unit VARCHAR,

            route VARCHAR NOT NULL,

            source VARCHAR NOT NULL,
            ingested_at TIMESTAMP NOT NULL,

            PRIMARY KEY (
                period,
                series_id,
                frequency
            )
        )
        """
    )


# ============================================================
# TRANSFORMATION
# ============================================================

def normalize_period(
    value: str,
    frequency: str,
):
    """
    Convert EIA period strings into a DuckDB-compatible date.

    Weekly:
        2026-09-11

    Monthly:
        2026-06
        ->
        2026-06-01
    """

    if frequency == "monthly":
        return pd.to_datetime(
            f"{value}-01"
        ).date()

    return pd.to_datetime(
        value
    ).date()


def prepare_rows(
    data: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:

    if data.empty:
        return pd.DataFrame()

    rows = data.copy()

    rows["raw_period"] = (
        rows["period"]
        .astype(str)
    )

    rows["period"] = rows[
        "raw_period"
    ].apply(
        lambda value: normalize_period(
            value,
            config["frequency"],
        )
    )

    rows["metric"] = config["metric"]
    rows["series_id"] = config["series_id"]
    rows["frequency"] = config["frequency"]
    rows["route"] = config["route"]

    rows["value"] = pd.to_numeric(
        rows["value"],
        errors="coerce",
    )

    # EIA normally returns a "units" field.
    # Handle either spelling safely.
    if "units" in rows.columns:
        rows["unit"] = rows["units"]

    elif "unit" not in rows.columns:
        rows["unit"] = None

    rows["source"] = "EIA"

    rows["ingested_at"] = (
        datetime.now(timezone.utc)
        .replace(tzinfo=None)
    )

    columns = [
        "period",
        "raw_period",
        "metric",
        "series_id",
        "frequency",
        "value",
        "unit",
        "route",
        "source",
        "ingested_at",
    ]

    return rows[columns]


# ============================================================
# UPSERT
# ============================================================

def upsert_rows(
    connection: duckdb.DuckDBPyConnection,
    rows: pd.DataFrame,
) -> None:

    if rows.empty:
        return

    connection.register(
        "eia_rows",
        rows,
    )

    try:

        connection.execute(
            """
            INSERT OR REPLACE INTO eia_natural_gas
            SELECT *
            FROM eia_rows
            """
        )

    finally:

        connection.unregister(
            "eia_rows"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    provider = EIAProvider()

    with duckdb.connect(
        str(DATABASE_PATH)
    ) as connection:

        create_schema(
            connection
        )

        print()
        print("=" * 70)
        print("EIA NATURAL GAS HISTORICAL BACKFILL")
        print("=" * 70)

        for config in SERIES_CONFIG:

            print()
            print(
                f"Fetching "
                f"{config['metric']}"
            )

            print(
                f"  Route:     "
                f"{config['route']}"
            )

            print(
                f"  Series:    "
                f"{config['series_id']}"
            )

            print(
                f"  Frequency: "
                f"{config['frequency']}"
            )

            try:

                data = provider.fetch_series(
                    route=config["route"],
                    series_id=config["series_id"],
                    frequency=config["frequency"],
                    start=config["start"],
                )

            except Exception as error:

                print(
                    f"  ERROR: {error}"
                )

                continue

            print(
                f"  Returned "
                f"{len(data):,} rows"
            )

            if data.empty:
                continue

            rows = prepare_rows(
                data,
                config,
            )

            upsert_rows(
                connection,
                rows,
            )

            print(
                f"  Written "
                f"{len(rows):,} rows"
            )


        # ====================================================
        # SUMMARY
        # ====================================================

        print()
        print("=" * 70)
        print("BACKFILL COMPLETE")
        print("=" * 70)

        summary = connection.execute(
            """
            SELECT
                metric,
                series_id,
                frequency,
                unit,

                COUNT(*) AS observations,

                MIN(period) AS first_period,
                MAX(period) AS last_period

            FROM eia_natural_gas

            GROUP BY
                metric,
                series_id,
                frequency,
                unit

            ORDER BY metric
            """
        ).df()

        print()
        print(
            summary.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()