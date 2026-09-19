"""One-time historical futures backfill using the Yahoo Finance provider.

Run from the project directory:
    python scripts/backfill_yfinance.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from providers.yfinance_provider import YFinanceProvider


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_DIR / "data" / "commodities.duckdb"
START_YEAR = 2015
END_YEAR = 2028


def create_schema(connection: duckdb.DuckDBPyConnection) -> None:
    """Create the shared daily-futures table if it does not yet exist."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS futures_daily (
            date DATE NOT NULL,
            symbol VARCHAR NOT NULL,
            root VARCHAR NOT NULL,
            contract_year INTEGER NOT NULL,
            contract_month INTEGER NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            adj_close DOUBLE,
            volume BIGINT,
            source VARCHAR NOT NULL,
            ingested_at TIMESTAMP NOT NULL,
            PRIMARY KEY (date, symbol)
        )
        """
    )


def prepare_rows(
    data: pd.DataFrame, root: str, contract_year: int, contract_month: int
) -> pd.DataFrame:
    """Shape provider output to the database schema for any futures root."""
    rows = data.copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.date
    rows["root"] = root
    rows["contract_year"] = contract_year
    rows["contract_month"] = contract_month
    rows["ingested_at"] = datetime.now(timezone.utc).replace(tzinfo=None)

    columns = [
        "date",
        "symbol",
        "root",
        "contract_year",
        "contract_month",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "source",
        "ingested_at",
    ]
    return rows[columns]


def upsert_rows(connection: duckdb.DuckDBPyConnection, rows: pd.DataFrame) -> None:
    """Replace existing observations with the same (date, symbol) key."""
    connection.register("backfill_rows", rows)
    try:
        connection.execute("INSERT OR REPLACE INTO futures_daily SELECT * FROM backfill_rows")
    finally:
        connection.unregister("backfill_rows")


def backfill_ng_contracts(start_year: int = START_YEAR, end_year: int = END_YEAR) -> None:
    """Backfill all available NYMEX Henry Hub Natural Gas monthly contracts."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    provider = YFinanceProvider()

    with duckdb.connect(str(DATABASE_PATH)) as connection:
        create_schema(connection)

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                symbol = provider.ng_contract_symbol(year, month)
                print(f"Fetching {symbol}...")

                try:
                    data = provider.fetch_ng_contract(year, month)
                except Exception as error:
                    print(f"  Returned 0 rows; skipped ({error})")
                    continue

                print(f"  Returned {len(data)} rows")
                if data.empty:
                    print("  No data; skipped")
                    continue

                try:
                    upsert_rows(connection, prepare_rows(data, "NG", year, month))
                    print("  Written successfully")
                except Exception as error:
                    print(f"  Write failed: {error}")


if __name__ == "__main__":
    backfill_ng_contracts()
