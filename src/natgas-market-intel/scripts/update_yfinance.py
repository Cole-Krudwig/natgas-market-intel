"""Update active NYMEX natural-gas contracts and rebuild the daily curve.

Run from the project root with:
    python scripts/update_yfinance.py
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from providers.yfinance_provider import YFinanceProvider
from models.curve_relative_value import run_curve_relative_value


DATABASE_PATH = PROJECT_DIR / "data" / "commodities.duckdb"
LOOKBACK_DAYS = 7
CURVE_MONTHS = 12


def create_schema(connection: duckdb.DuckDBPyConnection) -> None:
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


def estimated_last_trade_day(year: int, month: int) -> date:
    """Use the usual NG convention: roughly three business days before delivery."""
    delivery_start = pd.Timestamp(year=year, month=month, day=1)
    return (delivery_start - pd.offsets.BDay(3)).date()


def active_contracts(as_of: date, count: int = CURVE_MONTHS) -> list[tuple[int, int]]:
    """Return the next contracts whose estimated last trade date has not passed."""
    contracts: list[tuple[int, int]] = []
    year, month = as_of.year, as_of.month
    while len(contracts) < count:
        if estimated_last_trade_day(year, month) >= as_of:
            contracts.append((year, month))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return contracts


def prepare_rows(data: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    rows = data.copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.date
    rows["root"] = "NG"
    rows["contract_year"] = year
    rows["contract_month"] = month
    rows["ingested_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    return rows[
        [
            "date", "symbol", "root", "contract_year", "contract_month",
            "open", "high", "low", "close", "adj_close", "volume", "source",
            "ingested_at",
        ]
    ]


def upsert_rows(connection: duckdb.DuckDBPyConnection, rows: pd.DataFrame) -> None:
    connection.register("update_rows", rows)
    try:
        connection.execute("INSERT OR REPLACE INTO futures_daily SELECT * FROM update_rows")
    finally:
        connection.unregister("update_rows")


def refresh_ng_curve_daily(connection: duckdb.DuckDBPyConnection) -> None:
    """Build M1--M12 by contract expiration order for each trading date.

    A contract is excluded on/after its delivery month. This simple rule removes
    clearly expired contracts without depending on a holiday calendar.
    """
    price_columns = ",\n            ".join(
        f"MAX(CASE WHEN maturity = {maturity} THEN close END) AS m{maturity}"
        for maturity in range(1, CURVE_MONTHS + 1)
    )
    symbol_columns = ",\n            ".join(
        f"MAX(CASE WHEN maturity = {maturity} THEN symbol END) AS m{maturity}_symbol"
        for maturity in range(1, CURVE_MONTHS + 1)
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE ng_curve_daily AS
        WITH active_contract_data AS (
            SELECT
                date,
                symbol,
                contract_year,
                contract_month,
                close,
                ROW_NUMBER() OVER (
                    PARTITION BY date
                    ORDER BY contract_year, contract_month
                ) AS maturity
            FROM futures_daily
            WHERE root = 'NG'
              AND close IS NOT NULL
              AND date < make_date(contract_year, contract_month, 1)
        )
        SELECT
            date,
            {price_columns},
            {symbol_columns}
        FROM active_contract_data
        WHERE maturity <= {CURVE_MONTHS}
        GROUP BY date
        ORDER BY date
        """
    )


def update_yfinance(as_of: date | None = None) -> None:
    as_of = as_of or date.today()
    start = (pd.Timestamp(as_of) - pd.Timedelta(days=LOOKBACK_DAYS)).date().isoformat()
    # yfinance treats end as exclusive; include the current day when it is available.
    end = (pd.Timestamp(as_of) + pd.Timedelta(days=1)).date().isoformat()
    provider = YFinanceProvider()
    contracts = active_contracts(as_of)

    print(f"Updating {len(contracts)} active NG contracts ({start} to {as_of.isoformat()})")
    with duckdb.connect(str(DATABASE_PATH)) as connection:
        create_schema(connection)
        written = 0
        for year, month in contracts:
            symbol = provider.ng_contract_symbol(year, month)
            try:
                data = provider.fetch_ng_contract(year, month, start=start, end=end)
                if data.empty:
                    print(f"{symbol}: no rows")
                    continue
                rows = prepare_rows(data, year, month)
                upsert_rows(connection, rows)
                written += len(rows)
                print(f"{symbol}: upserted {len(rows)} rows")
            except Exception as error:
                print(f"{symbol}: skipped ({error})")

        refresh_ng_curve_daily(connection)
        print(f"Refreshed ng_curve_daily; upserted {written} rows total")

    # Rebuild derived spreads only after the futures writer connection is closed.
    try:
        run_curve_relative_value()
    except Exception as error:
        print(f"Curve relative value: skipped ({error})")


if __name__ == "__main__":
    update_yfinance()
