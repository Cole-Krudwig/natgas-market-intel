"""Derived natural-gas curve relative-value features and Winter-Summer fair value.

Run from the project root with:
    python models/curve_relative_value.py

Spread definitions:
* M1-M2, M1-M3, M1-M6: nearest active contracts by delivery-month rank.
* Jan-Mar / Mar-Apr: the next unexpired January or March contract and the
  matching March or April contract in that delivery year.
* Winter-Summer: average Nov-Mar strip less average Apr-Oct strip immediately
  following that winter. For example, in Sep-2026 it is average(Nov-26..Mar-27)
  minus average(Apr-27..Oct-27). Contract years are chosen from each quote date.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "commodities.duckdb"
SEASONAL_MIN_OBSERVATIONS = 20
TRAIN_FRACTION = 0.80
FAIR_VALUE_FEATURES = [
    "storage_vs_prior_seasonal_normal",
    "hdd_anomaly",
    "cdd_anomaly",
    "week_sin",
    "week_cos",
]
SPREAD_NAMES = ["M1-M2", "M1-M3", "M1-M6", "Jan-Mar", "Mar-Apr", "Winter-Summer"]


@dataclass
class FairValueResult:
    rows: pd.DataFrame
    diagnostics: dict[str, object]
    coefficients: pd.DataFrame


def load_active_contracts(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load valid NG quotes; a delivery month is never active on or after its start."""
    rows = connection.execute(
        """
        SELECT date, symbol, contract_year, contract_month, close
        FROM futures_daily
        WHERE root = 'NG'
          AND close IS NOT NULL
          AND date < make_date(contract_year, contract_month, 1)
        ORDER BY date, contract_year, contract_month
        """
    ).fetchdf()
    rows["date"] = pd.to_datetime(rows["date"])
    return rows


def build_maturity_data(contracts: pd.DataFrame) -> pd.DataFrame:
    """Rank available contracts by delivery month for every quote date."""
    output = contracts.copy()
    output["maturity"] = output.groupby("date").cumcount() + 1
    return output[output["maturity"] <= 12].copy()


def _seasonal_spreads_for_date(day: pd.DataFrame, quote_date: pd.Timestamp) -> list[dict[str, object]]:
    """Calculate calendar and strip spreads from the contracts available that day."""
    by_key = day.set_index(["contract_year", "contract_month"])
    delivery_index = quote_date.year * 12 + quote_date.month
    records: list[dict[str, object]] = []

    def add_pair(name: str, first: pd.Series | None, second: pd.Series | None) -> None:
        if first is None or second is None:
            return
        records.append(
            {
                "date": quote_date,
                "spread_name": name,
                "spread_value": first.close - second.close,
                "leg1_symbol": first.symbol,
                "leg2_symbol": second.symbol,
            }
        )

    ranked = day.sort_values(["contract_year", "contract_month"]).reset_index(drop=True)
    for name, second_rank in [("M1-M2", 1), ("M1-M3", 2), ("M1-M6", 5)]:
        if len(ranked) > second_rank:
            add_pair(name, ranked.iloc[0], ranked.iloc[second_rank])

    def next_delivery_year(month: int) -> int | None:
        candidates = day[(day["contract_month"] == month) & ((day["contract_year"] * 12 + month) > delivery_index)]
        return int(candidates["contract_year"].min()) if not candidates.empty else None

    january_year = next_delivery_year(1)
    if january_year is not None and (january_year, 1) in by_key.index and (january_year, 3) in by_key.index:
        add_pair("Jan-Mar", by_key.loc[(january_year, 1)], by_key.loc[(january_year, 3)])

    march_year = next_delivery_year(3)
    if march_year is not None and (march_year, 3) in by_key.index and (march_year, 4) in by_key.index:
        add_pair("Mar-Apr", by_key.loc[(march_year, 3)], by_key.loc[(march_year, 4)])

    winter_year = next_delivery_year(11)
    if winter_year is not None:
        winter_keys = [(winter_year, 11), (winter_year, 12)] + [(winter_year + 1, month) for month in (1, 2, 3)]
        summer_keys = [(winter_year + 1, month) for month in range(4, 11)]
        if all(key in by_key.index for key in winter_keys + summer_keys):
            winter = by_key.loc[winter_keys]
            summer = by_key.loc[summer_keys]
            records.append(
                {
                    "date": quote_date,
                    "spread_name": "Winter-Summer",
                    "spread_value": winter["close"].mean() - summer["close"].mean(),
                    "leg1_symbol": ",".join(winter["symbol"].tolist()),
                    "leg2_symbol": ",".join(summer["symbol"].tolist()),
                }
            )
    return records


def build_spreads(contracts: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for quote_date, day in contracts.groupby("date", sort=True):
        records.extend(_seasonal_spreads_for_date(day, quote_date))
    return pd.DataFrame(records).sort_values(["spread_name", "date"]).reset_index(drop=True)


def add_seasonal_statistics(spreads: pd.DataFrame) -> pd.DataFrame:
    """Compare each quote only with observations from prior years in its calendar month."""
    output = spreads.copy()
    output["one_day_change"] = output.groupby("spread_name")["spread_value"].diff()
    output["five_day_change"] = output.groupby("spread_name")["spread_value"].diff(5)
    output["seasonal_mean"] = np.nan
    output["seasonal_std"] = np.nan
    output["seasonal_zscore"] = np.nan
    output["seasonal_percentile"] = np.nan
    output["seasonal_observations"] = 0
    output["calendar_month"] = output["date"].dt.month
    output["calendar_year"] = output["date"].dt.year

    for _, indices in output.groupby(["spread_name", "calendar_month"]).groups.items():
        group = output.loc[indices].sort_values("date")
        for calendar_year, current in group.groupby("calendar_year"):
            prior = group.loc[group["calendar_year"] < calendar_year, "spread_value"].dropna()
            if len(prior) < SEASONAL_MIN_OBSERVATIONS:
                continue
            mean = prior.mean()
            std = prior.std(ddof=1)
            current_values = current["spread_value"]
            output.loc[current.index, "seasonal_mean"] = mean
            output.loc[current.index, "seasonal_std"] = std
            output.loc[current.index, "seasonal_observations"] = len(prior)
            output.loc[current.index, "seasonal_percentile"] = [
                (prior <= value).mean() * 100 for value in current_values
            ]
            if std and np.isfinite(std):
                output.loc[current.index, "seasonal_zscore"] = (current_values - mean) / std
    return output.drop(columns=["calendar_month", "calendar_year"])


def build_fair_value_dataset(connection: duckdb.DuckDBPyConnection, spreads: pd.DataFrame) -> pd.DataFrame:
    """Join spread quotes to the most recently released weekly storage observation.

    Storage periods are made available seven days after their Friday period end,
    which is deliberately conservative relative to the usual EIA release timing.
    """
    winter_summer = spreads[spreads["spread_name"] == "Winter-Summer"].copy()
    storage = connection.execute(
        """
        SELECT period, storage_vs_prior_seasonal_normal, hdd_anomaly, cdd_anomaly
        FROM ng_storage_balance_model
        ORDER BY period
        """
    ).fetchdf()
    if winter_summer.empty or storage.empty:
        return pd.DataFrame()
    storage["available_date"] = pd.to_datetime(storage["period"]) + pd.Timedelta(days=7)
    winter_summer = winter_summer.sort_values("date")
    storage = storage.sort_values("available_date")
    output = pd.merge_asof(
        winter_summer,
        storage.drop(columns="period"),
        left_on="date",
        right_on="available_date",
        direction="backward",
    )
    week = output["date"].dt.isocalendar().week.astype(int)
    angle = 2 * np.pi * week / 52.0
    output["week_sin"] = np.sin(angle)
    output["week_cos"] = np.cos(angle)
    return output.dropna(subset=[*FAIR_VALUE_FEATURES, "spread_value"]).reset_index(drop=True)


def fit_fair_value_model(dataset: pd.DataFrame) -> FairValueResult:
    if len(dataset) < 50:
        raise ValueError("At least 50 complete Winter-Summer observations are required.")
    split_index = min(int(len(dataset) * TRAIN_FRACTION), len(dataset) - 1)
    train, test = dataset.iloc[:split_index], dataset.iloc[split_index:]
    model = LinearRegression().fit(train[FAIR_VALUE_FEATURES], train["spread_value"])
    output = dataset.copy()
    output["fair_value"] = model.predict(output[FAIR_VALUE_FEATURES])
    output["deviation"] = output["spread_value"] - output["fair_value"]
    output["sample"] = "in_sample"
    output.loc[output.index >= split_index, "sample"] = "out_of_sample"
    holdout = output.iloc[split_index:]
    diagnostics: dict[str, object] = {
        "model_version": "winter_summer_linear_v1",
        "trained_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "train_start": train["date"].iloc[0].date(),
        "train_end": train["date"].iloc[-1].date(),
        "test_start": test["date"].iloc[0].date(),
        "test_end": test["date"].iloc[-1].date(),
        "observations": len(output),
        "test_observations": len(holdout),
        "mae": mean_absolute_error(holdout["spread_value"], holdout["fair_value"]),
        "rmse": mean_squared_error(holdout["spread_value"], holdout["fair_value"]) ** 0.5,
        "r2": r2_score(holdout["spread_value"], holdout["fair_value"]),
        "features": json.dumps(FAIR_VALUE_FEATURES),
    }
    coefficients = pd.DataFrame({"feature": FAIR_VALUE_FEATURES, "coefficient": model.coef_})
    coefficients.loc[len(coefficients)] = {"feature": "intercept", "coefficient": model.intercept_}
    return FairValueResult(output, diagnostics, coefficients)


def refresh_tables(connection: duckdb.DuckDBPyConnection, maturities: pd.DataFrame, spreads: pd.DataFrame, fair_value: FairValueResult) -> None:
    """Refresh derived curve/model outputs without modifying source futures or fundamentals."""
    maturities = maturities.copy()
    spreads = spreads.copy()
    fair_rows = fair_value.rows.copy()
    for frame in (maturities, spreads, fair_rows):
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
    connection.register("curve_maturities", maturities)
    connection.register("curve_relative_value", spreads)
    connection.register("winter_summer_fair_value", fair_rows)
    connection.register("winter_summer_diagnostics", pd.DataFrame([fair_value.diagnostics]))
    connection.register("winter_summer_coefficients", fair_value.coefficients.assign(model_version="winter_summer_linear_v1"))
    try:
        connection.execute(
            """CREATE OR REPLACE TABLE ng_curve_maturity_daily AS
            SELECT date, maturity, close, symbol, contract_year, contract_month FROM curve_maturities ORDER BY date, maturity"""
        )
        connection.execute(
            """CREATE OR REPLACE TABLE ng_curve_relative_value AS
            SELECT date, spread_name, spread_value, leg1_symbol, leg2_symbol,
                   one_day_change, five_day_change, seasonal_mean, seasonal_std,
                   seasonal_zscore, seasonal_percentile, seasonal_observations
            FROM curve_relative_value ORDER BY date, spread_name"""
        )
        connection.execute(
            """CREATE OR REPLACE TABLE ng_winter_summer_fair_value AS
            SELECT date, spread_value AS actual_winter_summer, fair_value, deviation,
                   storage_vs_prior_seasonal_normal, hdd_anomaly, cdd_anomaly,
                   week_sin, week_cos, sample
            FROM winter_summer_fair_value ORDER BY date"""
        )
        connection.execute("CREATE OR REPLACE TABLE ng_winter_summer_fair_value_diagnostics AS SELECT * FROM winter_summer_diagnostics")
        connection.execute("CREATE OR REPLACE TABLE ng_winter_summer_fair_value_coefficients AS SELECT * FROM winter_summer_coefficients ORDER BY feature")
    finally:
        connection.unregister("curve_maturities")
        connection.unregister("curve_relative_value")
        connection.unregister("winter_summer_fair_value")
        connection.unregister("winter_summer_diagnostics")
        connection.unregister("winter_summer_coefficients")


def run_curve_relative_value() -> FairValueResult:
    with duckdb.connect(str(DATABASE_PATH)) as connection:
        contracts = load_active_contracts(connection)
        maturities = build_maturity_data(contracts)
        spreads = add_seasonal_statistics(build_spreads(contracts))
        fair_dataset = build_fair_value_dataset(connection, spreads)
        fair_value = fit_fair_value_model(fair_dataset)
        refresh_tables(connection, maturities, spreads, fair_value)
    print("Curve spread definitions: M1-M2, M1-M3, M1-M6, next Jan-Mar, next Mar-Apr, Winter(Nov-Mar)-Summer(Apr-Oct)")
    print(f"Winter-Summer model variables: {', '.join(FAIR_VALUE_FEATURES)}")
    print(f"Train: {fair_value.diagnostics['train_start']} to {fair_value.diagnostics['train_end']}")
    print(f"Test: {fair_value.diagnostics['test_start']} to {fair_value.diagnostics['test_end']}")
    print(f"OOS MAE: {fair_value.diagnostics['mae']:.4f}; RMSE: {fair_value.diagnostics['rmse']:.4f}; R²: {fair_value.diagnostics['r2']:.3f}")
    return fair_value


if __name__ == "__main__":
    run_curve_relative_value()
