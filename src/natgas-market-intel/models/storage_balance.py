"""Weather-adjusted weekly Lower-48 natural-gas storage balance model.

Run from the project root with:
    python models/storage_balance.py

The model uses only data known on or before each EIA storage observation date.
Monthly EIA production, export, and power-burn series are intentionally not used
in V1 because this database does not contain their publication dates.
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
MODEL_VERSION = "storage_balance_linear_v1"
TRAIN_FRACTION = 0.80
FEATURE_COLUMNS = [
    "hdd",
    "cdd",
    "hdd_anomaly",
    "cdd_anomaly",
    "lag_storage_level",
    "storage_vs_prior_seasonal_normal",
    "week_sin",
    "week_cos",
]


@dataclass
class ModelResult:
    rows: pd.DataFrame
    diagnostics: dict[str, object]
    coefficients: pd.DataFrame


def load_modeling_dataset(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Align CONUS weather through each Friday storage observation date.

    The storage target is the change in Lower-48 working gas from the previous
    weekly observation. Weather is summed over the seven calendar days ending
    on that observation date, so it contains no days after the target period.
    """
    storage = connection.execute(
        """
        SELECT period, value AS storage_level
        FROM eia_natural_gas
        WHERE metric = 'working_gas_storage_lower48'
          AND frequency = 'weekly'
          AND value IS NOT NULL
        ORDER BY period
        """
    ).fetchdf()
    weather = connection.execute(
        """
        SELECT date, hdd, cdd, hdd_anomaly, cdd_anomaly
        FROM noaa_degree_days_daily
        WHERE region_code = 'CONUS'
        ORDER BY date
        """
    ).fetchdf()

    if storage.empty or weather.empty:
        return pd.DataFrame()

    storage["period"] = pd.to_datetime(storage["period"])
    weather["date"] = pd.to_datetime(weather["date"])
    storage["lag_storage_level"] = storage["storage_level"].shift(1)
    storage["actual_storage_change"] = storage["storage_level"].diff()
    storage["week_of_year"] = storage["period"].dt.isocalendar().week.astype(int)

    # Shift before expanding so each week-of-year normal uses prior years only.
    storage["prior_seasonal_storage_normal"] = (
        storage.groupby("week_of_year")["storage_level"]
        .transform(lambda values: values.shift(1).expanding().mean())
    )
    storage["storage_vs_prior_seasonal_normal"] = (
        storage["lag_storage_level"] - storage["prior_seasonal_storage_normal"]
    )

    weekly_weather = []
    for period in storage["period"]:
        week = weather[(weather["date"] >= period - pd.Timedelta(days=6)) & (weather["date"] <= period)]
        if len(week) == 7:
            weekly_weather.append(
                {
                    "period": period,
                    "hdd": week["hdd"].sum(),
                    "cdd": week["cdd"].sum(),
                    "hdd_anomaly": week["hdd_anomaly"].sum(),
                    "cdd_anomaly": week["cdd_anomaly"].sum(),
                }
            )
    weather_frame = pd.DataFrame(weekly_weather)
    if weather_frame.empty:
        return pd.DataFrame()

    dataset = storage.merge(weather_frame, on="period", how="inner")
    angle = 2 * np.pi * dataset["week_of_year"] / 52.0
    dataset["week_sin"] = np.sin(angle)
    dataset["week_cos"] = np.cos(angle)
    return dataset.dropna(subset=["actual_storage_change", *FEATURE_COLUMNS]).reset_index(drop=True)


def fit_storage_balance_model(dataset: pd.DataFrame) -> ModelResult:
    """Fit a chronological 80/20 LinearRegression model and score its holdout."""
    if len(dataset) < 30:
        raise ValueError("At least 30 complete weekly observations are required.")

    split_index = int(len(dataset) * TRAIN_FRACTION)
    if split_index >= len(dataset):
        split_index = len(dataset) - 1
    train = dataset.iloc[:split_index]
    test = dataset.iloc[split_index:]
    model = LinearRegression()
    model.fit(train[FEATURE_COLUMNS], train["actual_storage_change"])

    output = dataset.copy()
    output["predicted_storage_change"] = model.predict(output[FEATURE_COLUMNS])
    output["storage_surprise"] = (
        output["actual_storage_change"] - output["predicted_storage_change"]
    )
    output["sample"] = "in_sample"
    output.loc[output.index >= split_index, "sample"] = "out_of_sample"

    test_predictions = output.iloc[split_index:]
    diagnostics: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "train_start": train["period"].iloc[0].date(),
        "train_end": train["period"].iloc[-1].date(),
        "test_start": test["period"].iloc[0].date(),
        "test_end": test["period"].iloc[-1].date(),
        "observations": len(output),
        "test_observations": len(test_predictions),
        "mae": mean_absolute_error(test_predictions["actual_storage_change"], test_predictions["predicted_storage_change"]),
        "rmse": mean_squared_error(test_predictions["actual_storage_change"], test_predictions["predicted_storage_change"]) ** 0.5,
        "r2": r2_score(test_predictions["actual_storage_change"], test_predictions["predicted_storage_change"]),
        "features": json.dumps(FEATURE_COLUMNS),
    }
    coefficients = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "coefficient": model.coef_}
    )
    coefficients.loc[len(coefficients)] = {"feature": "intercept", "coefficient": model.intercept_}
    return ModelResult(rows=output, diagnostics=diagnostics, coefficients=coefficients)


def refresh_model_tables(connection: duckdb.DuckDBPyConnection, result: ModelResult) -> None:
    """Replace only derived model outputs; raw EIA and NOAA tables remain untouched."""
    rows = result.rows.copy()
    rows["period"] = pd.to_datetime(rows["period"]).dt.date
    rows["model_version"] = MODEL_VERSION
    connection.register("model_rows", rows)
    connection.register("model_diagnostics", pd.DataFrame([result.diagnostics]))
    connection.register("model_coefficients", result.coefficients.assign(model_version=MODEL_VERSION))
    try:
        connection.execute(
            """
            CREATE OR REPLACE TABLE ng_storage_balance_model AS
            SELECT
                period, actual_storage_change, predicted_storage_change, storage_surprise,
                storage_level, lag_storage_level, prior_seasonal_storage_normal,
                storage_vs_prior_seasonal_normal, hdd, cdd, hdd_anomaly, cdd_anomaly,
                week_of_year, week_sin, week_cos, sample, model_version
            FROM model_rows
            ORDER BY period
            """
        )
        connection.execute(
            "CREATE OR REPLACE TABLE ng_storage_balance_model_diagnostics AS SELECT * FROM model_diagnostics"
        )
        connection.execute(
            "CREATE OR REPLACE TABLE ng_storage_balance_model_coefficients AS SELECT * FROM model_coefficients ORDER BY feature"
        )
    finally:
        connection.unregister("model_rows")
        connection.unregister("model_diagnostics")
        connection.unregister("model_coefficients")


def run_model() -> ModelResult:
    """Build the dataset, fit the model, save outputs, and print concise diagnostics."""
    with duckdb.connect(str(DATABASE_PATH)) as connection:
        dataset = load_modeling_dataset(connection)
        result = fit_storage_balance_model(dataset)
        refresh_model_tables(connection, result)

    diagnostics = result.diagnostics
    print(f"Storage balance model: {diagnostics['observations']} weekly observations")
    print(f"Features: {', '.join(FEATURE_COLUMNS)}")
    print(f"Train: {diagnostics['train_start']} to {diagnostics['train_end']}")
    print(f"Test: {diagnostics['test_start']} to {diagnostics['test_end']}")
    print(f"OOS MAE: {diagnostics['mae']:.2f}; RMSE: {diagnostics['rmse']:.2f}; R²: {diagnostics['r2']:.3f}")
    return result


if __name__ == "__main__":
    run_model()
