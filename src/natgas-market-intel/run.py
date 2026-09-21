"""Run with: streamlit run run.py"""

from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st


DATABASE_PATH = Path(__file__).resolve().parent / "data" / "commodities.duckdb"

st.set_page_config(page_title="Natural Gas Market Intel", layout="wide")
st.title("Natural Gas Futures")
st.caption("Daily NYMEX Henry Hub futures data")
st.markdown(
    "[Market Overview](#market-overview) &nbsp;|&nbsp; "
    "[Futures Curve](#futures-curve) &nbsp;|&nbsp; "
    "[Curve & Relative Value](#curve-relative-value) &nbsp;|&nbsp; "
    "[Weather-Adjusted Storage Balance](#weather-adjusted-storage-balance)",
)


@st.cache_data(ttl=300)
def load_curve_data() -> pd.DataFrame:
    with duckdb.connect(str(DATABASE_PATH), read_only=True) as connection:
        return connection.execute("SELECT * FROM ng_curve_daily ORDER BY date").fetchdf()


@st.cache_data(ttl=300)
def load_storage_balance_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with duckdb.connect(str(DATABASE_PATH), read_only=True) as connection:
        model_rows = connection.execute(
            "SELECT * FROM ng_storage_balance_model ORDER BY period"
        ).fetchdf()
        diagnostics = connection.execute(
            "SELECT * FROM ng_storage_balance_model_diagnostics"
        ).fetchdf()
        coefficients = connection.execute(
            "SELECT feature, coefficient FROM ng_storage_balance_model_coefficients ORDER BY feature"
        ).fetchdf()
    return model_rows, diagnostics, coefficients


@st.cache_data(ttl=300)
def load_curve_relative_value_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with duckdb.connect(str(DATABASE_PATH), read_only=True) as connection:
        latest_date = connection.execute("SELECT max(date) FROM ng_curve_maturity_daily").fetchone()[0]
        current_curve = connection.execute(
            "SELECT maturity, close FROM ng_curve_maturity_daily WHERE date = ? ORDER BY maturity", [latest_date]
        ).fetchdf()
        seasonal_curve = connection.execute(
            """
            SELECT maturity, avg(close) AS seasonal_average
            FROM ng_curve_maturity_daily
            WHERE extract(month FROM date) = extract(month FROM ?::DATE)
              AND extract(year FROM date) < extract(year FROM ?::DATE)
            GROUP BY maturity ORDER BY maturity
            """, [latest_date, latest_date]
        ).fetchdf()
        latest_spreads = connection.execute(
            "SELECT * FROM ng_curve_relative_value WHERE date = (SELECT max(date) FROM ng_curve_relative_value)"
        ).fetchdf()
        winter_history = connection.execute(
            "SELECT * FROM ng_curve_relative_value WHERE spread_name = 'Winter-Summer' ORDER BY date"
        ).fetchdf()
        fair_value = connection.execute("SELECT * FROM ng_winter_summer_fair_value ORDER BY date").fetchdf()
        diagnostics = connection.execute("SELECT * FROM ng_winter_summer_fair_value_diagnostics").fetchdf()
        coefficients = connection.execute(
            "SELECT feature, coefficient FROM ng_winter_summer_fair_value_coefficients ORDER BY feature"
        ).fetchdf()
    return current_curve, seasonal_curve, latest_spreads, winter_history, fair_value, diagnostics, coefficients


try:
    curve = load_curve_data()
except Exception as error:
    st.error(f"Could not read ng_curve_daily: {error}")
    st.info("Run `python scripts/update_yfinance.py` first.")
    st.stop()

if curve.empty or curve["m1"].dropna().empty:
    st.info("No natural-gas curve data is available yet. Run the updater first.")
    st.stop()

curve["date"] = pd.to_datetime(curve["date"])
history = curve.dropna(subset=["m1"]).copy()
latest = history.iloc[-1]
previous = history.iloc[-2] if len(history) > 1 else None
daily_change = ((latest.m1 / previous.m1) - 1) if previous is not None and previous.m1 else None
month_ago = history[history["date"] <= latest.date - pd.DateOffset(months=1)]
one_month_return = ((latest.m1 / month_ago.iloc[-1].m1) - 1) if not month_ago.empty else None

if pd.notna(latest.m2):
    curve_state = "Contango" if latest.m2 > latest.m1 else "Backwardation" if latest.m2 < latest.m1 else "Flat"
else:
    curve_state = "Unavailable"

st.markdown('<a id="market-overview"></a>', unsafe_allow_html=True)
st.subheader("Market Overview")
metrics = st.columns(4)
metrics[0].metric("Front month", f"${latest.m1:,.3f}")
metrics[1].metric("Daily change", "—" if daily_change is None else f"{daily_change:.2%}")
metrics[2].metric("1-month return", "—" if one_month_return is None else f"{one_month_return:.2%}")
metrics[3].metric("M1 vs M2", curve_state)
st.caption(f"Latest curve date: {latest.date.date():%b %d, %Y}")

st.markdown('<a id="futures-curve"></a>', unsafe_allow_html=True)
st.subheader("Futures Curve")
left, right = st.columns(2)
with left:
    chart_history = history[["date", "m1"]].rename(columns={"m1": "Front month"})
    st.plotly_chart(
        px.line(chart_history, x="date", y="Front month", title="Recent front-month history"),
        use_container_width=True,
    )

with right:
    maturities = [f"m{i}" for i in range(1, 13) if pd.notna(latest.get(f"m{i}"))]
    latest_curve = pd.DataFrame({"Maturity": [label.upper() for label in maturities], "Price": [latest[label] for label in maturities]})
    st.plotly_chart(px.line(latest_curve, x="Maturity", y="Price", markers=True, title="Latest futures curve"), use_container_width=True)

spread = history.dropna(subset=["m2"])[["date", "m1", "m2"]].copy()
spread["M1 - M2"] = spread["m1"] - spread["m2"]
st.plotly_chart(px.line(spread, x="date", y="M1 - M2", title="Historical M1–M2 spread"), use_container_width=True)

st.markdown('<a id="curve-relative-value"></a>', unsafe_allow_html=True)
st.header("Curve & Relative Value")
try:
    current_rv_curve, seasonal_rv_curve, latest_rv_spreads, winter_rv_history, winter_fair_value, winter_fair_diagnostics, winter_fair_coefficients = load_curve_relative_value_data()
except Exception:
    current_rv_curve = pd.DataFrame()
    seasonal_rv_curve = pd.DataFrame()
    latest_rv_spreads = pd.DataFrame()
    winter_rv_history = pd.DataFrame()
    winter_fair_value = pd.DataFrame()
    winter_fair_diagnostics = pd.DataFrame()
    winter_fair_coefficients = pd.DataFrame()

if current_rv_curve.empty or latest_rv_spreads.empty:
    st.info("Curve relative-value outputs are not available yet. Run `python models/curve_relative_value.py` after updating futures data.")
else:
    latest_rv_spreads = latest_rv_spreads.set_index("spread_name")
    winter_latest = latest_rv_spreads.loc["Winter-Summer"] if "Winter-Summer" in latest_rv_spreads.index else None
    fair_latest = winter_fair_value.iloc[-1] if not winter_fair_value.empty else None
    rv_metrics = st.columns(7)
    rv_metrics[0].metric("Curve regime", curve_state)
    rv_metrics[1].metric("M1-M2", "--" if "M1-M2" not in latest_rv_spreads.index else f"{latest_rv_spreads.loc['M1-M2', 'spread_value']:.3f}")
    rv_metrics[2].metric("Winter-Summer", "--" if winter_latest is None else f"{winter_latest.spread_value:.3f}")
    rv_metrics[3].metric("Winter-Summer z-score", "--" if winter_latest is None or pd.isna(winter_latest.seasonal_zscore) else f"{winter_latest.seasonal_zscore:.2f}")
    rv_metrics[4].metric("Winter-Summer percentile", "--" if winter_latest is None or pd.isna(winter_latest.seasonal_percentile) else f"{winter_latest.seasonal_percentile:.1f}%")
    rv_metrics[5].metric("Model fair value", "--" if fair_latest is None else f"{fair_latest.fair_value:.3f}")
    rv_metrics[6].metric("Deviation from fair value", "--" if fair_latest is None else f"{fair_latest.deviation:.3f}")

    current_curve_chart = current_rv_curve.copy()
    current_curve_chart["Maturity"] = "M" + current_curve_chart["maturity"].astype(str)
    st.plotly_chart(px.line(current_curve_chart, x="Maturity", y="close", markers=True, title="Current futures curve"), use_container_width=True)

    if not seasonal_rv_curve.empty:
        curve_comparison = current_rv_curve.merge(seasonal_rv_curve, on="maturity", how="inner")
        curve_comparison["Maturity"] = "M" + curve_comparison["maturity"].astype(str)
        curve_comparison = curve_comparison.melt(id_vars="Maturity", value_vars=["close", "seasonal_average"], var_name="Series", value_name="Price")
        curve_comparison["Series"] = curve_comparison["Series"].map({"close": "Current", "seasonal_average": "Prior-year same-month average"})
        st.plotly_chart(px.line(curve_comparison, x="Maturity", y="Price", color="Series", markers=True, title="Current curve vs seasonal historical average"), use_container_width=True)

    winter_rv_history["date"] = pd.to_datetime(winter_rv_history["date"])
    winter_chart = winter_rv_history.melt(id_vars="date", value_vars=["spread_value", "seasonal_mean"], var_name="Series", value_name="Spread")
    winter_chart["Series"] = winter_chart["Series"].map({"spread_value": "Winter-Summer", "seasonal_mean": "Prior-year same-month mean"})
    st.plotly_chart(px.line(winter_chart, x="date", y="Spread", color="Series", title="Winter-Summer spread history"), use_container_width=True)
    if winter_latest is not None and pd.notna(winter_latest.seasonal_zscore):
        st.caption(f"Latest Winter-Summer seasonal z-score: {winter_latest.seasonal_zscore:.2f}, based on {int(winter_latest.seasonal_observations)} prior-year same-month observations.")

    if not winter_fair_value.empty:
        winter_fair_value["date"] = pd.to_datetime(winter_fair_value["date"])
        fair_chart = winter_fair_value.melt(id_vars="date", value_vars=["actual_winter_summer", "fair_value"], var_name="Series", value_name="Spread")
        fair_chart["Series"] = fair_chart["Series"].map({"actual_winter_summer": "Actual", "fair_value": "Model fair value"})
        st.plotly_chart(px.line(fair_chart, x="date", y="Spread", color="Series", title="Winter-Summer actual vs fair value"), use_container_width=True)
        deviation_chart = px.line(winter_fair_value, x="date", y="deviation", title="Winter-Summer deviation from fair value")
        deviation_chart.add_hline(y=0, line_color="gray", line_dash="dash")
        st.plotly_chart(deviation_chart, use_container_width=True)

    st.subheader("Spread relative-value summary")
    rv_table = latest_rv_spreads.reset_index()
    rv_table = rv_table[rv_table["seasonal_zscore"].notna()][["spread_name", "spread_value", "one_day_change", "five_day_change", "seasonal_mean", "seasonal_zscore", "seasonal_percentile"]].rename(columns={"spread_name": "Spread", "spread_value": "Current", "one_day_change": "1-day change", "five_day_change": "5-day change", "seasonal_mean": "Seasonal mean", "seasonal_zscore": "Z-score", "seasonal_percentile": "Percentile"})
    st.dataframe(rv_table, hide_index=True, use_container_width=True)

    st.subheader("Winter-Summer model diagnostics")
    if not winter_fair_diagnostics.empty:
        fair_diagnostic = winter_fair_diagnostics.iloc[0]
        fair_diagnostic_metrics = st.columns(3)
        fair_diagnostic_metrics[0].metric("Out-of-sample MAE", f"{fair_diagnostic.mae:.3f}")
        fair_diagnostic_metrics[1].metric("Out-of-sample RMSE", f"{fair_diagnostic.rmse:.3f}")
        fair_diagnostic_metrics[2].metric("Out-of-sample R^2", f"{fair_diagnostic.r2:.3f}")
        st.caption(f"Chronological training: {pd.to_datetime(fair_diagnostic.train_start).date()} to {pd.to_datetime(fair_diagnostic.train_end).date()}; test: {pd.to_datetime(fair_diagnostic.test_start).date()} to {pd.to_datetime(fair_diagnostic.test_end).date()}.")
    st.dataframe(winter_fair_coefficients.rename(columns={"coefficient": "Linear coefficient"}), hide_index=True, use_container_width=True)

st.markdown('<a id="weather-adjusted-storage-balance"></a>', unsafe_allow_html=True)
st.header("Weather-Adjusted Storage Balance")
try:
    storage_model, storage_diagnostics, storage_coefficients = load_storage_balance_data()
except Exception:
    storage_model = pd.DataFrame()
    storage_diagnostics = pd.DataFrame()
    storage_coefficients = pd.DataFrame()

if storage_model.empty:
    st.info("Storage model outputs are not available yet. Run `python models/storage_balance.py` after loading EIA and NOAA data.")
else:
    storage_model["period"] = pd.to_datetime(storage_model["period"])
    latest_storage = storage_model.iloc[-1]
    storage_metrics = st.columns(5)
    storage_metrics[0].metric("Actual storage change", f"{latest_storage.actual_storage_change:,.1f} Bcf")
    storage_metrics[1].metric("Model expected change", f"{latest_storage.predicted_storage_change:,.1f} Bcf")
    storage_metrics[2].metric("Storage surprise", f"{latest_storage.storage_surprise:,.1f} Bcf")
    storage_metrics[3].metric("HDD anomaly", f"{latest_storage.hdd_anomaly:,.1f}")
    storage_metrics[4].metric("CDD anomaly", f"{latest_storage.cdd_anomaly:,.1f}")
    st.caption(f"Latest EIA storage observation: {latest_storage.period.date():%b %d, %Y}")

    balance_chart = storage_model.melt(
        id_vars="period",
        value_vars=["actual_storage_change", "predicted_storage_change"],
        var_name="Series",
        value_name="Bcf",
    )
    balance_chart["Series"] = balance_chart["Series"].map(
        {"actual_storage_change": "Actual", "predicted_storage_change": "Model expected"}
    )
    st.plotly_chart(
        px.line(balance_chart, x="period", y="Bcf", color="Series", title="Weekly storage change: actual vs model expected"),
        use_container_width=True,
    )

    surprise_figure = px.line(
        storage_model, x="period", y="storage_surprise", title="Storage surprise history"
    )
    surprise_figure.add_hline(y=0, line_color="gray", line_dash="dash")
    surprise_figure.update_yaxes(title="Actual minus model expected (Bcf)")
    st.plotly_chart(surprise_figure, use_container_width=True)

    st.subheader("Weather Context")
    recent_weather = storage_model.tail(26).copy()
    recent_weather["HDD normal"] = recent_weather["hdd"] - recent_weather["hdd_anomaly"]
    recent_weather["CDD normal"] = recent_weather["cdd"] - recent_weather["cdd_anomaly"]
    weather_left, weather_right = st.columns(2)
    with weather_left:
        hdd_chart = recent_weather.melt(
            id_vars="period", value_vars=["hdd", "HDD normal"], var_name="Series", value_name="Degree days"
        )
        st.plotly_chart(px.line(hdd_chart, x="period", y="Degree days", color="Series", title="Weekly HDD vs seasonal normal"), use_container_width=True)
    with weather_right:
        cdd_chart = recent_weather.melt(
            id_vars="period", value_vars=["cdd", "CDD normal"], var_name="Series", value_name="Degree days"
        )
        st.plotly_chart(px.line(cdd_chart, x="period", y="Degree days", color="Series", title="Weekly CDD vs seasonal normal"), use_container_width=True)
    st.caption("Anomalies are observed degree days less NOAA's 1981–2010 seasonal normal, summed over each storage week.")

    st.subheader("Model Diagnostics")
    if not storage_diagnostics.empty:
        diagnostic = storage_diagnostics.iloc[0]
        diagnostic_metrics = st.columns(3)
        diagnostic_metrics[0].metric("Out-of-sample MAE", f"{diagnostic.mae:,.2f} Bcf")
        diagnostic_metrics[1].metric("Out-of-sample RMSE", f"{diagnostic.rmse:,.2f} Bcf")
        diagnostic_metrics[2].metric("Out-of-sample R²", f"{diagnostic.r2:,.3f}")
        st.caption(
            f"Chronological training: {pd.to_datetime(diagnostic.train_start).date()} to "
            f"{pd.to_datetime(diagnostic.train_end).date()}; test: "
            f"{pd.to_datetime(diagnostic.test_start).date()} to {pd.to_datetime(diagnostic.test_end).date()}."
        )
    st.dataframe(storage_coefficients.rename(columns={"coefficient": "Linear coefficient (Bcf per unit)"}), hide_index=True, use_container_width=True)
    st.caption(
        "Negative surprise = less gas added to storage, or more gas withdrawn, than the model expected. "
        "Positive surprise = more gas added, or less withdrawn, than expected. This is a model estimate, not necessarily market consensus."
    )
