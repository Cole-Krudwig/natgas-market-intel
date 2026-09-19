from __future__ import annotations

from datetime import date, datetime, timezone
from io import StringIO
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

# Change this if your .env is stored somewhere else.
load_dotenv(PROJECT_DIR / ".env")

NOAA_KEY = os.getenv("NOAA_KEY")


# ---------------------------------------------------------------------
# NOAA CPC configuration
# ---------------------------------------------------------------------

CPC_BASE_URL = (
    "https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted"
)

HISTORICAL_BASE_URL = f"{CPC_BASE_URL}/daily_data"

NORMALS_BASE_URL = (
    f"{CPC_BASE_URL}/daily_data/climatology/1981-2010"
)

FORECAST_BASE_URL = (
    f"{CPC_BASE_URL}/daily_forecasts_7day/latest"
)


CENSUS_DIVISIONS = {
    "1": "New England",
    "2": "Middle Atlantic",
    "3": "East North Central",
    "4": "West North Central",
    "5": "South Atlantic",
    "6": "East South Central",
    "7": "West South Central",
    "8": "Mountain",
    "9": "Pacific",
    "CONUS": "CONUS",
}


class NOAAProvider:
    """
    Provider for NOAA/CPC population-weighted degree-day data.

    Main data returned:
        - Historical realized HDD
        - Historical realized CDD
        - 1981-2010 normal HDD
        - 1981-2010 normal CDD
        - Latest 7-day forecast HDD
        - Latest 7-day forecast CDD

    Regions:
        - 9 U.S. Census divisions
        - CONUS

    Notes:
        CPC degree-day files are public and do not require NOAA_KEY.
        NOAA_KEY is loaded for future use with NOAA CDO endpoints.
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()

        # Keep available for future NOAA CDO calls.
        self.noaa_key = NOAA_KEY

    # -----------------------------------------------------------------
    # HTTP helpers
    # -----------------------------------------------------------------

    def _get_text(self, url: str) -> str:
        response = self.session.get(
            url,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.text

    # -----------------------------------------------------------------
    # CPC file parser
    # -----------------------------------------------------------------

    @staticmethod
    def _parse_cpc_wide_file(
        text: str,
        value_name: str,
        date_format: str,
        year: int | None = None,
    ) -> pd.DataFrame:
        """
        Convert NOAA CPC pipe-delimited wide format into long format.

        CPC files look roughly like:

            Product: Daily Heating Degree Days
            Regions: CPC::Regions::CensusDivisions
            Weights: Population
            Region|20260101|20260102|...
            1|41|47|...
            2|43|44|...
            ...
            CONUS|29|28|...

        Returns:
            date | region_code | region | value_name
        """

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        # First three lines contain metadata.
        data_text = "\n".join(lines[3:])

        wide = pd.read_csv(
            StringIO(data_text),
            sep="|",
            dtype={"Region": str},
        )

        long = wide.melt(
            id_vars="Region",
            var_name="date_raw",
            value_name=value_name,
        )

        long.rename(
            columns={"Region": "region_code"},
            inplace=True,
        )

        long["region_code"] = (
            long["region_code"]
            .astype(str)
            .str.strip()
        )

        long["region"] = long["region_code"].map(
            CENSUS_DIVISIONS
        )

        # Historical / forecast:
        # YYYYMMDD
        if date_format == "%Y%m%d":
            long["date"] = pd.to_datetime(
                long["date_raw"],
                format="%Y%m%d",
                errors="coerce",
            )

        # Climatology:
        # MMDD only
        elif date_format == "%m%d":
            if year is None:
                # Leap year chosen deliberately so Feb 29 is valid.
                year = 2000

            long["date"] = pd.to_datetime(
                str(year) + long["date_raw"].astype(str),
                format="%Y%m%d",
                errors="coerce",
            )

            long["month"] = long["date"].dt.month
            long["day"] = long["date"].dt.day

        else:
            raise ValueError(
                f"Unsupported date format: {date_format}"
            )

        long[value_name] = pd.to_numeric(
            long[value_name],
            errors="coerce",
        )

        columns = [
            "date",
            "region_code",
            "region",
            value_name,
        ]

        if "month" in long.columns:
            columns += ["month", "day"]

        return (
            long[columns]
            .dropna(subset=["date", value_name])
            .sort_values(["date", "region_code"])
            .reset_index(drop=True)
        )

    # -----------------------------------------------------------------
    # Historical realized HDD / CDD
    # -----------------------------------------------------------------

    def fetch_historical_year(
        self,
        year: int,
    ) -> pd.DataFrame:
        """
        Download one calendar year of population-weighted
        HDD and CDD.

        Returns:
            date
            region_code
            region
            hdd
            cdd
        """

        hdd_url = (
            f"{HISTORICAL_BASE_URL}/{year}/"
            "Population.Heating.txt"
        )

        cdd_url = (
            f"{HISTORICAL_BASE_URL}/{year}/"
            "Population.Cooling.txt"
        )

        hdd_text = self._get_text(hdd_url)
        cdd_text = self._get_text(cdd_url)

        hdd = self._parse_cpc_wide_file(
            text=hdd_text,
            value_name="hdd",
            date_format="%Y%m%d",
        )

        cdd = self._parse_cpc_wide_file(
            text=cdd_text,
            value_name="cdd",
            date_format="%Y%m%d",
        )

        df = hdd.merge(
            cdd[
                [
                    "date",
                    "region_code",
                    "cdd",
                ]
            ],
            on=[
                "date",
                "region_code",
            ],
            how="outer",
        )

        return (
            df.sort_values(
                ["date", "region_code"]
            )
            .reset_index(drop=True)
        )

    def fetch_historical(
        self,
        start_year: int,
        end_year: int | None = None,
    ) -> pd.DataFrame:
        """
        Download several years of historical degree days.

        Example:
            provider.fetch_historical(2015, 2026)
        """

        if end_year is None:
            end_year = date.today().year

        frames = []

        for year in range(
            start_year,
            end_year + 1,
        ):
            print(
                f"Fetching NOAA CPC historical degree days "
                f"for {year}..."
            )

            try:
                yearly = self.fetch_historical_year(year)
                frames.append(yearly)

            except requests.HTTPError as exc:
                print(
                    f"Skipping {year}: {exc}"
                )

        if not frames:
            return pd.DataFrame()

        return (
            pd.concat(
                frames,
                ignore_index=True,
            )
            .drop_duplicates(
                subset=[
                    "date",
                    "region_code",
                ],
                keep="last",
            )
            .sort_values(
                ["date", "region_code"]
            )
            .reset_index(drop=True)
        )

    # -----------------------------------------------------------------
    # 1981-2010 climatological normals
    # -----------------------------------------------------------------

    def fetch_normals(self) -> pd.DataFrame:
        """
        Download NOAA CPC 1981-2010 normal HDD/CDD.

        NOAA stores climatology by month/day rather than actual year.

        Returns:
            month
            day
            region_code
            region
            normal_hdd
            normal_cdd
        """

        hdd_url = (
            f"{NORMALS_BASE_URL}/"
            "Population.Heating.txt"
        )

        cdd_url = (
            f"{NORMALS_BASE_URL}/"
            "Population.Cooling.txt"
        )

        hdd_text = self._get_text(hdd_url)
        cdd_text = self._get_text(cdd_url)

        hdd = self._parse_cpc_wide_file(
            text=hdd_text,
            value_name="normal_hdd",
            date_format="%m%d",
            year=2000,
        )

        cdd = self._parse_cpc_wide_file(
            text=cdd_text,
            value_name="normal_cdd",
            date_format="%m%d",
            year=2000,
        )

        df = hdd.merge(
            cdd[
                [
                    "month",
                    "day",
                    "region_code",
                    "normal_cdd",
                ]
            ],
            on=[
                "month",
                "day",
                "region_code",
            ],
            how="outer",
        )

        return (
            df[
                [
                    "month",
                    "day",
                    "region_code",
                    "region",
                    "normal_hdd",
                    "normal_cdd",
                ]
            ]
            .sort_values(
                [
                    "month",
                    "day",
                    "region_code",
                ]
            )
            .reset_index(drop=True)
        )

    # -----------------------------------------------------------------
    # Latest 7-day forecast
    # -----------------------------------------------------------------

    def fetch_latest_forecast(
        self,
    ) -> pd.DataFrame:
        """
        Download the latest population-weighted 7-day HDD/CDD
        forecast from NOAA CPC.

        IMPORTANT:
        Store every result in DuckDB with forecast_date.
        Do not overwrite yesterday's forecast. That history lets us
        calculate forecast revisions.

        Returns:
            forecast_date
            target_date
            region_code
            region
            forecast_hdd
            forecast_cdd
        """

        hdd_url = (
            f"{FORECAST_BASE_URL}/"
            "Population.Heating.txt"
        )

        cdd_url = (
            f"{FORECAST_BASE_URL}/"
            "Population.Cooling.txt"
        )

        hdd_text = self._get_text(hdd_url)
        cdd_text = self._get_text(cdd_url)

        hdd = self._parse_cpc_wide_file(
            text=hdd_text,
            value_name="forecast_hdd",
            date_format="%Y%m%d",
        )

        cdd = self._parse_cpc_wide_file(
            text=cdd_text,
            value_name="forecast_cdd",
            date_format="%Y%m%d",
        )

        df = hdd.merge(
            cdd[
                [
                    "date",
                    "region_code",
                    "forecast_cdd",
                ]
            ],
            on=[
                "date",
                "region_code",
            ],
            how="outer",
        )

        df.rename(
            columns={
                "date": "target_date",
            },
            inplace=True,
        )

        df["forecast_date"] = pd.Timestamp.now(
            tz="UTC"
        ).normalize().tz_localize(None)

        return (
            df[
                [
                    "forecast_date",
                    "target_date",
                    "region_code",
                    "region",
                    "forecast_hdd",
                    "forecast_cdd",
                ]
            ]
            .sort_values(
                [
                    "target_date",
                    "region_code",
                ]
            )
            .reset_index(drop=True)
        )

    # -----------------------------------------------------------------
    # Add normals and anomalies
    # -----------------------------------------------------------------

    def add_normals(
        self,
        df: pd.DataFrame,
        date_column: str,
    ) -> pd.DataFrame:
        """
        Attach climatological normal HDD/CDD and calculate anomalies.

        Works with either historical data or forecast data.
        """

        output = df.copy()

        output[date_column] = pd.to_datetime(
            output[date_column]
        )

        output["month"] = (
            output[date_column].dt.month
        )

        output["day"] = (
            output[date_column].dt.day
        )

        normals = self.fetch_normals()

        output = output.merge(
            normals[
                [
                    "month",
                    "day",
                    "region_code",
                    "normal_hdd",
                    "normal_cdd",
                ]
            ],
            on=[
                "month",
                "day",
                "region_code",
            ],
            how="left",
        )

        if "hdd" in output.columns:
            output["hdd_anomaly"] = (
                output["hdd"]
                - output["normal_hdd"]
            )

        if "cdd" in output.columns:
            output["cdd_anomaly"] = (
                output["cdd"]
                - output["normal_cdd"]
            )

        if "forecast_hdd" in output.columns:
            output["hdd_anomaly"] = (
                output["forecast_hdd"]
                - output["normal_hdd"]
            )

        if "forecast_cdd" in output.columns:
            output["cdd_anomaly"] = (
                output["forecast_cdd"]
                - output["normal_cdd"]
            )

        output.drop(
            columns=[
                "month",
                "day",
            ],
            inplace=True,
        )

        return output

    # -----------------------------------------------------------------
    # Convenience functions
    # -----------------------------------------------------------------

    def fetch_historical_with_normals(
        self,
        start_year: int,
        end_year: int | None = None,
    ) -> pd.DataFrame:
        historical = self.fetch_historical(
            start_year=start_year,
            end_year=end_year,
        )

        return self.add_normals(
            historical,
            date_column="date",
        )

    def fetch_forecast_with_normals(
        self,
    ) -> pd.DataFrame:
        forecast = self.fetch_latest_forecast()

        return self.add_normals(
            forecast,
            date_column="target_date",
        )

    def fetch_conus_forecast(
        self,
    ) -> pd.DataFrame:
        """
        Convenience method for dashboard-level U.S. weather.
        """

        df = self.fetch_forecast_with_normals()

        return (
            df[
                df["region_code"] == "CONUS"
            ]
            .copy()
            .reset_index(drop=True)
        )


if __name__ == "__main__":
    provider = NOAAProvider()

    print("\nLatest CONUS forecast:")
    forecast = provider.fetch_conus_forecast()
    print(forecast)

    print("\nLatest forecast for all Census divisions:")
    forecast_all = provider.fetch_forecast_with_normals()
    print(forecast_all.head(20))