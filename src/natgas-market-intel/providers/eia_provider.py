from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# Local development: load .env if present
current_dir = Path(__file__).resolve().parent
env_path = current_dir.parent / ".env"

if env_path.exists():
    load_dotenv(env_path)

# Works both locally and in GitHub Actions
EIA_KEY = os.getenv("EIA_KEY")

if not EIA_KEY:
    raise RuntimeError(
        "EIA_KEY environment variable is not set."
    )


# ============================================================
# CONFIG
# ============================================================

BASE_URL = "https://api.eia.gov/v2"


class EIAProvider:
    """
    Simple provider for EIA API v2 data.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or EIA_KEY

    def fetch_series(
        self,
        route: str,
        series_id: str,
        frequency: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch one EIA series from an API v2 route.

        Example route:
            natural-gas/prod/sum

        Example series:
            N9070US2
        """

        url = f"{BASE_URL}/{route}/data/"

        # Use tuples because EIA requires bracketed parameter names.
        params = [
            ("api_key", self.api_key),
            ("frequency", frequency),
            ("data[0]", "value"),
            ("facets[series][]", series_id),
            ("sort[0][column]", "period"),
            ("sort[0][direction]", "asc"),
            ("offset", "0"),
            ("length", "5000"),
        ]

        if start:
            params.append(("start", start))

        if end:
            params.append(("end", end))

        response = requests.get(
            url,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        payload = response.json()

        rows = payload.get(
            "response",
            {},
        ).get(
            "data",
            [],
        )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        return df