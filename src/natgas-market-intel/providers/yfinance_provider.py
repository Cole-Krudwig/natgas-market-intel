import pandas as pd
import yfinance as yf


MONTH_CODES = {
    1: "F",
    2: "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z",
}


class YFinanceProvider:
    """
    Provider for historical market data from Yahoo Finance.
    """

    @staticmethod
    def ng_contract_symbol(year: int, month: int) -> str:
        """
        Generate a Yahoo Finance ticker for an individual
        NYMEX Henry Hub Natural Gas futures contract.

        Example:
            year=2026, month=12 -> NGZ26.NYM
        """

        if month not in MONTH_CODES:
            raise ValueError("month must be between 1 and 12")

        month_code = MONTH_CODES[month]
        year_code = str(year)[-2:]

        return f"NG{month_code}{year_code}.NYM"

    def fetch_daily(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data for a Yahoo Finance ticker.

        Parameters
        ----------
        symbol : str
            Yahoo Finance ticker.
        start : str | None
            Start date in YYYY-MM-DD format.
        end : str | None
            End date in YYYY-MM-DD format.

        Returns
        -------
        pd.DataFrame
            Clean dataframe containing:
            date, open, high, low, close, adj_close,
            volume, symbol, source
        """

        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df.empty:
            return pd.DataFrame()

        # yfinance can return MultiIndex columns.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                col[0] if isinstance(col, tuple) else col
                for col in df.columns
            ]

        df = df.reset_index()

        # Standardize column names.
        df.columns = [
            str(col)
            .strip()
            .lower()
            .replace(" ", "_")
            for col in df.columns
        ]

        # Standardize Yahoo's date column.
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})

        # Ensure date is a proper pandas datetime.
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        df["symbol"] = symbol
        df["source"] = "yfinance"

        return df

    def fetch_ng_contract(
        self,
        year: int,
        month: int,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch an individual NYMEX Henry Hub Natural Gas contract.
        """

        symbol = self.ng_contract_symbol(
            year=year,
            month=month,
        )

        df = self.fetch_daily(
            symbol=symbol,
            start=start,
            end=end,
        )

        if df.empty:
            return df

        df["contract_year"] = year
        df["contract_month"] = month

        return df