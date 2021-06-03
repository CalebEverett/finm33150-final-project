from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timezone
import hmac
import gzip
import os
from typing import Dict, List, Tuple
from urllib.request import urlretrieve

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from canvasapi import Canvas
import numpy as np
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly import colors
from plotly.subplots import make_subplots
import quandl
import requests
from requests import Request
from scipy import stats
from tabulate import tabulate
from tqdm.notebook import tqdm

# =============================================================================
# Credentials
# =============================================================================

quandl.ApiConfig.api_key = os.getenv("QUANDL_API_KEY")


# =============================================================================
# Canvas
# =============================================================================


def download_files(filename_frag: str):
    """Downloads files from Canvas with `filename_frag` in filename."""

    url = os.getenv("CANVAS_URL")
    token = os.getenv("CANVAS_TOKEN")

    course_id = 33395
    canvas = Canvas(url, token)
    course = canvas.get_course(course_id)

    for f in course.get_files():
        if filename_frag in f.filename:
            print(f.filename, f.id)
            file = course.get_file(f.id)
            file.download(file.filename)


# =============================================================================
# Reading Data
# =============================================================================


def get_trade_data(pair: str, year: str, path: str = "accumulation_opportunity/data"):
    """Reads local gzipped trade data file and return dataframe."""

    dtypes = {
        "PriceMillionths": int,
        "Side": int,
        "SizeBillionths": int,
        "timestamp_utc_nanoseconds": int,
    }

    filename = f"trades_narrow_{pair}_{year}.delim.gz"
    delimiter = {"2018": "|", "2021": "\t"}[year]

    with gzip.open(f"{path}/{filename}") as f:
        df = pd.read_csv(f, delimiter=delimiter, usecols=dtypes.keys(), dtype=dtypes)

    df.timestamp_utc_nanoseconds = pd.to_datetime(df.timestamp_utc_nanoseconds)

    return df.set_index("timestamp_utc_nanoseconds")


# =============================================================================
# Price Data
# =============================================================================


def get_table(dataset_code: str, database_code: str = "ZACKS"):
    """Downloads Zacks fundamental table from export api to local zip file."""

    url = (
        f"https://www.quandl.com/api/v3/datatables/{database_code}/{dataset_code}.json"
    )
    r = requests.get(
        url, params={"api_key": os.getenv("QUANDL_API_KEY"), "qopts.export": "true"}
    )
    data = r.json()
    urlretrieve(
        data["datatable_bulk_download"]["file"]["link"],
        f"zacks_{dataset_code.lower()}.zip",
    )


def load_table_files(table_filenames: Dict):
    """Loads Zacks fundamentals tables from csv files."""

    dfs = []
    for v in tqdm(table_filenames.values()):
        dfs.append(pd.read_csv(v, low_memory=False))

    return dfs


def get_hash(string: str) -> str:
    """Returns md5 hash of string."""

    return hashlib.md5(str(string).encode()).hexdigest()


def fetch_ticker(
    dataset_code: str, query_params: Dict = None, database_code: str = "EOD"
):
    """Fetches price data for a single ticker."""

    url = f"https://www.quandl.com/api/v3/datasets/{database_code}/{dataset_code}.json"

    params = dict(api_key=os.getenv("QUANDL_API_KEY"))
    if query_params is not None:
        params = dict(**params, **query_params)

    r = requests.get(url, params=params)

    dataset = r.json()["dataset"]
    df = pd.DataFrame(
        dataset["data"], columns=[c.lower() for c in dataset["column_names"]]
    )
    df["ticker"] = dataset["dataset_code"]

    return df.sort_values("date")


def fetch_all_tickers(tickers: List, query_params: Dict) -> pd.DataFrame:
    """Fetches price data from Quandl for each ticker in provide list and
    returns a dataframe of them concatenated together.
    """

    df_prices = pd.DataFrame()
    for t in tqdm(tickers):
        try:
            df = fetch_ticker(t, query_params)
            df_prices = pd.concat([df_prices, df])
        except:
            print(f"Couldn't get prices for {t}.")

    not_missing_data = (
        df_prices.set_index(["ticker", "date"])[["adj_close"]]
        .unstack("date")
        .isna()
        .sum(axis=1)
        == 0
    )

    df_prices = df_prices[
        df_prices.ticker.isin(not_missing_data[not_missing_data].index)
    ]

    return df_prices.set_index(["ticker", "date"])


# =============================================================================
# Download Preprocessed Files from S3
# =============================================================================


def upload_s3_file(filename: str):
    """Uploads file to S3. Requires credentials with write permissions to exist
    as environment variables.
    """

    client = boto3.client("s3")
    client.upload_file(filename, "finm33150", filename)


def download_s3_file(filename: str):
    """Downloads file from read only S3 bucket."""

    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    client.download_file("finm33150", filename, filename)


# =============================================================================
# Binance Futures API Calls
# =============================================================================


def sign_request(tbd):

    base_url = "https://fapi.binance.com"
    end_point = "/fapi/v1/continuousKlines"

    params = dict()
    request = Request("GET", f"{base_url}{end_point}", params=params)
    prepared = request.prepare()

    signature_payload = f"{prepared.path_url.split('?')[-1]}".encode()
    params["signature"] = hmac.new(
        os.getenv("BINANCE_API_SECRET").encode(), signature_payload, "sha256"
    ).hexdigest()


def get_coin_futures_positions():

    base_url = "https://dapi.binance.com"
    end_point = "/dapi/v1/positionRisk"

    params = dict(timestamp=round(datetime.now(timezone.utc).timestamp() * 1000))
    request = Request("GET", f"{base_url}{end_point}", params=params)
    prepared = request.prepare()

    signature_payload = f"{prepared.path_url.split('?')[-1]}".encode()
    params["signature"] = hmac.new(
        os.getenv("BINANCE_API_SECRET").encode(), signature_payload, "sha256"
    ).hexdigest()

    r = requests.get(
        f"{base_url}{end_point}",
        params=params,
        headers={"X-MBX-APIKEY": os.getenv("BINANCE_API_KEY")},
    )

    return r.json()


def get_margin_account_details():

    base_url = "https://api.binance.com"
    end_point = "/sapi/v1/margin/account"

    params = dict(timestamp=round(datetime.now(timezone.utc).timestamp() * 1000))
    request = Request("GET", f"{base_url}{end_point}", params=params)
    prepared = request.prepare()

    signature_payload = f"{prepared.path_url.split('?')[-1]}".encode()
    params["signature"] = hmac.new(
        os.getenv("BINANCE_API_SECRET").encode(), signature_payload, "sha256"
    ).hexdigest()

    r = requests.get(
        f"{base_url}{end_point}",
        params=params,
        headers={"X-MBX-APIKEY": os.getenv("BINANCE_API_KEY")},
    )

    return r.json()


def get_exchange_info():
    """
    Gets current exchange trading rules and symbol information.
    """

    base_url = "https://api.binance.com"
    end_point = "/api/v3/exchangeInfo"

    r = requests.get(
        f"{base_url}{end_point}",
        headers={"X-MBX-APIKEY": os.getenv("BINANCE_API_KEY")},
    )

    df = pd.DataFrame(
        r.json()["symbols"],
    ).set_index("symbol")

    return df


def get_candlestick_df(resp_json: List[Dict]) -> pd.DataFrame:
    """
    Returns dataframe from candlestick data json response:
    """

    df = pd.DataFrame(
        resp_json,
        columns=[
            "openTime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "closeTime",
            "quoteAssetVolume",
            "numTrades",
            "takerBuyBaseAssetVolume",
            "takerBuyQuoteAssetVolume",
            "ignore",
        ],
    ).astype(
        {
            "openTime": int,
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
            "closeTime": int,
            "quoteAssetVolume": float,
            "numTrades": int,
            "takerBuyBaseAssetVolume": float,
            "takerBuyQuoteAssetVolume": float,
            "ignore": object,
        },
    )

    df.openTime = pd.to_datetime(df.openTime, unit="ms")
    df.closeTime = pd.to_datetime(df.closeTime, unit="ms")

    df["per_return"] = np.log(df.close / df.close.shift())

    return df.set_index("openTime")


def get_utc_timestamp(iso_format_datetime: str):
    return int(
        datetime.fromisoformat(iso_format_datetime)
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1000
    )


def get_utc_timestamp_now():
    return int(
        datetime.now(timezone.utc).replace(tzinfo=timezone.utc).timestamp() * 1000
    )


def get_times(start_time: str, end_time: str):
    """
    Return utc timestamps from isoformat datetime strings.
    """

    if start_time is not None:
        start_time = get_utc_timestamp(start_time)

    if end_time is None:
        end_time = get_utc_timestamp_now()
    else:
        end_time = min(get_utc_timestamp_now(), get_utc_timestamp(end_time))

    return start_time, end_time


def fetch_request(base_url: str, end_point: str, params: Dict) -> pd.DataFrame:
    """
    Returns list of records iterating through multiple requests to
    fetch continuous history from params.startTime to params.endTime.
    """

    r = requests.get(
        f"{base_url}{end_point}",
        params=params,
        headers={"X-MBX-APIKEY": os.getenv("BINANCE_API_KEY")},
    )

    records = r.json()

    while r.json()[-1][6] < params["endTime"]:
        params["startTime"] = r.json()[-1][6] + 1
        r = requests.get(
            f"{base_url}{end_point}",
            params=params,
            headers={"X-MBX-APIKEY": os.getenv("BINANCE_API_KEY")},
        )
        records.extend(r.json())

    return records


def get_continuous_contracts(
    pair: str,
    interval: str = "8h",
    contract_type: str = "PERPETUAL",
    limit: int = None,
    start_time: str = None,
    end_time: str = None,
):
    """
    Fetches continuous contract data.

    Args:
        interval: Must be one of the following, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h
                    6h, 8h, 12h, 1d, 3d, 1w, 1M.
        contact_type: Must be one of PERPETUAL, CURRENT_MONTH, NEXT_MONTH,
            CURRENT_QUARTER, NEXT_QUARTER

    """

    base_url = "https://fapi.binance.com"
    end_point = "/fapi/v1/continuousKlines"

    start_time, end_time = get_times(start_time, end_time)

    params = {
        "limit": limit,
        "pair": pair,
        "interval": interval,
        "contractType": contract_type,
        "startTime": start_time,
        "endTime": end_time,
    }

    records = fetch_request(base_url, end_point, params)

    return get_candlestick_df(records)


def get_klines(
    symbol: str,
    interval: str = "8h",
    limit: int = None,
    start_time: str = None,
    end_time: str = None,
    verbose: bool = False,
):
    """
    Fetches kline/candlestick data for symbol.

    Args:
        interval: Must be one of the following, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h
                    6h, 8h, 12h, 1d, 3d, 1w, 1M.
        contact_type: Must be one of PERPETUAL, CURRENT_MONTH, NEXT_MONTH,
            CURRENT_QUARTER, NEXT_QUARTER

    """

    base_url = "https://api.binance.com"
    end_point = "/api/v3/klines"

    start_time, end_time = get_times(start_time, end_time)

    params = {
        "limit": limit,
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
    }

    records = fetch_request(base_url, end_point, params)

    return get_candlestick_df(records)


def get_funding_rate_history(
    symbol: str,
    limit: int = None,
    start_time: str = None,
    end_time: str = None,
):
    """
    Fetches funding rate history. Times are rounded to neaerest second to
    facilitate comparison with prices on the same index.
    """

    base_url = "https://fapi.binance.com"
    end_point = "/fapi/v1/fundingRate"

    start_time, end_time = get_times(start_time, end_time)

    params = {
        "limit": limit,
        "symbol": symbol,
        "startTime": start_time,
        "endTime": end_time,
    }

    r = requests.get(
        f"{base_url}{end_point}",
        params=params,
        headers={"X-MBX-APIKEY": os.getenv("BINANCE_API_KEY")},
    )

    records = r.json()

    while r.json()[-1]["fundingTime"] < (params["endTime"] - 8 * 60 * 60 * 1000):

        params["startTime"] = r.json()[-1]["fundingTime"] + 1
        r = requests.get(
            f"{base_url}{end_point}",
            params=params,
            headers={"X-MBX-APIKEY": os.getenv("BINANCE_API_KEY")},
        )
        records.extend(r.json())

    df = pd.DataFrame(records)

    df.fundingRate = df.fundingRate.astype(float)

    df.fundingTime = pd.to_datetime(df.fundingTime, unit="ms").round("1s")

    return df.set_index("fundingTime")[["fundingRate"]]


# =============================================================================
# Strategy
# =============================================================================


class PositionType(Enum):
    LONG: str = "long"
    SHORT: str = "short"


@dataclass
class Position:
    position_type: PositionType
    open_date: str
    security: str
    shares: int
    open_price: float
    open_transact_cost: float = 0
    close_price: float = None
    close_transact_cost: float = 0
    closed: bool = False
    close_date: str = None
    transact_cost_percent: float = 0.0003

    def __post_init__(self):
        self.open_transact_cost = (
            self.shares * self.transact_cost_percent * self.open_price
        )

    @property
    def open_value(self):
        return self.shares * self.open_price

    @property
    def close_value(self):
        return self.shares * self.close_price

    @property
    def transact_cost(self):
        return self.open_transact_cost + self.close_transact_cost

    @property
    def gross_profit(self):
        if self.closed:
            if self.position_type == PositionType.LONG:
                profit = self.close_value - self.open_value
            else:
                profit = self.open_value - self.close_value
            return profit
        else:
            raise Exception("Position is still open.")

    @property
    def net_profit(self):
        if self.closed:
            return self.gross_profit - self.transact_cost
        else:
            raise Exception("Position is still open.")

    def unrealized_profit(self, current_price: float):
        if self.closed:
            raise Exception("Position is closed.")
        else:
            if self.position_type == PositionType.LONG:
                mv = self.shares * (current_price - self.open_price)
            else:
                mv = self.shares * (self.open_price - current_price)
            return mv - self.shares * self.transact_cost_percent * current_price

    def close(self, close_date: str, close_price: float):
        self.close_date = close_date
        self.close_price = close_price
        self.close_transact_cost = (
            self.shares * self.transact_cost_percent * self.close_price
        )
        self.closed = True


class Strategy:
    """Class for conducting a backtest of a spread trading strategy.

    Properties:
        pair: Tuple of securities
        ticks: Pandas DataFrame with `adj_close`, `position_size`, and `adj_returns`
            columns.
        open_threshold: Absolute value of difference in returns above which
            a position will be opened if one is not already open.
        close_threshold: Absolute value of difference in returns above which
            a position will be opened if one is not already open.
        gross_profit: Realized gross profit as of `current_date`
        transact_cost: Cash transaction costs incurred as of `current_date`
        start_date: Starting date of the strategy
        current_date: Current date of strategy
        end_date: str = Ending date of strategy

        long_position = Open long position as of `current_date` if any
        short_position = Open short position as of `current_date` if any
        closed_positions = Closed positions as of `current date`
    """

    def __init__(
        self,
        pair: Tuple,
        df_ticks: pd.DataFrame,
        spread_column: Tuple,
        open_threshold: float,
        close_threshold: float,
        closed_positions: list,
        run: bool = True,
        transact_cost_percent: float = 0.0003,
        funding_rate_freq: str = "8h",
    ):

        self.pair = pair
        self.df_ticks = df_ticks
        self.spread_column = spread_column
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        self.position_profit: float = 0
        self.funding_rate_profit: float = 0
        self.transact_cost: float = 0
        self.current_date: str = None
        self.long_position: Position = None
        self.short_position: Position = None
        self.closed_positions = closed_positions
        self.transact_cost_percent = transact_cost_percent

        self.start_date = self.df_ticks.index.min().strftime("%Y-%m-%d %H:%M")
        self.end_date = self.df_ticks.index.max().strftime("%Y-%m-%d %H:%M")

        self.capital = (
            self.df_ticks["position_size"] * self.df_ticks["adj_close"]
        ).max().max() * 2

        # Frequency of `BM` is last business day of month to make sure
        # positions are closed if the calendar last day of the month
        # is not a trading day.
        self.funding_rate_dates = (
            pd.date_range(self.start_date, self.end_date, freq=funding_rate_freq)
            .strftime("%Y-%m-%d %H:%M")
            .to_list()
        )

        if run:
            self.run()

    @property
    def net_profit(self):
        return self.position_profit + self.funding_rate_profit - self.transact_cost

    @property
    def unrealized_profit(self):
        if self.long_position:
            return self.long_position.unrealized_profit(
                self.current_prices[self.long_position.security]
            ) + self.short_position.unrealized_profit(
                self.current_prices[self.short_position.security]
            )
        else:
            return 0

    def open_long_position(
        self, open_date: str, security: str, shares: int, open_price
    ):
        if self.long_position is not None:
            raise Exception("An open long position already exists.")

        if security not in self.pair:
            raise Exception(
                f"{security} is not included in strategy securities:"
                f" {str(self.pair)}"
            )

        if open_date < self.current_date:
            raise Exception(
                f"Position open date of {open_date} is before strategy current"
                f"date of {self.current_date}"
            )

        self.long_position = Position(
            position_type=PositionType.LONG,
            open_date=open_date,
            security=security,
            shares=shares,
            open_price=open_price,
            transact_cost_percent=self.transact_cost_percent,
        )

        self.transact_cost += self.long_position.open_transact_cost

    def open_short_position(
        self, open_date: str, security: str, shares: int, open_price
    ):
        if self.short_position is not None:
            raise Exception("An open short position already exists.")

        self.short_position = Position(
            position_type=PositionType.SHORT,
            open_date=open_date,
            security=security,
            shares=shares,
            open_price=open_price,
            transact_cost_percent=self.transact_cost_percent,
        )

        self.transact_cost += self.short_position.open_transact_cost

    def close_long_position(self, close_date: str, close_price: float):
        if self.long_position is None:
            raise Exception("There is no open long position.")

        self.long_position.close(close_date, close_price)

        self.position_profit += self.long_position.gross_profit
        self.transact_cost += self.long_position.close_transact_cost

        self.closed_positions.append(self.long_position)
        self.long_position = None

    def close_short_position(self, close_date: str, close_price: float):
        if self.short_position is None:
            raise Exception("There is no open long short.")

        self.short_position.close(close_date, close_price)

        self.position_profit += self.short_position.gross_profit
        self.transact_cost += self.short_position.close_transact_cost

        self.closed_positions.append(self.short_position)
        self.short_position = None

    def record_trade(
        self,
        open_position: bool,
        date: str,
        spread: int,
        long_security: str,
        short_security: str,
    ):
        if open_position:
            if short_security == self.pair[1]:
                trade_type = "short"
            else:
                trade_type = "buy"
        else:
            if short_security == self.pair[1]:
                trade_type = "buy"
            else:
                trade_type = "short"

        self.trades.append(
            {
                "date": date,
                "trade_type": trade_type,
                "spread": spread,
                "long_security": long_security,
                "short_security": short_security,
                "open_position": open_position,
            }
        )

    def run(self):
        self.stats = []
        for tick in self.df_ticks.iterrows():
            date, tick = tick
            spread = tick[self.spread_column]
            self.current_date = date.strftime("%Y-%m-%d %H:%M")
            self.current_prices = tick.adj_close
            short_security = self.pair[1] if spread > 0 else self.pair[0]
            long_security = self.pair[0] if spread > 0 else self.pair[1]

            # Just testing long position since both long and short positions
            # are always open or None. Don't open a position on the last day
            # of the strategy.

            # Closing positions first so that opening logic works whether opening
            # from not having an open position or from after having sold because the
            # spread reversed.

            if self.current_date == self.start_date:
                prior_total_profit = 0
            else:
                prior_total_profit = self.stats[-1]["total_profit"]

            if (
                self.long_position is not None
                and self.current_date in self.funding_rate_dates
            ):
                if self.long_position.security == self.pair[1]:
                    self.funding_rate_profit -= (
                        self.long_position.shares
                        * self.current_prices[self.pair[1]]
                        * tick.adj_return.funding_rate_prior
                    )
                else:
                    self.funding_rate_profit += (
                        self.short_position.shares
                        * self.current_prices[self.pair[1]]
                        * tick.adj_return.funding_rate_prior
                    )

            if self.long_position is not None and (
                (
                    abs(spread) < self.close_threshold
                    or self.current_date == self.end_date
                )
                or self.long_position.security != long_security
            ):

                self.close_long_position(
                    close_date=self.current_date,
                    close_price=tick.adj_close[self.long_position.security],
                )

                self.close_short_position(
                    close_date=self.current_date,
                    close_price=tick.adj_close[self.short_position.security],
                )

            if (
                abs(spread) > self.open_threshold
                and self.long_position is None
                and self.current_date != self.end_date
            ):

                self.open_long_position(
                    open_date=self.current_date,
                    security=long_security,
                    shares=tick.position_size[long_security],
                    open_price=tick.adj_close[long_security],
                )

                self.open_short_position(
                    open_date=self.current_date,
                    security=short_security,
                    shares=tick.position_size[short_security],
                    open_price=tick.adj_close[short_security],
                )

            total_profit = self.net_profit + self.unrealized_profit
            tick_profit = total_profit - prior_total_profit
            total_return = np.log(total_profit + self.capital) - np.log(self.capital)
            tick_return = np.log(tick_profit + self.capital) - np.log(self.capital)

            self.stats.append(
                {
                    "date": date,
                    "funding_rate_profit": self.funding_rate_profit,
                    "position_profit": self.position_profit,
                    "transact_cost": -self.transact_cost,
                    "realized_profit": self.net_profit,
                    "unrealized_profit": self.unrealized_profit,
                    "total_profit": total_profit,
                    "tick_profit": tick_profit,
                    "total_return": total_return,
                    "tick_return": tick_return,
                }
            )

    def get_day_trades(self, date: str):
        trades = []
        headers = ["trans", "sec", "shrs", "price", "profit"]
        opened_positions = [p for p in self.closed_positions if p.open_date == date]
        closed_positions = [p for p in self.closed_positions if p.close_date == date]

        for p in opened_positions:
            trans = "buy" if p.position_type.value == "long" else "short"
            trades.append([trans, p.security, p.shares, p.open_price, -p.transact_cost])

        for p in closed_positions:
            trans = "sell" if p.position_type.value == "long" else "cover"
            trades.append([trans, p.security, p.shares, p.close_price, p.net_profit])

        hover_text = tabulate(
            trades,
            tablefmt="plain",
            headers=headers,
            floatfmt=("", "", ",.4f", ",.0f", ",.0f"),
        ).replace("\n", "<br>")

        if opened_positions:
            long_position = [
                p for p in opened_positions if p.position_type.value == "long"
            ][0]
            if long_position.security == self.pair[0]:
                trade_type = "short"
            else:
                trade_type = "buy"
        elif closed_positions:
            long_position = [
                p for p in closed_positions if p.position_type.value == "long"
            ][0]
            if long_position.security == self.pair[0]:
                trade_type = "buy"
            else:
                trade_type = "short"

        size = 6.5 if opened_positions and closed_positions else 4.5

        return (
            date,
            self.df_ticks.loc[date, self.spread_column],
            trade_type,
            size,
            hover_text,
        )

    def plot(
        self,
        title_text: str = "Spread Trading Chart",
        date_fmt: str = "%Y-%m-%d %H:%M",
    ) -> go.Figure:
        """

        Returns:
            A plotly Figure ready for plotting

        """

        dates = self.df_ticks.index.get_level_values("date")
        start_date = dates.min()
        end_date = dates.max()
        date_range = pd.date_range(start_date, end_date, freq="D")
        range_breaks = date_range[~date_range.isin(dates)]

        label_fn = lambda p: f"{p[1]}-{p[0]}"

        fig = go.Figure()

        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=[
                f"Trades: Total = {len(self.closed_positions) // 2}",
                f"Returns: Total = {self.stats[-1]['total_return']:0.4f}",
                f"Profit: Total = ${self.stats[-1]['total_profit']:0.0f}",
            ],
            shared_xaxes=True,
            vertical_spacing=0.10,
            specs=[
                [dict(secondary_y=True)],
                [dict(secondary_y=True)],
                [dict(secondary_y=True)],
            ],
        )

        # =======================
        # Spread
        # =======================

        fig.append_trace(
            go.Scatter(
                y=self.df_ticks.adj_return.funding_rate,
                x=dates,
                name="funding_rate",
                line=dict(width=1),
            ),
            row=1,
            col=1,
        )

        fig.append_trace(
            go.Scatter(
                y=self.df_ticks.adj_return.spread,
                x=dates,
                name="spread",
                line=dict(width=1),
            ),
            row=1,
            col=1,
        )

        def add_band(positive: int = 1):
            fig.append_trace(
                go.Scatter(
                    y=[self.close_threshold * positive] * len(dates),
                    x=dates,
                    name="close_threshold",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            fig.append_trace(
                go.Scatter(
                    y=[self.open_threshold * positive] * len(dates),
                    x=dates,
                    name="open_threshold",
                    fill="tonexty",
                    line=dict(width=0),
                    line_color=COLORS[4],
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        add_band()
        add_band(-1)

        # =======================
        # Trades
        # =======================

        trade_dates = sorted(
            sum([[p.open_date, p.close_date] for p in self.closed_positions], [])
        )

        df_trades = pd.DataFrame(
            map(self.get_day_trades, trade_dates),
            columns=["date", "spread", "trans", "marker_size", "text"],
        )

        fig.append_trace(
            go.Scatter(
                y=df_trades.spread,
                x=df_trades.date,
                name="trades",
                mode="markers",
                marker=dict(
                    color=df_trades.trans.map({"buy": "green", "short": "red"}),
                    size=df_trades.marker_size,
                    line=dict(width=0),
                ),
                text=df_trades.text,
                hovertemplate="%{text}",
            ),
            row=1,
            col=1,
        )

        # =======================
        # Returns
        # =======================

        df_stats = pd.DataFrame(self.stats)
        fig.add_trace(
            go.Scatter(
                y=df_stats["total_return"],
                x=df_stats["date"],
                name="total_return",
                line=dict(width=1),
                line_color=COLORS[2],
            ),
            secondary_y=False,
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                y=df_stats["tick_return"],
                x=df_stats["date"],
                name="tick_return",
                line=dict(width=1),
                line_color=COLORS[1],
            ),
            row=2,
            col=1,
            secondary_y=True,
        )

        # =======================
        # Profit
        # =======================

        fig.add_trace(
            go.Scatter(
                y=df_stats["transact_cost"],
                x=df_stats["date"],
                name="transact_cost",
                line=dict(width=1),
                fill="tonexty",
            ),
            secondary_y=False,
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                y=df_stats["position_profit"] + df_stats["transact_cost"],
                x=df_stats["date"],
                name="net_position_profit",
                line=dict(width=1),
                fill="tonexty",
                line_color=COLORS[0],
            ),
            secondary_y=False,
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                y=df_stats["position_profit"]
                + df_stats["funding_rate_profit"]
                + df_stats["transact_cost"],
                x=df_stats["date"],
                name="fund_plus_pos_profit",
                line=dict(width=1),
                fill="tonexty",
                line_color=COLORS[1],
            ),
            secondary_y=False,
            row=3,
            col=1,
        )

        # =======================
        # Figure
        # =======================

        title_text = (
            f"{title_text}: {label_fn(self.pair)}: {start_date.strftime(date_fmt)}"
            f" - {end_date.strftime(date_fmt)}"
        )

        fig.update_yaxes(range=(-0.002, 0.004), row=1, col=1)

        fig.update_layout(
            template="none",
            autosize=True,
            height=1200,
            title_text=title_text,
        )

        # shapes = []
        # for month_end in self.month_ends:
        #     shapes.append(
        #         dict(
        #             type="line",
        #             yref="paper",
        #             y0=0.55,
        #             y1=0.98,
        #             xref="x",
        #             line_dash="dot",
        #             line_width=1,
        #             x0=month_end,
        #             x1=month_end,
        #         )
        #     )
        #     shapes.append(
        #         dict(
        #             type="line",
        #             yref="paper",
        #             y0=0.05,
        #             y1=0.42,
        #             xref="x",
        #             line_dash="dot",
        #             line_width=1,
        #             x0=month_end,
        #             x1=month_end,
        #         )
        #     )

        fig.update_layout(
            # shapes=shapes,
            hoverlabel=dict(font_family="Courier New, monospace"),
            # hovermode="x unified",
        )
        fig.update_xaxes(rangebreaks=[dict(values=range_breaks)])

        returns_annotation = get_moments_annotation(
            df_stats.tick_return,
            xref="paper",
            yref="paper",
            x=1,
            y=0.4,
            xanchor="left",
            labels=IS_labels,
            title="",
        )
        fig.add_annotation(returns_annotation)

        # spread_annotation = get_moments_annotation(
        #     self.df_ticks.adj_return.funding_rate,
        #     xref="paper",
        #     yref="paper",
        #     x=1,
        #     y=0.8,
        #     xanchor="left",
        #     labels=IS_labels,
        #     title="",
        # )
        # fig.add_annotation(spread_annotation)

        return fig


def get_ticks(
    df_perpetual: pd.DataFrame,
    df_spot: pd.DataFrame,
    df_funding: pd.DataFrame,
    dollar_position_size: float,
):
    """Creates table of spread, returns, closing prices and trade amounts to be processed
    iteratively by a Strategy instance.
    """

    df_ticks = pd.DataFrame()
    for asset, df in {"perpetual": df_perpetual, "spot": df_spot}.items():
        df = df[["open", "close", "per_return"]].copy()
        df["position_size"] = dollar_position_size / df["close"]
        df.columns = pd.MultiIndex.from_tuples(
            [
                ("adj_open", asset),
                ("adj_close", asset),
                ("adj_return", asset),
                ("position_size", asset),
            ],
            names=["series", "asset"],
        )
        df_ticks = pd.concat([df_ticks, df], axis=1)

    df_ticks[("adj_return", "funding_rate")] = df_funding.fundingRate
    df_ticks[("adj_return", "funding_rate")] = df_ticks[
        ("adj_return", "funding_rate")
    ].ffill()
    df_ticks[("adj_return", "funding_rate_prior")] = df_ticks[
        ("adj_return", "funding_rate")
    ].shift()

    df_ticks[("adj_return", "spread")] = (
        df_ticks.adj_close.perpetual / df_ticks.adj_close.spot - 1
    )

    df_ticks.index.name = "date"
    df_ticks = df_ticks.sort_index(axis=1)

    return df_ticks.dropna()


# =============================================================================
# Charts
# =============================================================================

COLORS = colors.qualitative.T10


def make_price_volume_chart(df_pv: pd.DataFrame, title: str):
    """
    Returns figure for price volume chart.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.7]
    )
    fig.add_trace(
        go.Candlestick(
            x=df_pv.index,
            open=df_pv.open,
            low=df_pv.low,
            high=df_pv.high,
            close=df_pv.close,
            line=dict(width=1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=df_pv.index, y=df_pv.volume, marker=dict(color=COLORS[0])),
        row=2,
        col=1,
    )

    title_text = (
        f"{title}<br>"
        f"{df_pv.index.min().strftime('%Y-%m-%d %H:%M')}"
        f" - {df_pv.index.max().strftime('%Y-%m-%d %H:%M')}"
    )

    fig.update(
        layout_xaxis_rangeslider_visible=False,
        layout_title=title_text,
        layout_showlegend=False,
    )

    return fig


IS_labels = [
    ("obs", lambda x: f"{x:>7d}"),
    ("min:max", lambda x: f"{x[0]:>0.4f}:{x[1]:>0.3f}"),
    ("mean", lambda x: f"{x:>7.4f}"),
    ("std", lambda x: f"{x:>7.4f}"),
    ("skewness", lambda x: f"{x:>7.4f}"),
    ("kurtosis", lambda x: f"{x:>7.4f}"),
]


def get_moments_annotation(
    s: pd.Series,
    xref: str,
    yref: str,
    x: float,
    y: float,
    xanchor: str,
    title: str,
    labels: List,
) -> go.layout.Annotation:
    """Calculates summary statistics for a series and returns and
    Annotation object.
    """
    moments = list(stats.describe(s.to_numpy()))
    moments[3] = np.sqrt(moments[3])

    sharpe = s.mean() / s.std()

    return go.layout.Annotation(
        text=(
            f"<b>sharpe: {sharpe:>8.4f}</b><br>"
            + ("<br>").join(
                [f"{k[0]:<9}{k[1](moments[i])}" for i, k in enumerate(labels)]
            )
        ),
        align="left",
        showarrow=False,
        xref=xref,
        yref=yref,
        x=x,
        y=y,
        bordercolor="black",
        borderwidth=0.5,
        borderpad=2,
        bgcolor="white",
        xanchor=xanchor,
        yanchor="top",
        font_family="Courier New, monospace",
    )


def make_overview_chart(
    series: pd.DataFrame, title: str, subtitle_base: str = "Log Returns"
) -> go.Figure:

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            subtitle_base,
            f"{subtitle_base} Distribution",
            f"Cumulative {subtitle_base}",
            f"Q/Q Plot",
        ],
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
    )

    # Returns Distribution
    series_cuts = pd.cut(series, 100).value_counts().sort_index()
    midpoints = series_cuts.index.map(lambda interval: interval.right).to_numpy()
    norm_dist = stats.norm.pdf(midpoints, loc=series.mean(), scale=series.std())

    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series,
            line=dict(width=1, color=COLORS[0]),
            name="return",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.cumsum(),
            line=dict(width=1, color=COLORS[0]),
            name="cum. return",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=[interval.mid for interval in series_cuts.index],
            y=series_cuts / series_cuts.sum(),
            name="pct. of returns",
            marker=dict(color=COLORS[0]),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=[interval.mid for interval in series_cuts.index],
            y=norm_dist / norm_dist.sum(),
            name="normal",
            line=dict(width=1, color=COLORS[1]),
        ),
        row=1,
        col=2,
    )

    # Q/Q Data
    returns_norm = ((series - series.mean()) / series.std()).sort_values()
    norm_dist = pd.Series(
        list(map(stats.norm.ppf, np.linspace(0.001, 0.999, len(series)))),
        name="normal",
    )

    fig.append_trace(
        go.Scatter(
            x=norm_dist,
            y=returns_norm,
            name="return norm.",
            mode="markers",
            marker=dict(color=COLORS[0], size=3),
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=norm_dist,
            y=norm_dist,
            name="norm.",
            line=dict(width=1, color=COLORS[1]),
        ),
        row=2,
        col=2,
    )

    fig.add_annotation(
        text=(f"{series.cumsum()[-1] * 100:0.1f}%"),
        xref="paper",
        yref="y3",
        x=0.465,
        y=series.cumsum()[-1],
        xanchor="left",
        showarrow=False,
        align="left",
    )

    fig.add_annotation(
        get_moments_annotation(
            series.dropna(),
            xref="paper",
            yref="paper",
            x=0.55,
            y=0.45,
            xanchor="left",
            title="Returns",
            labels=IS_labels,
        ),
        font=dict(size=6, family="Courier New, monospace"),
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

    fig.update_layout(
        title_text=(
            f"{title}<br>"
            f"{series.index.min().strftime('%Y-%m-%d %H:%M')}"
            f" - {series.index.max().strftime('%Y-%m-%d %H:%M')}"
        ),
        showlegend=False,
        height=600,
        font=dict(size=10),
        margin=dict(l=50, r=50, b=50, t=100),
        yaxis=dict(tickformat="0.3f"),
        yaxis3=dict(tickformat="0.3f"),
        yaxis2=dict(tickformat="0.3f"),
        yaxis4=dict(tickformat="0.1f"),
        xaxis2=dict(tickformat="0.3f"),
        xaxis4=dict(tickformat="0.1f"),
    )

    for i in fig["layout"]["annotations"]:
        i["font"]["size"] = 12

    fig.update_annotations(font=dict(size=10))

    return fig


def make_2_yaxis_lines(
    series1: pd.Series, series2: pd.Series, title: str, secondary_y: bool = False
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": secondary_y}]])

    fig.add_trace(
        go.Scatter(x=series1.index, y=series1, name=series1.name, line=dict(width=1)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=series2.index, y=series2, name=series2.name, line=dict(width=1)),
        secondary_y=False,
    )

    title_text = (
        f"{title}<br>"
        f"{series1.index.min().strftime('%Y-%m-%d %H:%M')}"
        f" - {series1.index.max().strftime('%Y-%m-%d %H:%M')}"
    )

    fig.update_layout(title=title_text)
    return fig
