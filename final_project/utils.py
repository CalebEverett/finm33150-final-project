from datetime import datetime
import hmac
import gzip
import os
from typing import Dict, List
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

    if start_time is not None:
        start_time = int(datetime.fromisoformat(start_time).timestamp() * 1000)

    if end_time is not None:
        end_time = int(datetime.fromisoformat(end_time).timestamp() * 1000)

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
    df = pd.DataFrame(
        r.json(),
    )

    df.fundingTime = pd.to_datetime(df.fundingTime, unit="ms").round("1s")

    return df.set_index("fundingTime")[["fundingRate"]]


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

    return df.set_index("openTime")


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

    if start_time is not None:
        start_time = int(datetime.fromisoformat(start_time).timestamp() * 1000)

    if end_time is not None:
        end_time = int(datetime.fromisoformat(end_time).timestamp() * 1000)

    params = {
        "limit": limit,
        "pair": pair,
        "interval": interval,
        "contractType": contract_type,
        "startTime": start_time,
        "endTime": end_time,
    }

    r = requests.get(
        f"{base_url}{end_point}",
        params=params,
        headers={"X-MBX-APIKEY": os.getenv("BINANCE_API_KEY")},
    )

    df_pv = get_candlestick_df(r.json())

    return df_pv


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

    if start_time is not None:
        start_time = int(datetime.fromisoformat(start_time).timestamp() * 1000)

    if end_time is not None:
        end_time = int(datetime.fromisoformat(end_time).timestamp() * 1000)

    params = {
        "limit": limit,
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
    }

    r = requests.get(
        f"{base_url}{end_point}",
        params=params,
        headers={"X-MBX-APIKEY": os.getenv("BINANCE_API_KEY")},
    )
    if verbose:
        print(r.text)

    df_pv = get_candlestick_df(r.json())

    return df_pv


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
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=df_pv.index, y=df_pv.volume, marker=dict(color=COLORS[0])),
        row=2,
        col=1,
    )
    fig.update(
        layout_xaxis_rangeslider_visible=False,
        layout_title=title,
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
    )


def make_components_chart(
    yc_L: str,
    fx_B: str,
    fx_L: str,
    libor: str,
    leverage: float,
    date_range: pd.date_range,
    dfs_yc: Dict,
    dfs_fx: Dict,
    dfs_libor: Dict,
) -> go.Figure:

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"5-Year Yield: {yc_L}",
            f"FX Rate: {fx_L}:{fx_B}",
            f"3 Month Libor: {libor}",
            f"FX Rate: {fx_B}:USD",
        ],
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
        specs=[
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": False}, {"secondary_y": True}],
        ],
    )

    # Lend market yield
    # =================
    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=dfs_yc[yc_L].loc[date_range]["5-year"],
            line=dict(width=1, color=COLORS[0]),
            name=yc_L,
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=dfs_yc[yc_L].loc[date_range]["5-year"].pct_change() * 100,
            line=dict(width=1, color=COLORS[1], dash="dot"),
            name=yc_L,
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # Borrow market fx
    # =================
    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=dfs_fx[fx_B].loc[date_range].rate,
            line=dict(width=1, color=COLORS[0]),
            name=fx_B,
        ),
        row=2,
        col=2,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=dfs_fx[fx_B].loc[date_range].rate.pct_change() * 100,
            line=dict(width=1, color=COLORS[1], dash="dot"),
            name=fx_B,
        ),
        row=2,
        col=2,
        secondary_y=True,
    )

    # Borrow market funding cost
    # =================
    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=dfs_libor[libor].loc[date_range].value,
            line=dict(width=1, color=COLORS[0]),
            name=libor,
        ),
        row=2,
        col=1,
    )

    # Lend market fx cost
    # =================
    fx_BL = (
        dfs_fx[fx_L].loc[date_range].loc[date_range].rate
        / dfs_fx[fx_B].loc[date_range].rate
    )

    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=fx_BL,
            line=dict(width=1, color=COLORS[0]),
            name=fx_L,
        ),
        row=1,
        col=2,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=fx_BL.pct_change() * 100,
            line=dict(width=1, color=COLORS[1], dash="dot"),
            name=fx_L,
        ),
        row=1,
        col=2,
        secondary_y=True,
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor="grey", mirror=True)
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor="grey", mirror=True, tickformat="0.1f"
    )

    fig.update_layout(
        title_text=(
            f"Weekly Carry Trade: Borrow {fx_B}, Lend {yc_L}"
            "<br>Underlying Securities: "
            f"{date_range.min().strftime('%Y-%m-%d')}"
            f" - {date_range.max().strftime('%Y-%m-%d')}"
        ),
        showlegend=False,
        height=600,
        font=dict(size=10),
        margin=dict(l=50, r=10, b=40, t=90),
        yaxis3=dict(tickformat="0.3f"),
    )

    for i in fig["layout"]["annotations"]:
        i["font"]["size"] = 12

    return fig


def make_returns_chart(df_ret: pd.DataFrame) -> go.Figure:

    fx_B, yc_L = df_ret.name.split(",")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"Weekly Returns",
            f"Returns Distribution",
            f"Cumulative Returns",
            f"Q/Q Plot",
        ],
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
    )

    # Returns Distribution
    returns = pd.cut(df_ret.per_return, 50).value_counts().sort_index()
    midpoints = returns.index.map(lambda interval: interval.right).to_numpy()
    norm_dist = stats.norm.pdf(
        midpoints, loc=df_ret.per_return.mean(), scale=df_ret.per_return.std()
    )

    fig.add_trace(
        go.Scatter(
            x=df_ret.index,
            y=df_ret.per_return * 100,
            line=dict(width=1, color=COLORS[0]),
            name="return",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_ret.index,
            y=df_ret.per_return.cumsum() * 100,
            line=dict(width=1, color=COLORS[0]),
            name="cum. return",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=[interval.mid for interval in returns.index],
            y=returns / returns.sum() * 100,
            name="pct. of returns",
            marker=dict(color=COLORS[0]),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=[interval.mid for interval in returns.index],
            y=norm_dist / norm_dist.sum() * 100,
            name="normal",
            line=dict(width=1, color=COLORS[1]),
        ),
        row=1,
        col=2,
    )

    # Q/Q Data
    returns_norm = (
        (df_ret.per_return - df_ret.per_return.mean()) / df_ret.per_return.std()
    ).sort_values()
    norm_dist = pd.Series(
        list(map(stats.norm.ppf, np.linspace(0.001, 0.999, len(df_ret.per_return)))),
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
        text=(f"{df_ret.per_return.cumsum()[-1] * 100:0.2f}"),
        xref="paper",
        yref="y3",
        x=0.465,
        y=df_ret.per_return.cumsum()[-1] * 100,
        xanchor="left",
        showarrow=False,
        align="left",
    )

    fig.add_annotation(
        get_moments_annotation(
            df_ret.per_return,
            xref="paper",
            yref="paper",
            x=0.81,
            y=0.23,
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
            f"Weekly Carry Trade: Borrow {fx_B}, Lend {yc_L}"
            "<br>Returns: "
            f"{df_ret.index.min().strftime('%Y-%m-%d')}"
            f" - {df_ret.index.max().strftime('%Y-%m-%d')}"
        ),
        showlegend=False,
        height=600,
        font=dict(size=10),
        margin=dict(l=50, r=50, b=50, t=100),
        yaxis=dict(tickformat="0.1f"),
        yaxis3=dict(tickformat="0.1f"),
        yaxis2=dict(tickformat="0.1f"),
        yaxis4=dict(tickformat="0.1f"),
        xaxis2=dict(tickformat="0.1f"),
        xaxis4=dict(tickformat="0.1f"),
    )

    for i in fig["layout"]["annotations"]:
        i["font"]["size"] = 12

    fig.update_annotations(font=dict(size=10))

    return fig
