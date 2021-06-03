import numpy as np
import pandas as pd
import plotly.io as pio


from ..final_project import utils

pd.options.plotting.backend = "plotly"
pio.templates.default = "seaborn"

tick_params = dict(interval="8h", start_time="2018-09-08")

df_funding = utils.get_funding_rate_history(
    symbol="BTCUSDT", start_time=tick_params["start_time"]
)


def show_funding_rate():
    fig = df_funding.fundingRate.plot(title="BTCUSDT Funding Rate")
    fig.update_traces(line=dict(width=1))
    fig.update(layout_showlegend=False)
    return fig


# df_perpetual = utils.get_continuous_contracts(pair="BTCUSDT", **tick_params)


# def show_btc_perpetual_chart():
#     return utils.make_overview_chart(
#         df_perpetual.per_return, title="BTCUSDT Perpetual", subtitle_base="Log Returns"
#     )


# df_ticks = (
#     pd.read_csv("../tests/df_ticks.csv", parse_dates=["date"])
#     .set_index(["date", "series", "asset"])["0"]
#     .unstack(["series", "asset"])
# )
