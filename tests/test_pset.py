#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest import TestCase

import pandas as pd

from final_project.utils import Strategy


class StrategyTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df_ticks = (
            pd.read_csv("tests/df_ticks.csv", parse_dates=["date"])
            .set_index(["date", "series", "asset"])["0"]
            .unstack(["series", "asset"])
        )

        strategy_params = dict(
            pair=("spot", "perpetual"),
            df_ticks=cls.df_ticks,
            window=9,
            open_threshold=0.0005,
            close_threshold=0.0002,
            run=True,
            transact_cost_percent=0.0005,
            closed_positions=[],
        )

        cls.strategy = Strategy(**strategy_params)

        cls.df_stats = pd.DataFrame(cls.strategy.stats).set_index("date")

    def test_open_trans_cost(self):
        """
        Ensures that transaction costs are recorded correctly.
        """

        long_position = self.strategy.closed_positions[0]
        short_position = self.strategy.closed_positions[1]

        open_trans_cost = (
            -long_position.open_transact_cost - short_position.open_transact_cost
        )

        unrealized_close_trans_cost = (
            -long_position.transact_cost_percent
            * long_position.shares
            * long_position.open_price
            - short_position.transact_cost_percent
            * short_position.shares
            * short_position.open_price
        )

        self.assertEqual(
            open_trans_cost,
            self.df_stats.loc[long_position.open_date].realized_profit,
        )

        self.assertEqual(
            unrealized_close_trans_cost,
            self.df_stats.loc[long_position.open_date].unrealized_profit,
        )

        self.assertEqual(
            open_trans_cost + unrealized_close_trans_cost,
            self.df_stats.loc[long_position.open_date].total_profit,
        )

        self.assertEqual(
            open_trans_cost + unrealized_close_trans_cost,
            self.df_stats.loc[long_position.open_date].tick_profit,
        )

    def test_funding_rate_profit(self):
        """
        Ensures that funding rate profits are recorded correctly.

        Funding Rate Profit = Shares * Closing Price at end of Funding Rate Period
        * Funding Rate set at beginning of period.
        """

        short_position = self.strategy.closed_positions[1]

        self.assertEqual(short_position.security, "perpetual")

        prior_tick = self.df_ticks.loc[short_position.open_date]
        prior_stats = self.df_stats.loc[short_position.open_date]
        next_tick = self.df_ticks.shift(-1).loc[short_position.open_date]
        next_stats = self.df_stats.shift(-1).loc[short_position.open_date]

        funding_rate_profit = (
            short_position.shares
            * next_tick.adj_close.perpetual
            * next_tick.adj_return.prior_funding_rate
        )

        self.assertEqual(
            prior_stats.realized_profit + funding_rate_profit,
            next_stats.realized_profit,
        )

    def test_unrealized_profit(self):
        """
        Ensures that unrealized profits based on changes in asset prices are
        recorded correctly.

        Unrealized Profit = Change in Long Position Value + Change in Short Position Value
            - Unrealized Closing Transaction Costs
        Change in Long Position Value = Long Position Shares * (Closing Price - Open Price)
        Change in Short Position Value = Short Position Shares * (Open Price - Closing Price)
        """

        long_position = self.strategy.closed_positions[0]
        short_position = self.strategy.closed_positions[1]

        prior_tick = self.df_ticks.loc[short_position.open_date]
        next_tick = self.df_ticks.shift(-1).loc[short_position.open_date]
        next_stats = self.df_stats.shift(-1).loc[short_position.open_date]

        long_position_change = long_position.shares * (
            next_tick.adj_close[long_position.security]
            - prior_tick.adj_close[long_position.security]
        )

        short_position_change = short_position.shares * (
            prior_tick.adj_close[short_position.security]
            - next_tick.adj_close[short_position.security]
        )

        unrealized_close_trans_cost = (
            -long_position.transact_cost_percent
            * long_position.shares
            * next_tick.adj_close[long_position.security]
            - short_position.transact_cost_percent
            * short_position.shares
            * next_tick.adj_close[short_position.security]
        )

        self.assertAlmostEqual(
            long_position_change + short_position_change + unrealized_close_trans_cost,
            next_stats.unrealized_profit,
            5,
        )

    def test_realized_profit(self):
        """
        Ensures that realized profits from opening to closing of pair of positions
        is recorded correctly.


        Realized Profit = Change in Long Position Value + Change in Short Position Value
            + Funding Rate Profit - Transaction Costs
        Change in Long Position Value = Long Position Shares * (Closing Price - Open Price)
        Change in Short Position Value = Short Position Shares * (Open Price - Closing Price)
        Funding Rate Profit = Sum(Perpetual Shares * Perpetual Closing Price * Prior Funding Rate)
            for each tick after the first
        Transaction Costs = Opening Transaction Cost + Closing Transation Costs
        """

        long_position = self.strategy.closed_positions[0]
        short_position = self.strategy.closed_positions[1]

        ticks = self.df_ticks.loc[short_position.open_date : short_position.close_date]

        open_trans_cost = (
            -long_position.open_transact_cost - short_position.open_transact_cost
        )

        long_position_change = long_position.shares * (
            ticks.iloc[-1].adj_close[long_position.security]
            - ticks.iloc[0].adj_close[long_position.security]
        )

        short_position_change = short_position.shares * (
            ticks.iloc[0].adj_close[short_position.security]
            - ticks.iloc[-1].adj_close[short_position.security]
        )

        funding_rate_profit = (
            ticks.iloc[1:].adj_return.prior_funding_rate
            * short_position.shares
            * ticks.iloc[1:].adj_close.perpetual
        ).sum()

        close_trans_cost = (
            -long_position.transact_cost_percent
            * long_position.shares
            * ticks.iloc[-1].adj_close[long_position.security]
            - short_position.transact_cost_percent
            * short_position.shares
            * ticks.iloc[-1].adj_close[short_position.security]
        )

        self.assertEqual(
            close_trans_cost,
            -long_position.close_transact_cost - short_position.close_transact_cost,
        )

        self.assertAlmostEqual(
            open_trans_cost
            + long_position_change
            + short_position_change
            + funding_rate_profit
            + close_trans_cost,
            self.df_stats.loc[short_position.close_date].total_profit,
            5,
        )
