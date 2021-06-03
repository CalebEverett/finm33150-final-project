Contents
========

..    toctree::
      :numbered:
      :maxdepth: 2
   
      assets
      literature
      strategy
      returns
      sizing
      risks
      enhancements
      references
      tables
      figures
      plotly

Overview
========

This is a quantitative trading strategy involving perpetual futures contracts :cite:`wiki:Perpetual_futures` on crypto tokens. Perpetual futures contracts are instruments that trade on the major crypto token exchanges that are similar to traditional futures contracts, but have no expiration date. In order to keep the futures contract price in line with the spot price a funding rate is employed. Long holders of the perpetual futures contracts pay the funding rate to the short holders of the perpetual futures contracts. The funding rate is determined by formula based on the spread between the perpetual futures contract price and the spot price, increasing as the spread increases and contracting as the spread narrows, and flipping such that the short holders pay the long holders if the spread becomes sufficiently negative.

This strategy is designed to capture the funding rate while limiting risk related to movements in the prices of the crypto tokens themselves to de minimis levels. It is effected by establishing a short position in the perpetual futures contract when the funding rate is attractive in order to capture the funding rate payments while establishing a corresponding short position of equivalent size in the underlying crypto token in order to hedge against price fluctuations.


Assets
======

Our initial execution of this strategy includes five crypto tokens on three different exchanges as detailed in the table below.

[insert table of crypto tokens and exchanges]

Returns
=======

Our backtesting indicates that we would have generated annualized returns of x% over the last three years on a capital base of $10 million with conservative leverage of 1.5x. We believe we would have been able to put an additional $100 million to work with a reduction in returns to y% per annum. Below is a summary of our base case returns:

[number of trades, returns, sharpe ratio, information ratio, sortino ratio, beta vs. crypto, beta vs. spy, downside beta vs crypto, downside beta vs.spy]

Risks
=====

The primary risks to this strategy are:

1. Need to transact from jurisdiction outside US to comply with derivatives trading regulations in the US

    * Should be doable for multinational asset managers
    * Explore options to establish offshore domicile from which to transact legally

2. Risk of not being able to keep positions perfectly hedged in the event of extreme volatility - liquidation price and automatic deleveraging

    * Conservative leverage - quantify % change until unhedged
    * Rebalance evaluated a very frequent intervals
    * How many price changes of liquidation price magnitude in last three years

3. Risk of exchanges collapsing / being hacked

    * Mitigate by trading across multiple exchanges

4. Macro / regulatory risk of crypto going away that decimates values to the point that the exchanges collapse so quickly that positions can’t be liquidated

5. Counterparty risk on the futures contracts

    * Insurance fund

6. Trading volumes aren’t as big as they are reported to be


:cite:`RePEc:bla:jfinan:v:48:y:1993:i:3:p:911-31`

.. math:: \frac{1}{2}
    :label: 

Further Enhancements
====================

1. As the academic literature indicates, changes in the price of the perpetual futures contract are not perfectly correlated with changes in the spot price. As such, there me be opportunity to increase returns at the margin by effecting a mean-variance optimized hedging strategy that results in a short position that is not the same size as the futures contract.
