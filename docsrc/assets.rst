******
Assets
******

A `perpetual futures contract`_ is like a traditional futures contract except that it has no expiration date They were first proposed by Robert Schiller in 1992 to enable derivative markets for illiquid assets :cite:p:`RePEc:bla:jfinan:v:48:y:1993:i:3:p:911-31`. To date, they have only developed for the crypto token markets and first started trading on the BitMEX exchange in 2016 :cite:p:`Alexander:BitMEX`. Today, they are traded on all the major crypto exchanges for all of the major crypto tokens in significant aggregate volume. The average daily dollar volume of traded perpetual futures year to date is over [$100 billion] as of 2021-06-02. Another indicator of the rapid pace at which the market is maturing, Bitcoin futures with fixed expiration dates also trade on the `CME`_, albeit at lower volumes (~$3bn per day year to date) than the crypto exchanges.


[Top five tokens on top five exchange in the last 24 hours]

[Growth of BTCUSDT on Binance since 2019]

Regulation
==========
Derivative trading is heavily regulated in the United States and requires being registered with the Commodity Futures Trading Commission among other requirements. None of the crypto exchanges are registered and consequently, it is not possible to trade on them from the United States. Our assumption is that this strategy would nonetheless appeal to institutional investors with existing offshore operations or that if the opportunity is deemed sufficiently attractive, such operations could be established. This is also relevant as a risk to the overall strategy as some exchanges have run afoul of regulators for making their services available in the United States, which may impact their operations in the future in a manner adverse users of their services (see `What’s at Stake in the U.S. Case Against a Crypto Rebel`_ for a current description of BitMEX's trouble).


Expiration
=============
With no expiration date, there is no need to roll fixed expiration date contracts forward.

Pricing
=======
Many of the futures contracts are priced in units of USD or `USDT`_, Tether, a `stablecoin`_ (a crypto token that by contract maintains a value of essentially $1) with the crypto token as the base currency.

[Candlestick charts for top six coins on binance]

Leverage
========
Most of the contracts offer the ability to utilize high levels of leverage. Up to 125x leverage can be used for the BTCUSDT perpetual futures contract on `Binance`_, for example. Higher leverage levels obviously increase the risk of margin calls, which can result in forced liquidation. One of they key executional aspects of this strategy is managing the dollar balances of the spot and futures positions to avoid forced liquidation and to keep them matched.

Liquidation
===========
Another key feature of perpetual futures contracts is that there is no central counter party in the middle of the clearing. Transactions are settled automatically directly between end market participants by contract. The benefit of this is the the risk of default is essentially non-existent, since everything happens digitally, by code. Positions are liquidated automatically before they reach a zero balance - referred to as the maintenance margin. On Binance the maintenance margin is typically 50% of the initial margin. There is some risk in light of the inherently high volatility, that positions are not able to be liquidated before they reach a negative balance. The exchanges set up insurance pools, funded by the maintenance margin on automatically liquidated positions and potentially from the liquidation of profitable, highly leveraged positions if the insurance pool is ever short of funds (`auto de-leveraging`_). Our strategy entails relatively modest leverage (less than 10x on the perpetual futures contracts) and given our frequent rebalancing of our spot and futures balances, we are unlikely to be subject to auto de-leveraging.

To reduce volatility and the number of forced liquidation events, the liquidation price is typically based on a mark price as opposed to last price, where the mark price is based on a `composite index of prices`_ from multiple exchanges.


Funding Rate
============
The `funding rate`_ is the mechanism that keeps the futures price tethered to the spot price absent a set expiration date upon which the contract is settled and the spot must equal the futures price. As Schiller envisioned it, the funding rate represented the flow of value from one side to the other. The crypto exchanges use a formula based on the order book that results in a bigger funding rate if the market is more bullish and a smaller, or even negative rate in which the shorts pay the longs, when the market is more bearish. Funding on most exchanges occurs every eight hours. The rates are set in advance and are paid based on the notional value of the contract at the time of payment. The amounts are paid directly from one side to the other without the exchanges extracting any rent.

[Funding rate versus the spread for top six coins on binance]


.. _`perpetual futures contract`: http://en.wikipedia.org/w/index.php?title=Perpetual\%20futures&oldid=1006938691
.. _`stablecoin`: http://en.wikipedia.org/w/index.php?title=Stablecoin&oldid=1024888758
.. _`USDT`: http://cnn.com/
.. _`Binance`: https://www.binance.com/en/support/faq/360033162192
.. _`CME`: https://www.cmegroup.com/trading/equity-index/us-index/bitcoin.html
.. _`What’s at Stake in the U.S. Case Against a Crypto Rebel`: https://www.bloomberg.com/news/articles/2021-04-15/will-crypto-be-regulated-the-bitmex-case-could-bring-laws-to-bitcoin
.. _`auto de-leveraging`: https://academy.binance.com/en/articles/the-ultimate-guide-to-trading-on-binance-futures
.. _`funding rate`: https://www.binance.com/en/support/faq/360033525031
.. _`composite index of prices`: https://www.binance.com/en/support/faq/547ba48141474ab3bddc5d7898f97928