"""
Market microstructure engine.
Calculates OBI (Order Book Imbalance), CVD (Cumulative Volume Delta),
bid-ask spread, and related signals from raw order book + trade data.
"""
from __future__ import annotations

from collections import deque

import config
from core.data_feed import OrderBook


def calculate_obi(orderbook: OrderBook, levels: int = config.OBI_LEVELS) -> float:
    """
    Order Book Imbalance: ratio of bid volume vs ask volume across top N levels.

    Returns value in [-1, +1]:
        +1 = 100% bid pressure (strong buy)
        -1 = 100% ask pressure (strong sell)
         0 = perfectly balanced
    """
    bids = orderbook.bids[:levels]
    asks = orderbook.asks[:levels]

    bid_vol = sum(b[1] for b in bids if len(b) >= 2)
    ask_vol = sum(a[1] for a in asks if len(a) >= 2)
    total   = bid_vol + ask_vol

    if total == 0:
        return 0.0
    return round((bid_vol - ask_vol) / total, 4)


def calculate_spread_bps(orderbook: OrderBook) -> float:
    """
    Bid-ask spread in basis points.
    spread_bps = (best_ask - best_bid) / mid_price * 10_000
    """
    if not orderbook.bids or not orderbook.asks:
        return 999.0   # unknown — treat as too wide

    best_bid = orderbook.bids[0][0]
    best_ask = orderbook.asks[0][0]
    mid      = (best_bid + best_ask) / 2

    if mid == 0:
        return 999.0
    return round(((best_ask - best_bid) / mid) * 10_000, 4)


def calculate_cvd(recent_trades: deque, lookback: int = config.CVD_LOOKBACK * 60) -> dict:
    """
    Cumulative Volume Delta over the recent trade window.

    Aggressive buys (taker hits ask) = positive delta.
    Aggressive sells (taker hits bid) = negative delta.

    lookback: approximate number of seconds of trades to include.

    Returns:
        cvd:        raw cumulative delta value
        cvd_trend:  "rising" | "falling" | "flat"
        buy_vol:    total aggressive buy volume
        sell_vol:   total aggressive sell volume
    """
    if not recent_trades:
        return {"cvd": 0.0, "cvd_trend": "flat", "buy_vol": 0.0, "sell_vol": 0.0}

    buy_vol  = 0.0
    sell_vol = 0.0

    trades_list = list(recent_trades)

    # Use last N trades as a proxy for the lookback window
    window = trades_list[-max(1, lookback):]

    for trade in window:
        amount = float(trade.get("amount", 0) or 0)
        side   = trade.get("side", "")
        if side == "buy":
            buy_vol += amount
        elif side == "sell":
            sell_vol += amount

    cvd = buy_vol - sell_vol

    # Trend: compare first-half vs second-half delta
    mid  = len(window) // 2
    first_half_delta  = sum(
        (float(t.get("amount", 0)) if t.get("side") == "buy" else -float(t.get("amount", 0)))
        for t in window[:mid]
    )
    second_half_delta = sum(
        (float(t.get("amount", 0)) if t.get("side") == "buy" else -float(t.get("amount", 0)))
        for t in window[mid:]
    )

    if second_half_delta > first_half_delta * 1.1:
        trend = "rising"
    elif second_half_delta < first_half_delta * 0.9:
        trend = "falling"
    else:
        trend = "flat"

    return {
        "cvd":      round(cvd, 4),
        "cvd_trend": trend,
        "buy_vol":  round(buy_vol, 4),
        "sell_vol": round(sell_vol, 4),
    }


def get_mid_price(orderbook: OrderBook) -> float:
    """Return the mid price from the order book."""
    if not orderbook.bids or not orderbook.asks:
        return 0.0
    return (orderbook.bids[0][0] + orderbook.asks[0][0]) / 2
