"""
SMC Liquidity Pool Mapping
===========================
Maps where stop-loss clusters are sitting in the market.

Sell-side liquidity (above price): shorts' stop-losses → equal highs, swing highs
Buy-side  liquidity (below price): longs' stop-losses  → equal lows,  swing lows

Smart money sweeps these pools before the real move.
Detecting a completed sweep (spike + close back) = high-conviction entry signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class LiquidityPool:
    kind: str           # "buy_side" | "sell_side"
    price: float        # level where liquidity is clustered
    strength: int       # number of equal touch-points
    source: str         # "equal_highs" | "equal_lows" | "prev_swing" | "round_number"
    age: int            # candles since most recent touch


@dataclass
class LiquiditySweep:
    kind: str               # "swept_buy_side" | "swept_sell_side"
    price: float            # swept level
    sweep_idx: int          # candle index of the sweep
    age: int                # candles ago
    reversal_confirmed: bool  # next candle closed back beyond the level


def find_liquidity_pools(
    candles: pd.DataFrame,
    swing_highs: list,
    swing_lows: list,
    cluster_pct: float = 0.03,
    lookback: int = 100,
) -> tuple[list[LiquidityPool], list[LiquidityPool]]:
    """
    Identify buy-side and sell-side liquidity pools.

    Returns:
        (buy_side, sell_side) sorted by strength (strongest first)
    """
    if len(candles) < 10:
        return [], []

    n             = len(candles)
    current_price = candles["close"].iloc[-1]

    buy_side:  list[LiquidityPool] = []
    sell_side: list[LiquidityPool] = []

    # ── Equal highs (sell-side, above) ───────────────────────────────────────
    sh_prices = [sh.price for sh in swing_highs if sh.price > current_price * 1.0005]
    _cluster_into_pools(sh_prices, "sell_side", "equal_highs", swing_highs, sell_side, cluster_pct, n)

    # ── Equal lows (buy-side, below) ──────────────────────────────────────────
    sl_prices = [sl.price for sl in swing_lows if sl.price < current_price * 0.9995]
    _cluster_into_pools(sl_prices, "buy_side", "equal_lows", swing_lows, buy_side, cluster_pct, n)

    # ── Individual swing points as liquidity ──────────────────────────────────
    for sh in swing_highs[-8:]:
        if sh.price > current_price * 1.001:
            sell_side.append(LiquidityPool(
                kind="sell_side", price=sh.price,
                strength=1, source="prev_swing", age=n - 1 - sh.index,
            ))

    for sl in swing_lows[-8:]:
        if sl.price < current_price * 0.999:
            buy_side.append(LiquidityPool(
                kind="buy_side", price=sl.price,
                strength=1, source="prev_swing", age=n - 1 - sl.index,
            ))

    # ── Round number levels ───────────────────────────────────────────────────
    _add_round_numbers(current_price, buy_side, sell_side)

    # Deduplicate, sort, restrict to ±2% of price
    buy_side  = _dedup_pools(buy_side,  cluster_pct)
    sell_side = _dedup_pools(sell_side, cluster_pct)

    buy_side.sort(key=lambda x: -x.strength)
    sell_side.sort(key=lambda x: -x.strength)

    buy_side  = [p for p in buy_side  if p.price >= current_price * 0.98][:6]
    sell_side = [p for p in sell_side if p.price <= current_price * 1.02][:6]

    return buy_side, sell_side


def detect_liquidity_sweeps(
    candles: pd.DataFrame,
    buy_pools: list[LiquidityPool],
    sell_pools: list[LiquidityPool],
    lookback: int = 6,
) -> list[LiquiditySweep]:
    """
    Detect recent stop-hunt sweeps: wick through a liquidity level, close back.

    Pattern: price wick breaks the level but the candle CLOSES on the near side
    → the stops were grabbed, smart money is reversing.
    """
    if len(candles) < 3:
        return []

    highs  = candles["high"].values
    lows   = candles["low"].values
    closes = candles["close"].values
    n      = len(candles)
    start  = max(0, n - lookback)
    sweeps: list[LiquiditySweep] = []

    for i in range(start, n):
        # Swept sell-side (wick above, close below)
        for pool in sell_pools:
            if highs[i] > pool.price and closes[i] < pool.price:
                reversal_ok = (i < n - 1) and (closes[min(i + 1, n - 1)] < pool.price)
                sweeps.append(LiquiditySweep(
                    kind="swept_sell_side", price=pool.price,
                    sweep_idx=i, age=n - 1 - i,
                    reversal_confirmed=reversal_ok,
                ))

        # Swept buy-side (wick below, close above)
        for pool in buy_pools:
            if lows[i] < pool.price and closes[i] > pool.price:
                reversal_ok = (i < n - 1) and (closes[min(i + 1, n - 1)] > pool.price)
                sweeps.append(LiquiditySweep(
                    kind="swept_buy_side", price=pool.price,
                    sweep_idx=i, age=n - 1 - i,
                    reversal_confirmed=reversal_ok,
                ))

    sweeps.sort(key=lambda x: x.age)
    return sweeps


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cluster_into_pools(
    prices: list[float],
    pool_kind: str,
    source: str,
    swing_points: list,
    result: list[LiquidityPool],
    cluster_pct: float,
    n: int,
) -> None:
    """Group nearby price levels into clustered pools (equal highs / lows)."""
    if len(prices) < 2:
        return

    prices_s = sorted(prices)
    i = 0
    while i < len(prices_s):
        cluster = [prices_s[i]]
        j = i + 1
        while j < len(prices_s):
            if (prices_s[j] - cluster[0]) / max(cluster[0], 1e-9) * 100 <= cluster_pct:
                cluster.append(prices_s[j])
                j += 1
            else:
                break

        if len(cluster) >= 2:
            avg_price = sum(cluster) / len(cluster)
            ages = [
                n - 1 - sp.index for sp in swing_points
                if abs(sp.price - avg_price) / max(avg_price, 1e-9) * 100 <= cluster_pct
            ]
            result.append(LiquidityPool(
                kind=pool_kind, price=avg_price,
                strength=len(cluster), source=source,
                age=min(ages) if ages else n,
            ))
        i = j if j > i else i + 1


def _add_round_numbers(
    price: float,
    buy_side: list[LiquidityPool],
    sell_side: list[LiquidityPool],
) -> None:
    """Add psychological round-number levels as moderate-strength liquidity."""
    if price > 50_000:
        interval = 1_000
    elif price > 10_000:
        interval = 500
    elif price > 1_000:
        interval = 100
    elif price > 100:
        interval = 10
    elif price > 10:
        interval = 1
    else:
        interval = 0.1

    lower = (price // interval) * interval
    upper = lower + interval

    if lower > 0 and lower < price * 0.999:
        buy_side.append(LiquidityPool(
            kind="buy_side", price=lower, strength=2,
            source="round_number", age=0,
        ))
    if upper > price * 1.001:
        sell_side.append(LiquidityPool(
            kind="sell_side", price=upper, strength=2,
            source="round_number", age=0,
        ))


def _dedup_pools(pools: list[LiquidityPool], cluster_pct: float) -> list[LiquidityPool]:
    """Merge pools that are within cluster_pct% of each other."""
    if not pools:
        return pools
    pools_s = sorted(pools, key=lambda x: x.price)
    merged = [pools_s[0]]
    for pool in pools_s[1:]:
        last = merged[-1]
        if abs(pool.price - last.price) / max(last.price, 1e-9) * 100 <= cluster_pct:
            last.price    = (last.price + pool.price) / 2
            last.strength = last.strength + pool.strength
        else:
            merged.append(pool)
    return merged
