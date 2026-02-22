"""
SMC Fair Value Gap (FVG) Detection
====================================
3-candle imbalances left by fast, institutional moves.

Bullish FVG : candle[i-2].high  <  candle[i].low   (gap upward — unfilled demand)
Bearish FVG : candle[i-2].low   >  candle[i].high  (gap downward — unfilled supply)

Price is attracted back to these gaps (partial or full fill) before continuing.
Fresh, unmitigated FVGs at the current price = high-probability entry zone.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FVG:
    kind: str           # "bullish" | "bearish"
    top: float          # upper boundary of the gap
    bottom: float       # lower boundary of the gap
    mid: float          # midpoint (50% level)
    size_pct: float     # gap size as % of price
    age: int            # candles since the gap formed
    filled_pct: float   # 0.0 – 1.0, portion already filled
    active: bool        # False when filled_pct >= 0.75


def find_fvgs(
    candles: pd.DataFrame,
    min_size_pct: float = 0.03,
    lookback: int = 30,
) -> tuple[list[FVG], list[FVG]]:
    """
    Detect active bullish and bearish FVGs.

    Args:
        candles      : OHLCV DataFrame
        min_size_pct : minimum gap as % of price (filters noise)
        lookback     : how many candles back to scan

    Returns:
        (bullish_fvgs, bearish_fvgs) — sorted freshest first, max 5 each
    """
    if len(candles) < 5:
        return [], []

    highs  = candles["high"].values
    lows   = candles["low"].values
    closes = candles["close"].values
    n      = len(candles)
    start  = max(0, n - lookback - 2)

    bullish_fvgs: list[FVG] = []
    bearish_fvgs: list[FVG] = []

    for i in range(start + 2, n):
        ref_price = closes[i] or 1.0

        # ── Bullish FVG ───────────────────────────────────────────────────────
        gap_bot = highs[i - 2]
        gap_top = lows[i]
        if gap_top > gap_bot:
            size_pct = (gap_top - gap_bot) / gap_bot * 100
            if size_pct >= min_size_pct:
                mid = (gap_top + gap_bot) / 2
                age = n - 1 - i
                filled_pct = _measure_fill(lows, closes, i + 1, n, gap_top, gap_bot, direction="bullish")
                if filled_pct < 0.75:
                    bullish_fvgs.append(FVG(
                        kind="bullish", top=gap_top, bottom=gap_bot,
                        mid=mid, size_pct=round(size_pct, 4),
                        age=age, filled_pct=round(filled_pct, 3), active=True,
                    ))

        # ── Bearish FVG ───────────────────────────────────────────────────────
        gap_top = lows[i - 2]
        gap_bot = highs[i]
        if gap_top > gap_bot:
            size_pct = (gap_top - gap_bot) / max(gap_top, 1e-9) * 100
            if size_pct >= min_size_pct:
                mid = (gap_top + gap_bot) / 2
                age = n - 1 - i
                filled_pct = _measure_fill(highs, closes, i + 1, n, gap_top, gap_bot, direction="bearish")
                if filled_pct < 0.75:
                    bearish_fvgs.append(FVG(
                        kind="bearish", top=gap_top, bottom=gap_bot,
                        mid=mid, size_pct=round(size_pct, 4),
                        age=age, filled_pct=round(filled_pct, 3), active=True,
                    ))

    # Sort freshest first
    bullish_fvgs.sort(key=lambda x: x.age)
    bearish_fvgs.sort(key=lambda x: x.age)

    return bullish_fvgs[:5], bearish_fvgs[:5]


def price_in_fvg(price: float, fvgs: list[FVG]) -> FVG | None:
    """Return the first FVG whose zone contains price, or None."""
    for f in fvgs:
        if f.bottom <= price <= f.top:
            return f
    return None


# ── Internal helpers ─────────────────────────────────────────────────────────

def _measure_fill(
    wick_arr,
    close_arr,
    start_k: int,
    n: int,
    gap_top: float,
    gap_bot: float,
    direction: str,
) -> float:
    """Measure what fraction of a gap has been filled by subsequent candles."""
    gap_size = gap_top - gap_bot
    if gap_size <= 0:
        return 1.0

    max_fill = 0.0
    for k in range(start_k, n):
        if direction == "bullish":
            penetration = gap_top - wick_arr[k]   # how far into gap from top
        else:
            penetration = wick_arr[k] - gap_bot    # how far into gap from bottom

        if penetration > 0:
            max_fill = max(max_fill, penetration / gap_size)

    return min(max_fill, 1.0)
