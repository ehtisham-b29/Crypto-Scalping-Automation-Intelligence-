"""
SMC Market Structure Analysis
==============================
Detects swing highs/lows, Break of Structure (BOS), and Change of Character (CHoCH).

BOS  = price closes beyond last significant swing → continuation
CHoCH = sequence of swings shifts direction → structural reversal

Used to determine higher-timeframe bias: bullish | bearish | ranging
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


@dataclass
class SwingPoint:
    index: int
    price: float
    kind: Literal["high", "low"]
    timestamp: object


@dataclass
class StructureBreak:
    kind: Literal["BOS", "CHoCH"]
    direction: Literal["bullish", "bearish"]
    price: float
    index: int


@dataclass
class MarketStructure:
    bias: Literal["bullish", "bearish", "ranging"]
    swing_highs: list[SwingPoint] = field(default_factory=list)
    swing_lows: list[SwingPoint] = field(default_factory=list)
    last_break: StructureBreak | None = None
    last_swing_high: SwingPoint | None = None
    last_swing_low: SwingPoint | None = None
    structure_breaks: list[StructureBreak] = field(default_factory=list)


def analyze(candles: pd.DataFrame, swing_n: int = 3) -> MarketStructure:
    """
    Detect swing structure and determine directional bias.

    Args:
        candles : OHLCV DataFrame (indexed by timestamp)
        swing_n : bars each side required to confirm a swing (default 3)

    Returns:
        MarketStructure with bias, swing lists, and last structural break
    """
    min_candles = swing_n * 2 + 6
    if len(candles) < min_candles:
        return MarketStructure(bias="ranging")

    highs = candles["high"].values
    lows  = candles["low"].values
    n     = len(candles)

    swing_highs: list[SwingPoint] = []
    swing_lows:  list[SwingPoint] = []

    # ── Detect swing points ────────────────────────────────────────────────────
    for i in range(swing_n, n - swing_n):
        h = highs[i]
        l = lows[i]

        # Guard against NaN values in real market data
        if h != h or l != l:   # NaN check (NaN != NaN is True)
            continue

        is_sh = all(bool(h >= highs[i - j]) for j in range(1, swing_n + 1)) and \
                all(bool(h >= highs[i + j]) for j in range(1, swing_n + 1))
        is_sl = all(bool(l <= lows[i - j])  for j in range(1, swing_n + 1)) and \
                all(bool(l <= lows[i + j])  for j in range(1, swing_n + 1))

        if is_sh:
            # Avoid duplicate adjacent swings (must differ by at least 0.01%)
            if not swing_highs or abs(h - swing_highs[-1].price) / h > 0.0001:
                swing_highs.append(SwingPoint(
                    index=i, price=h, kind="high",
                    timestamp=candles.index[i],
                ))

        if is_sl:
            if not swing_lows or abs(l - swing_lows[-1].price) / l > 0.0001:
                swing_lows.append(SwingPoint(
                    index=i, price=l, kind="low",
                    timestamp=candles.index[i],
                ))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return MarketStructure(
            bias="ranging",
            swing_highs=swing_highs,
            swing_lows=swing_lows,
        )

    # ── Score structure from recent swings ────────────────────────────────────
    recent_highs = swing_highs[-4:]
    recent_lows  = swing_lows[-4:]

    hh = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i].price > recent_highs[i-1].price)
    lh = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i].price < recent_highs[i-1].price)
    hl = sum(1 for i in range(1, len(recent_lows))  if recent_lows[i].price  > recent_lows[i-1].price)
    ll = sum(1 for i in range(1, len(recent_lows))  if recent_lows[i].price  < recent_lows[i-1].price)

    bullish_score = hh + hl
    bearish_score = lh + ll

    # ── Detect structural breaks on latest price ───────────────────────────────
    last_close = candles["close"].iloc[-1]
    last_high  = highs[-1]
    last_low   = lows[-1]
    structure_breaks: list[StructureBreak] = []

    # BOS: clean close beyond last swing
    if last_close > swing_highs[-1].price:
        structure_breaks.append(StructureBreak(
            kind="BOS", direction="bullish",
            price=swing_highs[-1].price, index=n - 1,
        ))
    elif last_close < swing_lows[-1].price:
        structure_breaks.append(StructureBreak(
            kind="BOS", direction="bearish",
            price=swing_lows[-1].price, index=n - 1,
        ))

    # CHoCH: structure shift (wick breaks prior swing without close)
    if not structure_breaks:
        if bullish_score > bearish_score and last_high > swing_highs[-1].price:
            structure_breaks.append(StructureBreak(
                kind="CHoCH", direction="bullish",
                price=swing_highs[-1].price, index=n - 1,
            ))
        elif bearish_score > bullish_score and last_low < swing_lows[-1].price:
            structure_breaks.append(StructureBreak(
                kind="CHoCH", direction="bearish",
                price=swing_lows[-1].price, index=n - 1,
            ))

    # ── Final bias ────────────────────────────────────────────────────────────
    if bullish_score >= 2 and bullish_score > bearish_score:
        bias: Literal["bullish", "bearish", "ranging"] = "bullish"
    elif bearish_score >= 2 and bearish_score > bullish_score:
        bias = "bearish"
    else:
        bias = "ranging"

    return MarketStructure(
        bias=bias,
        swing_highs=swing_highs[-10:],
        swing_lows=swing_lows[-10:],
        last_break=structure_breaks[-1] if structure_breaks else None,
        last_swing_high=swing_highs[-1],
        last_swing_low=swing_lows[-1],
        structure_breaks=structure_breaks,
    )
