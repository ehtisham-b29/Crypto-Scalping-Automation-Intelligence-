"""
SMC Order Block Detection
==========================
Identifies institutional order blocks — zones where large players placed orders
that caused a significant structural break (BOS/CHoCH).

Bullish OB : last bearish (red) candle BEFORE a bullish impulse that breaks structure
Bearish OB : last bullish (green) candle BEFORE a bearish impulse that breaks structure

When price returns to an unmitigated OB → high-probability reversal / continuation entry.
Mitigation: OB is invalidated when price CLOSES through its opposite boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class OrderBlock:
    kind: str           # "bullish" | "bearish"
    high: float         # OB upper boundary
    low: float          # OB lower boundary
    mid: float          # midpoint
    origin_idx: int     # candle index where OB formed
    strength: float     # 0.0 – 1.0 (volume × displacement normalised)
    mitigated: bool = False
    touches: int = 0    # how many times price visited without breaking


def find_order_blocks(
    candles: pd.DataFrame,
    swing_highs: list,
    swing_lows: list,
    lookback: int = 50,
) -> tuple[list[OrderBlock], list[OrderBlock]]:
    """
    Scan recent candles for active (unmitigated) bullish and bearish order blocks.

    Returns:
        (bullish_obs, bearish_obs) — sorted most-recent first, max 5 each
    """
    if len(candles) < 10:
        return [], []

    opens   = candles["open"].values
    highs   = candles["high"].values
    lows    = candles["low"].values
    closes  = candles["close"].values
    volumes = candles["volume"].values
    n       = len(candles)
    start   = max(0, n - lookback)

    avg_vol = volumes[start:].mean() or 1.0

    bullish_obs: list[OrderBlock] = []
    bearish_obs: list[OrderBlock] = []

    for i in range(start + 3, n - 1):
        # Skip candles with NaN OHLCV (can occur with real exchange data)
        if closes[i] != closes[i] or opens[i] != opens[i]:
            continue

        # ── Bullish OB: bearish candle before bullish impulse ─────────────────
        if closes[i] > opens[i]:                          # current = bullish
            ob_idx = None
            for j in range(i - 1, max(i - 7, start - 1), -1):
                if closes[j] < opens[j]:                  # last bearish before impulse
                    ob_idx = j
                    break

            if ob_idx is not None:
                displacement = (closes[i] - opens[ob_idx]) / max(opens[ob_idx], 1e-9) * 100
                if displacement >= 0.10:                   # meaningful move ≥ 0.10%
                    ob_low  = lows[ob_idx]
                    ob_high = highs[ob_idx]
                    # Mitigated = any later candle closes below OB low
                    mitigated = any(closes[k] < ob_low for k in range(ob_idx + 1, n))
                    if not mitigated:
                        touches = sum(
                            1 for k in range(ob_idx + 1, n)
                            if lows[k] <= ob_high and closes[k] >= ob_low
                        )
                        vol_str  = min(volumes[ob_idx] / avg_vol, 3.0) / 3.0
                        disp_str = min(displacement / 0.50, 1.0)
                        strength = round((vol_str + disp_str) / 2, 3)
                        bullish_obs.append(OrderBlock(
                            kind="bullish", high=ob_high, low=ob_low,
                            mid=(ob_high + ob_low) / 2,
                            origin_idx=ob_idx, strength=strength,
                            mitigated=False, touches=touches,
                        ))

        # ── Bearish OB: bullish candle before bearish impulse ─────────────────
        if closes[i] < opens[i]:                          # current = bearish
            ob_idx = None
            for j in range(i - 1, max(i - 7, start - 1), -1):
                if closes[j] > opens[j]:                  # last bullish before impulse
                    ob_idx = j
                    break

            if ob_idx is not None:
                displacement = (opens[ob_idx] - closes[i]) / max(opens[ob_idx], 1e-9) * 100
                if displacement >= 0.10:
                    ob_low  = lows[ob_idx]
                    ob_high = highs[ob_idx]
                    # Mitigated = any later candle closes above OB high
                    mitigated = any(closes[k] > ob_high for k in range(ob_idx + 1, n))
                    if not mitigated:
                        touches = sum(
                            1 for k in range(ob_idx + 1, n)
                            if highs[k] >= ob_low and closes[k] <= ob_high
                        )
                        vol_str  = min(volumes[ob_idx] / avg_vol, 3.0) / 3.0
                        disp_str = min(displacement / 0.50, 1.0)
                        strength = round((vol_str + disp_str) / 2, 3)
                        bearish_obs.append(OrderBlock(
                            kind="bearish", high=ob_high, low=ob_low,
                            mid=(ob_high + ob_low) / 2,
                            origin_idx=ob_idx, strength=strength,
                            mitigated=False, touches=touches,
                        ))

    # Deduplicate overlapping OBs, keep most recent 5
    bullish_obs = _deduplicate(bullish_obs)
    bearish_obs = _deduplicate(bearish_obs)

    bullish_obs.sort(key=lambda x: x.origin_idx, reverse=True)
    bearish_obs.sort(key=lambda x: x.origin_idx, reverse=True)

    return bullish_obs[:5], bearish_obs[:5]


def price_in_ob(price: float, obs: list[OrderBlock]) -> OrderBlock | None:
    """Return the first OB whose zone contains price, or None."""
    for ob in obs:
        if ob.low <= price <= ob.high:
            return ob
    return None


# ── Internal helpers ─────────────────────────────────────────────────────────

def _deduplicate(obs: list[OrderBlock]) -> list[OrderBlock]:
    """Merge OBs that overlap more than 50% of their range."""
    if len(obs) <= 1:
        return obs
    unique: list[OrderBlock] = [obs[0]]
    for ob in obs[1:]:
        merged = False
        for u in unique:
            rng = ob.high - ob.low
            if rng > 0:
                overlap = min(ob.high, u.high) - max(ob.low, u.low)
                if overlap / rng > 0.50:
                    merged = True
                    break
        if not merged:
            unique.append(ob)
    return unique
