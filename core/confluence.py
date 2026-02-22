"""
Confluence scoring engine.
Evaluates 7 independent signals and returns a score [-7, +7].
Positive = bullish bias, Negative = bearish bias.
The score is used to pre-filter before calling the AI.
"""
from __future__ import annotations

import config


def score(indicators: dict, micro: dict) -> dict:
    """
    Evaluate all 7 signals. Each signal contributes +1 (bullish), -1 (bearish), or 0 (neutral).

    Args:
        indicators: output of core.indicators.calculate()
        micro:      dict with keys: obi, spread_bps, cvd, cvd_trend, buy_vol, sell_vol

    Returns dict with:
        score:       int in range [-7, +7]
        direction:   "long" | "short" | "neutral"
        signals:     list of signal breakdown dicts
        tradeable:   bool — True if |score| >= MIN_CONFLUENCE_SCORE and spread OK
    """
    signals = []
    total   = 0

    price     = indicators.get("price", 0)
    rsi       = indicators.get("rsi", 50)
    ema_fast  = indicators.get("ema_fast", price)
    ema_slow  = indicators.get("ema_slow", price)
    bb_upper  = indicators.get("bb_upper", price * 1.02)
    bb_lower  = indicators.get("bb_lower", price * 0.98)
    vwap_dev  = indicators.get("vwap_deviation", 0.0)   # % deviation from VWAP
    vol_ratio = indicators.get("vol_ratio", 1.0)

    obi       = micro.get("obi", 0.0)
    spread    = micro.get("spread_bps", 999.0)
    cvd_trend = micro.get("cvd_trend", "flat")

    # ── Signal 1: EMA trend ────────────────────────────────────────────────────
    if ema_fast > ema_slow * 1.0001:
        s1 = 1
    elif ema_fast < ema_slow * 0.9999:
        s1 = -1
    else:
        s1 = 0
    signals.append({"name": "EMA trend (9 vs 21)", "value": s1,
                    "detail": f"EMA9={ema_fast:.4f} EMA21={ema_slow:.4f}"})
    total += s1

    # ── Signal 2: RSI ──────────────────────────────────────────────────────────
    if rsi <= config.RSI_OVERSOLD:
        s2 = 1
    elif rsi >= config.RSI_OVERBOUGHT:
        s2 = -1
    else:
        s2 = 0
    signals.append({"name": "RSI", "value": s2,
                    "detail": f"RSI={rsi:.1f} (OS<{config.RSI_OVERSOLD} OB>{config.RSI_OVERBOUGHT})"})
    total += s2

    # ── Signal 3: Bollinger Band position ──────────────────────────────────────
    if price <= bb_lower * 1.001:
        s3 = 1
    elif price >= bb_upper * 0.999:
        s3 = -1
    else:
        s3 = 0
    signals.append({"name": "Bollinger Band position", "value": s3,
                    "detail": f"price={price:.4f} lower={bb_lower:.4f} upper={bb_upper:.4f}"})
    total += s3

    # ── Signal 4: VWAP deviation ───────────────────────────────────────────────
    if vwap_dev <= -0.10:          # price below VWAP by 0.10%+ → bullish
        s4 = 1
    elif vwap_dev >= 0.10:         # price above VWAP by 0.10%+ → bearish
        s4 = -1
    else:
        s4 = 0
    signals.append({"name": "VWAP deviation", "value": s4,
                    "detail": f"dev={vwap_dev:.3f}%"})
    total += s4

    # ── Signal 5: Order Book Imbalance ─────────────────────────────────────────
    if obi >= config.OBI_THRESHOLD:
        s5 = 1
    elif obi <= -config.OBI_THRESHOLD:
        s5 = -1
    else:
        s5 = 0
    signals.append({"name": "OBI", "value": s5,
                    "detail": f"OBI={obi:.3f} (threshold=±{config.OBI_THRESHOLD})"})
    total += s5

    # ── Signal 6: CVD trend ────────────────────────────────────────────────────
    if cvd_trend == "rising":
        s6 = 1
    elif cvd_trend == "falling":
        s6 = -1
    else:
        s6 = 0
    signals.append({"name": "CVD trend", "value": s6,
                    "detail": f"cvd_trend={cvd_trend}"})
    total += s6

    # ── Signal 7: Volume confirmation ─────────────────────────────────────────
    if vol_ratio >= config.VOLUME_RATIO_MIN:
        # Volume spike — directional based on price position vs VWAP
        s7 = 1 if vwap_dev <= 0 else -1
    else:
        s7 = 0
    signals.append({"name": "Volume", "value": s7,
                    "detail": f"vol_ratio={vol_ratio:.2f}x (min={config.VOLUME_RATIO_MIN})"})
    total += s7

    # ── Direction ──────────────────────────────────────────────────────────────
    if total >= config.MIN_CONFLUENCE_SCORE:
        direction = "long"
    elif total <= -config.MIN_CONFLUENCE_SCORE:
        direction = "short"
    else:
        direction = "neutral"

    spread_ok = spread <= config.SPREAD_MAX_BPS
    tradeable = (abs(total) >= config.MIN_CONFLUENCE_SCORE and spread_ok
                 and not indicators.get("atr_elevated", False))

    return {
        "score":      total,
        "direction":  direction,
        "signals":    signals,
        "spread_bps": spread,
        "spread_ok":  spread_ok,
        "tradeable":  tradeable,
    }
