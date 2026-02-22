"""
Rule-Based Decision Engine
==========================
Zero-latency replacement for the OpenAI AI engine.

Produces an identical output dict to the former ai_engine so main.py
needs only a single import change.

Decision pipeline
-----------------
1. Guard checks  — immediate WAIT on bad market/account conditions
2. Trend filter  — confirm trade direction with EMA stack + MACD
3. Quality score — 0-100 confidence built from 10+ independent factors
4. Risk rating   — LOW / MEDIUM / HIGH from opposing signal count
5. SL / TP calc  — ATR-based with configurable R:R floor
6. Decision      — BUY / SELL / WAIT

Design goals
------------
- Maximum accuracy: only trade when multiple independent factors agree
- Minimum loss: aggressive filtering of counter-trend and low-quality signals
- Instant: no network calls, deterministic, runs in microseconds
"""
from __future__ import annotations

from loguru import logger

import config

# Risk:Reward applied to ATR-derived stop loss.
# Higher = more profit per win, fewer wins needed to cover losses.
_BASE_RR  = 1.6          # default (raised to 1.7 for very high-confidence signals)
_HIGH_RR  = 1.8          # used when confidence >= 80


async def get_decision(
    symbol: str,
    indicators: dict,
    micro: dict,
    confluence: dict,
    portfolio_state: dict,
    fear_greed: dict | None = None,
    funding_rate: float = 0.0,
    open_interest_change: float = 0.0,
) -> dict:
    """
    Async wrapper so main.py can `await` this identically to the old ai_engine.
    All logic is synchronous — the async shell costs nothing.
    """
    return _decide(
        symbol, indicators, micro, confluence,
        portfolio_state, fear_greed, funding_rate,
    )


# ---------------------------------------------------------------------------

def _decide(
    symbol: str,
    ind: dict,
    micro: dict,
    conf: dict,
    port: dict,
    fear_greed: dict | None,
    funding_rate: float,
) -> dict:

    score     = conf.get("score", 0)
    direction = conf.get("direction", "neutral")
    signals   = conf.get("signals", [])
    spread    = micro.get("spread_bps", 999.0)

    bullish: list[str] = []
    bearish: list[str] = []

    # -- Extract indicators ---------------------------------------------------
    price     = ind.get("price", 0.0)
    rsi       = ind.get("rsi", 50.0)
    ema_fast  = ind.get("ema_fast", price)
    ema_slow  = ind.get("ema_slow", price)
    bb_upper  = ind.get("bb_upper", price * 1.02)
    bb_lower  = ind.get("bb_lower", price * 0.98)
    vwap_dev  = ind.get("vwap_deviation", 0.0)
    vol_ratio = ind.get("vol_ratio", 1.0)
    atr       = ind.get("atr", 0.0)
    macd      = ind.get("macd", 0.0)
    macd_sig  = ind.get("macd_signal", 0.0)
    macd_hist = ind.get("macd_hist", 0.0)
    stoch_k   = ind.get("stoch_k", 50.0)
    stoch_d   = ind.get("stoch_d", 50.0)
    bb_width  = ind.get("bb_width", 0.02)

    obi       = micro.get("obi", 0.0)
    cvd_trend = micro.get("cvd_trend", "flat")

    # =========================================================================
    # STEP 1 — Hard guard checks (immediate WAIT)
    # =========================================================================

    if direction == "neutral":
        return _wait("Neutral — no directional bias from confluence")

    if spread > config.SPREAD_MAX_BPS:
        return _wait(f"Spread {spread:.1f} bps > max {config.SPREAD_MAX_BPS} bps")

    if ind.get("atr_elevated", False):
        return _wait("ATR elevated — avoid trading during volatility spikes")

    # Funding rate: longs pay when positive, shorts pay when negative
    if direction == "long" and funding_rate > 0.03:
        return _wait(f"Funding {funding_rate:+.3f}% penalises new longs")
    if direction == "short" and funding_rate < -0.01:
        return _wait(f"Funding {funding_rate:+.3f}% penalises new shorts")

    # Bollinger Band squeeze — low volatility, breakout direction unclear
    if bb_width < 0.005:
        return _wait("BB squeeze — wait for breakout direction to confirm")

    # Portfolio safety valve: back off when close to consecutive-loss limit
    if port.get("consecutive_losses", 0) >= config.MAX_CONSECUTIVE_LOSSES - 1:
        return _wait("Approaching consecutive-loss limit — sitting out")

    # =========================================================================
    # STEP 2 — Trend quality filter
    # Must have EMA alignment OR MACD crossover to trade WITH the signal.
    # Counter-trend trades are only allowed with extreme RSI + BB touch.
    # =========================================================================

    ema_bullish  = ema_fast > ema_slow * 1.0001
    ema_bearish  = ema_fast < ema_slow * 0.9999
    macd_bullish = macd_hist > 0 and macd > macd_sig
    macd_bearish = macd_hist < 0 and macd < macd_sig

    if direction == "long":
        trend_aligned = ema_bullish or macd_bullish
        extreme_reversal = rsi <= 28 and price <= bb_lower * 1.003
        if not trend_aligned and not extreme_reversal:
            return _wait(
                "Long signal but EMA/MACD trend not aligned — "
                "needs RSI<=28 + BB-lower touch for counter-trend entry"
            )

    else:  # short
        trend_aligned = ema_bearish or macd_bearish
        extreme_reversal = rsi >= 72 and price >= bb_upper * 0.997
        if not trend_aligned and not extreme_reversal:
            return _wait(
                "Short signal but EMA/MACD trend not aligned — "
                "needs RSI>=72 + BB-upper touch for counter-trend entry"
            )

    # =========================================================================
    # STEP 3 — Confidence scoring (0 - 100)
    # Base from raw confluence count, then adjusted by quality of each factor.
    # =========================================================================

    # Base: abs(score)/7 × 100  →  4/7=57.1  5/7=71.4  6/7=85.7  7/7=100
    confidence = (abs(score) / 7) * 100

    # ── RSI ──────────────────────────────────────────────────────────────────
    if direction == "long":
        if rsi <= 25:
            confidence += 14
            bullish.append(f"RSI deep oversold {rsi:.1f} — high-probability bounce")
        elif rsi <= 35:
            confidence += 8
            bullish.append(f"RSI oversold {rsi:.1f}")
        elif rsi <= 45:
            confidence += 3
            bullish.append(f"RSI neutral-low {rsi:.1f}")
        elif rsi >= 65:
            confidence -= 14
            bearish.append(f"RSI {rsi:.1f} over-extended — poor long entry")
        elif rsi >= 55:
            confidence -= 6
            bearish.append(f"RSI {rsi:.1f} elevated for long")
    else:
        if rsi >= 75:
            confidence += 14
            bearish.append(f"RSI deep overbought {rsi:.1f} — high-probability reversal")
        elif rsi >= 65:
            confidence += 8
            bearish.append(f"RSI overbought {rsi:.1f}")
        elif rsi >= 55:
            confidence += 3
            bearish.append(f"RSI neutral-high {rsi:.1f}")
        elif rsi <= 35:
            confidence -= 14
            bullish.append(f"RSI {rsi:.1f} over-extended — poor short entry")
        elif rsi <= 45:
            confidence -= 6
            bullish.append(f"RSI {rsi:.1f} low for short")

    # ── Stochastic ────────────────────────────────────────────────────────────
    if direction == "long" and stoch_k <= 20 and stoch_d <= 20:
        confidence += 9
        bullish.append(f"Stochastic oversold K={stoch_k:.0f}/D={stoch_d:.0f}")
    elif direction == "long" and stoch_k >= 80:
        confidence -= 8
        bearish.append(f"Stochastic overbought K={stoch_k:.0f} — avoid long")
    elif direction == "short" and stoch_k >= 80 and stoch_d >= 80:
        confidence += 9
        bearish.append(f"Stochastic overbought K={stoch_k:.0f}/D={stoch_d:.0f}")
    elif direction == "short" and stoch_k <= 20:
        confidence -= 8
        bullish.append(f"Stochastic oversold K={stoch_k:.0f} — avoid short")

    # ── Bollinger Band position ───────────────────────────────────────────────
    if direction == "long":
        if price <= bb_lower * 1.001:
            confidence += 12
            bullish.append("Price at lower BB — classic mean-reversion buy zone")
        elif price >= bb_upper * 0.999:
            confidence -= 12
            bearish.append("Price at upper BB — risky long entry")
    else:
        if price >= bb_upper * 0.999:
            confidence += 12
            bearish.append("Price at upper BB — classic mean-reversion sell zone")
        elif price <= bb_lower * 1.001:
            confidence -= 12
            bullish.append("Price at lower BB — risky short entry")

    # ── EMA alignment ────────────────────────────────────────────────────────
    if direction == "long":
        if ema_bullish:
            confidence += 8
            bullish.append(f"EMA9 > EMA21 — uptrend structure intact")
        else:
            confidence -= 5
            bearish.append("EMA9 < EMA21 — counter-trend long")
    else:
        if ema_bearish:
            confidence += 8
            bearish.append("EMA9 < EMA21 — downtrend structure intact")
        else:
            confidence -= 5
            bullish.append("EMA9 > EMA21 — counter-trend short")

    # ── MACD histogram momentum ───────────────────────────────────────────────
    if direction == "long":
        if macd_hist > 0:
            confidence += 6
            bullish.append("MACD histogram positive — bullish momentum")
        else:
            confidence -= 6
            bearish.append("MACD histogram negative — momentum opposes long")
    else:
        if macd_hist < 0:
            confidence += 6
            bearish.append("MACD histogram negative — bearish momentum")
        else:
            confidence -= 6
            bullish.append("MACD histogram positive — momentum opposes short")

    # ── VWAP positioning ──────────────────────────────────────────────────────
    if direction == "long":
        if vwap_dev <= -0.08:
            confidence += 7
            bullish.append(f"Price {abs(vwap_dev):.2f}% below VWAP — value zone")
        elif vwap_dev >= 0.15:
            confidence -= 7
            bearish.append(f"Price {vwap_dev:.2f}% above VWAP — over-extended for long")
    else:
        if vwap_dev >= 0.08:
            confidence += 7
            bearish.append(f"Price {vwap_dev:.2f}% above VWAP — premium zone")
        elif vwap_dev <= -0.15:
            confidence -= 7
            bullish.append(f"Price {abs(vwap_dev):.2f}% below VWAP — over-extended for short")

    # ── Order Book Imbalance ──────────────────────────────────────────────────
    if direction == "long":
        if obi >= 0.45:
            confidence += 9
            bullish.append(f"Strong buy-side order book OBI={obi:+.3f}")
        elif obi >= 0.25:
            confidence += 4
            bullish.append(f"Mild buy pressure OBI={obi:+.3f}")
        elif obi <= -0.35:
            confidence -= 9
            bearish.append(f"Sell pressure opposes long OBI={obi:+.3f}")
    else:
        if obi <= -0.45:
            confidence += 9
            bearish.append(f"Strong sell-side order book OBI={obi:+.3f}")
        elif obi <= -0.25:
            confidence += 4
            bearish.append(f"Mild sell pressure OBI={obi:+.3f}")
        elif obi >= 0.35:
            confidence -= 9
            bullish.append(f"Buy pressure opposes short OBI={obi:+.3f}")

    # ── CVD (Cumulative Volume Delta) ─────────────────────────────────────────
    if direction == "long":
        if cvd_trend == "rising":
            confidence += 7
            bullish.append("CVD rising — aggressive buyers dominating tape")
        elif cvd_trend == "falling":
            confidence -= 7
            bearish.append("CVD falling — sellers dominate despite long signal")
    else:
        if cvd_trend == "falling":
            confidence += 7
            bearish.append("CVD falling — aggressive sellers dominating tape")
        elif cvd_trend == "rising":
            confidence -= 7
            bullish.append("CVD rising — buyers dominate despite short signal")

    # ── Volume ───────────────────────────────────────────────────────────────
    if vol_ratio >= 2.0:
        confidence += 8
        (bullish if direction == "long" else bearish).append(
            f"High-volume surge {vol_ratio:.1f}x avg — strong conviction"
        )
    elif vol_ratio >= 1.3:
        confidence += 4
        (bullish if direction == "long" else bearish).append(
            f"Above-average volume {vol_ratio:.1f}x — confirmed move"
        )
    elif vol_ratio < 0.7:
        confidence -= 7
        bearish.append(f"Very low volume {vol_ratio:.1f}x — weak conviction, avoid")
    elif vol_ratio < 1.0:
        confidence -= 3

    # ── Fear & Greed macro ───────────────────────────────────────────────────
    if fear_greed:
        try:
            fg = int(fear_greed.get("value", 50))
        except (ValueError, TypeError):
            fg = 50
        if direction == "long" and fg <= 20:
            confidence += 6
            bullish.append(f"Extreme Fear ({fg}) — contrarian long opportunity")
        elif direction == "long" and fg >= 85:
            confidence -= 6
            bearish.append(f"Extreme Greed ({fg}) — caution on longs")
        elif direction == "short" and fg >= 80:
            confidence += 6
            bearish.append(f"Extreme Greed ({fg}) — contrarian short opportunity")
        elif direction == "short" and fg <= 15:
            confidence -= 6
            bullish.append(f"Extreme Fear ({fg}) — caution on shorts")

    # =========================================================================
    # STEP 4 — Risk rating from opposing signal count
    # =========================================================================

    if direction == "long":
        opposing = sum(1 for s in signals if s.get("value", 0) < 0)
    else:
        opposing = sum(1 for s in signals if s.get("value", 0) > 0)

    if opposing >= 3:
        risk = "HIGH"
    elif opposing == 2:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # =========================================================================
    # STEP 5 — Clamp confidence and apply minimum threshold
    # =========================================================================

    confidence = max(0.0, min(100.0, round(confidence, 1)))

    if confidence < config.MIN_AI_CONFIDENCE:
        return _wait(
            f"Confidence {confidence:.0f}% below {config.MIN_AI_CONFIDENCE:.0f}% threshold",
            confidence=confidence,
            bullish=bullish,
            bearish=bearish,
        )

    if risk == "HIGH":
        return _wait(
            f"Risk HIGH — {opposing} signals oppose direction",
            confidence=confidence,
            bullish=bullish,
            bearish=bearish,
        )

    # =========================================================================
    # STEP 6 — SL / TP calculation
    # =========================================================================

    if atr > 0 and price > 0:
        sl_pct = (config.ATR_STOP_MULTIPLIER * atr / price) * 100
        sl_pct = min(sl_pct, config.MAX_STOP_LOSS_PCT * 100)
    else:
        sl_pct = config.MAX_STOP_LOSS_PCT * 100

    rr       = _HIGH_RR if confidence >= 80 else _BASE_RR
    tp_pct   = sl_pct * rr

    # Must clear round-trip fees with margin
    min_tp = config.MIN_PROFIT_TARGET_PCT * 100
    if tp_pct < min_tp:
        return _wait(
            f"TP {tp_pct:.3f}% < min profit target {min_tp:.3f}% after fees",
            confidence=confidence,
            bullish=bullish,
            bearish=bearish,
        )

    # =========================================================================
    # STEP 7 — Final decision
    # =========================================================================

    action   = "BUY" if direction == "long" else "SELL"
    key_risk = (
        bearish[0] if direction == "long" and bearish else
        bullish[0] if direction == "short" and bullish else
        "No significant opposing factors"
    )

    logger.debug(
        f"[RuleEngine] {symbol} -> {action} | conf={confidence:.0f}% | "
        f"risk={risk} | SL={sl_pct:.3f}% TP={tp_pct:.3f}% (1:{rr} R:R)"
    )

    return {
        "action":                action,
        "confidence":            confidence,
        "confluence_score":      score,
        "stop_loss_pct":         round(sl_pct, 4),
        "take_profit_pct":       round(tp_pct, 4),
        "estimated_net_profit_pct": round(tp_pct - 0.04, 4),
        "risk_assessment":       risk,
        "hold_duration_estimate": "10-30 minutes",
        "reasoning": {
            "bullish_factors": bullish,
            "bearish_factors": bearish,
            "key_risk":        key_risk,
        },
    }


# ---------------------------------------------------------------------------

def _wait(
    reason: str,
    confidence: float = 0.0,
    bullish: list | None = None,
    bearish: list | None = None,
) -> dict:
    return {
        "action":           "WAIT",
        "confidence":       confidence,
        "wait_reason":      reason,
        "confluence_score": 0,
        "risk_assessment":  "HIGH",
        "reasoning": {
            "bullish_factors": bullish or [],
            "bearish_factors": bearish or [],
            "key_risk":        reason,
        },
    }
