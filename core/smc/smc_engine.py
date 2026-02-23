"""
SMC Engine — Master Signal Brain
==================================
Combines all 6 SMC modules into a single, unified trading decision.

Entry criteria (all must align for a trade):
  1. Market Structure — HTF (15m) bias determines direction
  2. Order Block      — price must be inside an unmitigated OB (primary POI)
  3. Fair Value Gap   — price inside an FVG (secondary / additive POI)
  4. Liquidity Sweep  — recent stop-hunt sweep confirms institutional reversal intent
  5. Premium/Discount — longs only in discount, shorts only in premium
  6. Killzone         — higher confidence during London / NY open

Output dict is identical in structure to the old decision_engine so main.py
needs only a single import swap.
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

import config
from core.smc import (
    market_structure,
    order_blocks,
    fvg,
    liquidity,
    premium_discount,
    killzone,
)


# ── Public entry point ────────────────────────────────────────────────────────

async def analyze(
    symbol: str,
    candles_5m: pd.DataFrame,
    indicators: dict,
    micro: dict,
    portfolio_state: dict,
    fear_greed: dict | None = None,
    funding_rate: float = 0.0,
) -> dict:
    """
    Full SMC analysis pipeline.

    Async wrapper so main.py can `await` this identically to the old engines.
    All computation is synchronous — zero network calls.
    Never raises: any internal exception is caught, logged, and returns WAIT.
    """
    try:
        return _smc_decide(
            symbol=symbol,
            candles=candles_5m,
            ind=indicators,
            micro=micro,
            port=portfolio_state,
            fear_greed=fear_greed,
            funding_rate=funding_rate,
        )
    except Exception:
        import traceback as _tb
        err = _tb.format_exc()
        logger.error(f"[SMC] {symbol} internal error — returning WAIT:\n{err}")
        return _wait(f"Internal SMC error (see log): {err.splitlines()[-1]}")


# ── Core decision logic ───────────────────────────────────────────────────────

def _smc_decide(
    symbol: str,
    candles: pd.DataFrame,
    ind: dict,
    micro: dict,
    port: dict,
    fear_greed: dict | None,
    funding_rate: float,
) -> dict:

    if len(candles) < 30:
        return _wait("Insufficient candle data for SMC analysis")

    price          = ind.get("price",          0.0)
    atr            = ind.get("atr",            0.0)
    rsi            = ind.get("rsi",            50.0)
    rsi_prev       = ind.get("rsi_prev",       50.0)
    macd_hist      = ind.get("macd_hist",      0.0)
    macd_hist_prev = ind.get("macd_hist_prev", 0.0)
    stoch_k        = ind.get("stoch_k",        50.0)
    spread         = micro.get("spread_bps",   999.0)

    if price <= 0:
        return _wait("Invalid price data")

    # ── Hard guards (same rules as old engine) ────────────────────────────────
    if spread > config.SPREAD_MAX_BPS:
        return _wait(f"Spread {spread:.1f} bps > max {config.SPREAD_MAX_BPS} bps")

    if ind.get("atr_elevated", False):
        return _wait("ATR elevated — volatility spike, standing aside")

    if port.get("consecutive_losses", 0) >= config.MAX_CONSECUTIVE_LOSSES - 1:
        return _wait("Approaching consecutive-loss limit — sitting out")

    # Funding rate guards
    long_blocked  = funding_rate > 0.03
    short_blocked = funding_rate < -0.01

    # ── Step 1 — Market Structure ─────────────────────────────────────────────
    candles_15m = _resample_to_15m(candles)
    htf_ms = market_structure.analyze(candles_15m, swing_n=config.SMC_SWING_N)
    ltf_ms = market_structure.analyze(candles,     swing_n=config.SMC_SWING_N)

    htf_bias = htf_ms.bias   # "bullish" | "bearish" | "ranging"
    ltf_bias = ltf_ms.bias

    if htf_bias == "ranging" and ltf_bias == "ranging":
        return _wait("Both HTF and LTF ranging — no structural edge")

    # Primary bias = HTF; fall back to LTF only when HTF is ranging
    bias = htf_bias if htf_bias != "ranging" else ltf_bias

    # ── Step 2 — Order Blocks ─────────────────────────────────────────────────
    all_sh = (htf_ms.swing_highs or []) + (ltf_ms.swing_highs or [])
    all_sl = (htf_ms.swing_lows  or []) + (ltf_ms.swing_lows  or [])

    bull_obs, bear_obs = order_blocks.find_order_blocks(
        candles, all_sh, all_sl, lookback=config.SMC_OB_LOOKBACK,
    )

    active_bull_ob = order_blocks.price_in_ob(price, bull_obs)
    active_bear_ob = order_blocks.price_in_ob(price, bear_obs)

    # ── Step 3 — Fair Value Gaps ──────────────────────────────────────────────
    bull_fvgs, bear_fvgs = fvg.find_fvgs(
        candles,
        min_size_pct=config.SMC_FVG_MIN_SIZE_PCT,
        lookback=config.SMC_FVG_LOOKBACK,
    )

    active_bull_fvg = fvg.price_in_fvg(price, bull_fvgs)
    active_bear_fvg = fvg.price_in_fvg(price, bear_fvgs)

    # ── Step 4 — Liquidity ────────────────────────────────────────────────────
    buy_pools, sell_pools = liquidity.find_liquidity_pools(
        candles, all_sh, all_sl,
        cluster_pct=config.SMC_LIQUIDITY_CLUSTER_PCT,
    )

    sweeps = liquidity.detect_liquidity_sweeps(
        candles, buy_pools, sell_pools,
        lookback=config.SMC_SWEEP_LOOKBACK,
    )

    recent_bull_sweep = next(
        (s for s in sweeps if s.kind == "swept_buy_side"  and s.age <= 8), None
    )
    recent_bear_sweep = next(
        (s for s in sweeps if s.kind == "swept_sell_side" and s.age <= 8), None
    )

    # ── Step 5 — Premium / Discount ───────────────────────────────────────────
    if ltf_ms.last_swing_high and ltf_ms.last_swing_low:
        zone_info = premium_discount.classify(
            price,
            ltf_ms.last_swing_high.price,
            ltf_ms.last_swing_low.price,
        )
    else:
        zone_info = premium_discount.PriceZone(
            zone="equilibrium", fib_level=0.50, in_ote=False,
            swing_high=price, swing_low=price, equilibrium=price,
        )

    # ── Step 6 — Killzone ─────────────────────────────────────────────────────
    session = killzone.current_session()

    # =========================================================================
    # DIRECTION DECISION
    # =========================================================================

    # ── Reversal candle confirmation (prevents falling-knife entries) ────────
    # Check the last CLOSED candle (index -2) and the one before it (index -3)
    _c = candles["close"].values
    _o = candles["open"].values
    _l = candles["low"].values
    _h = candles["high"].values
    _last_bullish  = len(_c) >= 2 and _c[-2] > _o[-2]   # last closed candle green
    _last_bearish  = len(_c) >= 2 and _c[-2] < _o[-2]   # last closed candle red
    _higher_low    = len(_l) >= 3 and _l[-2] > _l[-3]   # price making higher low
    _lower_high    = len(_h) >= 3 and _h[-2] < _h[-3]   # price making lower high
    _rsi_turning_up   = rsi > rsi_prev + 0.3             # RSI slope positive
    _rsi_turning_down = rsi < rsi_prev - 0.3             # RSI slope negative

    # RSI extreme conditions — requires at least one reversal confirmation signal
    # to prevent entering while price is still in freefall / still at peak
    rsi_long_poi  = (rsi <= 30) and (_last_bullish or _higher_low or _rsi_turning_up)
    rsi_short_poi = (rsi >= 70) and (_last_bearish or _lower_high or _rsi_turning_down)

    # A valid POI must exist in the direction of bias (OB, FVG, or RSI extreme)
    long_poi  = active_bull_ob or active_bull_fvg or rsi_long_poi
    short_poi = active_bear_ob or active_bear_fvg or rsi_short_poi

    # RSI extremes allow counter-trend entries (RSI=25 overrides bearish structure for longs)
    can_long  = (bias == "bullish" or rsi_long_poi)  and bool(long_poi) and not long_blocked
    can_short = (bias == "bearish" or rsi_short_poi) and bool(short_poi) and not short_blocked

    if not can_long and not can_short:
        detail_parts = []
        if not long_poi and not short_poi:
            detail_parts.append("no OB or FVG at price")
        if bias == "ranging":
            detail_parts.append("structure ranging")
        if long_blocked and bias == "bullish":
            detail_parts.append(f"funding {funding_rate:+.3f}% blocks longs")
        if short_blocked and bias == "bearish":
            detail_parts.append(f"funding {funding_rate:+.3f}% blocks shorts")
        return _wait("; ".join(detail_parts) or "No SMC confluence aligned")

    direction = "long" if can_long else "short"

    # Flag whether this is an RSI-extreme entry (possibly counter-trend)
    rsi_extreme_entry = (
        (direction == "long"  and rsi <= 30) or
        (direction == "short" and rsi >= 70)
    )
    counter_trend = (
        (direction == "long"  and bias != "bullish") or
        (direction == "short" and bias != "bearish")
    )

    # =========================================================================
    # CONFIDENCE SCORING  (0 – 100)
    # =========================================================================

    # RSI-driven counter-trend entries start with a 40-point base so the
    # reversal signal can overcome structural opposition penalties.
    confidence = 40.0 if (rsi_extreme_entry and counter_trend) else 0.0
    bullish_f: list[str] = []
    bearish_f: list[str] = []
    factors   = bullish_f if direction == "long" else bearish_f
    counter_f = bearish_f if direction == "long" else bullish_f

    # ── Market structure (up to 35 pts) ──────────────────────────────────────
    if htf_bias == bias:
        confidence += 25
        factors.append(f"HTF {htf_bias} structure confirmed (15m)")
    elif htf_bias == "ranging":
        confidence += 12
        factors.append(f"HTF ranging — LTF {ltf_bias} bias only")
    else:
        penalty = -4 if (rsi_extreme_entry and counter_trend) else -10
        confidence += penalty
        counter_f.append(f"HTF {htf_bias} opposes {direction} — counter-trend risk")

    if ltf_bias == bias:
        confidence += 10
        factors.append(f"LTF {ltf_bias} structure aligned (5m)")
    elif ltf_bias != "ranging":
        penalty = -3 if (rsi_extreme_entry and counter_trend) else -8
        confidence += penalty
        counter_f.append(f"LTF structure ({ltf_bias}) diverges from bias")

    # ── RSI extreme reversal signal (up to 25 pts) ───────────────────────────
    if direction == "long" and rsi <= 30:
        rsi_pts = 25 if rsi <= 25 else 15
        confidence += rsi_pts
        factors.append(f"RSI deeply oversold ({rsi:.1f}) — reversal bounce (+{rsi_pts}pts)")
    elif direction == "short" and rsi >= 70:
        rsi_pts = 25 if rsi >= 75 else 15
        confidence += rsi_pts
        factors.append(f"RSI deeply overbought ({rsi:.1f}) — reversal short (+{rsi_pts}pts)")
    elif direction == "long" and rsi < 40:
        confidence += 8
        factors.append(f"RSI bearish ({rsi:.1f}) — approaching oversold, supports long")
    elif direction == "short" and rsi > 60:
        confidence += 8
        factors.append(f"RSI bullish ({rsi:.1f}) — approaching overbought, supports short")

    # ── Order Block (up to 25 pts) ────────────────────────────────────────────
    active_ob = active_bull_ob if direction == "long" else active_bear_ob
    if active_ob:
        _str = active_ob.strength if (active_ob.strength == active_ob.strength) else 0.0
        ob_pts = int(15 + _str * 10)   # 15 base + up to 10 for strong OB
        ob_pts = min(ob_pts, 25)
        confidence += ob_pts
        factors.append(
            f"Price inside {direction} OB [{active_ob.low:.4f}–{active_ob.high:.4f}] "
            f"str={active_ob.strength:.2f} touches={active_ob.touches}"
        )

    # ── Fair Value Gap (up to 20 pts) ─────────────────────────────────────────
    active_fvg = active_bull_fvg if direction == "long" else active_bear_fvg
    if active_fvg:
        fvg_pts = max(int(20 - active_fvg.filled_pct * 12), 5)
        confidence += fvg_pts
        factors.append(
            f"Price inside {direction} FVG [{active_fvg.bottom:.4f}–{active_fvg.top:.4f}] "
            f"filled={active_fvg.filled_pct:.0%} age={active_fvg.age}c"
        )

    # OB + FVG confluence bonus
    if active_ob and active_fvg:
        confidence += 10
        factors.append("OB + FVG overlap — institutional double-confirmation zone")

    # ── Liquidity sweep (up to 20 pts) ────────────────────────────────────────
    relevant_sweep = recent_bull_sweep if direction == "long" else recent_bear_sweep
    if relevant_sweep:
        sweep_pts = 20 if relevant_sweep.reversal_confirmed else 10
        confidence += sweep_pts
        factors.append(
            f"Liquidity sweep ({relevant_sweep.kind}) at {relevant_sweep.price:.4f} "
            f"{relevant_sweep.age}c ago"
            + (" ✓ reversal confirmed" if relevant_sweep.reversal_confirmed else "")
        )
    else:
        if not rsi_extreme_entry:
            confidence -= 5
        counter_f.append("No recent liquidity sweep")

    # ── Premium / Discount zone (up to 15 pts) ────────────────────────────────
    if (direction == "long"  and zone_info.zone == "discount") or \
       (direction == "short" and zone_info.zone == "premium"):
        confidence += 10
        factors.append(
            f"Price in {zone_info.zone} zone (fib {zone_info.fib_level:.3f})"
            + (" — OTE ✓" if zone_info.in_ote else "")
        )
        if zone_info.in_ote:
            confidence += 5
    elif zone_info.zone == "equilibrium":
        confidence -= 5
        counter_f.append("Price at equilibrium — slight deduction")

    # ── Killzone timing (±15 pts) ─────────────────────────────────────────────
    # For RSI-extreme entries, cap the dead-zone penalty so strong reversal
    # signals are not completely suppressed during low-liquidity hours.
    kz_adj = session.confidence_adjustment
    if rsi_extreme_entry and kz_adj < 0:
        kz_adj = max(kz_adj, -8)
    confidence += kz_adj
    if session.in_killzone:
        factors.append(
            f"In {session.session} killzone "
            f"({session.minutes_into_session}m elapsed) "
            f"+{kz_adj}pts"
        )
    else:
        counter_f.append(f"Outside killzone ({session.session}) {kz_adj:+d}pts")

    # ── Microstructure confirmation (up to ±10 pts) ───────────────────────────
    obi       = micro.get("obi", 0.0)
    cvd_trend = micro.get("cvd_trend", "flat")

    if direction == "long":
        if obi >= 0.25:
            confidence += 5
            factors.append(f"Buy-side order book pressure OBI={obi:+.3f}")
        elif obi <= -0.35:
            confidence -= 5
            counter_f.append(f"Sell pressure in order book OBI={obi:+.3f}")

        if cvd_trend == "rising":
            confidence += 5
            factors.append("CVD rising — buyers absorbing sellers")
        elif cvd_trend == "falling":
            confidence -= 5
            counter_f.append("CVD falling — sellers still active")
    else:
        if obi <= -0.25:
            confidence += 5
            factors.append(f"Sell-side order book pressure OBI={obi:+.3f}")
        elif obi >= 0.35:
            confidence -= 5
            counter_f.append(f"Buy pressure in order book OBI={obi:+.3f}")

        if cvd_trend == "falling":
            confidence += 5
            factors.append("CVD falling — sellers dominate")
        elif cvd_trend == "rising":
            confidence -= 5
            counter_f.append("CVD rising — buyers dominate, opposes short")

    # ── Fear & Greed macro sentiment (±6 pts) ────────────────────────────────
    if fear_greed:
        try:
            fg = int(fear_greed.get("value", 50))
        except (ValueError, TypeError):
            fg = 50

        if direction == "long" and fg <= 25:
            confidence += 6
            factors.append(f"Extreme Fear ({fg}) — contrarian long aligns with SMC")
        elif direction == "long" and fg >= 80:
            confidence -= 6
            counter_f.append(f"Extreme Greed ({fg}) — caution on longs at this level")
        elif direction == "short" and fg >= 80:
            confidence += 6
            factors.append(f"Extreme Greed ({fg}) — contrarian short aligns with SMC")
        elif direction == "short" and fg <= 25:
            confidence -= 6
            counter_f.append(f"Extreme Fear ({fg}) — caution on shorts at this level")

    # ── MACD momentum confirmation (±8 pts) ──────────────────────────────────
    macd_turning_up   = macd_hist > macd_hist_prev and macd_hist > -0.001 * price
    macd_turning_down = macd_hist < macd_hist_prev and macd_hist <  0.001 * price

    if direction == "long":
        if macd_turning_up:
            confidence += 8
            factors.append(f"MACD histogram turning up ({macd_hist:.4f}) — momentum confirming")
        elif macd_hist < macd_hist_prev:
            confidence -= 4
            counter_f.append("MACD histogram still falling — momentum not confirmed")
    else:
        if macd_turning_down:
            confidence += 8
            factors.append(f"MACD histogram turning down ({macd_hist:.4f}) — momentum confirming")
        elif macd_hist > macd_hist_prev:
            confidence -= 4
            counter_f.append("MACD histogram still rising — momentum not confirmed")

    # ── Stochastic confirmation (±6 pts) ─────────────────────────────────────
    if direction == "long":
        if stoch_k <= 25:
            confidence += 6
            factors.append(f"Stochastic oversold ({stoch_k:.1f}) — supports long")
        elif stoch_k >= 75:
            confidence -= 4
            counter_f.append(f"Stochastic overbought ({stoch_k:.1f}) — caution on long")
    else:
        if stoch_k >= 75:
            confidence += 6
            factors.append(f"Stochastic overbought ({stoch_k:.1f}) — supports short")
        elif stoch_k <= 25:
            confidence -= 4
            counter_f.append(f"Stochastic oversold ({stoch_k:.1f}) — caution on short")

    # =========================================================================
    # THRESHOLD GATE
    # =========================================================================

    confidence = max(0.0, min(100.0, round(confidence, 1)))

    # Count SMC confluence components for display (analogous to old 0-7 score)
    smc_score = _count_smc_score(active_ob, active_fvg, relevant_sweep, zone_info, htf_bias, bias, rsi)

    logger.debug(
        f"[SMC] {symbol} → {direction.upper()} | conf={confidence:.0f}% | "
        f"htf={htf_bias} ltf={ltf_bias} | "
        f"ob={'✓' if active_ob else '✗'} fvg={'✓' if active_fvg else '✗'} "
        f"sweep={'✓' if relevant_sweep else '✗'} "
        f"zone={zone_info.zone}({zone_info.fib_level:.2f}) | "
        f"session={session.session} | score={smc_score}/7"
    )

    if confidence < config.MIN_AI_CONFIDENCE:
        return _wait(
            f"Confidence {confidence:.0f}% below {config.MIN_AI_CONFIDENCE:.0f}% threshold — "
            f"ob={'✓' if active_ob else '✗'} fvg={'✓' if active_fvg else '✗'} "
            f"sweep={'✓' if relevant_sweep else '✗'} zone={zone_info.zone}",
            confidence=confidence,
            bullish=bullish_f,
            bearish=bearish_f,
        )

    # =========================================================================
    # SL / TP CALCULATION (SMC-style)
    # =========================================================================

    sl_pct, tp_pct = _calculate_sl_tp(
        direction=direction,
        price=price,
        active_ob=active_ob,
        active_fvg=active_fvg,
        sell_pools=sell_pools,
        buy_pools=buy_pools,
        ltf_ms=ltf_ms,
        atr=atr,
        confidence=confidence,
    )

    if sl_pct <= 0 or tp_pct <= 0:
        return _wait("Could not compute valid SMC SL/TP levels")

    min_tp = config.MIN_PROFIT_TARGET_PCT * 100
    if tp_pct < min_tp:
        return _wait(
            f"TP {tp_pct:.3f}% < minimum {min_tp:.3f}% — risk:reward too small",
            confidence=confidence,
        )

    action  = "BUY" if direction == "long" else "SELL"
    key_risk = (
        bearish_f[0] if direction == "long"  and bearish_f else
        bullish_f[0] if direction == "short" and bullish_f else
        "No significant opposing factors identified"
    )

    logger.info(
        f"[SMC] {symbol} → {action} | conf={confidence:.0f}% | "
        f"SL={sl_pct:.3f}% TP={tp_pct:.3f}% (1:{tp_pct/sl_pct:.1f} R:R) | "
        f"zone={zone_info.zone} session={session.session}"
    )

    return {
        "action":                   action,
        "confidence":               confidence,
        "confluence_score":         smc_score,
        "stop_loss_pct":            round(sl_pct, 4),
        "take_profit_pct":          round(tp_pct, 4),
        "estimated_net_profit_pct": round(tp_pct - 0.04, 4),
        "risk_assessment":          "LOW" if len(counter_f) <= 1 else "MEDIUM",
        "hold_duration_estimate":   "5-30 minutes",
        "reasoning": {
            "bullish_factors": bullish_f,
            "bearish_factors": bearish_f,
            "key_risk":        key_risk,
            "smc_details": {
                "htf_bias":    htf_bias,
                "ltf_bias":    ltf_bias,
                "order_block": (
                    f"{active_ob.low:.4f}–{active_ob.high:.4f} str={active_ob.strength:.2f}"
                    if active_ob else None
                ),
                "fvg": (
                    f"{active_fvg.bottom:.4f}–{active_fvg.top:.4f} "
                    f"filled={active_fvg.filled_pct:.0%}"
                    if active_fvg else None
                ),
                "sweep":       relevant_sweep.kind if relevant_sweep else None,
                "zone":        zone_info.zone,
                "fib":         zone_info.fib_level,
                "in_ote":      zone_info.in_ote,
                "session":     session.session,
                "in_killzone": session.in_killzone,
            },
        },
    }


# ── SL / TP calculation ───────────────────────────────────────────────────────

def _calculate_sl_tp(
    direction: str,
    price: float,
    active_ob,
    active_fvg,
    sell_pools: list,
    buy_pools: list,
    ltf_ms,
    atr: float,
    confidence: float,
) -> tuple[float, float]:
    """
    SMC-specific stop loss and take profit placement.

    SL: just beyond the invalidation boundary of the OB / FVG / last swing
    TP: at the nearest opposing liquidity pool, or R:R derived if no pool found
    """
    BUFFER = 0.0002   # 0.02% buffer beyond the OB for breathing room

    if direction == "long":
        # SL — below OB low (invalidation) or ATR fallback
        if active_ob:
            sl_price = active_ob.low * (1 - BUFFER)
        elif active_fvg:
            sl_price = active_fvg.bottom * (1 - BUFFER)
        elif ltf_ms.last_swing_low:
            sl_price = ltf_ms.last_swing_low.price * (1 - BUFFER)
        else:
            sl_price = price * (1 - max(atr / price, 0.002)) if atr > 0 else price * 0.995

        sl_pct = abs(price - sl_price) / price * 100

        # TP — at nearest sell-side liquidity above price
        tp_candidates = [p.price for p in sell_pools if p.price > price * 1.001]
        if tp_candidates:
            tp_price = min(tp_candidates)
        elif ltf_ms.last_swing_high:
            tp_price = ltf_ms.last_swing_high.price
        else:
            rr = 2.2 if confidence >= 80 else 1.8
            tp_price = price + (price - sl_price) * rr

        tp_pct = abs(tp_price - price) / price * 100

    else:  # short
        # SL — above OB high (invalidation)
        if active_ob:
            sl_price = active_ob.high * (1 + BUFFER)
        elif active_fvg:
            sl_price = active_fvg.top * (1 + BUFFER)
        elif ltf_ms.last_swing_high:
            sl_price = ltf_ms.last_swing_high.price * (1 + BUFFER)
        else:
            sl_price = price * (1 + max(atr / price, 0.002)) if atr > 0 else price * 1.005

        sl_pct = abs(sl_price - price) / price * 100

        # TP — at nearest buy-side liquidity below price
        tp_candidates = [p.price for p in buy_pools if p.price < price * 0.999]
        if tp_candidates:
            tp_price = max(tp_candidates)
        elif ltf_ms.last_swing_low:
            tp_price = ltf_ms.last_swing_low.price
        else:
            rr = 2.2 if confidence >= 80 else 1.8
            tp_price = price - (sl_price - price) * rr

        tp_pct = abs(price - tp_price) / price * 100

    # Cap SL at configured maximum
    sl_pct = min(sl_pct, config.MAX_STOP_LOSS_PCT * 100)

    # Enforce minimum R:R of 1.5
    if sl_pct > 0 and tp_pct < sl_pct * 1.5:
        tp_pct = sl_pct * 1.8

    return sl_pct, tp_pct


# ── Helpers ──────────────────────────────────────────────────────────────────

def _resample_to_15m(candles_5m: pd.DataFrame) -> pd.DataFrame:
    """Resample 5m OHLCV data into 15m candles using pandas resampling."""
    if len(candles_5m) < 3:
        return candles_5m
    try:
        resampled = candles_5m.resample("15min").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna()
        return resampled if len(resampled) >= 5 else candles_5m
    except Exception:
        return candles_5m


def _count_smc_score(active_ob, active_fvg, sweep, zone_info, htf_bias, bias, rsi: float = 50.0) -> int:
    """
    Return a 0-7 score analogous to the old confluence score, for display purposes.
    """
    score = 0
    if htf_bias == bias:                          score += 1
    if active_ob:                                 score += 2
    if active_fvg:                                score += 2
    if sweep:                                     score += 1
    if zone_info.zone in ("discount", "premium"): score += 1
    if rsi <= 30 or rsi >= 70:                    score = max(score, 2)  # RSI extreme = at least 2/7
    return min(score, 7)


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
