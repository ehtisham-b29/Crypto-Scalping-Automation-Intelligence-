"""
AI Decision Engine -- sends aggregated market data to OpenAI and
receives a structured trading decision via function calling (guaranteed JSON).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError
from loguru import logger

import config

# -- OpenAI async client (lazy init) ------------------------------------------
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")
        _client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    return _client


# -- System prompt -------------------------------------------------------------
SYSTEM_PROMPT = f"""You are an expert AI scalping trader operating on MEXC Futures (perpetual swaps).
Your role is to analyze multi-indicator market data and make precise, disciplined trading decisions.

TRADING RULES (non-negotiable):
1. Never enter when ATR is elevated more than 2x its average (avoid news events)
2. Never enter when bid-ask spread exceeds {config.SPREAD_MAX_BPS} basis points
3. Minimum Risk:Reward ratio is 1:1.5 (e.g., risk 0.20%, target 0.30%)
4. Stop loss must never exceed {config.MAX_STOP_LOSS_PCT * 100:.2f}% from entry
5. Account for round-trip fee of 0.04% (0.020% entry + 0.020% exit on MEXC Futures)
6. Minimum net profit target after fees: {config.MIN_PROFIT_TARGET_PCT * 100:.2f}%
7. If signals are unclear or confidence is below {config.MIN_AI_CONFIDENCE}, output action=WAIT
8. Never force a trade -- a missed opportunity costs nothing; a bad trade costs capital
9. Consider funding rate: avoid new longs when funding > 0.03%, avoid new shorts when < -0.01%
10. Check if daily loss limit or consecutive loss limit is already hit before entering

GOAL: Accumulate ${config.PROFIT_TARGET_USDT:.0f} profit through disciplined small-gain trades.
Starting capital: ${config.STARTING_CAPITAL:.0f} USDT with {config.LEVERAGE}x leverage.

You must call the make_trade_decision function with your decision. No other text."""

# -- Function schema (OpenAI function calling format) -------------------------
TRADE_DECISION_TOOL = {
    "type": "function",
    "function": {
        "name": "make_trade_decision",
        "description": "Output the trading decision as structured data after analyzing all market signals.",
        "parameters": {
            "type": "object",
            "required": ["action", "confidence", "confluence_score", "reasoning", "risk_assessment"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["BUY", "SELL", "WAIT"],
                    "description": "BUY=long, SELL=short, WAIT=no trade",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence 0-100. Below threshold -> WAIT regardless.",
                },
                "confluence_score": {
                    "type": "integer",
                    "description": "How many of the 7 signals align (0-7)",
                },
                "entry_price": {
                    "type": "number",
                    "description": "Suggested entry price",
                },
                "stop_loss": {
                    "type": "number",
                    "description": "Stop loss price level",
                },
                "take_profit": {
                    "type": "number",
                    "description": "Take profit price level",
                },
                "stop_loss_pct": {
                    "type": "number",
                    "description": "Stop loss as % from entry (e.g. 0.20 means 0.20%)",
                },
                "take_profit_pct": {
                    "type": "number",
                    "description": "Take profit as % from entry (e.g. 0.30 means 0.30%)",
                },
                "estimated_net_profit_pct": {
                    "type": "number",
                    "description": "Expected net profit % after deducting 0.04% round-trip fee",
                },
                "reasoning": {
                    "type": "object",
                    "required": ["bullish_factors", "bearish_factors", "key_risk"],
                    "properties": {
                        "bullish_factors": {"type": "array", "items": {"type": "string"}},
                        "bearish_factors": {"type": "array", "items": {"type": "string"}},
                        "key_risk":        {"type": "string"},
                    },
                },
                "risk_assessment": {
                    "type": "string",
                    "enum": ["LOW", "MEDIUM", "HIGH", "EXTREME"],
                },
                "hold_duration_estimate": {
                    "type": "string",
                    "description": "Expected hold time, e.g. '2-5 minutes'",
                },
                "position_size_pct": {
                    "type": "number",
                    "description": "Suggested % of capital to risk (0.5 to 2.0)",
                },
            },
        },
    },
}


# -- Main decision function ----------------------------------------------------

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
    Build the market context message and call OpenAI.
    Returns the parsed decision dict (guaranteed JSON via function calling).
    """
    prompt = _build_prompt(
        symbol, indicators, micro, confluence,
        portfolio_state, fear_greed, funding_rate, open_interest_change,
    )

    # o-series models (o1, o3, o4-mini) don't support temperature and
    # require max_completion_tokens instead of max_tokens.
    _o_series = config.AI_MODEL.startswith(("o1", "o3", "o4"))
    _extra: dict = {} if _o_series else {"temperature": 0.2}

    try:
        client = _get_client()
        response = await client.chat.completions.create(
            model=config.AI_MODEL,
            max_completion_tokens=config.AI_MAX_COMPLETION_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            tools=[TRADE_DECISION_TOOL],
            tool_choice={"type": "function", "function": {"name": "make_trade_decision"}},
            **_extra,
        )

        # Extract the function call result
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            return _wait_decision("OpenAI returned no function call")

        decision = json.loads(tool_calls[0].function.arguments)

        # Hard confidence gate
        if decision.get("confidence", 0) < config.MIN_AI_CONFIDENCE:
            decision["action"] = "WAIT"
            decision["wait_reason"] = (
                f"Confidence {decision.get('confidence')} < {config.MIN_AI_CONFIDENCE}"
            )

        # Hard risk gate -- EXTREME risk always waits
        if decision.get("risk_assessment") == "EXTREME":
            decision["action"] = "WAIT"
            decision["wait_reason"] = "Risk assessment = EXTREME"

        logger.debug(
            f"AI decision [{symbol}]: {decision.get('action')} "
            f"conf={decision.get('confidence')} risk={decision.get('risk_assessment')}"
        )
        return decision

    except RateLimitError as e:
        logger.error(f"OpenAI rate limit hit: {e}")
    except APIConnectionError as e:
        logger.error(f"OpenAI connection error: {e}")
    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse OpenAI function call JSON: {e}")
    except Exception as e:
        logger.error(f"AI engine unexpected error: {e}")

    return _wait_decision("AI call failed")


# -- Prompt builder ------------------------------------------------------------

def _build_prompt(
    symbol: str,
    ind: dict,
    micro: dict,
    conf: dict,
    port: dict,
    fear_greed: dict | None,
    funding_rate: float,
    oi_change: float,
) -> str:
    ts       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    fg_value = fear_greed.get("value", "N/A") if fear_greed else "N/A"
    fg_label = fear_greed.get("value_classification", "N/A") if fear_greed else "N/A"

    signal_breakdown = "\n".join(
        f"  [{'+' if s['value'] > 0 else ('-' if s['value'] < 0 else '0')}] {s['name']}: {s['detail']}"
        for s in conf.get("signals", [])
    )

    return f"""Analyze this market snapshot and decide: BUY, SELL, or WAIT.

SYMBOL: {symbol}
TIMEFRAME: {config.TIMEFRAME}
TIMESTAMP: {ts}

--- PRICE ---
Current Price:  {ind.get('price', 0):.6f} USDT
VWAP:           {ind.get('vwap', 0):.6f} USDT
VWAP Deviation: {ind.get('vwap_deviation', 0):+.3f}%

--- TECHNICAL INDICATORS ---
RSI({config.RSI_PERIOD}):       {ind.get('rsi', 0):.2f}
EMA{config.EMA_FAST}:           {ind.get('ema_fast', 0):.6f}
EMA{config.EMA_SLOW}:           {ind.get('ema_slow', 0):.6f}
BB Upper:       {ind.get('bb_upper', 0):.6f}
BB Middle:      {ind.get('bb_mid', 0):.6f}
BB Lower:       {ind.get('bb_lower', 0):.6f}
BB Width:       {ind.get('bb_width', 0):.4f}
MACD:           {ind.get('macd', 0):.6f}
MACD Signal:    {ind.get('macd_signal', 0):.6f}
MACD Histogram: {ind.get('macd_hist', 0):.6f}
ATR({config.ATR_PERIOD}):       {ind.get('atr', 0):.6f} (avg={ind.get('atr_avg', 0):.6f}, elevated={ind.get('atr_elevated', False)})
Stoch K/D:      {ind.get('stoch_k', 0):.2f} / {ind.get('stoch_d', 0):.2f}

--- MICROSTRUCTURE ---
OBI:            {micro.get('obi', 0):+.4f}  (buy pressure: +1, sell: -1)
Spread:         {micro.get('spread_bps', 0):.2f} bps  (max allowed: {config.SPREAD_MAX_BPS} bps)
CVD:            {micro.get('cvd', 0):+.4f} ({micro.get('cvd_trend', 'flat')})
Buy Volume:     {micro.get('buy_vol', 0):.4f}
Sell Volume:    {micro.get('sell_vol', 0):.4f}

--- CONFLUENCE SCORING ---
Score:          {conf.get('score', 0)} / 7  (need >= {config.MIN_CONFLUENCE_SCORE} to trade)
Direction bias: {conf.get('direction', 'neutral').upper()}
Signals:
{signal_breakdown}

--- DERIVATIVES ---
Funding Rate:   {funding_rate:+.4f}% per 8h
OI Change (1h): {oi_change:+.2f}%
Volume Ratio:   {ind.get('vol_ratio', 1.0):.2f}x vs 20-period avg

--- MACRO ---
Fear & Greed:   {fg_value} ({fg_label})

--- ACCOUNT STATE ---
Capital:        ${port.get('capital', 0):.2f} USDT
Net P&L today:  ${port.get('daily_pnl', 0):+.2f} USDT
Profit vs goal: ${port.get('total_pnl', 0):.2f} / ${config.PROFIT_TARGET_USDT:.0f} target
Open positions: {port.get('open_positions', 0)} (max {config.MAX_OPEN_POSITIONS})
Trades today:   {port.get('daily_trades', 0)} (max {config.MAX_DAILY_TRADES})
Consec. losses: {port.get('consecutive_losses', 0)} (max {config.MAX_CONSECUTIVE_LOSSES})
Daily loss:     ${port.get('daily_loss', 0):.2f} (limit: {config.DAILY_LOSS_LIMIT_PCT}% = ${port.get('capital', config.STARTING_CAPITAL) * config.DAILY_LOSS_LIMIT_PCT / 100:.2f})
Max risk/trade: ${port.get('max_risk_usd', 0):.2f} USDT

--- FEE REMINDER ---
Round-trip fee: 0.04% (0.020% entry + 0.020% exit on MEXC Futures)
Breakeven move: 0.04%
Min net target: {config.MIN_PROFIT_TARGET_PCT * 100:.2f}% after fees

Analyze all signals holistically. Consider confluence, momentum, risk, and account state.
Call make_trade_decision with your structured decision."""


# -- Fallback ------------------------------------------------------------------

def _wait_decision(reason: str) -> dict:
    return {
        "action": "WAIT",
        "confidence": 0,
        "confluence_score": 0,
        "risk_assessment": "HIGH",
        "reasoning": {
            "bullish_factors": [],
            "bearish_factors": [],
            "key_risk": reason,
        },
        "wait_reason": reason,
    }
