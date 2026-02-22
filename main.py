"""
SMC Scalping Bot — Main Entry Point
=====================================
Architecture:
  DataFeed      → streams OHLCV, order book, trades from MEXC Futures
  Indicators    → RSI, EMA, BB, MACD, ATR, VWAP, Stochastic (used by SMC engine)
  Microstructure→ OBI, CVD, spread
  SMC Engine    → Smart Money Concepts analysis:
                  market structure → order blocks → FVGs →
                  liquidity sweeps → premium/discount → killzone
  Executor      → places/monitors orders (paper or live)
  Portfolio     → P&L tracking, SQLite persistence
  RiskManager   → enforces all circuit breakers

Usage:
  python main.py

Configure via .env (copy from .env.example).
"""
import asyncio
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

import config
import core.indicators as indicators
import core.microstructure as microstructure
import core.setup_wizard as setup_wizard
import core.display as display
from core.data_feed import DataFeed
from core.executor import Executor
from core.external import ExternalSignals
from core.portfolio import Portfolio
from core.risk_manager import RiskManager
from core.smc import smc_engine

# ── Logging setup ─────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
logger.add(display.loguru_sink, level=config.LOG_LEVEL, format="{message}", colorize=False)
logger.add(
    config.LOG_FILE, level="DEBUG", rotation="10 MB", retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}",
)


class ScalpingBot:
    def __init__(self) -> None:
        self.portfolio = Portfolio()
        self.risk_mgr  = RiskManager()
        self.data_feed = DataFeed()
        self.executor  = Executor(self.portfolio, self.risk_mgr)
        self.external  = ExternalSignals()
        self._running  = True

        # Per-symbol cooldown: evaluate once per closed candle
        self._last_eval: dict[str, str] = {}   # symbol → closed candle timestamp str

    # ── Startup ───────────────────────────────────────────────────────────────

    async def run(self) -> None:
        mode = "PAPER TRADING" if config.IS_PAPER else "LIVE TRADING"
        logger.info("=" * 60)
        logger.info(f"  SMC Scalping Bot starting — {mode}")
        logger.info(f"  Strategy:       Smart Money Concepts (SMC)")
        logger.info(f"  Exchange:       MEXC Futures")
        logger.info(f"  Pairs:          {', '.join(config.TRADING_PAIRS)}")
        logger.info(f"  Capital:        ${config.STARTING_CAPITAL} USDT")
        logger.info(f"  Leverage:       {config.LEVERAGE}x")
        logger.info(f"  Profit target:  ${config.PROFIT_TARGET_USDT} USDT")
        logger.info(f"  Min confidence: {config.MIN_AI_CONFIDENCE}%")
        logger.info(f"  SMC modules:    structure → OB → FVG → liquidity → zone → killzone")
        logger.info(f"  MEXC taker fee: {config.TAKER_FEE * 100:.3f}%")
        logger.info("=" * 60)

        await self.external.start()

        await asyncio.gather(
            self.data_feed.start(),
            self.external.poll_loop(),
            self._analysis_loop(),
            self._status_loop(),
            display.run_live(
                self.data_feed.all_states,
                lambda: self.portfolio,
                lambda: self._running,
            ),
        )

    # ── Main analysis loop ────────────────────────────────────────────────────

    async def _analysis_loop(self) -> None:
        """
        Polls each symbol after every candle close.
        Pipeline: indicators → microstructure → SMC engine → execution.
        """
        logger.info("Waiting for data feed to seed candles...")
        await asyncio.sleep(10)

        while self._running:
            await asyncio.sleep(3)

            if self.portfolio.halted:
                continue

            if self.portfolio.state().get("target_reached"):
                logger.success(
                    f"PROFIT TARGET ${config.PROFIT_TARGET_USDT} REACHED! "
                    f"Total P&L = ${self.portfolio.total_pnl:.2f} USDT"
                )
                self._print_summary()
                break

            # Check exits on all open positions first
            for state in self.data_feed.all_states():
                if state.ready and state.last_price > 0:
                    await self.executor.check_exits(state.symbol, state.last_price)

            # Evaluate each symbol for new entries
            for state in self.data_feed.all_states():
                if not state.ready or state.candles.empty:
                    continue
                if len(state.candles) < 2:
                    continue

                # Key off the last CLOSED candle (index[-2]).
                # index[-1] is the currently forming candle (real-time tick updates).
                # index[-2] is the fully-closed candle with complete OHLCV data.
                closed_ts = str(state.candles.index[-2])
                if self._last_eval.get(state.symbol) == closed_ts:
                    continue

                await self._evaluate_symbol(state)
                self._last_eval[state.symbol] = closed_ts

    async def _evaluate_symbol(self, state) -> None:
        """Full SMC evaluation pipeline for one symbol."""
        symbol = state.symbol
        try:
            await self._evaluate_symbol_inner(state)
        except Exception:
            import traceback
            logger.error(
                f"{symbol} | _evaluate_symbol crashed — full traceback:\n"
                + traceback.format_exc()
            )

    async def _evaluate_symbol_inner(self, state) -> None:
        """Inner evaluation — exceptions caught by _evaluate_symbol wrapper."""
        symbol = state.symbol

        # 1. Standard indicators (ATR, RSI, EMA, BB etc — used by SMC engine)
        ind = indicators.calculate(state.candles)
        if ind is None:
            return

        # 2. Microstructure (OBI, CVD, spread)
        obi        = microstructure.calculate_obi(state.orderbook)
        spread_bps = microstructure.calculate_spread_bps(state.orderbook)
        cvd_data   = microstructure.calculate_cvd(state.recent_trades)
        micro = {"obi": obi, "spread_bps": spread_bps, **cvd_data}

        # 3. SMC analysis — replaces old confluence + decision engine
        port_state = self.portfolio.state()
        decision = await smc_engine.analyze(
            symbol=symbol,
            candles_5m=state.candles,
            indicators=ind,
            micro=micro,
            portfolio_state=port_state,
            fear_greed=self.external.get_fear_greed(),
            funding_rate=self.external.get_funding_rate(symbol),
        )

        action     = decision.get("action",     "WAIT")
        confidence = decision.get("confidence", 0.0)
        smc_score  = decision.get("confluence_score", 0)
        smc_det    = decision.get("reasoning", {}).get("smc_details", {})

        # 4. Update display cache
        state.cached_rsi       = ind["rsi"]
        state.cached_conf      = smc_score
        state.cached_dir       = (
            "long"    if action == "BUY"  else
            "short"   if action == "SELL" else
            "neutral"
        )
        state.cached_tradeable = action in ("BUY", "SELL")

        logger.debug(
            f"{symbol} | price={ind['price']:.4f} | RSI={ind['rsi']:.1f} | "
            f"SMC={smc_score}/7 conf={confidence:.0f}% | {action} | "
            f"htf={smc_det.get('htf_bias','?')} ltf={smc_det.get('ltf_bias','?')} | "
            f"ob={'✓' if smc_det.get('order_block') else '✗'} "
            f"fvg={'✓' if smc_det.get('fvg') else '✗'} "
            f"sweep={'✓' if smc_det.get('sweep') else '✗'} "
            f"zone={smc_det.get('zone','?')} "
            f"session={smc_det.get('session','?')}"
        )

        if action == "WAIT":
            wait_reason = decision.get("wait_reason", "no SMC setup")
            logger.info(f"{symbol} | SMC → WAIT (conf={confidence:.0f}%) | {wait_reason}")
            return

        # 5. Risk check before executing
        allowed, reason = self.risk_mgr.check(port_state, state.cached_dir)
        if not allowed:
            logger.info(f"{symbol} RISK BLOCK: {reason}")
            return

        # 6. SL / TP levels from SMC engine
        entry_price = ind["price"]
        atr         = ind["atr"]

        sl_pct = decision.get("stop_loss_pct",  config.ATR_STOP_MULTIPLIER * (atr / entry_price) * 100)
        tp_pct = decision.get("take_profit_pct", sl_pct * 1.8)
        sl_pct = min(sl_pct, config.MAX_STOP_LOSS_PCT * 100)

        if action == "BUY":
            stop_loss   = entry_price * (1 - sl_pct / 100)
            take_profit = entry_price * (1 + tp_pct / 100)
        else:   # SELL
            stop_loss   = entry_price * (1 + sl_pct / 100)
            take_profit = entry_price * (1 - tp_pct / 100)

        direction = "long" if action == "BUY" else "short"

        # Log the setup before entering
        reasoning  = decision.get("reasoning", {})
        key_factors = reasoning.get(
            "bullish_factors" if action == "BUY" else "bearish_factors", []
        )
        logger.info(
            f"{symbol} | SMC → {action} conf={confidence:.0f}% | "
            f"SL={stop_loss:.4f} TP={take_profit:.4f} "
            f"(1:{tp_pct/sl_pct:.1f} R:R) | "
            f"htf={smc_det.get('htf_bias','?')} "
            f"zone={smc_det.get('zone','?')} "
            f"session={smc_det.get('session','?')} | "
            f"factors={key_factors[:2]}"
        )

        # 7. Execute
        pos_id = await self.executor.enter(
            symbol=symbol,
            direction=direction,
            current_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            ai_confidence=confidence,
            confluence_score=smc_score,
        )

        if not pos_id:
            logger.warning(f"{symbol} | Executor rejected entry (position sizing or duplicate)")

    # ── Status loop ───────────────────────────────────────────────────────────

    async def _status_loop(self) -> None:
        """Heartbeat every 60 s; full status summary every 2 min."""
        tick = 0
        while self._running:
            await asyncio.sleep(60)
            tick += 1
            self._print_scan_heartbeat()
            if tick % 2 == 0:
                self._print_status()

    def _print_scan_heartbeat(self) -> None:
        """One-line status per pair using cached SMC analysis values."""
        lines = []
        for state in self.data_feed.all_states():
            if not state.ready or state.candles.empty:
                lines.append(f"  {state.symbol}: warming up")
                continue

            price     = state.last_price or 0.0
            rsi       = getattr(state, "cached_rsi",       0.0)
            smc_score = getattr(state, "cached_conf",      0)
            direction = getattr(state, "cached_dir",       "neutral").upper()
            tradeable = getattr(state, "cached_tradeable", False)

            lines.append(
                f"  {state.symbol:<20} price=${price:>12,.4f}  "
                f"RSI={rsi:>5.1f}  SMC={smc_score:>+2}/7 {direction:<7}"
                + ("  ** TRADEABLE **" if tradeable else "")
            )

        logger.info(f"SCANNING — {config.TIMEFRAME} candles | SMC engine active")
        for ln in lines:
            logger.info(ln)

    def _print_status(self) -> None:
        s        = self.portfolio.summary()
        pnl_sign = "+" if self.portfolio.total_pnl >= 0 else ""
        logger.info(
            f"── STATUS ─── "
            f"P&L={pnl_sign}${self.portfolio.total_pnl:.2f} | "
            f"capital=${self.portfolio.capital:.2f} | "
            f"trades={s['total_trades']} (W:{s['wins']} L:{s['losses']}) | "
            f"WR={s['win_rate']:.1%} | "
            f"fees=${self.portfolio.total_fees_paid:.4f} | "
            f"target={s['target_progress_pct']:.1f}%"
        )

    def _print_summary(self) -> None:
        s = self.portfolio.summary()
        logger.info("=" * 60)
        logger.info("  FINAL SUMMARY")
        logger.info(f"  Total trades:    {s['total_trades']}")
        logger.info(f"  Win rate:        {s['win_rate']:.1%}")
        logger.info(f"  Total net P&L:   ${s['total_pnl']:.4f} USDT")
        logger.info(f"  Total fees paid: ${s['total_fees_paid']:.4f} USDT")
        logger.info(f"  Final capital:   ${s['current_capital']:.4f} USDT")
        logger.info(f"  Avg win:         ${s['avg_win']:.4f} USDT")
        logger.info(f"  Avg loss:        ${s['avg_loss']:.4f} USDT")
        logger.info(f"  Profit factor:   {s['profit_factor']:.2f}")
        logger.info("=" * 60)

    def stop(self) -> None:
        logger.info("Stopping bot...")
        self._running = False


# ── Entry point ───────────────────────────────────────────────────────────────

async def _main() -> None:
    settings = await setup_wizard.run()
    for key, value in settings.items():
        setattr(config, key, value)

    bot = ScalpingBot()
    loop = asyncio.get_event_loop()

    def _quiet_exc_handler(loop, context):
        exc = context.get("exception")
        if isinstance(exc, (asyncio.CancelledError, RuntimeError)):
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(_quiet_exc_handler)

    def _shutdown(sig, frame):
        logger.info(f"Signal {sig} received — shutting down")
        bot.stop()
        loop.stop()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        await bot.run()
    finally:
        await bot.data_feed.close()
        await bot.external.close()
        bot._print_summary()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except (KeyboardInterrupt, RuntimeError):
        pass
