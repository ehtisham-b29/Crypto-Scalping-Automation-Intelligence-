"""
Order executor — supports two modes:
  paper: simulates all orders locally with realistic fee accounting
  live:  places real orders on MEXC Futures via ccxt REST

In paper mode, fills are assumed at the current market price immediately.
Stop loss and take profit are monitored on every price update.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import ccxt
from loguru import logger

import config
from core.portfolio import Portfolio, Position
from core.risk_manager import RiskManager


class Executor:
    def __init__(self, portfolio: Portfolio, risk_manager: RiskManager) -> None:
        self.portfolio    = portfolio
        self.risk_manager = risk_manager
        self._exchange: Optional[ccxt.mexc] = None

        if not config.IS_PAPER:
            self._exchange = ccxt.mexc({
                "apiKey": config.MEXC_API_KEY,
                "secret": config.MEXC_SECRET,
                "options": {"defaultType": "swap"},
            })
            logger.info("Executor: LIVE mode — connected to MEXC Futures")
        else:
            logger.info("Executor: PAPER mode — all trades are simulated")

    # ── Entry ─────────────────────────────────────────────────────────────────

    async def enter(
        self,
        symbol: str,
        direction: str,         # "long" | "short"
        current_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        ai_confidence: float = 0.0,
        confluence_score: int = 0,
    ) -> Optional[str]:
        """
        Open a new position. Returns position_id or None if rejected.
        """
        port_state = self.portfolio.state()

        # Risk gate
        allowed, reason = self.risk_manager.check(port_state, direction)
        if not allowed:
            logger.warning(f"[BLOCKED] {symbol} {direction}: {reason}")
            return None

        # Position sizing
        sizing = self.risk_manager.calculate_position_size(
            capital=self.portfolio.capital,
            entry_price=current_price,
            stop_loss_price=stop_loss_price,
        )
        if sizing["quantity"] <= 0:
            logger.warning(f"[BLOCKED] {symbol}: position size = 0")
            return None

        entry_fee = sizing["position_usd"] * config.TAKER_FEE

        position = Position(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            quantity=sizing["quantity"],
            position_usd=sizing["position_usd"],
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            entry_time=datetime.now(timezone.utc).isoformat(),
            entry_fee=entry_fee,
        )

        if config.IS_PAPER:
            self.portfolio.open_position(position)
            return position.id

        # Live order placement
        try:
            side = "buy" if direction == "long" else "sell"
            order = self._exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=sizing["quantity"],
            )
            logger.info(f"Live order placed: {order.get('id')} {symbol} {side}")
            self.portfolio.open_position(position)
            return position.id
        except ccxt.BaseError as e:
            logger.error(f"Order placement failed ({symbol}): {e}")
            return None

    # ── Exit ──────────────────────────────────────────────────────────────────

    async def exit_position(
        self,
        position_id: str,
        current_price: float,
        exit_reason: str,
        ai_confidence: float = 0.0,
        confluence_score: int = 0,
    ) -> None:
        """Close an open position."""
        pos = self.portfolio.open_positions.get(position_id)
        if pos is None:
            return

        if not config.IS_PAPER:
            try:
                side = "sell" if pos.direction == "long" else "buy"
                self._exchange.create_order(
                    symbol=pos.symbol,
                    type="market",
                    side=side,
                    amount=pos.quantity,
                )
            except ccxt.BaseError as e:
                logger.error(f"Exit order failed ({pos.symbol}): {e}")
                return

        self.portfolio.close_position(
            position_id=position_id,
            exit_price=current_price,
            exit_reason=exit_reason,
            ai_confidence=ai_confidence,
            confluence_score=confluence_score,
        )

    # ── Stop loss / Take profit monitor ───────────────────────────────────────

    async def check_exits(self, symbol: str, current_price: float) -> None:
        """
        Called every time a new price arrives for a symbol.
        Checks open positions for that symbol against their SL/TP levels.
        """
        to_close = []
        for pos_id, pos in self.portfolio.open_positions.items():
            if pos.symbol != symbol:
                continue

            if pos.direction == "long":
                if current_price <= pos.stop_loss:
                    to_close.append((pos_id, "stop_loss"))
                elif current_price >= pos.take_profit:
                    to_close.append((pos_id, "take_profit"))
            else:  # short
                if current_price >= pos.stop_loss:
                    to_close.append((pos_id, "stop_loss"))
                elif current_price <= pos.take_profit:
                    to_close.append((pos_id, "take_profit"))

        for pos_id, reason in to_close:
            await self.exit_position(pos_id, current_price, reason)
