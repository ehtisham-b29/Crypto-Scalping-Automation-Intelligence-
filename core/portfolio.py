"""
Portfolio & P&L tracker.
Tracks capital, open positions, trade history, fees paid, and progress
toward the $500 profit target. Persists every trade to SQLite.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

import config


@dataclass
class Position:
    id: str
    symbol: str
    direction: str          # "long" | "short"
    entry_price: float
    quantity: float
    position_usd: float
    stop_loss: float
    take_profit: float
    entry_time: str
    entry_fee: float = 0.0


@dataclass
class ClosedTrade:
    id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    position_usd: float
    gross_pnl: float
    fees: float
    net_pnl: float
    outcome: str            # "win" | "loss" | "breakeven"
    entry_time: str
    exit_time: str
    hold_seconds: float
    ai_confidence: float
    confluence_score: int
    exit_reason: str        # "take_profit" | "stop_loss" | "manual"


class Portfolio:
    """Tracks all trading state and writes history to SQLite."""

    def __init__(self) -> None:
        self.capital: float = config.STARTING_CAPITAL
        self.total_pnl: float = 0.0
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.daily_loss: float = 0.0
        self.consecutive_losses: int = 0
        self.total_fees_paid: float = 0.0
        self.open_positions: dict[str, Position] = {}    # id → Position
        self.trade_history: list[ClosedTrade] = []
        self.halted: bool = False
        self._session_date: str = _today()
        self._init_db()

    # ── State snapshot for AI / risk manager ──────────────────────────────────

    def state(self) -> dict:
        return {
            "capital":            self.capital,
            "total_pnl":          self.total_pnl,
            "daily_pnl":          self.daily_pnl,
            "daily_loss":         self.daily_loss,
            "daily_trades":       self.daily_trades,
            "open_positions":     len(self.open_positions),
            "consecutive_losses": self.consecutive_losses,
            "halted":             self.halted,
            "total_fees_paid":    self.total_fees_paid,
            "max_risk_usd":       self.capital * (config.RISK_PER_TRADE_PCT / 100),
            "profit_remaining":   max(0.0, config.PROFIT_TARGET_USDT - self.total_pnl),
            "target_reached":     self.total_pnl >= config.PROFIT_TARGET_USDT,
        }

    # ── Position lifecycle ────────────────────────────────────────────────────

    def open_position(self, position: Position) -> None:
        self._check_date_rollover()
        self.open_positions[position.id] = position
        self.daily_trades += 1
        self.capital -= position.entry_fee
        self.total_fees_paid += position.entry_fee
        logger.info(
            f"[OPEN] {position.symbol} {position.direction.upper()} "
            f"@ {position.entry_price:.4f} | qty={position.quantity:.6f} "
            f"| notional=${position.position_usd:.2f} | SL={position.stop_loss:.4f} "
            f"| TP={position.take_profit:.4f} | fee=${position.entry_fee:.4f}"
        )

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str,
        ai_confidence: float = 0.0,
        confluence_score: int = 0,
    ) -> Optional[ClosedTrade]:
        pos = self.open_positions.pop(position_id, None)
        if pos is None:
            logger.warning(f"Position {position_id} not found")
            return None

        exit_time  = datetime.now(timezone.utc).isoformat()
        entry_dt   = datetime.fromisoformat(pos.entry_time)
        exit_dt    = datetime.fromisoformat(exit_time)
        hold_secs  = (exit_dt - entry_dt).total_seconds()

        # P&L calculation
        if pos.direction == "long":
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity

        exit_fee  = pos.position_usd * config.TAKER_FEE
        total_fee = pos.entry_fee + exit_fee
        net_pnl   = gross_pnl - exit_fee   # entry_fee already deducted on open

        if net_pnl > 0.0:
            outcome = "win"
            self.consecutive_losses = 0
        elif net_pnl < 0.0:
            outcome = "loss"
            self.consecutive_losses += 1
        else:
            outcome = "breakeven"

        self.capital         += net_pnl
        self.total_pnl       += net_pnl
        self.daily_pnl       += net_pnl
        self.total_fees_paid += exit_fee

        if net_pnl < 0:
            self.daily_loss += abs(net_pnl)

        # Check circuit breakers after close
        self._check_circuit_breakers()

        trade = ClosedTrade(
            id=position_id,
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            position_usd=pos.position_usd,
            gross_pnl=round(gross_pnl, 6),
            fees=round(total_fee, 6),
            net_pnl=round(net_pnl, 6),
            outcome=outcome,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            hold_seconds=hold_secs,
            ai_confidence=ai_confidence,
            confluence_score=confluence_score,
            exit_reason=exit_reason,
        )
        self.trade_history.append(trade)
        self._persist_trade(trade)

        pnl_emoji = "+" if net_pnl >= 0 else ""
        logger.info(
            f"[CLOSE] {pos.symbol} {pos.direction.upper()} | {exit_reason.upper()} | "
            f"net_pnl={pnl_emoji}{net_pnl:.4f} USDT | fees={total_fee:.4f} | "
            f"outcome={outcome.upper()} | hold={hold_secs:.0f}s | "
            f"total_pnl={self.total_pnl:.4f} / {config.PROFIT_TARGET_USDT}"
        )

        if self.total_pnl >= config.PROFIT_TARGET_USDT:
            logger.success(
                f"TARGET REACHED! Total P&L = ${self.total_pnl:.2f} USDT "
                f"(goal was ${config.PROFIT_TARGET_USDT})"
            )

        return trade

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        wins   = [t for t in self.trade_history if t.outcome == "win"]
        losses = [t for t in self.trade_history if t.outcome == "loss"]
        total  = len(self.trade_history)

        return {
            "total_trades":       total,
            "wins":               len(wins),
            "losses":             len(losses),
            "win_rate":           len(wins) / total if total else 0.0,
            "total_pnl":          self.total_pnl,
            "total_fees_paid":    self.total_fees_paid,
            "current_capital":    self.capital,
            "avg_win":            sum(t.net_pnl for t in wins)  / len(wins)  if wins   else 0.0,
            "avg_loss":           sum(t.net_pnl for t in losses)/ len(losses) if losses else 0.0,
            "profit_factor":      (
                sum(t.net_pnl for t in wins) / abs(sum(t.net_pnl for t in losses))
                if losses and sum(t.net_pnl for t in losses) != 0 else float("inf")
            ),
            "consecutive_losses": self.consecutive_losses,
            "target_progress_pct": (self.total_pnl / config.PROFIT_TARGET_USDT) * 100,
        }

    # ── Circuit breakers ──────────────────────────────────────────────────────

    def _check_circuit_breakers(self) -> None:
        capital         = self.capital
        daily_loss_limit = config.STARTING_CAPITAL * (config.DAILY_LOSS_LIMIT_PCT / 100)

        if self.consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
            self.halted = True
            logger.warning(
                f"CIRCUIT BREAKER: {self.consecutive_losses} consecutive losses — bot halted"
            )
        elif self.daily_loss >= daily_loss_limit:
            self.halted = True
            logger.warning(
                f"CIRCUIT BREAKER: Daily loss ${self.daily_loss:.2f} >= limit ${daily_loss_limit:.2f}"
            )
        elif capital < config.STARTING_CAPITAL * 0.70:
            self.halted = True
            logger.warning(
                f"CIRCUIT BREAKER: Capital ${capital:.2f} below 70% of start"
            )

    def reset_daily(self) -> None:
        """Call at the start of each new trading day."""
        logger.info(f"Daily reset — pnl={self.daily_pnl:.4f} trades={self.daily_trades}")
        self.daily_pnl    = 0.0
        self.daily_loss   = 0.0
        self.daily_trades = 0
        self.halted       = False    # allow fresh start each day
        self._session_date = _today()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(config.DB_PATH)
        con.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                position_usd REAL,
                gross_pnl REAL,
                fees REAL,
                net_pnl REAL,
                outcome TEXT,
                entry_time TEXT,
                exit_time TEXT,
                hold_seconds REAL,
                ai_confidence REAL,
                confluence_score INTEGER,
                exit_reason TEXT
            )
        """)
        con.commit()
        con.close()

    def _persist_trade(self, trade: ClosedTrade) -> None:
        try:
            con = sqlite3.connect(config.DB_PATH)
            con.execute("""
                INSERT OR REPLACE INTO trades VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                trade.id, trade.symbol, trade.direction,
                trade.entry_price, trade.exit_price, trade.quantity,
                trade.position_usd, trade.gross_pnl, trade.fees, trade.net_pnl,
                trade.outcome, trade.entry_time, trade.exit_time,
                trade.hold_seconds, trade.ai_confidence, trade.confluence_score,
                trade.exit_reason,
            ))
            con.commit()
            con.close()
        except Exception as e:
            logger.error(f"Failed to persist trade {trade.id}: {e}")

    def _check_date_rollover(self) -> None:
        if _today() != self._session_date:
            self.reset_daily()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
