"""
Risk management — evaluates account state against configured limits.
Acts as a final gate before any trade is allowed to execute.
"""
from __future__ import annotations

from loguru import logger

import config


class RiskManager:
    """
    Checks all risk rules before allowing a trade.
    All state is tracked externally (in Portfolio); this class just evaluates rules.
    """

    def check(self, portfolio_state: dict, direction: str) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        If allowed=False, the trade must not be placed.
        """
        capital         = portfolio_state.get("capital", config.STARTING_CAPITAL)
        daily_pnl       = portfolio_state.get("daily_pnl", 0.0)
        open_positions  = portfolio_state.get("open_positions", 0)
        daily_trades    = portfolio_state.get("daily_trades", 0)
        consec_losses   = portfolio_state.get("consecutive_losses", 0)
        halted          = portfolio_state.get("halted", False)

        if halted:
            return False, "Bot is halted — manual review required"

        # Daily loss limit
        daily_loss_limit = capital * (config.DAILY_LOSS_LIMIT_PCT / 100)
        if daily_pnl <= -daily_loss_limit:
            return False, (
                f"Daily loss limit hit: {daily_pnl:.2f} <= -{daily_loss_limit:.2f} USDT"
            )

        # Max open positions
        if open_positions >= config.MAX_OPEN_POSITIONS:
            return False, f"Max open positions ({config.MAX_OPEN_POSITIONS}) already reached"

        # Max daily trades
        if daily_trades >= config.MAX_DAILY_TRADES:
            return False, f"Max daily trades ({config.MAX_DAILY_TRADES}) reached"

        # Consecutive losses
        if consec_losses >= config.MAX_CONSECUTIVE_LOSSES:
            return False, (
                f"Consecutive losses ({consec_losses}) hit limit ({config.MAX_CONSECUTIVE_LOSSES})"
            )

        # Account balance floor (stop if account drops 30%)
        min_balance = config.STARTING_CAPITAL * 0.70
        if capital < min_balance:
            return False, (
                f"Account below minimum balance: {capital:.2f} < {min_balance:.2f} USDT"
            )

        return True, "OK"

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: float,
        risk_pct: float | None = None,
    ) -> dict:
        """
        ATR-based position sizing.

        Returns dict with:
            risk_usd:       dollar amount being risked
            position_usd:   notional position value (with leverage)
            quantity:       position size in base asset units
            leverage_used:  actual leverage implied
        """
        risk_pct   = risk_pct or config.RISK_PER_TRADE_PCT
        risk_usd   = capital * (risk_pct / 100)

        price_risk = abs(entry_price - stop_loss_price)
        if price_risk == 0 or entry_price == 0:
            return {"risk_usd": 0, "position_usd": 0, "quantity": 0, "leverage_used": 0}

        # Base quantity from risk
        quantity_from_risk = risk_usd / price_risk

        # Apply leverage cap
        position_usd   = quantity_from_risk * entry_price
        max_notional   = capital * config.LEVERAGE
        if position_usd > max_notional:
            position_usd      = max_notional
            quantity_from_risk = max_notional / entry_price

        leverage_used = position_usd / capital if capital > 0 else 1.0

        logger.debug(
            f"Position sizing: risk_usd={risk_usd:.2f} pos={position_usd:.2f} "
            f"qty={quantity_from_risk:.6f} lev={leverage_used:.1f}x"
        )

        return {
            "risk_usd":      round(risk_usd, 4),
            "position_usd":  round(position_usd, 4),
            "quantity":      quantity_from_risk,
            "leverage_used": round(leverage_used, 2),
        }
