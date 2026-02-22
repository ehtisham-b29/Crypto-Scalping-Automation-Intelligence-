"""
SMC Killzone (Session Timing) Filter
======================================
Institutional order flow is heaviest at session opens.
Trades taken during killzones have statistically higher win rates
because liquidity, volatility, and trend momentum all peak.

Killzone schedule (UTC):
  London Open  : 07:00 – 10:00  → +15 confidence (highest priority)
  New York Open: 12:00 – 15:00  → +15 confidence (highest priority)
  London Close : 15:00 – 16:00  → +5  confidence  (medium)
  Asia Open    : 23:00 – 02:00  → +0  confidence  (low, volatile)
  Dead Zone    : 16:00 – 23:00  → -15 confidence  (avoid — thin market)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal


@dataclass
class SessionInfo:
    session: Literal[
        "london_open", "new_york_open", "london_close",
        "asia_open", "dead_zone"
    ]
    in_killzone: bool
    confidence_adjustment: int   # points added/subtracted from SMC confidence
    minutes_into_session: int    # elapsed minutes since session open


def current_session(dt: datetime | None = None) -> SessionInfo:
    """
    Determine current trading session and killzone status.

    Args:
        dt : UTC datetime to evaluate (defaults to now)

    Returns:
        SessionInfo with session name, killzone flag, and confidence adjustment
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    h   = dt.hour
    m   = dt.minute
    tot = h * 60 + m    # total minutes since midnight UTC

    # London Open  07:00 – 10:00
    if 7 * 60 <= tot < 10 * 60:
        return SessionInfo(
            session="london_open", in_killzone=True,
            confidence_adjustment=+15,
            minutes_into_session=tot - 7 * 60,
        )

    # New York Open  12:00 – 15:00
    if 12 * 60 <= tot < 15 * 60:
        return SessionInfo(
            session="new_york_open", in_killzone=True,
            confidence_adjustment=+15,
            minutes_into_session=tot - 12 * 60,
        )

    # London Close  15:00 – 16:00
    if 15 * 60 <= tot < 16 * 60:
        return SessionInfo(
            session="london_close", in_killzone=True,
            confidence_adjustment=+5,
            minutes_into_session=tot - 15 * 60,
        )

    # Asia Open  23:00 – 02:00  (wraps midnight)
    if tot >= 23 * 60 or tot < 2 * 60:
        mins_in = (tot - 23 * 60) if tot >= 23 * 60 else (tot + 60)
        return SessionInfo(
            session="asia_open", in_killzone=True,
            confidence_adjustment=0,
            minutes_into_session=max(0, mins_in),
        )

    # Dead zone  (everything else)
    return SessionInfo(
        session="dead_zone", in_killzone=False,
        confidence_adjustment=-15,
        minutes_into_session=0,
    )
