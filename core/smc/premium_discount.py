"""
SMC Premium / Discount Zone Classification
===========================================
Applies Fibonacci retracement to the most recent swing range to classify
whether price is cheap (discount) or expensive (premium).

SMC rule:
  Long  → only in DISCOUNT  (price below 50% of range, ideally below 38.2%)
  Short → only in PREMIUM   (price above 50% of range, ideally above 61.8%)

OTE (Optimal Trade Entry):
  Bullish OTE : 61.8% – 78.6% retracement back INTO discount (fib 0.214 – 0.382)
  Bearish OTE : 61.8% – 78.6% retracement back INTO premium  (fib 0.618 – 0.786)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class PriceZone:
    zone: Literal["premium", "discount", "equilibrium"]
    fib_level: float    # 0.0 = at swing low, 1.0 = at swing high
    in_ote: bool        # True if price is in the Optimal Trade Entry band
    swing_high: float
    swing_low: float
    equilibrium: float  # 0.50 fib price


def classify(
    current_price: float,
    swing_high: float,
    swing_low: float,
) -> PriceZone:
    """
    Classify current price into premium / discount / equilibrium.

    Args:
        current_price : latest traded price
        swing_high    : most recent significant swing high
        swing_low     : most recent significant swing low

    Returns:
        PriceZone with zone, fibonacci level, and OTE flag
    """
    swing_range = swing_high - swing_low
    equilibrium = swing_low + swing_range * 0.50

    if swing_range <= 0 or swing_high <= 0:
        return PriceZone(
            zone="equilibrium", fib_level=0.50, in_ote=False,
            swing_high=swing_high, swing_low=swing_low, equilibrium=equilibrium,
        )

    fib = (current_price - swing_low) / swing_range
    fib = max(0.0, min(1.0, fib))

    if fib < 0.382:
        zone: Literal["premium", "discount", "equilibrium"] = "discount"
    elif fib > 0.618:
        zone = "premium"
    else:
        zone = "equilibrium"

    # OTE bands (from ICT / SMC methodology)
    # Discount OTE : price between 21.4% – 38.2% of range (deep pullback into discount)
    # Premium OTE  : price between 61.8% – 78.6% of range (deep pullback into premium)
    in_ote = (0.214 <= fib <= 0.382) or (0.618 <= fib <= 0.786)

    return PriceZone(
        zone=zone,
        fib_level=round(fib, 4),
        in_ote=in_ote,
        swing_high=swing_high,
        swing_low=swing_low,
        equilibrium=equilibrium,
    )
