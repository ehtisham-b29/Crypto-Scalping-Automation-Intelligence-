"""
Interactive Setup Wizard -- runs before the bot starts.

Asks the user for capital, profit target, risk preference, and daily loss
tolerance. Fetches live pair prices from MEXC. Derives a complete trading
strategy and returns a settings dict that main.py applies to config.
"""
from __future__ import annotations

import sys

import ccxt.pro as ccxtpro

import config

# -- Pairs shown in price table ------------------------------------------------
_PRICE_PAIRS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
]

_PAIR_LABELS = {
    "BTC/USDT:USDT": "Bitcoin   (BTC)",
    "ETH/USDT:USDT": "Ethereum  (ETH)",
    "SOL/USDT:USDT": "Solana    (SOL)",
    "BNB/USDT:USDT": "BNB Chain (BNB)",
}

# -- Risk profiles -------------------------------------------------------------
_PROFILES: dict[str, dict] = {
    "1": {
        "name":               "Conservative",
        "leverage":           3,
        "risk_per_trade_pct": 1.0,
        "sl_pct":             0.25,
        "rr_ratio":           1.8,
        "daily_loss_pct":     2.0,
        "max_open_positions": 1,
        "min_confluence":     5,
        "min_ai_confidence":  72,
        "desc": "Low leverage (3x), tight stops -- slow and steady capital preservation",
    },
    "2": {
        "name":               "Moderate",
        "leverage":           5,
        "risk_per_trade_pct": 1.5,
        "sl_pct":             0.35,
        "rr_ratio":           1.6,
        "daily_loss_pct":     3.0,
        "max_open_positions": 2,
        "min_confluence":     4,
        "min_ai_confidence":  65,
        "desc": "Balanced risk/reward -- recommended for most users",
    },
    "3": {
        "name":               "Aggressive",
        "leverage":           10,
        "risk_per_trade_pct": 2.0,
        "sl_pct":             0.45,
        "rr_ratio":           1.5,
        "daily_loss_pct":     5.0,
        "max_open_positions": 2,
        "min_confluence":     4,
        "min_ai_confidence":  60,
        "desc": "Higher leverage (10x), larger positions -- faster gains AND bigger drawdowns",
    },
}

# Realistic win-rate assumption used only for projections
_WIN_RATE = 0.55


# -- Print helpers -------------------------------------------------------------

def _sep(char: str = "-", width: int = 62) -> None:
    print(char * width)


def _header(title: str, width: int = 62) -> None:
    _sep("=", width)
    print(f"  {title}")
    _sep("=", width)


def _ask(prompt: str, validator=None, error: str = "Invalid input -- try again.") -> str:
    """Print prompt, read a line, validate, retry on failure."""
    while True:
        try:
            val = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Setup cancelled.")
            sys.exit(0)
        if validator is None or validator(val):
            return val
        print(f"  X  {error}")


# -- Price fetcher -------------------------------------------------------------

async def _fetch_prices() -> dict[str, float]:
    """Fetch last traded price for each display pair from MEXC Futures."""
    prices: dict[str, float] = {}
    exchange = ccxtpro.mexc({"options": {"defaultType": "swap"}})
    try:
        for symbol in _PRICE_PAIRS:
            try:
                ticker = await exchange.fetch_ticker(symbol)
                prices[symbol] = float(ticker.get("last") or ticker.get("close") or 0)
            except Exception:
                prices[symbol] = 0.0
    finally:
        await exchange.close()
    return prices


# -- Strategy calculator -------------------------------------------------------

def _calc(capital: float, profit_target: float, profile: dict, daily_loss_pct: float) -> dict:
    """Return projection numbers based on capital, target, and risk profile."""
    risk_usd     = capital * (profile["risk_per_trade_pct"] / 100)
    avg_win_usd  = risk_usd * profile["rr_ratio"]
    avg_loss_usd = risk_usd
    tp_pct       = profile["sl_pct"] * profile["rr_ratio"]
    ev_trade     = (_WIN_RATE * avg_win_usd) - ((1 - _WIN_RATE) * avg_loss_usd)

    trades_needed = max(1, int(profit_target / ev_trade)) if ev_trade > 0 else 99999
    days_est      = max(1, trades_needed // 20)

    return {
        "risk_usd":       round(risk_usd, 2),
        "avg_win_usd":    round(avg_win_usd, 2),
        "avg_loss_usd":   round(avg_loss_usd, 2),
        "tp_pct":         round(tp_pct, 3),
        "ev_trade":       round(ev_trade, 4),
        "trades_needed":  trades_needed,
        "days_est":       days_est,
        "daily_loss_usd": round(capital * daily_loss_pct / 100, 2),
        "max5_loss_usd":  round(avg_loss_usd * 5, 2),
    }


# -- Main wizard ---------------------------------------------------------------

async def run() -> dict:
    """
    Run the full interactive setup flow.
    Returns a dict of resolved settings to be applied to the config module.
    """
    print()
    _header("AI SCALPING BOT  --  INTERACTIVE SETUP WIZARD")
    print()
    print("  This wizard designs a personalized trading strategy for you.")
    print("  Answer each question and the bot will configure itself accordingly.")
    print()

    # Step 1: Live prices
    print("  Fetching live prices from MEXC Futures...")
    print()
    prices = await _fetch_prices()

    _sep()
    print("  LIVE MARKET PRICES  (MEXC Futures -- Perpetual Swaps)")
    _sep()
    for sym, label in _PAIR_LABELS.items():
        p = prices.get(sym, 0)
        if p > 0:
            print(f"  {label}   ->   ${p:>12,.2f} USDT")
        else:
            print(f"  {label}   ->   price unavailable")
    _sep()
    print()

    # Step 2: Capital
    print("  STEP 1  --  Starting Capital")
    print("  How much USDT do you want to trade with?")
    print("  Accepted range: $10 to $500")
    print()

    def _valid_capital(v: str) -> bool:
        try:
            return 10.0 <= float(v) <= 500.0
        except ValueError:
            return False

    capital = float(_ask("  Your capital ($10 - $500): $",
                         _valid_capital, "Enter a number between 10 and 500."))
    print(f"  OK  Capital: ${capital:.2f} USDT")
    print()

    # Step 3: Profit target
    print("  STEP 2  --  Profit Target")
    print("  How much profit do you want the bot to chase before reporting success?")
    print("  Enter a dollar amount (e.g. 50, 100) OR a % return (e.g. 20%)")
    print()

    def _valid_target(v: str) -> bool:
        try:
            return float(v.rstrip("%")) > 0
        except ValueError:
            return False

    raw_target = _ask("  Profit target ($ or %): ",
                      _valid_target, "Enter a positive number like 50 or 20%")

    if raw_target.endswith("%"):
        target_pct    = float(raw_target.rstrip("%"))
        profit_target = round(capital * (target_pct / 100), 2)
        print(f"  OK  Target: {target_pct}% return  =  ${profit_target:.2f} USDT")
    else:
        profit_target = float(raw_target)
        target_pct    = round((profit_target / capital) * 100, 1)
        print(f"  OK  Target: ${profit_target:.2f} USDT  ({target_pct}% return on your capital)")
    print()

    # Step 4: Risk profile
    print("  STEP 3  --  Risk Preference")
    print()
    for key, p in _PROFILES.items():
        rr = p["rr_ratio"]
        print(f"  [{key}]  {p['name']}")
        print(f"        Leverage:     {p['leverage']}x")
        print(f"        Risk/trade:   {p['risk_per_trade_pct']}% of capital"
              f"  =  ${capital * p['risk_per_trade_pct'] / 100:.2f} USDT per trade")
        print(f"        Stop loss:    {p['sl_pct']:.2f}%"
              f"  |  Take profit: {p['sl_pct'] * rr:.3f}%  (1:{rr} R:R)")
        print(f"        Daily cap:    {p['daily_loss_pct']}%"
              f"  =  ${capital * p['daily_loss_pct'] / 100:.2f} USDT max loss/day")
        print(f"        {p['desc']}")
        print()

    risk_choice = _ask("  Choose risk level [1 / 2 / 3]: ",
                       lambda v: v in _PROFILES, "Enter 1, 2, or 3.")
    profile = _PROFILES[risk_choice]
    print(f"  OK  Risk profile: {profile['name']}")
    print()

    # Step 5: Daily loss limit
    print("  STEP 4  --  Daily Loss Limit")
    default_dl = profile["daily_loss_pct"]
    print(f"  Your chosen profile suggests {default_dl}%"
          f"  =  ${capital * default_dl / 100:.2f} USDT/day.")
    print("  The bot halts for the day if this limit is hit.")
    print(f"  Press Enter to keep {default_dl}%, or type a custom % (1.0 - 10.0):")
    print()

    def _valid_dl(v: str) -> bool:
        if v.lower() in ("", "y", "yes", "keep"):
            return True
        try:
            return 1.0 <= float(v.rstrip("%")) <= 10.0
        except ValueError:
            return False

    raw_dl = _ask(f"  Daily loss limit [{default_dl}%] (Enter to keep): ",
                  _valid_dl, "Enter a number 1.0 - 10.0")

    daily_loss_pct = default_dl if raw_dl.lower() in ("", "y", "yes", "keep") \
                     else float(raw_dl.rstrip("%"))
    print(f"  OK  Daily loss cap: {daily_loss_pct}%"
          f"  =  ${capital * daily_loss_pct / 100:.2f} USDT")
    print()

    # Step 6: Trade quality
    print("  STEP 5  --  Trade Quality Preference")
    print("  Should the bot take more trades or only very high-quality setups?")
    print()
    print("  [1]  More trades   -- lower confluence bar (4/7 signals), more opportunities")
    print("  [2]  Fewer trades  -- higher bar (5/7 signals), only the clearest setups")
    print()

    quality_choice = _ask("  Trade quality [1 / 2]: ",
                          lambda v: v in ("1", "2"), "Enter 1 or 2.")
    if quality_choice == "2":
        profile = dict(profile)   # don't mutate the global
        profile["min_confluence"]    = min(7, profile["min_confluence"] + 1)
        profile["min_ai_confidence"] = min(90, profile["min_ai_confidence"] + 5)
    quality_label = "More trades (wider filter)" if quality_choice == "1" \
                    else "Fewer trades (tighter filter)"
    print(f"  OK  {quality_label}")
    print(f"      Confluence required: {profile['min_confluence']}/7"
          f"  |  Min AI confidence: {profile['min_ai_confidence']}%")
    print()

    # Step 7: Trading mode
    print("  STEP 6  --  Trading Mode")
    print()
    print("  [1]  PAPER  -- fully simulated, no real money, no MEXC keys needed")
    print("  [2]  LIVE   -- real orders on MEXC Futures (requires API keys in .env)")
    print()

    mode_choice = _ask("  Mode [1 / 2]: ",
                       lambda v: v in ("1", "2"), "Enter 1 or 2.")
    is_paper   = mode_choice == "1"
    mode_label = "PAPER (simulated)" if is_paper else "LIVE (real money)"

    if not is_paper and (not config.MEXC_API_KEY or not config.MEXC_SECRET):
        print()
        print("  WARNING: MEXC_API_KEY / MEXC_SECRET are not set in your .env file.")
        print("  The bot will fail to place real orders. Consider paper mode first.")

    print(f"  OK  Mode: {mode_label}")
    print()

    # Strategy summary
    calc = _calc(capital, profit_target, profile, daily_loss_pct)

    _header("STRATEGY SUMMARY")
    print()
    print(f"  Capital:              ${capital:.2f} USDT")
    print(f"  Profit target:        ${profit_target:.2f} USDT  ({target_pct}% return)")
    print(f"  Trading mode:         {mode_label}")
    print(f"  Risk profile:         {profile['name']}")
    print(f"  Trade filter:         {quality_label}")
    print()
    print(f"  Leverage:             {profile['leverage']}x")
    print(f"  Risk per trade:       {profile['risk_per_trade_pct']}%"
          f"  =  ${calc['risk_usd']:.2f} USDT risked per trade")
    print(f"  Stop loss:            {profile['sl_pct']:.2f}%  from entry")
    print(f"  Take profit:          {calc['tp_pct']:.3f}%  from entry"
          f"  (1:{profile['rr_ratio']} risk-to-reward)")
    print(f"  Avg win per trade:    +${calc['avg_win_usd']:.2f} USDT")
    print(f"  Avg loss per trade:   -${calc['avg_loss_usd']:.2f} USDT")
    print(f"  Daily loss cap:       ${calc['daily_loss_usd']:.2f} USDT  ({daily_loss_pct}%)")
    print(f"  Max 5-loss streak:    -${calc['max5_loss_usd']:.2f} USDT")
    print()
    print(f"  Pairs scanned:        BTC, ETH, SOL, BNB  (1-minute candles)")
    print(f"  Min confluence:       {profile['min_confluence']}/7 signals must agree")
    print(f"  Min AI confidence:    {profile['min_ai_confidence']}%")
    print()
    _sep()
    print(f"  PROJECTIONS  (assumes {_WIN_RATE * 100:.0f}% win rate -- realistic for scalping)")
    _sep()
    print(f"  Expected value/trade: ${calc['ev_trade']:.4f} USDT")
    if calc["ev_trade"] > 0:
        print(f"  Trades to reach target:  ~{calc['trades_needed']} trades")
        print(f"  Estimated time:          ~{calc['days_est']} day(s)"
              f"  (assuming ~20 good setups/day)")
    else:
        print("  WARNING: Negative expected value -- consider adjusting risk settings")
    _sep()
    print()
    print("  Note: Projections are estimates only. Use paper mode first to validate")
    print("  before going live with real money.")
    print()

    # Confirm
    confirm = _ask("  Start the bot with these settings? [y / n]: ",
                   lambda v: v.lower() in ("y", "yes", "n", "no"),
                   "Enter y or n.")

    if confirm.lower() in ("n", "no"):
        print("\n  Setup cancelled. Re-run main.py to start over.\n")
        sys.exit(0)

    print()
    print("  Launching bot...")
    print()

    # Return resolved settings
    return {
        "STARTING_CAPITAL":      capital,
        "LEVERAGE":              profile["leverage"],
        "RISK_PER_TRADE_PCT":    profile["risk_per_trade_pct"],
        "PROFIT_TARGET_USDT":    profit_target,
        "DAILY_LOSS_LIMIT_PCT":  daily_loss_pct,
        "MAX_STOP_LOSS_PCT":     profile["sl_pct"] / 100,
        "MIN_CONFLUENCE_SCORE":  profile["min_confluence"],
        "MIN_AI_CONFIDENCE":     float(profile["min_ai_confidence"]),
        "IS_PAPER":              is_paper,
        "TRADING_MODE":          "paper" if is_paper else "live",
        "MAX_OPEN_POSITIONS":    profile["max_open_positions"],
        "ATR_STOP_MULTIPLIER":   round(profile["sl_pct"] / 0.10, 2),
        "MIN_PROFIT_TARGET_PCT": max(0.0010, (calc["tp_pct"] / 100) - 0.0004),
    }
