"""
Central configuration — loads from .env file.
All other modules import from here.
"""
import os
from dotenv import load_dotenv

load_dotenv()


# ── Exchange ──────────────────────────────────────────────────────────────────
MEXC_API_KEY: str = os.getenv("MEXC_API_KEY", "")
MEXC_SECRET: str = os.getenv("MEXC_SECRET", "")

# ── Trading mode ──────────────────────────────────────────────────────────────
TRADING_MODE: str = os.getenv("TRADING_MODE", "paper")   # "paper" | "live"
IS_PAPER: bool = TRADING_MODE.lower() == "paper"

# ── Capital & sizing ──────────────────────────────────────────────────────────
STARTING_CAPITAL: float = float(os.getenv("STARTING_CAPITAL", "500"))
LEVERAGE: int = int(os.getenv("LEVERAGE", "10"))
RISK_PER_TRADE_PCT: float = float(os.getenv("RISK_PER_TRADE_PCT", "1.5"))
PROFIT_TARGET_USDT: float = float(os.getenv("PROFIT_TARGET_USDT", "500"))

# ── Risk limits ───────────────────────────────────────────────────────────────
DAILY_LOSS_LIMIT_PCT: float = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "3.0"))
MAX_CONSECUTIVE_LOSSES: int = 5
MAX_DAILY_TRADES: int = 20           # fewer trades on 5m — each one is higher quality
MAX_OPEN_POSITIONS: int = 2

# ── Pairs & timeframe ─────────────────────────────────────────────────────────
_pairs_env = os.getenv("TRADING_PAIRS", "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,BNB/USDT:USDT")
TRADING_PAIRS: list[str] = [p.strip() for p in _pairs_env.split(",")]
TIMEFRAME: str = os.getenv("TIMEFRAME", "5m")   # 5m candles: less noise, better signals
CANDLE_LIMIT: int = 200              # 200 × 5m = 16+ hours of history for indicators

# ── Signal thresholds ─────────────────────────────────────────────────────────
MIN_CONFLUENCE_SCORE: int = int(os.getenv("MIN_CONFLUENCE_SCORE", "4"))
# Renamed: MIN_AI_CONFIDENCE retained for backward compat with wizard + decision engine
MIN_AI_CONFIDENCE: float = float(os.getenv("MIN_AI_CONFIDENCE", "58"))

# ── Fees (MEXC Futures) ───────────────────────────────────────────────────────
MAKER_FEE: float = 0.0000           # 0% — limit orders
TAKER_FEE: float = 0.0002           # 0.020% — market orders
SLIPPAGE_BUFFER: float = 0.0001     # 0.01% extra buffer for slippage

# ── Trade parameters ──────────────────────────────────────────────────────────
# Round-trip fee = TAKER_FEE * 2 + SLIPPAGE_BUFFER = 0.05%
MIN_PROFIT_TARGET_PCT: float = 0.0020    # 0.20% net — well above 0.05% breakeven
MAX_STOP_LOSS_PCT: float = 0.0050        # never wider than 0.50% stop
ATR_STOP_MULTIPLIER: float = 1.2         # stop = 1.2 × ATR

# ── Indicator settings ────────────────────────────────────────────────────────
# Tuned for 5-minute candles: wider RSI bands reduce false signals on higher TF
RSI_PERIOD: int = 14
RSI_OVERSOLD: float = 30.0           # 30 on 5m (was 35 on 1m — cleaner signal)
RSI_OVERBOUGHT: float = 70.0         # 70 on 5m (was 65 on 1m)
EMA_FAST: int = 9
EMA_SLOW: int = 21
BB_PERIOD: int = 20
BB_STD: float = 2.0
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
ATR_PERIOD: int = 14
STOCH_K: int = 14
STOCH_D: int = 3
CVD_LOOKBACK: int = 10               # 10 candles × 5m = 50 min CVD window
OBI_LEVELS: int = 10
OBI_THRESHOLD: float = 0.40
SPREAD_MAX_BPS: float = 5.0
VOLUME_RATIO_MIN: float = 1.2

# ── SMC (Smart Money Concepts) parameters ─────────────────────────────────────
# Swing detection: bars each side needed to confirm a swing high/low
SMC_SWING_N: int = 3
# Order block lookback: how many candles to scan for OB formation
SMC_OB_LOOKBACK: int = 50
# FVG minimum size as % of price (filters micro-noise gaps)
SMC_FVG_MIN_SIZE_PCT: float = 0.03
# FVG lookback: candles to scan for fair value gaps
SMC_FVG_LOOKBACK: int = 30
# Liquidity cluster: price % range within which swing points are merged into one pool
SMC_LIQUIDITY_CLUSTER_PCT: float = 0.03
# Sweep lookback: how many recent candles to check for liquidity sweeps
SMC_SWEEP_LOOKBACK: int = 15

# ── External signals ──────────────────────────────────────────────────────────
FEAR_GREED_URL: str = "https://api.alternative.me/fng/?limit=1"
FEAR_GREED_POLL_SECONDS: int = 300

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: str = "logs/bot.log"

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH: str = "data/trades.db"
