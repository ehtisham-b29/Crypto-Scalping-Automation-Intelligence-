"""
Technical indicator engine — pure pandas/numpy, no pandas-ta dependency.
Works on Python 3.14+. Calculates RSI, EMA, BB, MACD, ATR, VWAP, Stochastic.
Returns a flat dict of current values for the latest closed candle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

import config


def calculate(df: pd.DataFrame) -> dict | None:
    """
    Calculate all indicators on the provided OHLCV DataFrame.
    Returns a dict of indicator values for the most recent candle,
    or None if there are insufficient candles.
    """
    if df is None or len(df) < 30:
        return None

    try:
        df = df.copy()

        # ── RSI (Wilder smoothing) ──────────────────────────────────────────────
        delta      = df["close"].diff()
        gain       = delta.clip(lower=0)
        loss       = (-delta).clip(lower=0)
        alpha      = 1.0 / config.RSI_PERIOD
        avg_gain   = gain.ewm(alpha=alpha, min_periods=config.RSI_PERIOD, adjust=False).mean()
        avg_loss   = loss.ewm(alpha=alpha, min_periods=config.RSI_PERIOD, adjust=False).mean()
        rs         = avg_gain / avg_loss.replace(0, 1e-10)
        df["rsi"]  = 100.0 - (100.0 / (1.0 + rs))

        # ── EMAs ───────────────────────────────────────────────────────────────
        df["ema_fast"] = df["close"].ewm(span=config.EMA_FAST, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=config.EMA_SLOW, adjust=False).mean()

        # ── Bollinger Bands ────────────────────────────────────────────────────
        bb_mid         = df["close"].rolling(config.BB_PERIOD).mean()
        bb_std         = df["close"].rolling(config.BB_PERIOD).std(ddof=0)
        df["bb_upper"] = bb_mid + config.BB_STD * bb_std
        df["bb_mid"]   = bb_mid
        df["bb_lower"] = bb_mid - config.BB_STD * bb_std
        # Bandwidth as % (matches pandas-ta BBB output)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid * 100

        # ── MACD ───────────────────────────────────────────────────────────────
        ema_fast_m      = df["close"].ewm(span=config.MACD_FAST, adjust=False).mean()
        ema_slow_m      = df["close"].ewm(span=config.MACD_SLOW, adjust=False).mean()
        df["macd"]      = ema_fast_m - ema_slow_m
        df["macd_signal"] = df["macd"].ewm(span=config.MACD_SIGNAL, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # ── ATR (Wilder smoothing) ──────────────────────────────────────────────
        hl   = df["high"] - df["low"]
        hpc  = (df["high"] - df["close"].shift(1)).abs()
        lpc  = (df["low"]  - df["close"].shift(1)).abs()
        tr   = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        atr_alpha  = 1.0 / config.ATR_PERIOD
        df["atr"]  = tr.ewm(alpha=atr_alpha, min_periods=config.ATR_PERIOD, adjust=False).mean()
        df["atr_avg"] = df["atr"].rolling(20).mean()

        # ── VWAP (daily reset) ─────────────────────────────────────────────────
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        df["_tp"] = typical
        try:
            date_key = df.index.normalize()            # works for DatetimeTzAware index
            df["_date"] = date_key
            cum_tp_vol  = df.groupby("_date").apply(
                lambda g: (g["_tp"] * g["volume"]).cumsum()
            ).reset_index(level=0, drop=True)
            cum_vol     = df.groupby("_date")["volume"].cumsum()
            df["vwap"]  = cum_tp_vol / cum_vol.replace(0, np.nan)
        except Exception:
            # Fallback: session VWAP (no daily reset)
            df["vwap"] = (typical * df["volume"]).cumsum() / df["volume"].cumsum()
        df.drop(columns=["_tp", "_date"], errors="ignore", inplace=True)

        # ── Stochastic (%K smoothed, %D) ───────────────────────────────────────
        smooth_k    = 3
        low_min     = df["low"].rolling(config.STOCH_K).min()
        high_max    = df["high"].rolling(config.STOCH_K).max()
        raw_k       = 100.0 * (df["close"] - low_min) / (high_max - low_min).replace(0, 1e-10)
        df["stoch_k"] = raw_k.rolling(smooth_k).mean()
        df["stoch_d"] = df["stoch_k"].rolling(config.STOCH_D).mean()

        # ── Volume ratio ───────────────────────────────────────────────────────
        df["vol_avg"]   = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_avg"].replace(0, 1e-10)

        # ── Pull last row and previous closed row ──────────────────────────────
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else last

        price = float(last["close"])
        vwap  = float(last["vwap"]) if pd.notna(last.get("vwap")) else price
        vwap_deviation = ((price - vwap) / vwap * 100) if vwap != 0 else 0.0

        atr     = float(last["atr"])     if pd.notna(last.get("atr"))     else 0.0
        atr_avg = float(last["atr_avg"]) if pd.notna(last.get("atr_avg")) else atr
        atr_elevated = atr > (atr_avg * 2.0)

        return {
            "price":          price,
            "rsi":            _safe(last, "rsi"),
            "rsi_prev":       _safe(prev, "rsi"),          # RSI one candle ago
            "ema_fast_prev":  _safe(prev, "ema_fast"),     # EMA fast one candle ago
            "ema_slow_prev":  _safe(prev, "ema_slow"),     # EMA slow one candle ago
            "ema_fast":       _safe(last, "ema_fast"),
            "ema_slow":       _safe(last, "ema_slow"),
            "bb_upper":       _safe(last, "bb_upper"),
            "bb_mid":         _safe(last, "bb_mid"),
            "bb_lower":       _safe(last, "bb_lower"),
            "bb_width":       _safe(last, "bb_width"),
            "macd":           _safe(last, "macd"),
            "macd_signal":    _safe(last, "macd_signal"),
            "macd_hist":      _safe(last, "macd_hist"),
            "macd_hist_prev": _safe(prev, "macd_hist"),    # MACD hist one candle ago
            "atr":            atr,
            "atr_avg":        atr_avg,
            "atr_elevated":   atr_elevated,
            "vwap":           vwap,
            "vwap_deviation": round(vwap_deviation, 4),
            "stoch_k":        _safe(last, "stoch_k"),
            "stoch_d":        _safe(last, "stoch_d"),
            "vol_ratio":      _safe(last, "vol_ratio"),
            "vol_avg":        _safe(last, "vol_avg"),
        }

    except Exception as e:
        logger.error(f"Indicator calculation error: {e}")
        return None


def _safe(row: pd.Series, col: str, default: float = 0.0) -> float:
    val = row.get(col, default)
    return round(float(val), 6) if pd.notna(val) else default
