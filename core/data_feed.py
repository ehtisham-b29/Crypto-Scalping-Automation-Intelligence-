"""
Data feed layer — streams OHLCV, order book, and trades from MEXC Futures
via ccxt.pro WebSocket. Maintains a shared MarketState for each symbol.
"""
import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import ccxt.pro as ccxtpro
import pandas as pd
from loguru import logger

import config


@dataclass
class OrderBook:
    bids: list = field(default_factory=list)   # [[price, volume], ...]
    asks: list = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class MarketState:
    """All live market data for one symbol, updated by WebSocket streams."""
    symbol: str
    candles: pd.DataFrame = field(default_factory=pd.DataFrame)
    orderbook: OrderBook = field(default_factory=OrderBook)
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=500))
    last_price: float = 0.0
    last_update: float = 0.0
    ready: bool = False              # True once enough candles are loaded

    # Display cache — written by analysis loop, read by live display
    cached_rsi: float = 0.0
    cached_conf: int = 0
    cached_dir: str = "neutral"
    cached_tradeable: bool = False


class DataFeed:
    """
    Manages WebSocket connections to MEXC Futures for all configured symbols.
    Call .start() to begin streaming; reads state via .get_state(symbol).
    """

    def __init__(self) -> None:
        self._states: dict[str, MarketState] = {
            sym: MarketState(symbol=sym) for sym in config.TRADING_PAIRS
        }
        self._exchange: Optional[ccxtpro.mexc] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def get_state(self, symbol: str) -> Optional[MarketState]:
        return self._states.get(symbol)

    def all_states(self) -> list[MarketState]:
        return list(self._states.values())

    async def start(self) -> None:
        """Connect to MEXC and start all streams concurrently."""
        self._exchange = ccxtpro.mexc({
            "apiKey": config.MEXC_API_KEY or None,
            "secret": config.MEXC_SECRET or None,
            "options": {"defaultType": "swap"},    # perpetual futures
        })

        logger.info(f"DataFeed starting — mode={config.TRADING_MODE}, "
                    f"pairs={config.TRADING_PAIRS}")

        # Seed each symbol with historical candles first
        await self._seed_candles()

        # Then launch all live WebSocket streams
        stream_tasks = []
        for symbol in config.TRADING_PAIRS:
            stream_tasks += [
                self._stream_ohlcv(symbol),
                self._stream_orderbook(symbol),
                self._stream_trades(symbol),
            ]
        await asyncio.gather(*stream_tasks)

    async def close(self) -> None:
        if self._exchange:
            await self._exchange.close()

    # ── Seeding ───────────────────────────────────────────────────────────────

    async def _seed_candles(self) -> None:
        """Fetch historical OHLCV to warm up all indicators before streaming."""
        exchange_rest = ccxtpro.mexc({
            "options": {"defaultType": "swap"},
        })
        try:
            for symbol in config.TRADING_PAIRS:
                try:
                    raw = await exchange_rest.fetch_ohlcv(
                        symbol, config.TIMEFRAME, limit=config.CANDLE_LIMIT
                    )
                    df = self._raw_to_df(raw)
                    self._states[symbol].candles = df
                    logger.info(f"Seeded {len(df)} candles for {symbol}")
                except Exception as e:
                    logger.warning(f"Could not seed candles for {symbol}: {e}")
        finally:
            await exchange_rest.close()

    # ── WebSocket streams ─────────────────────────────────────────────────────

    async def _stream_ohlcv(self, symbol: str) -> None:
        """Continuously receive 1-minute candles and update state."""
        while True:
            try:
                candles = await self._exchange.watch_ohlcv(
                    symbol, config.TIMEFRAME
                )
                df_new = self._raw_to_df(candles)
                state = self._states[symbol]

                if state.candles.empty:
                    state.candles = df_new
                else:
                    # Merge: drop rows with timestamps already present, append new
                    combined = pd.concat([state.candles, df_new])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined.sort_index(inplace=True)
                    # Keep only the last CANDLE_LIMIT rows to avoid unbounded growth
                    state.candles = combined.tail(config.CANDLE_LIMIT).copy()

                state.last_price = float(state.candles["close"].iloc[-1])
                state.last_update = datetime.now(timezone.utc).timestamp()

                if len(state.candles) >= 30:
                    state.ready = True

            except ccxtpro.NetworkError as e:
                logger.warning(f"OHLCV stream network error ({symbol}): {e} — retrying")
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"OHLCV stream error ({symbol}): {e}")
                await asyncio.sleep(5)

    async def _stream_orderbook(self, symbol: str) -> None:
        """Stream L2 order book updates."""
        while True:
            try:
                ob = await self._exchange.watch_order_book(
                    symbol, limit=config.OBI_LEVELS * 2
                )
                state = self._states[symbol]
                state.orderbook = OrderBook(
                    bids=ob.get("bids", [])[:config.OBI_LEVELS],
                    asks=ob.get("asks", [])[:config.OBI_LEVELS],
                    timestamp=ob.get("timestamp") or 0.0,
                )
            except ccxtpro.NetworkError as e:
                logger.warning(f"OrderBook stream error ({symbol}): {e} — retrying")
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"OrderBook stream error ({symbol}): {e}")
                await asyncio.sleep(5)

    async def _stream_trades(self, symbol: str) -> None:
        """Stream recent trades for CVD calculation and ms-level price updates."""
        while True:
            try:
                trades = await self._exchange.watch_trades(symbol)
                state = self._states[symbol]
                state.recent_trades.extend(trades)
                # Update last_price on every trade for millisecond-level display
                if trades:
                    state.last_price = float(trades[-1]["price"])
                    state.last_update = datetime.now(timezone.utc).timestamp()
            except ccxtpro.NetworkError as e:
                logger.warning(f"Trades stream error ({symbol}): {e} — retrying")
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Trades stream error ({symbol}): {e}")
                await asyncio.sleep(5)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _raw_to_df(raw: list) -> pd.DataFrame:
        """Convert raw OHLCV list to a labelled DataFrame with datetime index."""
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
