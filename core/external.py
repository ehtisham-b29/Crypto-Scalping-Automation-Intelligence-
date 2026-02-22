"""
External signal fetcher — Fear & Greed Index and MEXC funding rates.
All calls are async and cached to avoid hammering APIs.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

import aiohttp
from loguru import logger

import config


class ExternalSignals:
    def __init__(self) -> None:
        self._fear_greed: dict = {}
        self._fear_greed_ts: float = 0.0
        self._funding: dict[str, float] = {}   # symbol → rate
        self._funding_ts: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        # Initial fetch
        await asyncio.gather(
            self._fetch_fear_greed(),
            self._fetch_funding_rates(),
        )

    async def close(self) -> None:
        if self._session:
            await self._session.close()

    # ── Public getters ────────────────────────────────────────────────────────

    def get_fear_greed(self) -> dict:
        return self._fear_greed

    def get_funding_rate(self, symbol: str) -> float:
        return self._funding.get(symbol, 0.0)

    # ── Background polling ────────────────────────────────────────────────────

    async def poll_loop(self) -> None:
        """Run this as a background asyncio task."""
        while True:
            await asyncio.sleep(config.FEAR_GREED_POLL_SECONDS)
            await asyncio.gather(
                self._fetch_fear_greed(),
                self._fetch_funding_rates(),
            )

    # ── Fetchers ──────────────────────────────────────────────────────────────

    async def _fetch_fear_greed(self) -> None:
        try:
            async with self._session.get(
                config.FEAR_GREED_URL, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                data = await resp.json(content_type=None)
                entry = data.get("data", [{}])[0]
                self._fear_greed = {
                    "value": int(entry.get("value", 50)),
                    "value_classification": entry.get("value_classification", "Neutral"),
                    "timestamp": entry.get("timestamp", ""),
                }
                self._fear_greed_ts = time.time()
                logger.debug(
                    f"Fear & Greed: {self._fear_greed['value']} "
                    f"({self._fear_greed['value_classification']})"
                )
        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed: {e}")

    async def _fetch_funding_rates(self) -> None:
        """Fetch funding rates from MEXC for all configured pairs."""
        try:
            import ccxt
            ex = ccxt.mexc({"options": {"defaultType": "swap"}})
            for symbol in config.TRADING_PAIRS:
                try:
                    info = ex.fetch_funding_rate(symbol)
                    rate = float(info.get("fundingRate", 0) or 0)
                    self._funding[symbol] = rate
                except Exception:
                    self._funding[symbol] = 0.0
            self._funding_ts = time.time()
            logger.debug(f"Funding rates updated: {self._funding}")
        except Exception as e:
            logger.warning(f"Funding rate fetch failed: {e}")
