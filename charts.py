"""
Chart screenshot module using Playwright (headless Chromium).

Captures TradingView widget embeds and DexScreener chart pages as PNG images
and returns the raw bytes for sending via Telegram's send_photo API.

Browser instance is shared (singleton) and lazily launched on first use.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger("TB.charts")

# ── Shared browser instance ─────────────────────────────────────────────────

_browser = None
_playwright_ctx = None


async def _get_browser():
    """Lazily launch and return a shared headless Chromium instance."""
    global _browser, _playwright_ctx
    if _browser is None or not _browser.is_connected():
        from playwright.async_api import async_playwright

        _playwright_ctx = await async_playwright().start()
        _browser = await _playwright_ctx.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        logger.info("Chromium browser launched")
    return _browser


# ── TradingView interval → widget interval ──────────────────────────────────

_WIDGET_INTERVALS = {
    "1m": "1",   "5m": "5",   "15m": "15",  "30m": "30",
    "1h": "60",  "2h": "120", "4h": "240",
    "1d": "D",   "1w": "W",   "1W": "W",    "1M": "M",
}

# Some exchanges use different names in the widget vs. tradingview_ta
_WIDGET_EXCHANGE_MAP = {
    "FX_IDC": "FX",  # forex screener → widget exchange
}


# ── TradingView chart screenshot ────────────────────────────────────────────

async def screenshot_tradingview_chart(
    symbol: str,
    exchange: str,
    interval: str = "1h",
    width: int = 1280,
    height: int = 800,
) -> Optional[bytes]:
    """
    Screenshot a TradingView chart via their widget embed page.

    The chart renders in dark theme with RSI, MACD, and Bollinger Band
    overlays — the same indicators the AI analyses in text.

    Returns PNG bytes on success, None on failure.
    """
    page = None
    try:
        browser = await _get_browser()
        page = await browser.new_page(viewport={"width": width, "height": height})

        wi = _WIDGET_INTERVALS.get(interval, "60")
        widget_exchange = _WIDGET_EXCHANGE_MAP.get(exchange, exchange)

        url = (
            "https://s.tradingview.com/widgetembed/"
            f"?frameElementId=tv_chart"
            f"&symbol={widget_exchange}%3A{symbol}"
            f"&interval={wi}"
            "&theme=dark&style=1&locale=en"
            "&enable_publishing=false"
            "&hide_side_toolbar=true"
            "&hide_top_toolbar=false"
            "&save_image=false"
            "&hideideas=1"
            "&studies=%5B%22RSI%40tv-basicstudies%22%2C"
            "%22MACD%40tv-basicstudies%22%2C"
            "%22BB%40tv-basicstudies%22%5D"
        )

        await page.goto(url, wait_until="networkidle", timeout=20_000)

        # Wait for the chart canvas to appear (proves the widget loaded)
        try:
            await page.wait_for_selector("canvas", timeout=10_000)
        except Exception:
            logger.warning("TradingView canvas not found, using timed wait")

        # Extra time for candlestick data to stream in via WebSocket
        await page.wait_for_timeout(3_000)

        screenshot = await page.screenshot(type="png")
        await page.close()
        page = None

        logger.info(
            "TradingView chart captured: %s:%s (%s) — %d bytes",
            widget_exchange, symbol, interval, len(screenshot),
        )
        return screenshot

    except Exception as exc:
        logger.error("TradingView screenshot failed (%s:%s): %s", exchange, symbol, exc)
        if page:
            try:
                await page.close()
            except Exception:
                pass
        return None


# ── DexScreener chart screenshot ────────────────────────────────────────────

async def screenshot_dexscreener_chart(
    chain: str,
    pair_address: str,
    width: int = 1280,
    height: int = 900,
) -> Optional[bytes]:
    """
    Screenshot the DexScreener chart page for a token pair.

    Returns PNG bytes on success, None on failure.
    """
    page = None
    try:
        browser = await _get_browser()
        page = await browser.new_page(viewport={"width": width, "height": height})

        url = f"https://dexscreener.com/{chain}/{pair_address}"
        await page.goto(url, wait_until="networkidle", timeout=25_000)

        # Dismiss cookie / permission banners
        for btn_text in ["Accept", "Got it", "Close", "OK", "I understand"]:
            try:
                await page.click(f'button:has-text("{btn_text}")', timeout=1_500)
            except Exception:
                pass

        # Wait for chart to render
        await page.wait_for_timeout(4_000)

        screenshot = await page.screenshot(type="png")
        await page.close()
        page = None

        logger.info(
            "DexScreener chart captured: %s/%s — %d bytes",
            chain, pair_address, len(screenshot),
        )
        return screenshot

    except Exception as exc:
        logger.error("DexScreener screenshot failed (%s/%s): %s", chain, pair_address, exc)
        if page:
            try:
                await page.close()
            except Exception:
                pass
        return None


# ── Cleanup ──────────────────────────────────────────────────────────────────

async def close_browser():
    """Shut down the shared browser (call on app shutdown)."""
    global _browser, _playwright_ctx
    if _browser:
        try:
            await _browser.close()
        except Exception:
            pass
        _browser = None
    if _playwright_ctx:
        try:
            await _playwright_ctx.stop()
        except Exception:
            pass
        _playwright_ctx = None
    logger.info("Chromium browser closed")
