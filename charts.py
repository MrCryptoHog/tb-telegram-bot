"""
Chart screenshot module using Playwright (headless Chromium).

Captures TradingView widget embeds and GeckoTerminal chart embeds as PNG images
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
) -> tuple[bytes | None, str | None]:
    """
    Screenshot a TradingView chart via their widget embed page.

    The chart renders in dark theme with RSI, MACD, and Bollinger Band
    overlays — the same indicators the AI analyses in text.

    Returns (PNG bytes, live_price_string) on success.
    Either or both may be None on failure.
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

        # ---- Extract the live price shown on the chart ----
        chart_price = None
        try:
            # The widget header shows "C<price>" for the close of the current
            # bar being hovered / the last bar.  Try multiple selectors.
            price_text = await page.evaluate("""
                () => {
                    // Method 1: OHLC header values (the "C" value = current close)
                    const spans = document.querySelectorAll(
                        '[class*="headerWrapper"] [class*="value"]'
                    );
                    if (spans.length >= 4) {
                        // spans are O, H, L, C in order
                        return spans[3]?.textContent?.trim() || null;
                    }
                    // Method 2: last price label on the price axis
                    const priceLine = document.querySelector(
                        '[class*="lastPrice"], [class*="currentPrice"], '
                        + '[class*="price-axis"] [class*="last"]'
                    );
                    if (priceLine) return priceLine.textContent?.trim() || null;
                    // Method 3: try to find any element containing
                    // the close price text in the header row
                    const header = document.querySelector('[class*="headerWrapper"]');
                    if (header) {
                        const all = header.querySelectorAll('div, span');
                        for (const el of all) {
                            const t = el.textContent?.trim();
                            if (t && /^\d[\d,]*\.\d+$/.test(t)) return t;
                        }
                    }
                    return null;
                }
            """)
            if price_text:
                # Remove commas: "67,100.50" → "67100.50"
                chart_price = price_text.replace(",", "").strip()
                logger.info("Chart live price extracted: %s", chart_price)
        except Exception as price_exc:
            logger.warning("Could not extract chart price: %s", price_exc)

        screenshot = await page.screenshot(type="png")
        await page.close()
        page = None

        logger.info(
            "TradingView chart captured: %s:%s (%s) — %d bytes, price=%s",
            widget_exchange, symbol, interval, len(screenshot), chart_price,
        )
        return screenshot, chart_price

    except Exception as exc:
        logger.error("TradingView screenshot failed (%s:%s): %s", exchange, symbol, exc)
        if page:
            try:
                await page.close()
            except Exception:
                pass
        return None, None


# ── GeckoTerminal chart screenshot (replaces DexScreener — Cloudflare blocked) ─

# Map DexScreener chainId → GeckoTerminal network slug
_GT_CHAIN_MAP = {
    "ethereum": "eth",
    "bsc": "bsc",
    "solana": "solana",
    "arbitrum": "arbitrum",
    "polygon": "polygon_pos",
    "base": "base",
    "avalanche": "avax",
    "optimism": "optimism",
    "fantom": "ftm",
    "cronos": "cronos",
    "pulsechain": "pulsechain",
    "blast": "blast",
    "mantle": "mantle",
    "linea": "linea",
    "zksync": "zksync",
    "scroll": "scroll",
    "celo": "celo",
    "sui": "sui-network",
    "aptos": "aptos",
    "ton": "ton",
    "tron": "tron",
}


# ── GeckoTerminal timeframe button map ──
# Maps common interval strings to the button text visible in
# the GeckoTerminal TradingView embed toolbar.
_GT_TIMEFRAME_BUTTONS = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "D",
    "1D": "D",
    "1w": "D",     # GT embed doesn't have weekly — use daily
    "1W": "D",
    "1M": "D",
}


async def screenshot_geckoterminal_chart(
    chain: str,
    pair_address: str,
    interval: str = "1h",
    width: int = 1280,
    height: int = 900,
) -> Optional[bytes]:
    """
    Screenshot the GeckoTerminal embed chart for a token pair.

    The embed page loads a TradingView widget inside a child iframe.
    We must:
      1. Wait for the TradingView iframe to appear
      2. Wait for its toolbar buttons to load (~8s from page load)
      3. Click the correct timeframe button *inside the iframe*
      4. Wait for candlestick data to render
      5. Screenshot the full page

    Returns PNG bytes on success, None on failure.
    """
    page = None
    try:
        browser = await _get_browser()
        page = await browser.new_page(viewport={"width": width, "height": height})

        gt_chain = _GT_CHAIN_MAP.get(chain, chain)
        url = (
            f"https://www.geckoterminal.com/{gt_chain}/pools/{pair_address}"
            f"?embed=1&info=0&swaps=0"
        )
        logger.info("GeckoTerminal chart URL: %s (interval=%s)", url, interval)

        await page.goto(url, wait_until="domcontentloaded", timeout=25_000)

        # ── 1. Find the TradingView iframe by name ──
        tv_frame = None
        for _ in range(15):
            await page.wait_for_timeout(1_000)
            for f in page.frames:
                if "tradingview" in f.name.lower():
                    tv_frame = f
                    break
            if tv_frame:
                break

        if not tv_frame:
            logger.warning("TradingView iframe not found, taking plain screenshot")
            await page.wait_for_timeout(5_000)
            screenshot = await page.screenshot(type="png")
            await page.close()
            return screenshot

        logger.info("TradingView iframe found: %s", tv_frame.name)

        # ── 2. Wait for toolbar buttons to load inside the iframe ──
        tf_btn_text = _GT_TIMEFRAME_BUTTONS.get(interval, "1h")
        btn_ready = False
        for _ in range(12):
            await page.wait_for_timeout(1_000)
            has_btn = await tv_frame.evaluate("""
                (target) => {
                    const btns = document.querySelectorAll('button');
                    for (const b of btns) {
                        if (b.textContent?.trim() === target) return true;
                    }
                    return false;
                }
            """, tf_btn_text)
            if has_btn:
                btn_ready = True
                break

        # ── 3. Click the timeframe button inside the iframe ──
        if btn_ready:
            clicked = await tv_frame.evaluate("""
                (target) => {
                    const btns = document.querySelectorAll('button');
                    for (const b of btns) {
                        if (b.textContent?.trim() === target && b.offsetParent !== null) {
                            b.click();
                            return true;
                        }
                    }
                    return false;
                }
            """, tf_btn_text)
            if clicked:
                logger.info("Clicked timeframe button '%s' in TV iframe", tf_btn_text)
            else:
                logger.warning("Timeframe button '%s' found but click failed", tf_btn_text)
        else:
            logger.warning("Timeframe button '%s' never appeared in TV iframe", tf_btn_text)

        # ── 4. Wait for chart data to render ──
        # After clicking timeframe, new candles need to load.
        # Poll until the iframe reports canvases with actual drawn content.
        for attempt in range(15):
            await page.wait_for_timeout(1_000)
            canvas_count = await tv_frame.evaluate("""
                () => {
                    const canvases = document.querySelectorAll('canvas');
                    let loaded = 0;
                    for (const c of canvases) {
                        if (c.width > 200 && c.height > 200) loaded++;
                    }
                    return loaded;
                }
            """)
            if canvas_count >= 2:
                logger.info("Chart rendered (%d large canvases at attempt %d)",
                           canvas_count, attempt + 1)
                break
        else:
            logger.warning("Chart may not be fully rendered, proceeding anyway")

        # Final settle for rendering
        await page.wait_for_timeout(2_000)

        screenshot = await page.screenshot(type="png")
        await page.close()
        page = None

        logger.info(
            "GeckoTerminal chart captured: %s/%s — %d bytes",
            gt_chain, pair_address, len(screenshot),
        )
        return screenshot

    except Exception as exc:
        logger.error("GeckoTerminal screenshot failed (%s/%s): %s", chain, pair_address, exc)
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
