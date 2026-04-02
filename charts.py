"""
Chart module — generates candlestick chart images.

TradingView charts: Playwright screenshots of widget embeds (works great).
DEX token charts:   Generated via mplfinance from GeckoTerminal OHLCV API
                    data (Playwright screenshots of GT embeds fail because
                    the TradingView widget inside them doesn't render candle
                    data in headless Chromium).

Browser instance is shared (singleton) and lazily launched on first use.
"""

import asyncio
import logging
from io import BytesIO
from typing import Optional

import httpx

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


# ── DEX token chart via GeckoTerminal OHLCV API + mplfinance ─────────────────
# No browser needed — fetches candle data from the free GT API and renders
# a professional-looking candlestick chart with volume using matplotlib.

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

# Map interval strings → GeckoTerminal OHLCV API parameters (timeframe, aggregate)
_GT_OHLCV_PARAMS = {
    "1m":  ("minute", 1),
    "5m":  ("minute", 5),
    "15m": ("minute", 15),
    "30m": ("minute", 30),
    "1h":  ("hour", 1),
    "2h":  ("hour", 2),
    "4h":  ("hour", 4),
    "1d":  ("day", 1),
    "1D":  ("day", 1),
    "1w":  ("day", 1),   # GT has no weekly — show daily
    "1W":  ("day", 1),
    "1M":  ("day", 1),
}


async def generate_dex_chart(
    chain: str,
    pair_address: str,
    interval: str = "1h",
    token_symbol: str = "TOKEN",
) -> Optional[bytes]:
    """
    Generate a candlestick + volume chart image for a DEX token pair.

    Fetches OHLCV data from GeckoTerminal's free API, then renders
    a dark-themed candlestick chart using mplfinance.

    Returns PNG bytes on success, None on failure.
    """
    try:
        # ── 1. Fetch OHLCV candles from GeckoTerminal API ──
        gt_chain = _GT_CHAIN_MAP.get(chain, chain)
        timeframe, aggregate = _GT_OHLCV_PARAMS.get(interval, ("hour", 1))

        api_url = (
            f"https://api.geckoterminal.com/api/v2/networks/{gt_chain}"
            f"/pools/{pair_address}/ohlcv/{timeframe}"
            f"?aggregate={aggregate}&limit=80"
        )
        logger.info("Fetching OHLCV: %s", api_url)

        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(api_url)
            resp.raise_for_status()
            data = resp.json()

        ohlcv_list = (
            data.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
        )
        if not ohlcv_list or len(ohlcv_list) < 3:
            logger.warning("Not enough OHLCV data (%d candles)", len(ohlcv_list))
            return None

        # ── 2. Convert to pandas DataFrame ──
        import pandas as pd

        # Each candle: [timestamp, open, high, low, close, volume]
        # API returns newest-first — reverse to chronological order
        ohlcv_list.sort(key=lambda c: c[0])

        df = pd.DataFrame(ohlcv_list, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["Date"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("Date", inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

        # ── 3. Render chart with mplfinance ──
        import mplfinance as mpf
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend

        # Dark theme matching typical trading terminals
        mc = mpf.make_marketcolors(
            up="#26a69a", down="#ef5350",        # green / red candles
            edge="inherit",
            wick="inherit",
            volume={"up": "#26a69a", "down": "#ef5350"},
        )
        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=mc,
            facecolor="#131722",
            edgecolor="#131722",
            figcolor="#131722",
            gridcolor="#1e222d",
            gridstyle="--",
            gridaxis="both",
            y_on_right=True,
            rc={
                "font.size": 9,
                "axes.labelcolor": "#787b86",
                "xtick.color": "#787b86",
                "ytick.color": "#787b86",
            },
        )

        # Build the title
        tf_label = interval.upper() if len(interval) <= 3 else interval
        title = f"{token_symbol}/USD · {tf_label}"

        buf = BytesIO()
        mpf.plot(
            df,
            type="candle",
            style=style,
            volume=True,
            title=title,
            ylabel="Price (USD)",
            ylabel_lower="Volume",
            figsize=(12, 7),
            tight_layout=True,
            savefig=dict(fname=buf, dpi=120, bbox_inches="tight", pad_inches=0.3),
        )
        buf.seek(0)
        png_bytes = buf.read()

        logger.info(
            "DEX chart generated: %s %s/%s (%s) — %d candles, %d bytes",
            token_symbol, gt_chain, pair_address[:10], interval,
            len(ohlcv_list), len(png_bytes),
        )
        return png_bytes

    except Exception as exc:
        logger.error("DEX chart generation failed (%s/%s): %s", chain, pair_address, exc)
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
