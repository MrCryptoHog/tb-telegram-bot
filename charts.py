"""
Chart module — generates candlestick chart images + wick analysis.

TradingView charts: Playwright screenshots of widget embeds (works great).
DEX token charts:   Generated via mplfinance from GeckoTerminal OHLCV API
                    data (Playwright screenshots of GT embeds fail because
                    the TradingView widget inside them doesn't render candle
                    data in headless Chromium).
1-Min Wick Analysis: Fetches 1m OHLCV from GeckoTerminal and detects
                     consecutive upper/lower wicks indicating TP or
                     accumulation patterns.

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


# ── 1-Minute Wick Analysis (Tops & Bottoms Detection) ───────────────────────
# Fetches the last 60 one-minute candles and detects consecutive wick patterns
# at similar price levels — the hallmark of tops (rejection / distribution)
# and bottoms (absorption / accumulation).


def _detect_wick_clusters(candles: list[dict], wick_key: str, price_key: str,
                          threshold: float = 0.35) -> list[dict]:
    """
    Scan candles for clusters of consecutive significant wicks whose tip prices
    land in a similar zone (within 1.5× average candle range), indicating a
    top or bottom formation.

    Args:
        candles: list of {'o','h','l','c','vol','idx'} dicts
        wick_key: 'upper' or 'lower'
        price_key: 'h' for upper wick tips, 'l' for lower wick tips
        threshold: minimum wick/range ratio to count as significant

    Returns list of cluster dicts with 'count', 'price_zone', 'start_idx', 'end_idx'
    """
    clusters = []
    streak = []

    def _save_streak():
        if len(streak) >= 2:
            tips = [c[price_key] for c in streak]
            clusters.append({
                "count": len(streak),
                "price_zone": sum(tips) / len(tips),
                "price_low": min(tips),
                "price_high": max(tips),
                "start_idx": streak[0]["idx"],
                "end_idx": streak[-1]["idx"],
            })

    # Compute average candle range for zone tolerance
    ranges = [c["h"] - c["l"] for c in candles if c["h"] - c["l"] > 0]
    avg_range = sum(ranges) / len(ranges) if ranges else 0
    zone_tolerance = avg_range * 2.0  # wicks within 2× avg range = same zone

    for candle in candles:
        rng = candle["h"] - candle["l"]
        if rng == 0:
            _save_streak()
            streak = []
            continue

        upper_wick = candle["h"] - max(candle["o"], candle["c"])
        lower_wick = min(candle["o"], candle["c"]) - candle["l"]
        wick = upper_wick if wick_key == "upper" else lower_wick
        ratio = wick / rng

        if ratio >= threshold:
            # Check if this wick tip is in the same price zone as current streak
            if streak and zone_tolerance > 0:
                avg_tip = sum(c[price_key] for c in streak) / len(streak)
                if abs(candle[price_key] - avg_tip) <= zone_tolerance:
                    streak.append(candle)
                else:
                    _save_streak()
                    streak = [candle]
            else:
                streak.append(candle)
        else:
            _save_streak()
            streak = []

    _save_streak()
    return clusters


async def analyze_1m_wicks(
    chain: str,
    pair_address: str,
) -> str | None:
    """
    Fetch 60 x 1-minute candles from GeckoTerminal and detect consecutive
    wick patterns that indicate potential tops (upper wick clusters at
    similar highs = rejection/distribution) and bottoms (lower wick clusters
    at similar lows = absorption/accumulation).

    Returns a human-readable analysis string to inject into the AI context,
    or None if data is unavailable / insufficient.
    """
    try:
        gt_chain = _GT_CHAIN_MAP.get(chain, chain)
        api_url = (
            f"https://api.geckoterminal.com/api/v2/networks/{gt_chain}"
            f"/pools/{pair_address}/ohlcv/minute"
            f"?aggregate=1&limit=60"
        )

        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(api_url)
            resp.raise_for_status()
            data = resp.json()

        ohlcv_list = (
            data.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
        )
        if not ohlcv_list or len(ohlcv_list) < 10:
            return None

        # Sort chronological (oldest first)
        ohlcv_list.sort(key=lambda c: c[0])

        # Build candle dicts
        candles = []
        for i, raw in enumerate(ohlcv_list):
            _, o, h, l, c, vol = [float(x) for x in raw]
            candles.append({"o": o, "h": h, "l": l, "c": c, "vol": vol, "idx": i})

        # Detect clusters
        top_clusters = _detect_wick_clusters(candles, "upper", "h")
        bottom_clusters = _detect_wick_clusters(candles, "lower", "l")

        # Count total significant wicks
        total_upper = sum(1 for c in candles
                         if (c["h"] - c["l"]) > 0
                         and (c["h"] - max(c["o"], c["c"])) / (c["h"] - c["l"]) >= 0.35)
        total_lower = sum(1 for c in candles
                         if (c["h"] - c["l"]) > 0
                         and (min(c["o"], c["c"]) - c["l"]) / (c["h"] - c["l"]) >= 0.35)

        # Price change
        first_close = candles[0]["c"]
        last_close = candles[-1]["c"]
        pct_change = ((last_close - first_close) / first_close * 100) if first_close else 0

        # Overall high/low of the period
        period_high = max(c["h"] for c in candles)
        period_low = min(c["l"] for c in candles)

        n = len(candles)
        lines = [
            f"=== 1-MINUTE WICK ANALYSIS (last {n} candles) ===",
            f"Price change over period: {pct_change:+.2f}%",
            f"Period range: low {period_low:.8g} → high {period_high:.8g}",
            f"Upper wick candles (rejection/selling): {total_upper}/{n}",
            f"Lower wick candles (absorption/buying): {total_lower}/{n}",
        ]

        # Report TOP formations
        if top_clusters:
            best = max(top_clusters, key=lambda x: x["count"])
            lines.append(
                f"🔴 POTENTIAL TOP DETECTED: {best['count']} consecutive upper wicks "
                f"clustered around {best['price_zone']:.8g} "
                f"(range {best['price_low']:.8g}–{best['price_high']:.8g})"
            )
            if best["count"] >= 4:
                lines.append(
                    "⚠️ STRONG TOP SIGNAL: 4+ consecutive rejections at this level = "
                    "heavy selling / distribution — price struggling to break through"
                )
            elif best["count"] >= 3:
                lines.append(
                    "⚠️ MODERATE TOP SIGNAL: 3 rejections at this level = "
                    "sellers defending, watch for reversal"
                )
            if len(top_clusters) > 1:
                lines.append(
                    f"Total top formations found: {len(top_clusters)} "
                    f"(streaks: {[c['count'] for c in top_clusters]})"
                )
        else:
            lines.append("No consecutive upper wick clusters (no clear top forming)")

        # Report BOTTOM formations
        if bottom_clusters:
            best = max(bottom_clusters, key=lambda x: x["count"])
            lines.append(
                f"🟢 POTENTIAL BOTTOM DETECTED: {best['count']} consecutive lower wicks "
                f"clustered around {best['price_zone']:.8g} "
                f"(range {best['price_low']:.8g}–{best['price_high']:.8g})"
            )
            if best["count"] >= 4:
                lines.append(
                    "🟢 STRONG BOTTOM SIGNAL: 4+ consecutive absorptions at this level = "
                    "buyers aggressively defending — potential reversal / accumulation zone"
                )
            elif best["count"] >= 3:
                lines.append(
                    "🟢 MODERATE BOTTOM SIGNAL: 3 absorptions at this level = "
                    "dip buyers active, holding support"
                )
            if len(bottom_clusters) > 1:
                lines.append(
                    f"Total bottom formations found: {len(bottom_clusters)} "
                    f"(streaks: {[c['count'] for c in bottom_clusters]})"
                )
        else:
            lines.append("No consecutive lower wick clusters (no clear bottom forming)")

        # Overall assessment
        if top_clusters and not bottom_clusters:
            lines.append("OVERALL: Selling pressure dominant — potential top forming")
        elif bottom_clusters and not top_clusters:
            lines.append("OVERALL: Buying absorption dominant — potential bottom forming")
        elif top_clusters and bottom_clusters:
            lines.append("OVERALL: Both top and bottom signals present — choppy/ranging")
        elif total_upper > total_lower * 1.5:
            lines.append("OVERALL: Upper wicks frequent but no clear cluster — mild selling")
        elif total_lower > total_upper * 1.5:
            lines.append("OVERALL: Lower wicks frequent but no clear cluster — mild buying")
        else:
            lines.append("OVERALL: No significant wick patterns — clean price action")

        lines.append("=========================")

        result = "\n".join(lines)
        logger.info(
            "Wick analysis for %s/%s: %d tops, %d bottoms detected",
            gt_chain, pair_address[:10],
            len(top_clusters), len(bottom_clusters),
        )
        return result

    except Exception as exc:
        logger.warning("1m wick analysis failed (%s/%s): %s", chain, pair_address, exc)
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
