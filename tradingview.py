"""
TradingView Technical Analysis integration.

Uses the tradingview_ta library to fetch live indicator data from
TradingView's scanning API (free, no key needed).

Supports: forex pairs, crypto, indices, commodities, and DXY.
"""

import re
import logging
from dataclasses import dataclass

from tradingview_ta import TA_Handler, Interval

logger = logging.getLogger("TB.tradingview")

# ── Interval mapping ────────────────────────────────────────────────────────

INTERVAL_MAP: dict[str, str] = {
    "1m":  Interval.INTERVAL_1_MINUTE,
    "5m":  Interval.INTERVAL_5_MINUTES,
    "15m": Interval.INTERVAL_15_MINUTES,
    "30m": Interval.INTERVAL_30_MINUTES,
    "1h":  Interval.INTERVAL_1_HOUR,
    "2h":  Interval.INTERVAL_2_HOURS,
    "4h":  Interval.INTERVAL_4_HOURS,
    "1d":  Interval.INTERVAL_1_DAY,
    "1w":  Interval.INTERVAL_1_WEEK,
    "1M":  Interval.INTERVAL_1_MONTH,
}

# ── Symbol → TradingView config mapping ─────────────────────────────────────
# Each entry: alias → (tv_symbol, screener, exchange)

@dataclass
class TVSymbol:
    symbol: str
    screener: str
    exchange: str
    display_name: str


# Master symbol database — aliases map to TradingView configs
SYMBOL_DB: dict[str, TVSymbol] = {}


def _register(aliases: list[str], symbol: str, screener: str, exchange: str, display: str):
    """Register a symbol under multiple aliases."""
    tv = TVSymbol(symbol=symbol, screener=screener, exchange=exchange, display_name=display)
    for alias in aliases:
        SYMBOL_DB[alias.lower()] = tv


# ── Forex pairs ──
_register(["eurusd", "eur/usd", "euro", "eur"],
          "EURUSD", "forex", "FX_IDC", "EUR/USD")
_register(["gbpusd", "gbp/usd", "pound", "gbp", "cable"],
          "GBPUSD", "forex", "FX_IDC", "GBP/USD")
_register(["usdjpy", "usd/jpy", "jpy", "yen"],
          "USDJPY", "forex", "FX_IDC", "USD/JPY")
_register(["audusd", "aud/usd", "aud", "aussie"],
          "AUDUSD", "forex", "FX_IDC", "AUD/USD")
_register(["usdcad", "usd/cad", "cad", "loonie"],
          "USDCAD", "forex", "FX_IDC", "USD/CAD")
_register(["usdchf", "usd/chf", "chf", "swissy"],
          "USDCHF", "forex", "FX_IDC", "USD/CHF")
_register(["nzdusd", "nzd/usd", "nzd", "kiwi"],
          "NZDUSD", "forex", "FX_IDC", "NZD/USD")
_register(["gbpjpy", "gbp/jpy"],
          "GBPJPY", "forex", "FX_IDC", "GBP/JPY")
_register(["eurjpy", "eur/jpy"],
          "EURJPY", "forex", "FX_IDC", "EUR/JPY")
_register(["eurgbp", "eur/gbp"],
          "EURGBP", "forex", "FX_IDC", "EUR/GBP")

# ── Indices ──
_register(["spx", "sp500", "s&p", "s&p500", "s&p 500", "sp 500"],
          "SPX", "america", "SP", "S&P 500")
_register(["ndx", "nasdaq", "nasdaq100", "nasdaq 100", "nas100", "nq"],
          "NDX", "america", "NASDAQ", "NASDAQ 100")
_register(["dxy", "dollar", "dollar index", "usd index"],
          "DXY", "america", "TVC", "US Dollar Index (DXY)")

# ── Commodities ──
_register(["gold", "xauusd", "xau/usd", "xau"],
          "XAUUSD", "cfd", "OANDA", "Gold (XAU/USD)")

# ── Crypto (Binance USDT pairs) ──
_register(["btc", "bitcoin", "btcusd", "btc/usd", "btcusdt", "btc/usdt"],
          "BTCUSDT", "crypto", "BINANCE", "Bitcoin (BTC/USDT)")
_register(["eth", "ethereum", "ethusd", "eth/usd", "ethusdt", "eth/usdt"],
          "ETHUSDT", "crypto", "BINANCE", "Ethereum (ETH/USDT)")
_register(["sol", "solana", "solusd", "sol/usd", "solusdt", "sol/usdt"],
          "SOLUSDT", "crypto", "BINANCE", "Solana (SOL/USDT)")
_register(["bnb", "bnbusd", "bnb/usd", "bnbusdt"],
          "BNBUSDT", "crypto", "BINANCE", "BNB (BNB/USDT)")
_register(["xrp", "ripple", "xrpusd", "xrpusdt"],
          "XRPUSDT", "crypto", "BINANCE", "XRP (XRP/USDT)")
_register(["doge", "dogecoin", "dogeusd", "dogeusdt"],
          "DOGEUSDT", "crypto", "BINANCE", "Dogecoin (DOGE/USDT)")
_register(["ada", "cardano", "adausd", "adausdt"],
          "ADAUSDT", "crypto", "BINANCE", "Cardano (ADA/USDT)")
_register(["avax", "avalanche", "avaxusd", "avaxusdt"],
          "AVAXUSDT", "crypto", "BINANCE", "Avalanche (AVAX/USDT)")
_register(["dot", "polkadot", "dotusd", "dotusdt"],
          "DOTUSDT", "crypto", "BINANCE", "Polkadot (DOT/USDT)")
_register(["link", "chainlink", "linkusd", "linkusdt"],
          "LINKUSDT", "crypto", "BINANCE", "Chainlink (LINK/USDT)")
_register(["matic", "polygon", "maticusd", "maticusdt"],
          "MATICUSDT", "crypto", "BINANCE", "Polygon (MATIC/USDT)")
_register(["atom", "cosmos", "atomusd", "atomusdt"],
          "ATOMUSDT", "crypto", "BINANCE", "Cosmos (ATOM/USDT)")
_register(["near", "nearusd", "nearusdt"],
          "NEARUSDT", "crypto", "BINANCE", "NEAR Protocol (NEAR/USDT)")
_register(["apt", "aptos", "aptusd", "aptusdt"],
          "APTUSDT", "crypto", "BINANCE", "Aptos (APT/USDT)")
_register(["sui", "suiusd", "suiusdt"],
          "SUIUSDT", "crypto", "BINANCE", "Sui (SUI/USDT)")
_register(["arb", "arbitrum", "arbusd", "arbusdt"],
          "ARBUSDT", "crypto", "BINANCE", "Arbitrum (ARB/USDT)")
_register(["op", "optimism", "opusd", "opusdt"],
          "OPUSDT", "crypto", "BINANCE", "Optimism (OP/USDT)")
_register(["pepe", "pepeusd", "pepeusdt"],
          "PEPEUSDT", "crypto", "BINANCE", "PEPE (PEPE/USDT)")
_register(["shib", "shiba", "shibusd", "shibusdt"],
          "SHIBUSDT", "crypto", "BINANCE", "Shiba Inu (SHIB/USDT)")
_register(["wif", "wifusd", "wifusdt"],
          "WIFUSDT", "crypto", "BINANCE", "dogwifhat (WIF/USDT)")
_register(["inj", "injective", "injusd", "injusdt"],
          "INJUSDT", "crypto", "BINANCE", "Injective (INJ/USDT)")

# ── US Stocks ──
_register(["aapl", "apple"],
          "AAPL", "america", "NASDAQ", "Apple (AAPL)")
_register(["msft", "microsoft"],
          "MSFT", "america", "NASDAQ", "Microsoft (MSFT)")
_register(["googl", "google", "goog", "alphabet"],
          "GOOGL", "america", "NASDAQ", "Alphabet (GOOGL)")
_register(["amzn", "amazon"],
          "AMZN", "america", "NASDAQ", "Amazon (AMZN)")
_register(["tsla", "tesla"],
          "TSLA", "america", "NASDAQ", "Tesla (TSLA)")
_register(["nvda", "nvidia"],
          "NVDA", "america", "NASDAQ", "NVIDIA (NVDA)")
_register(["meta", "facebook"],
          "META", "america", "NASDAQ", "Meta Platforms (META)")
_register(["nflx", "netflix"],
          "NFLX", "america", "NASDAQ", "Netflix (NFLX)")
_register(["amd"],
          "AMD", "america", "NASDAQ", "AMD (AMD)")
_register(["intc", "intel"],
          "INTC", "america", "NASDAQ", "Intel (INTC)")
_register(["pltr", "palantir"],
          "PLTR", "america", "NASDAQ", "Palantir (PLTR)")
_register(["pypl", "paypal"],
          "PYPL", "america", "NASDAQ", "PayPal (PYPL)")
_register(["coin", "coinbase"],
          "COIN", "america", "NASDAQ", "Coinbase (COIN)")
_register(["hood", "robinhood"],
          "HOOD", "america", "NASDAQ", "Robinhood (HOOD)")
_register(["jpm", "jpmorgan", "jp morgan"],
          "JPM", "america", "NYSE", "JPMorgan (JPM)")
_register(["v", "visa"],
          "V", "america", "NYSE", "Visa (V)")
_register(["dis", "disney"],
          "DIS", "america", "NYSE", "Disney (DIS)")
_register(["ba", "boeing"],
          "BA", "america", "NYSE", "Boeing (BA)")
_register(["ko", "coca-cola", "coca cola"],
          "KO", "america", "NYSE", "Coca-Cola (KO)")
_register(["jnj", "johnson"],
          "JNJ", "america", "NYSE", "Johnson & Johnson (JNJ)")
_register(["gme", "gamestop"],
          "GME", "america", "NYSE", "GameStop (GME)")
_register(["amc"],
          "AMC", "america", "NYSE", "AMC Entertainment (AMC)")
_register(["nio"],
          "NIO", "america", "NYSE", "NIO Inc (NIO)")
_register(["baba", "alibaba"],
          "BABA", "america", "NYSE", "Alibaba (BABA)")


# ── Timeframe extraction from user text ─────────────────────────────────────

TIMEFRAME_PATTERN = re.compile(
    r'\b(\d+)\s*(?:min(?:ute)?s?|m)\b'   # "5m", "15 minutes", "1min"
    r'|\b(\d+)\s*(?:hour|hr|h)\b'         # "1h", "4 hour", "2hr"
    r'|\b(\d+)\s*(?:day|d)\b'             # "1d", "1 day"
    r'|\b(\d+)\s*(?:week|wk|w)\b'         # "1w", "1 week"
    r'|\b(daily)\b'                        # "daily"
    r'|\b(weekly)\b'                       # "weekly"
    r'|\b(monthly)\b',                     # "monthly"
    re.IGNORECASE
)


def extract_interval(text: str) -> str:
    """Extract a timeframe from user text. Default: 1h."""
    m = TIMEFRAME_PATTERN.search(text)
    if not m:
        return Interval.INTERVAL_1_HOUR  # sensible default

    if m.group(1):  # minutes
        mins = int(m.group(1))
        return INTERVAL_MAP.get(f"{mins}m", Interval.INTERVAL_1_HOUR)
    elif m.group(2):  # hours
        hrs = int(m.group(2))
        return INTERVAL_MAP.get(f"{hrs}h", Interval.INTERVAL_1_HOUR)
    elif m.group(3):  # days
        return Interval.INTERVAL_1_DAY
    elif m.group(4):  # weeks
        return Interval.INTERVAL_1_WEEK
    elif m.group(5):  # "daily"
        return Interval.INTERVAL_1_DAY
    elif m.group(6):  # "weekly"
        return Interval.INTERVAL_1_WEEK
    elif m.group(7):  # "monthly"
        return Interval.INTERVAL_1_MONTH

    return Interval.INTERVAL_1_HOUR


# ── Symbol extraction from user text ────────────────────────────────────────

# Sorted by length (longest first) so "s&p 500" matches before "s&p"
_SORTED_ALIASES = sorted(SYMBOL_DB.keys(), key=len, reverse=True)


def extract_symbol(text: str) -> TVSymbol | None:
    """Try to find a known trading symbol in the user's message."""
    text_lower = text.lower()
    for alias in _SORTED_ALIASES:
        # Word boundary check to avoid partial matches
        pattern = r'(?<!\w)' + re.escape(alias) + r'(?!\w)'
        if re.search(pattern, text_lower):
            return SYMBOL_DB[alias]
    return None


# ── Fetch TradingView TA ────────────────────────────────────────────────────

def fetch_tradingview_ta(tv_symbol: TVSymbol, interval: str) -> dict | None:
    """
    Fetch technical analysis data from TradingView.
    Returns a dict with summary, indicators, oscillators, moving_averages.
    Returns None on failure.
    """
    try:
        handler = TA_Handler(
            symbol=tv_symbol.symbol,
            screener=tv_symbol.screener,
            exchange=tv_symbol.exchange,
            interval=interval,
        )
        analysis = handler.get_analysis()

        return {
            "summary": analysis.summary,
            "oscillators": analysis.oscillators,
            "moving_averages": analysis.moving_averages,
            "indicators": analysis.indicators,
        }
    except Exception as exc:
        logger.warning("TradingView TA failed for %s: %s", tv_symbol.symbol, exc)
        return None


# ── Format TA data into a context block for the AI ──────────────────────────

_INTERVAL_LABELS = {
    Interval.INTERVAL_1_MINUTE: "1 Minute",
    Interval.INTERVAL_5_MINUTES: "5 Minutes",
    Interval.INTERVAL_15_MINUTES: "15 Minutes",
    Interval.INTERVAL_30_MINUTES: "30 Minutes",
    Interval.INTERVAL_1_HOUR: "1 Hour",
    Interval.INTERVAL_2_HOURS: "2 Hours",
    Interval.INTERVAL_4_HOURS: "4 Hours",
    Interval.INTERVAL_1_DAY: "Daily",
    Interval.INTERVAL_1_WEEK: "Weekly",
    Interval.INTERVAL_1_MONTH: "Monthly",
}


def format_tradingview_context(tv_symbol: TVSymbol, interval: str, data: dict) -> str:
    """Format TradingView TA data into a readable block for the AI prompt."""
    ind = data["indicators"]
    summary = data["summary"]
    osc = data["oscillators"]
    ma = data["moving_averages"]

    interval_label = _INTERVAL_LABELS.get(interval, interval)

    def fmt(val, decimals=2):
        if val is None:
            return "N/A"
        if isinstance(val, float):
            return f"{val:.{decimals}f}"
        return str(val)

    # Determine price decimal places based on asset type
    price_dp = 5 if tv_symbol.screener == "forex" else 2

    lines = [
        f"=== LIVE TRADINGVIEW TECHNICAL ANALYSIS ===",
        f"IMPORTANT: This is REAL live data from TradingView. Analyze it using Rule 10.",
        f"",
        f"Asset: {tv_symbol.display_name}",
        f"Timeframe: {interval_label}",
        f"Exchange: {tv_symbol.exchange}",
        f"",
        f"=== PRICE DATA ===",
        f"Open: {fmt(ind.get('open'), price_dp)}",
        f"High: {fmt(ind.get('high'), price_dp)}",
        f"Low: {fmt(ind.get('low'), price_dp)}",
        f"Close: {fmt(ind.get('close'), price_dp)}",
        f"Change: {fmt(ind.get('change'))}%",
        f"",
        f"=== OVERALL RECOMMENDATION ===",
        f"Summary: {summary.get('RECOMMENDATION', 'N/A')}",
        f"  Buy signals: {summary.get('BUY', 0)} | Neutral: {summary.get('NEUTRAL', 0)} | Sell signals: {summary.get('SELL', 0)}",
        f"",
        f"=== OSCILLATORS ({osc.get('RECOMMENDATION', 'N/A')}) ===",
        f"RSI(14): {fmt(ind.get('RSI'))} {'(overbought >70)' if ind.get('RSI') and ind['RSI'] > 70 else '(oversold <30)' if ind.get('RSI') and ind['RSI'] < 30 else ''}",
        f"MACD: {fmt(ind.get('MACD.macd'), 4)} | Signal: {fmt(ind.get('MACD.signal'), 4)}",
        f"Stochastic %K: {fmt(ind.get('Stoch.K'))} | %D: {fmt(ind.get('Stoch.D'))}",
        f"CCI(20): {fmt(ind.get('CCI20'))}",
        f"ADX(14): {fmt(ind.get('ADX'))} {'(strong trend >25)' if ind.get('ADX') and ind['ADX'] > 25 else '(weak trend)'}",
        f"ATR(14): {fmt(ind.get('ATR'), price_dp)}",
        f"Williams %R: {fmt(ind.get('W.R'))}",
        f"Momentum(10): {fmt(ind.get('Mom'), 4)}",
        f"",
        f"=== MOVING AVERAGES ({ma.get('RECOMMENDATION', 'N/A')}) ===",
        f"EMA10: {fmt(ind.get('EMA10'), price_dp)} | EMA20: {fmt(ind.get('EMA20'), price_dp)}",
        f"EMA50: {fmt(ind.get('EMA50'), price_dp)} | EMA200: {fmt(ind.get('EMA200'), price_dp)}",
        f"SMA10: {fmt(ind.get('SMA10'), price_dp)} | SMA20: {fmt(ind.get('SMA20'), price_dp)}",
        f"SMA50: {fmt(ind.get('SMA50'), price_dp)} | SMA200: {fmt(ind.get('SMA200'), price_dp)}",
        f"VWAP: {fmt(ind.get('VWAP'), price_dp)}",
        f"",
        f"=== BOLLINGER BANDS ===",
        f"Upper: {fmt(ind.get('BB.upper'), price_dp)} | Lower: {fmt(ind.get('BB.lower'), price_dp)}",
        f"",
        f"=== PIVOT POINTS (Classic Monthly) ===",
        f"R3: {fmt(ind.get('Pivot.M.Classic.R3'), price_dp)}",
        f"R2: {fmt(ind.get('Pivot.M.Classic.R2'), price_dp)}",
        f"R1: {fmt(ind.get('Pivot.M.Classic.R1'), price_dp)}",
        f"Pivot: {fmt(ind.get('Pivot.M.Classic.Middle'), price_dp)}",
        f"S1: {fmt(ind.get('Pivot.M.Classic.S1'), price_dp)}",
        f"S2: {fmt(ind.get('Pivot.M.Classic.S2'), price_dp)}",
        f"S3: {fmt(ind.get('Pivot.M.Classic.S3'), price_dp)}",
        f"=================================",
    ]
    return "\n".join(lines)


# ── TA request detection ────────────────────────────────────────────────────

TA_REQUEST_PATTERN = re.compile(
    r'\b('
    r'ta\b|technical.?analysis|analy[sz]e|chart|'
    r'indicators?|rsi|macd|bollinger|ema|sma|'
    r'overbought|oversold|momentum|trend|'
    r'support.?resistance|setup|signal'
    r')\b',
    re.IGNORECASE,
)


def is_ta_request(text: str) -> bool:
    """Check if the message is asking for technical analysis."""
    return bool(TA_REQUEST_PATTERN.search(text))


def get_supported_symbols_text() -> str:
    """Return a formatted string of all supported symbols for help messages."""
    # Indices are identified by their specific symbols
    _INDEX_SYMBOLS = {"SPX", "NDX", "DXY"}
    
    categories = {
        "Forex": [],
        "Crypto": [],
        "Stocks": [],
        "Indices": [],
        "Commodities": [],
    }
    seen = set()
    for alias, tv in sorted(SYMBOL_DB.items(), key=lambda x: x[1].display_name):
        if tv.display_name in seen:
            continue
        seen.add(tv.display_name)
        if tv.screener == "forex":
            categories["Forex"].append(tv.display_name)
        elif tv.screener == "crypto":
            categories["Crypto"].append(tv.display_name)
        elif tv.symbol in _INDEX_SYMBOLS:
            categories["Indices"].append(tv.display_name)
        elif tv.screener == "america":
            categories["Stocks"].append(tv.display_name)
        elif tv.screener == "cfd":
            categories["Commodities"].append(tv.display_name)

    lines = []
    for cat, symbols in categories.items():
        if symbols:
            lines.append(f"<b>{cat}:</b> {', '.join(symbols)}")
    return "\n".join(lines)
