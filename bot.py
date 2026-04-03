"""
TB – Telegram Trading & Crypto Education Bot
Multi-provider AI with intelligent fallback & rate-limit management.

Providers (in priority order):
  1. Groq  (Llama 3.3 70B)
  2. Gemini (gemini-2.0-flash)
  3. Cerebras (Llama 3.3 70B)
  4. Mistral (mistral-small-latest)
  5. SambaNova (Llama 3.1 70B)

The bot lives in a private group and only responds when @mentioned.
It provides educational / strategy-focused answers across the entire
crypto and day-trading spectrum. It does NOT provide live market data.
"""

import os
import logging
import re
import time
import hashlib
from collections import OrderedDict
from dotenv import load_dotenv
import asyncio
from io import BytesIO
import httpx
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)

from providers import ProviderManager
from rate_limiter import RateLimiter
from charts import screenshot_tradingview_chart, generate_dex_chart, analyze_1m_wicks
from tradingview import (
    extract_symbol,
    extract_interval,
    fetch_tradingview_ta,
    format_tradingview_context,
    is_ta_request,
    get_supported_symbols_text,
)

# ── Configuration ────────────────────────────────────────────────────────────

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BOT_USERNAME = os.getenv("BOT_USERNAME", "HeyTB_bot")  # without the @
WEBHOOK_URL = os.getenv("RAILWAY_PUBLIC_DOMAIN", os.getenv("WEBHOOK_URL", ""))
PORT = int(os.getenv("PORT", "8080"))

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set. Add it to your .env or Railway env vars.")

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("TB")

# ── AI Provider Manager ─────────────────────────────────────────────────────

provider_mgr = ProviderManager()

# ── Rate limiter ─────────────────────────────────────────────────────────────
# Math: 5 free-tier providers ≈ 8,700 safe API calls/day ≈ 362/hour.
# With 30-min response cache (~30% savings) → ~12,400 effective/day ≈ 517/hr.
# 2-hour window ≈ 1,034 available calls. We cap well below that for safety.
#
#   Per-user:  8 questions per 2-hour window
#   Anti-spam: 45-second minimum gap between questions
#   Global:    150 API calls per 2-hour window (keeps us at ~1,800/day max)
#   → At 1,800/day, we only use ~21% of the 8,700 daily budget
#   → Even if one provider dies, the remaining 4 can cover it easily

rate_limiter = RateLimiter(
    user_max_per_window=8,
    window_seconds=7200,       # 2 hours
    user_cooldown_seconds=45,  # min gap between messages
    global_max_per_window=150, # group-wide cap per 2h
)

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are **TB** – the resident trading & crypto education expert for a private \
community Telegram group.

## Your expertise
You have deep, comprehensive knowledge spanning:
- **Crypto**: micro-caps, low-caps, mid-caps, large-caps, DeFi, NFTs, \
tokenomics, on-chain analysis, smart-contract basics, CEX/DEX mechanics, \
yield farming, liquidity pools, airdrops, layer-1/layer-2 ecosystems, \
Bitcoin, Ethereum, Solana, and the broader altcoin landscape.
- **Day trading**: forex, commodities (gold, oil, agricultural), indices \
(S&P 500, NASDAQ, DAX, etc.), US stocks (AAPL, MSFT, TSLA, etc.), \
macroeconomics, interest-rate cycles, inflation metrics, central-bank policy.
- **Crypto leverage trading**: perpetual futures, funding rates, margin \
management, liquidation mechanics, risk-reward sizing, hedging strategies.
- **Technical analysis**: chart patterns, indicators (RSI, MACD, Bollinger \
Bands, Volume Profile, Ichimoku, etc.), order-flow, market structure, \
support/resistance, Fibonacci, Elliott Wave.
- **Risk management**: position sizing, stop-loss strategies, portfolio \
allocation, drawdown management, risk-of-ruin calculations.
- **Trading strategy development**: backtesting, journaling, edge \
identification, probability-based thinking, expectancy.

## Rules you MUST follow

1. **Educational only** – By default you do NOT have access to live market data. \
If someone asks for a current price, live chart, or real-time data AND no \
live token data block is provided below their message, politely explain: \
"I don't have access to live prices – we have other tools in the group for \
that! But I'd love to help you think through strategy, analysis, or risk \
management on that asset. What specifically are you working on?"\
\
HOWEVER: If the message includes a data block starting with \
"=== LIVE TOKEN DATA ===" or "=== LIVE TRADINGVIEW TECHNICAL ANALYSIS ===" \
then you DO have real data to work with. \
In that case, NEVER say you lack live data. NEVER mention the data block \
name to the user — just analyze the data directly using Rule 9 or Rule 10.

2. **Strategy-focused** – Always steer conversations toward actionable \
learning: refining strategy, improving risk management, understanding \
market structure, building good habits, etc.

3. **Helpful & concise** – Respect that this is a group chat. Keep answers \
clear, well-structured, and CONCISE. Aim for 150-300 words max. Use \
bullet points and numbered lists when it helps readability. Do NOT write \
essays. Get to the point quickly.

4. **Friendly tone** – Be approachable, supportive, and encouraging. \
Never condescending. You're a knowledgeable friend, not a lecturer.

5. **No financial advice** – You provide education and general information \
only. Remind users that nothing you say constitutes financial advice and \
they should always do their own research (DYOR).

6. **SPECIAL RULE – Trading Psychology**: If a user's question touches on \
psychology, mindset, discipline, emotional control, FOMO, fear, greed, \
revenge trading, tilt, burnout, confidence, or any other mental/emotional \
aspect of trading, give a full, helpful answer. Then, at the VERY END of \
your reply, add EXACTLY this paragraph on a new line:

"🧠 For a quick psychology check, type /psychology in the group to take \
our dedicated quiz bot's 10-question test and get your score out of 10!"

7. **Stay on topic** – If someone asks something completely unrelated to \
trading, investing, finance, or crypto, gently redirect: "That's a bit \
outside my wheelhouse! I'm here to help with trading, crypto, and market \
strategy questions. What can I help you with on that front?"\
\
HOWEVER: Questions about market hours, trading schedules, holidays \
(bank holidays, Easter, Christmas, etc.), session times, or whether \
markets are open/closed on specific days are ALWAYS on-topic. These \
are essential trading logistics. Answer them directly and helpfully. \
You know standard market hours: NYSE/NASDAQ 9:30-16:00 ET (Mon-Fri), \
Forex 24/5 (Sun 17:00-Fri 17:00 ET), crypto 24/7. You also know \
common market holidays (Good Friday, Easter Monday for UK/EU, \
US federal holidays, etc.). Answer confidently.

8. **Telegram HTML formatting** – You MUST format replies using Telegram HTML \
tags, NOT Markdown. Use these tags:\
  <b>bold</b> for emphasis\
  <i>italic</i> for secondary emphasis\
  <code>inline code</code> for technical terms\
  <pre>code blocks</pre> for multi-line code\
  DO NOT use *, **, _, __, `, ```, ##, or any Markdown syntax.\
  Use <b> and <i> tags instead of asterisks and underscores.\
  Use line breaks and <b>bold headers</b> to structure answers.\
  Bullet points with • or numbered lists are fine as plain text.

9. **Chart / Token Analysis — DEGEN FOCUS (IMPORTANT)** – When the message \
contains a "=== LIVE TOKEN DATA ===" block, you MUST use that data to provide \
a real analysis. This data is fetched live from DexScreener at the moment \
of the question. Most tokens shared here are degen/micro-cap plays. \
NEVER mention "LIVE TOKEN DATA", "data block", "wick analysis", or any \
internal mechanism to the user — just analyze the data directly. \
If the block is present, it means we successfully looked up their token. \
You MUST focus on these KEY METRICS for degen plays:\
  • <b>Market Cap</b>: State it clearly. Then estimate realistic X potential — \
    e.g. "At $500K mcap, a 10x to $5M is possible if volume sustains, but \
    50x+ would need major catalyst." Compare to similar tokens that have run.\
  • <b>Volume</b>: Is 24h volume healthy vs mcap? (>30% of mcap = hot, \
    <5% = dead). Buy:sell ratio — who's in control?\
  • <b>Liquidity</b>: Is it sufficient? Low liq + high mcap = dangerous.\
  • <b>Holder Count</b>: If holder data is provided, analyze distribution. \
    <200 holders = extremely early/risky. 500-2000 = growing. 5000+ = established.\
  • <b>1-Minute Wick Analysis</b>: If a WICK ANALYSIS section is provided, \
    use it! Consecutive upper wicks on the 1m = sellers taking profit / \
    distribution in progress. This is a KEY red flag for entries. \
    Consecutive lower wicks = buyers absorbing dips (accumulation). \
    Report what the wick data shows and what it means for the trade.\
  • <b>X Potential Estimate</b>: ALWAYS give a realistic X estimate from \
    current mcap with reasoning. Consider: volume trend, liquidity depth, \
    how old the token is (pair created date), and momentum.\
  • Highlight rug-pull red flags: low liq, mcap/fdv mismatch, dying volume.\
  NEVER say "I don't have access to live data" when this block is present.\
  Remind them this is a point-in-time snapshot, not live monitoring, and DYOR.\
  CRITICAL: Your entire response MUST be under 900 characters. A chart image \
  is shown alongside your text, so keep it tight — bullet points, no filler.

10. **TradingView Technical Analysis (IMPORTANT)** – When the message contains a \
"=== LIVE TRADINGVIEW TECHNICAL ANALYSIS ===" block, you MUST provide a \
detailed technical analysis using the real indicator data provided. This data \
is fetched live from TradingView at the moment of the question. A live chart \
image IS attached alongside your text reply — the user can see the candles. \
The price in the data block is the canonical source of truth (the chart may \
show a slightly different tick due to streaming delay — ignore that). \
You MUST:\
  • State the asset, timeframe, and current price (from the data block) right away\
  • Interpret the overall recommendation (BUY/SELL/NEUTRAL) and signal counts\
  • Analyze RSI: overbought (>70), oversold (<30), or neutral territory\
  • Analyze MACD: bullish/bearish crossover, histogram direction\
  • Analyze moving averages: price vs EMA/SMA 20/50/200, golden/death cross\
  • Note Bollinger Band position (near upper = overbought, near lower = oversold)\
  • Note ADX for trend strength (>25 = strong trend, <20 = ranging)\
  • Mention key pivot levels as support/resistance\
  • Summarize with 2-3 actionable takeaways\
  Keep your response concise — the chart image provides the visual context, \
  so focus your text on interpreting the numbers, not describing the chart.\
  NEVER say "I don't have access to live data" when this block is present.\
  This is a real-time snapshot, remind them to monitor for updates and DYOR.\
  CRITICAL: Your entire response MUST be under 900 characters. A chart image \
  is shown alongside your text, so keep it tight — bullet points, no filler.
"""


# ── DexScreener integration (free API, no key needed) ───────────────────────

# Regex to match DexScreener URLs and extract chain + pair address
# Supports both EVM (0x...) and Solana/base58 addresses in URLs
DEXSCREENER_URL_PATTERN = re.compile(
    r'https?://(?:www\.)?dexscreener\.com/([\w-]+)/([a-zA-Z0-9]{20,})',
    re.IGNORECASE,
)

# ── Raw contract address detection ──────────────────────────────────────────
# EVM: 0x followed by 40 hex chars
CONTRACT_EVM_PATTERN = re.compile(r'\b(0x[a-fA-F0-9]{40})\b')
# Solana/base58: 32-44 chars of base58 alphabet (no 0, O, I, l)
# Negative lookbehind/ahead for URL chars to avoid matching inside URLs
CONTRACT_SOL_PATTERN = re.compile(
    r'(?<![/\w])([1-9A-HJ-NP-Za-km-z]{32,44})(?![/\w])'
)


def extract_dexscreener_urls(text: str) -> list[tuple[str, str]]:
    """Extract (chain, pair_address) tuples from DexScreener URLs in text."""
    return DEXSCREENER_URL_PATTERN.findall(text)


def extract_contract_address(text: str) -> str | None:
    """
    Extract a raw contract address from the text (not inside a URL).
    Returns the address string, or None if not found.
    Checks for EVM (0x...) and Solana (base58) addresses.
    """
    # Remove DexScreener URLs first to avoid double-matching
    cleaned = DEXSCREENER_URL_PATTERN.sub('', text)

    # Check EVM first (unambiguous thanks to 0x prefix)
    m = CONTRACT_EVM_PATTERN.search(cleaned)
    if m:
        return m.group(1)

    # Check Solana-style base58 — require mixed case to avoid false positives
    m = CONTRACT_SOL_PATTERN.search(cleaned)
    if m:
        candidate = m.group(1)
        has_upper = any(c.isupper() for c in candidate)
        has_lower = any(c.islower() for c in candidate)
        has_digit = any(c.isdigit() for c in candidate)
        if (has_upper and has_lower) or (has_digit and (has_upper or has_lower)):
            return candidate

    return None


def has_question_with_ca(text: str) -> bool:
    """
    Return True if the text contains BOTH a contract address AND
    meaningful question/request text (not just a bare CA).
    E.g. 'TA on this please: 0xabc...' → True
         '0xabc...' → False (bare CA, no question)
    """
    ca = extract_contract_address(text)
    if not ca:
        return False
    # Remove the CA from the text and see if anything meaningful remains
    remaining = text.replace(ca, '').strip()
    # Strip common punctuation/colons left behind
    remaining = re.sub(r'^[:\s,;-]+|[:\s,;-]+$', '', remaining).strip()
    # Need at least 2 words of actual question/request text
    words = [w for w in remaining.split() if len(w) > 1]
    return len(words) >= 2


async def fetch_dexscreener_data(chain: str, pair_address: str) -> dict | None:
    """
    Fetch token pair data from DexScreener's free public API.
    Returns parsed JSON dict or None on failure.
    """
    url = f"https://api.dexscreener.com/latest/dex/pairs/{chain}/{pair_address}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            if data.get("pairs") and len(data["pairs"]) > 0:
                return data["pairs"][0]
            # Try the token endpoint as fallback
            url2 = f"https://api.dexscreener.com/latest/dex/tokens/{pair_address}"
            resp2 = await client.get(url2)
            resp2.raise_for_status()
            data2 = resp2.json()
            if data2.get("pairs") and len(data2["pairs"]) > 0:
                return data2["pairs"][0]
            return None
    except Exception as exc:
        logger.warning("DexScreener API error for %s/%s: %s", chain, pair_address, exc)
        return None


async def fetch_dexscreener_by_token(token_address: str) -> tuple[dict | None, str | None, str | None]:
    """
    Look up a raw contract address on DexScreener's token endpoint.
    Returns (pair_data, chain, pair_address) or (None, None, None).
    Picks the highest-liquidity pair when multiple exist.
    """
    url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            pairs = data.get("pairs") or []
            if not pairs:
                return None, None, None
            # Pick the highest-liquidity pair
            best = max(pairs, key=lambda p: (p.get("liquidity", {}).get("usd") or 0))
            chain_id = best.get("chainId", "unknown")
            pair_addr = best.get("pairAddress", token_address)
            return best, chain_id, pair_addr
    except Exception as exc:
        logger.warning("DexScreener token lookup error for %s: %s", token_address, exc)
        return None, None, None


def format_dexscreener_context(pair: dict) -> str:
    """
    Format DexScreener pair data into a readable context block
    that gets injected into the AI prompt.
    """
    base = pair.get("baseToken", {})
    quote = pair.get("quoteToken", {})
    price_change = pair.get("priceChange", {})
    volume = pair.get("volume", {})
    liquidity = pair.get("liquidity", {})
    txns = pair.get("txns", {})

    # Format transaction data
    txn_lines = []
    for period, label in [("m5", "5min"), ("h1", "1hr"), ("h6", "6hr"), ("h24", "24hr")]:
        t = txns.get(period, {})
        if t:
            txn_lines.append(f"  {label}: {t.get('buys', '?')} buys / {t.get('sells', '?')} sells")

    lines = [
        f"=== LIVE TOKEN DATA (from DexScreener) ===",
        f"IMPORTANT: This is REAL live data fetched just now. Analyze it using Rule 9.",
        f"Token: {base.get('name', '?')} ({base.get('symbol', '?')})",
        f"Pair: {base.get('symbol', '?')}/{quote.get('symbol', '?')}",
        f"Chain: {pair.get('chainId', '?')}",
        f"DEX: {pair.get('dexId', '?')}",
        f"Price (USD): ${pair.get('priceUsd', '?')}",
        f"Price ({quote.get('symbol', 'quote')}): {pair.get('priceNative', '?')}",
        f"",
        f"Price Changes:",
        f"  5min:  {price_change.get('m5', '?')}%",
        f"  1hr:   {price_change.get('h1', '?')}%",
        f"  6hr:   {price_change.get('h6', '?')}%",
        f"  24hr:  {price_change.get('h24', '?')}%",
        f"",
        f"Volume:",
        f"  5min:  ${volume.get('m5', '?')}",
        f"  1hr:   ${volume.get('h1', '?')}",
        f"  6hr:   ${volume.get('h6', '?')}",
        f"  24hr:  ${volume.get('h24', '?')}",
        f"",
        f"Transactions:",
    ]
    lines.extend(txn_lines if txn_lines else ["  No transaction data available"])
    lines.extend([
        f"",
        f"Liquidity (USD): ${liquidity.get('usd', '?')}",
        f"Market Cap: ${pair.get('marketCap', '?')}" if pair.get('marketCap') else "Market Cap: N/A",
        f"FDV: ${pair.get('fdv', '?')}" if pair.get('fdv') else "FDV: N/A",
        f"Pair created: {pair.get('pairCreatedAt', 'unknown')}",
        f"=========================",
    ])
    return "\n".join(lines)

# ── Response cache (saves rate limits for repeated questions) ────────────────

class LRUCache:
    """Simple LRU cache to avoid re-querying AI for identical/similar questions."""

    def __init__(self, max_size: int = 50, ttl_seconds: int = 3600):
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds

    @staticmethod
    def _key(text: str) -> str:
        """Normalize and hash the question for cache lookup."""
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, question: str) -> str | None:
        key = self._key(question)
        if key in self._cache:
            answer, ts = self._cache[key]
            if time.time() - ts < self._ttl:
                self._cache.move_to_end(key)
                return answer
            else:
                del self._cache[key]
        return None

    def put(self, question: str, answer: str):
        key = self._key(question)
        self._cache[key] = (answer, time.time())
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


response_cache = LRUCache(max_size=100, ttl_seconds=1800)  # 30 min TTL

# ── Psychology detection ─────────────────────────────────────────────────────

PSYCHOLOGY_KEYWORDS = re.compile(
    r"\b("
    r"psychology|mindset|discipline|disciplined|emotional|emotions|emotion|"
    r"fomo|fear|greed|revenge.?trad|tilt(?:ed|ing)?|burnout|burnt?.?out|"
    r"confidence|self.?doubt|anxiety|anxious|panic|impatien[ct]|"
    r"overtrading|over.?trad|impulsi(?:ve|vity)|hesitat|"
    r"mental(?:ity|ly)?|stress(?:ed|ful)?|frustrat|cope|coping|"
    r"bias(?:es)?|cognitive|gut.?feeling|intuition"
    r")\b",
    re.IGNORECASE,
)

PSYCHOLOGY_FOOTER = (
    "\n\n🧠 For a quick psychology check, type /psychology in the group to take "
    "our dedicated quiz bot's 10-question test and get your score out of 10!"
)


def sanitize_for_html(text: str) -> str:
    """
    Aggressive Markdown → Telegram-HTML converter.
    Converts what it can, then *strips* any surviving formatting markers
    so raw asterisks / underscores never reach the user.
    """
    import html as html_mod
    import uuid

    # ── 0. Bullet-point asterisks → • (before italic regex eats them) ──
    text = re.sub(r'^\s*\*\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*-\s+', '• ', text, flags=re.MULTILINE)

    # ── 1. Stash fenced code blocks ────────────────────────────────────
    code_blocks: dict[str, str] = {}

    def stash_code_block(m):
        key = f'__CB_{uuid.uuid4().hex[:8]}__'
        code_blocks[key] = f'<pre>{html_mod.escape(m.group(2))}</pre>'
        return key

    text = re.sub(r'```(?:\w*)\n?([\s\S]*?)```', stash_code_block, text)

    # ── 2. Stash inline code `…` ──────────────────────────────────────
    inline_codes: dict[str, str] = {}

    def stash_inline_code(m):
        key = f'__IC_{uuid.uuid4().hex[:8]}__'
        inline_codes[key] = f'<code>{html_mod.escape(m.group(1))}</code>'
        return key

    text = re.sub(r'`([^`\n]+)`', stash_inline_code, text)

    # ── 3. Preserve existing valid HTML tags ───────────────────────────
    placeholders: dict[str, str] = {}
    allowed_tags = ['b', 'i', 'code', 'pre', 'u', 's', 'a']
    tag_pattern = re.compile(
        r'(</?(?:' + '|'.join(allowed_tags) + r')(?:\s[^>]*)?>)',
        re.IGNORECASE,
    )

    def save_tag(m):
        key = f'__TG_{uuid.uuid4().hex[:8]}__'
        placeholders[key] = m.group(0)
        return key

    text = tag_pattern.sub(save_tag, text)

    # ── 4. Escape HTML-special characters ──────────────────────────────
    text = html_mod.escape(text)

    # ── 5. Restore all stashed content ─────────────────────────────────
    for key, val in placeholders.items():
        text = text.replace(html_mod.escape(key), val)
    for key, val in code_blocks.items():
        text = text.replace(html_mod.escape(key), val)
    for key, val in inline_codes.items():
        text = text.replace(html_mod.escape(key), val)

    # ── 6. Markdown → HTML (single-line only, safe patterns) ──────────
    # Headers  ### text
    text = re.sub(r'^#{1,3}\s*(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    # Bold-italic ***text***
    text = re.sub(r'\*{3}(.+?)\*{3}', r'<b><i>\1</i></b>', text)
    # Bold **text**  (single-line only — no DOTALL!)
    text = re.sub(r'\*{2}(.+?)\*{2}', r'<b>\1</b>', text)
    # Italic *text* (single-line, no asterisks inside)
    text = re.sub(r'(?<!\w)\*([^\n*]+?)\*(?!\w)', r'<i>\1</i>', text)
    # Bold __text__
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
    # Italic _text_ (single-line, no underscores inside)
    text = re.sub(r'(?<!\w)_([^\n_]+?)_(?!\w)', r'<i>\1</i>', text)
    # Links [text](url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)

    # ── 7. NUCLEAR CLEANUP: strip any surviving formatting markers ─────
    #   This ensures NO raw asterisks or underscores from Markdown ever
    #   reach the user. Targets only formatting positions, not math (5 * 3).
    #
    #   Opening markers:  whitespace/start + ***/** /*  + word-char
    text = re.sub(r'(?<=\s)\*{1,3}(?=\w)', '', text)
    text = re.sub(r'^\*{1,3}(?=\w)', '', text, flags=re.MULTILINE)
    #   Closing markers:  word-char/punct + ***/** /*  + whitespace/punct/end
    text = re.sub(r'(?<=[\w.,;:!?)])\*{1,3}(?=[\s.,;:!?)\]\}]|$)', '', text, flags=re.MULTILINE)
    #   Standalone lone asterisks on a line
    text = re.sub(r'^\*{1,3}\s*$', '', text, flags=re.MULTILINE)

    #   Same for underscores used as formatting (opening/closing)
    text = re.sub(r'(?<=\s)_{1,2}(?=\w)', '', text)
    text = re.sub(r'^_{1,2}(?=\w)', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?<=[\w.,;:!?)])_{1,2}(?=[\s.,;:!?)\]\}]|$)', '', text, flags=re.MULTILINE)

    return text


def mentions_psychology(text: str) -> bool:
    """Return True if the message touches on trading psychology topics."""
    return bool(PSYCHOLOGY_KEYWORDS.search(text))


# ── On-topic filter (rejects off-topic BEFORE hitting AI = free) ─────────────

ON_TOPIC_KEYWORDS = re.compile(
    r"\b("
    # Crypto
    r"crypt|bitcoin|btc|ethereum|eth|solana|sol|altcoin|token|defi|nft|"
    r"blockchain|web3|wallet|staking|yield|airdrop|dex|cex|swap|liquidity|"
    r"memecoin|meme.?coin|shitcoin|microcap|micro.?cap|lowcap|low.?cap|"
    r"midcap|mid.?cap|largecap|large.?cap|hodl|whale|pump|dump|rug.?pull|"
    r"tokenomics|smart.?contract|layer.?[12]|l[12]|bridge|chain|gas.?fee|"
    r"mining|halving|satoshi|gwei|binance|coinbase|kraken|bybit|okx|"
    r"uniswap|pancakeswap|raydium|jupiter|phantom|metamask|ledger|"
    r"usdt|usdc|stablecoin|pepe|doge|shib|xrp|ada|avax|matic|bnb|"
    r"link|dot|atom|near|apt|sui|arb|op|ftm|sei|inj|tia|jup|wif|bonk|"
    r"onchain|on.?chain|tvl|mcap|market.?cap|volume|pumpfun|pump\.fun|"
    # Trading general
    r"trad(?:e|ing|er)|forex|fx|commodit|indices|index|futures|options|"
    r"stock|equit|share|market|bull(?:ish)?|bear(?:ish)?|long|short|"
    r"leverage|margin|liquidat|perpetual|perp|spot|position|order|"
    r"limit.?order|market.?order|stop.?loss|take.?profit|tp|sl|"
    r"entry|exit|breakout|breakdown|pullback|retracement|reversal|"
    r"support|resistance|trend|channel|range|consolidat|accumul|distribut|"
    r"swing|scalp|day.?trad|intraday|timeframe|time.?frame|"
    # Technical analysis
    r"technical.?analy|\bta\b|chart|candle|candlestick|pattern|indicator|"
    r"rsi|macd|bollinger|moving.?average|ema|sma|vwap|volume.?profile|"
    r"fibonacci|fib|elliott|ichimoku|stochastic|atr|obv|divergen|"
    r"overbought|oversold|golden.?cross|death.?cross|"
    r"head.?and.?shoulder|double.?top|double.?bottom|triangle|wedge|flag|"
    r"pennant|cup.?and.?handle|gap|wick|doji|hammer|engulf|"
    r"order.?flow|order.?book|bid|ask|spread|slippage|"
    # Macro / fundamentals
    r"macro|econom|inflat|deflat|interest.?rate|fed|federal.?reserve|"
    r"central.?bank|gdp|cpi|ppi|employment|payroll|recession|"
    r"monetary|fiscal|quantitative|tapering|yield.?curve|bond|treasur|"
    r"dollar|dxy|eur|gbp|jpy|aud|cad|chf|nzd|gold|silver|oil|crude|"
    r"natural.?gas|wheat|corn|copper|platinum|palladium|"
    r"s&p|s.p.500|nasdaq|dow|russell|dax|ftse|nikkei|hang.?seng|"
    # Popular stock tickers (on-topic for TA requests)
    r"aapl|msft|googl|goog|amzn|tsla|nvda|meta|nflx|amd|intc|"
    r"pltr|pypl|coin|hood|jpm|dis|ba|ko|jnj|gme|amc|nio|baba|"
    # Risk & strategy
    r"risk|reward|r:r|rr|risk.?reward|position.?siz|portfolio|"
    r"diversif|hedge|correlat|drawdown|bankroll|capital|"
    r"backtest|journal|edge|expectancy|win.?rate|loss.?rate|"
    r"strategy|setup|plan|system|method|approach|"
    # Psychology (also on-topic)
    r"psychology|mindset|discipline|emotional|fomo|fear|greed|"
    r"revenge.?trad|tilt|burnout|confidence|stress|frustrat|"
    r"bias|cognitive|overtrading|impulsiv|"
    # General finance
    r"invest|profit|loss|pnl|p&l|roi|return|compound|dca|"
    r"dollar.?cost|buy|sell|accumulate|allocat|rebalanc|"
    r"fundamentals|valuation|earnings|revenue|"
    # Market schedule / holidays (on-topic — traders need to know)
    r"market.?open|market.?close|market.?hour|trading.?hour|session|"
    r"pre.?market|after.?hour|holiday|bank.?holiday|easter|christmas|"
    r"new.?year|thanks.?giving|memorial.?day|labor.?day|independence|"
    r"good.?friday|monday|tuesday|wednesday|thursday|friday|"
    r"weekend|tomorrow|today|open.?tomorrow|closed.?tomorrow|"
    r"nyse|london|tokyo|sydney|asian.?session|london.?session|"
    r"us.?session|new.?york.?session|exchange.?hours|"
    # Common question patterns about the above
    r"price.?action|pa|money.?management|mm|"
    r"technical|analysis|signal|setup|confluenc"
    r")\b",
    re.IGNORECASE,
)

# Short messages (< 4 words) that are greetings / noise — skip topic check
GREETING_PATTERN = re.compile(
    r"^(hi|hey|hello|yo|sup|thanks|thank you|gm|gn|wb|cheers|ok|okay)\s*[!?.]*$",
    re.IGNORECASE,
)

OFF_TOPIC_REPLY = (
    "Hey! I only answer questions about <b>crypto</b>, <b>trading</b>, <b>forex</b>, "
    "<b>macro</b>, and related topics like <b>technical analysis</b>, <b>risk management</b>, "
    "and <b>trading psychology</b>. 📊\n\n"
    "Try asking me something like:\n"
    "• <i>How do I manage risk on a leveraged trade?</i>\n"
    "• <i>What's a good strategy for swing trading altcoins?</i>\n"
    "• <i>How does the RSI indicator work?</i>"
)


def is_on_topic(text: str) -> bool:
    """
    Return True if the message is related to trading/crypto/finance.
    Very short messages (greetings) are treated as on-topic to avoid
    false-rejecting things like 'thanks' or 'hi'.
    DexScreener URLs and contract addresses are always on-topic.
    """
    cleaned = text.strip()
    # DexScreener / chart URLs are always on-topic
    if DEXSCREENER_URL_PATTERN.search(cleaned):
        return True
    # Raw contract addresses are always on-topic
    if extract_contract_address(cleaned):
        return True
    # Let very short messages through (greetings, follow-ups)
    if len(cleaned.split()) <= 3:
        return True
    # If it matches any on-topic keyword, it's good
    return bool(ON_TOPIC_KEYWORDS.search(cleaned))


# ── Helpers ──────────────────────────────────────────────────────────────────

def strip_mention(text: str, username: str) -> str:
    """Remove the @username mention from the message text."""
    pattern = re.compile(rf"@{re.escape(username)}\b", re.IGNORECASE)
    return pattern.sub("", text).strip()


def is_mention(update: Update, username: str) -> bool:
    """Check whether the bot was @mentioned in the message."""
    msg = update.effective_message
    if not msg or not msg.text:
        return False

    if msg.entities:
        for entity in msg.entities:
            if entity.type == "mention":
                mentioned = msg.text[entity.offset : entity.offset + entity.length]
                if mentioned.lower() == f"@{username.lower()}":
                    return True

    if f"@{username.lower()}" in msg.text.lower():
        return True

    return False


def smart_split(text: str, limit: int = 4096) -> list[str]:
    """Split text into chunks that respect paragraph boundaries."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        idx = text.rfind("\n\n", 0, limit)
        if idx == -1:
            idx = text.rfind("\n", 0, limit)
        if idx == -1:
            idx = text.rfind(" ", 0, limit)
        if idx == -1:
            idx = limit

        chunks.append(text[:idx])
        text = text[idx:].lstrip("\n")

    return chunks


# ── Core handler ─────────────────────────────────────────────────────────────

async def _safe_reply_text(update: Update, text: str, **kwargs):
    """Reply to a message, falling back to send_message if the original is gone."""
    try:
        await update.effective_message.reply_text(text, **kwargs)
    except Exception:
        try:
            await update.effective_chat.send_message(text, **kwargs)
        except Exception as exc:
            logger.warning("Failed to send message: %s", exc)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process group messages that @mention the bot."""
    if not is_mention(update, BOT_USERNAME):
        return

    user_text = strip_mention(update.effective_message.text, BOT_USERNAME)

    if not user_text:
        await _safe_reply_text(
            update,
            "Hey! Tag me with a question and I'll do my best to help. 💡"
        )
        return

    user_id = update.effective_user.id
    user_name = update.effective_user.first_name or "trader"
    logger.info("Question from %s (id=%s): %s", user_name, user_id, user_text[:120])

    # ── Off-topic filter (zero API cost!) ──
    if not is_on_topic(user_text):
        logger.info("Off-topic from %s — rejected locally (0 API calls used)", user_name)
        await _safe_reply_text(update, OFF_TOPIC_REPLY, parse_mode="HTML")
        return

    # ── Detect live-data requests (skip cache for these) ──
    has_dex_url = bool(extract_dexscreener_urls(user_text))
    raw_ca = extract_contract_address(user_text) if has_question_with_ca(user_text) else None
    has_raw_ca = bool(raw_ca)
    tv_symbol = extract_symbol(user_text)
    has_live_data = has_dex_url or has_raw_ca or (tv_symbol and is_ta_request(user_text))

    if has_live_data:
        logger.info("Live-data request detected — cache bypassed (dex=%s, ca=%s, tv=%s)",
                     has_dex_url, raw_ca,
                     tv_symbol.display_name if tv_symbol else None)

    # ── Check cache first (costs zero rate limit!) ──
    # Skip cache for live-data requests (DexScreener, TradingView)
    if not has_live_data:
        cached = response_cache.get(user_text)
        if cached:
            logger.info("Cache hit — serving cached response (no rate limit used)")
            await _send_reply(update, cached, context=context)
            return

    # ── Rate limit check (only for non-cached / actual API calls) ──
    allowed, denial_msg = rate_limiter.check(user_id, user_name)
    if not allowed:
        logger.info("Rate-limited %s (id=%s): %s", user_name, user_id, denial_msg)
        await _safe_reply_text(update, denial_msg)
        return

    # ── Show typing indicator ──
    chat_action = "upload_photo" if has_live_data else "typing"
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=chat_action
    )

    # ── Check for DexScreener URLs OR raw contract addresses → fetch live data ──
    dex_context = ""
    chain = pair_addr = None
    token_name = "TOKEN"
    dex_urls = extract_dexscreener_urls(user_text) if has_dex_url else []
    if dex_urls:
        chain, pair_addr = dex_urls[0]  # Use first URL found
        logger.info("DexScreener URL detected: %s/%s — fetching data", chain, pair_addr)
        pair_data = await fetch_dexscreener_data(chain, pair_addr)
        if pair_data:
            dex_context = "\n\n" + format_dexscreener_context(pair_data)
            token_name = pair_data.get("baseToken", {}).get("symbol", "token")
            logger.info("DexScreener data fetched for %s (%d bytes)",
                       token_name, len(dex_context))
        else:
            dex_context = ("\n\n[Note: User shared a DexScreener link but "
                          "the API returned no data for this pair. "
                          "Acknowledge the link and offer general guidance.]")
            logger.warning("DexScreener returned no data for %s/%s", chain, pair_addr)
    elif raw_ca:
        logger.info("Raw contract address detected: %s — looking up on DexScreener", raw_ca)
        pair_data, chain, pair_addr = await fetch_dexscreener_by_token(raw_ca)
        if pair_data:
            dex_context = "\n\n" + format_dexscreener_context(pair_data)
            token_name = pair_data.get("baseToken", {}).get("symbol", "token")
            logger.info("DexScreener data fetched for %s via token lookup (%d bytes)",
                       token_name, len(dex_context))
        else:
            dex_context = (f"\n\n[Note: User shared contract address {raw_ca} but "
                          f"DexScreener returned no data. This could mean the token "
                          f"is very new, unlisted, or the address is invalid. "
                          f"Let the user know the token wasn't found on DexScreener "
                          f"and suggest they double-check the address or try again later.]")
            logger.warning("DexScreener returned no data for token %s", raw_ca)

    # ── Fetch 1-minute wick analysis for degen tokens (parallel-safe) ──
    wick_context = ""
    if chain and pair_addr and dex_context:
        try:
            wick_data = await analyze_1m_wicks(chain, pair_addr)
            if wick_data:
                wick_context = "\n\n" + wick_data
                logger.info("Wick analysis attached (%d bytes)", len(wick_context))
        except Exception as exc:
            logger.warning("Wick analysis error (non-fatal): %s", exc)

    # ── Check for TradingView TA request → fetch live indicators + chart ──
    tv_context = ""
    interval = None
    chart_image = None       # will hold PNG bytes if we capture a chart
    chart_price = None       # live price scraped from the chart widget
    if tv_symbol and is_ta_request(user_text):
        interval = extract_interval(user_text)
        logger.info("TradingView TA request: %s (%s)", tv_symbol.display_name, interval)

        # Fetch indicators + chart screenshot in parallel.
        # We need the chart_price BEFORE building the AI prompt,
        # so the screenshot must finish before the AI call starts.
        import asyncio as _aio
        tv_data_result, chart_result = await _aio.gather(
            _aio.get_event_loop().run_in_executor(
                None, fetch_tradingview_ta, tv_symbol, interval
            ),
            screenshot_tradingview_chart(
                tv_symbol.symbol, tv_symbol.exchange, interval
            ),
            return_exceptions=True,
        )

        # Unpack TA data
        tv_data = tv_data_result if not isinstance(tv_data_result, Exception) else None
        if isinstance(tv_data_result, Exception):
            logger.warning("TradingView TA fetch error: %s", tv_data_result)

        # Unpack chart screenshot + price
        if isinstance(chart_result, Exception):
            logger.warning("Chart screenshot error: %s", chart_result)
        else:
            chart_image, chart_price = chart_result

        if tv_data:
            tv_context = "\n\n" + format_tradingview_context(
                tv_symbol, interval, tv_data, chart_price=chart_price
            )
            logger.info("TradingView data fetched for %s (%d bytes, chart_price=%s)",
                       tv_symbol.display_name, len(tv_context), chart_price)
        else:
            tv_context = (f"\n\n[Note: User asked for TA on {tv_symbol.display_name} "
                         f"but TradingView returned no data. Acknowledge the request "
                         f"and provide general TA guidance.]")
            logger.warning("TradingView returned no data for %s", tv_symbol.symbol)

    # ── Query AI with multi-provider fallback ──
    live_context = dex_context + wick_context + tv_context

    prompt = f"[Group member {user_name} asks]: {user_text}{live_context}"

    try:
        has_tv_chart = bool(tv_symbol and is_ta_request(user_text) and interval)
        has_dex_chart = bool((has_dex_url or has_raw_ca) and chain and pair_addr)
        has_chart = has_tv_chart or (has_dex_chart and not chart_image)
        tokens = 500 if has_chart else (1200 if live_context else 800)

        # ── DEX chart (mplfinance from GeckoTerminal API) + AI call in parallel ──
        if has_dex_chart and not chart_image:
            # Extract timeframe from user text for the chart (default 1h)
            dex_interval = extract_interval(user_text) if not interval else interval
            dex_token_sym = token_name
            dex_chart_coro = generate_dex_chart(
                chain, pair_addr, interval=dex_interval, token_symbol=dex_token_sym
            )
            results = await asyncio.gather(
                provider_mgr.generate(
                    system_prompt=SYSTEM_PROMPT,
                    user_message=prompt,
                    max_tokens=tokens,
                ),
                dex_chart_coro,
                return_exceptions=True,
            )
            if isinstance(results[0], Exception):
                raise results[0]
            answer, provider_name = results[0]
            if isinstance(results[1], Exception):
                logger.warning("DEX chart generation failed: %s", results[1])
            else:
                chart_image = results[1]
        else:
            answer, provider_name = await provider_mgr.generate(
                system_prompt=SYSTEM_PROMPT,
                user_message=prompt,
                max_tokens=tokens,
            )

        logger.info("Response from %s (%d chars)", provider_name, len(answer))

        # Record successful API call for rate limiting
        rate_limiter.record(user_id)

        # Append psychology footer if needed
        if mentions_psychology(user_text) and "/psychology" not in answer:
            answer += PSYCHOLOGY_FOOTER

        # Sanitize Markdown → HTML *before* caching so cache always stores clean HTML
        answer = sanitize_for_html(answer)

        # Cache the sanitized response (skip for live-data — always fetch fresh)
        if not has_live_data:
            response_cache.put(user_text, answer)

        # ── Send response (chart + text combined if available) ──
        if chart_image:
            await _send_reply(update, answer, chart_image=chart_image, context=context)
        else:
            await _send_reply(update, answer, context=context)

    except Exception as exc:
        logger.exception("All providers failed: %s", exc)
        await _safe_reply_text(
            update,
            "⚠️ I'm having a temporary issue. Please try again in a moment!"
        )


async def _send_reply(
    update: Update,
    answer: str,
    chart_image: bytes | None = None,
    context: ContextTypes.DEFAULT_TYPE | None = None,
):
    """
    Send the answer, optionally combined with a chart image.

    Telegram photo captions are limited to 1024 chars.
    If the text exceeds 1024, hard-truncate at the last complete line/sentence.
    No overflow messages — the AI is instructed to keep chart responses short.

    Falls back to send_message/send_photo (without reply-to) if the original
    message is gone (e.g. deleted, or stale update from deploy restart).
    """
    CAPTION_LIMIT = 1024
    chat_id = update.effective_chat.id
    bot = context.bot if context else update.get_bot()

    if chart_image:
        photo = BytesIO(chart_image)
        photo.name = "chart.png"

        caption = answer
        if len(caption) > CAPTION_LIMIT:
            # Hard truncate at last clean break
            truncated = caption[:CAPTION_LIMIT]
            brk = truncated.rfind("\n")
            if brk < CAPTION_LIMIT // 2:
                brk = truncated.rfind(". ")
            if brk < CAPTION_LIMIT // 3:
                brk = CAPTION_LIMIT - 3
            caption = truncated[:brk].rstrip() + "…"

        # Try reply_photo → send_photo (no reply-to) → send plain text
        for attempt, use_html in [(1, True), (2, True), (3, False), (4, False)]:
            try:
                clean_cap = caption if use_html else re.sub(r'<[^>]+>', '', caption)
                pm = "HTML" if use_html else None
                photo.seek(0)
                if attempt <= 2:
                    try:
                        await update.effective_message.reply_photo(
                            photo=photo, caption=clean_cap, parse_mode=pm,
                        )
                        return
                    except Exception:
                        if attempt == 1:
                            continue  # retry as send_photo
                        raise
                else:
                    await bot.send_photo(
                        chat_id=chat_id, photo=photo,
                        caption=clean_cap, parse_mode=pm,
                    )
                    return
            except Exception as exc:
                if attempt < 4:
                    continue
                # Last resort: send as plain text
                logger.warning("Photo send failed entirely: %s", exc)
                clean = re.sub(r'<[^>]+>', '', caption)
                try:
                    await bot.send_message(chat_id=chat_id, text=clean)
                except Exception:
                    logger.error("Could not send response at all")
        return

    # No chart image — plain text
    chunks = smart_split(answer, 4096)
    for chunk in chunks:
        try:
            await update.effective_message.reply_text(chunk, parse_mode="HTML")
        except Exception:
            clean = re.sub(r'<[^>]+>', '', chunk)
            try:
                await update.effective_message.reply_text(clean)
            except Exception:
                # Original message gone — send without reply-to
                try:
                    await bot.send_message(
                        chat_id=chat_id, text=clean,
                    )
                except Exception as exc:
                    logger.error("Could not send chunk: %s", exc)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    logger.info("Starting TB bot (@%s) …", BOT_USERNAME)
    logger.info("Available AI providers: %s", provider_mgr.list_providers())

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    app.add_handler(
        MessageHandler(
            filters.TEXT & (filters.ChatType.GROUP | filters.ChatType.SUPERGROUP),
            handle_message,
        )
    )

    # Use webhook mode on Railway (has RAILWAY_PUBLIC_DOMAIN),
    # fall back to polling for local development.
    if WEBHOOK_URL:
        webhook_full = f"https://{WEBHOOK_URL}/webhook"
        logger.info("Starting in WEBHOOK mode → %s", webhook_full)
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path="/webhook",
            webhook_url=webhook_full,
            drop_pending_updates=True,
        )
    else:
        logger.info("No WEBHOOK_URL set — starting in POLLING mode (local dev)")
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
