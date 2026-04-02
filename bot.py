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
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)

from providers import ProviderManager
from rate_limiter import RateLimiter

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
(S&P 500, NASDAQ, DAX, etc.), macroeconomics, interest-rate cycles, \
inflation metrics, central-bank policy.
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

1. **Educational only** – You do NOT have access to live market data. \
If someone asks for a current price, live chart, or real-time data, \
politely explain: "I don't have access to live prices – we have other \
tools in the group for that! But I'd love to help you think through \
strategy, analysis, or risk management on that asset. What specifically \
are you working on?"

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
strategy questions. What can I help you with on that front?"

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
"""

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
    Bullet-proof Markdown → Telegram-HTML converter.
    Handles: bullet lists, ```code blocks```, `inline code`, ***bold-italic***,
    **bold**, *italic*, _italic_, __bold__, ###headers, [links](url).
    Also escapes bare HTML so Telegram's parser never chokes.
    """
    import html as html_mod
    import uuid

    # ── 0. Convert bullet-point asterisks BEFORE anything else ──────────
    #   "* item"  →  "• item"   (prevents confusion with italic markers)
    text = re.sub(r'^\s*\*\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*-\s+', '• ', text, flags=re.MULTILINE)

    # ── 1. Stash fenced code blocks so nothing inside gets mangled ──────
    code_blocks: dict[str, str] = {}

    def stash_code_block(m):
        key = f'__CODEBLOCK_{uuid.uuid4().hex[:8]}__'
        # Keep content, escape HTML inside it
        code_blocks[key] = f'<pre>{html_mod.escape(m.group(2))}</pre>'
        return key

    text = re.sub(r'```(\w*)\n?([\s\S]*?)```', stash_code_block, text)

    # ── 2. Preserve any existing valid HTML tags ────────────────────────
    placeholders: dict[str, str] = {}
    allowed_tags = ['b', 'i', 'code', 'pre', 'u', 's', 'a']
    tag_pattern = re.compile(
        r'(</?(?:' + '|'.join(allowed_tags) + r')(?:\s[^>]*)?>)',
        re.IGNORECASE,
    )

    def save_tag(m):
        key = f'__TAG_{uuid.uuid4().hex[:8]}__'
        placeholders[key] = m.group(0)
        return key

    text = tag_pattern.sub(save_tag, text)

    # ── 3. Escape HTML-special characters (&, <, >) ────────────────────
    text = html_mod.escape(text)

    # ── 4. Restore stashed tags & code blocks ──────────────────────────
    for key, tag in placeholders.items():
        text = text.replace(html_mod.escape(key), tag)
    for key, block in code_blocks.items():
        text = text.replace(html_mod.escape(key), block)

    # ── 5. Markdown → HTML conversions (order matters!) ────────────────
    # Inline code  `code`
    text = re.sub(r'`([^`\n]+)`', r'<code>\1</code>', text)
    # Bold-italic  ***text***
    text = re.sub(r'\*{3}(.+?)\*{3}', r'<b><i>\1</i></b>', text, flags=re.DOTALL)
    # Bold  **text**
    text = re.sub(r'\*{2}(.+?)\*{2}', r'<b>\1</b>', text, flags=re.DOTALL)
    # Italic  *text*  (only within a single line; [^\n*] avoids crossing bullets)
    text = re.sub(r'(?<!\w)\*([^\n*]+?)\*(?!\w)', r'<i>\1</i>', text)
    # Italic  _text_
    text = re.sub(r'(?<!\w)_([^\n_]+?)_(?!\w)', r'<i>\1</i>', text)
    # Bold  __text__
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text, flags=re.DOTALL)
    # Headers  ### text
    text = re.sub(r'^#{1,3}\s*(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    # Links  [text](url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)

    # ── 6. Last-resort cleanup: strip stray lone formatting asterisks ──
    #   A line that is JUST a stray "*" or "**"
    text = re.sub(r'^\*{1,3}\s*$', '', text, flags=re.MULTILINE)

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
    r"technical.?analy|chart|candle|candlestick|pattern|indicator|"
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
    """
    cleaned = text.strip()
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

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process group messages that @mention the bot."""
    if not is_mention(update, BOT_USERNAME):
        return

    user_text = strip_mention(update.effective_message.text, BOT_USERNAME)

    if not user_text:
        await update.effective_message.reply_text(
            "Hey! Tag me with a question and I'll do my best to help. 💡"
        )
        return

    user_id = update.effective_user.id
    user_name = update.effective_user.first_name or "trader"
    logger.info("Question from %s (id=%s): %s", user_name, user_id, user_text[:120])

    # ── Off-topic filter (zero API cost!) ──
    if not is_on_topic(user_text):
        logger.info("Off-topic from %s — rejected locally (0 API calls used)", user_name)
        await update.effective_message.reply_text(OFF_TOPIC_REPLY, parse_mode="HTML")
        return

    # ── Check cache first (costs zero rate limit!) ──
    cached = response_cache.get(user_text)
    if cached:
        logger.info("Cache hit — serving cached response (no rate limit used)")
        await _send_reply(update, cached)
        return

    # ── Rate limit check (only for non-cached / actual API calls) ──
    allowed, denial_msg = rate_limiter.check(user_id, user_name)
    if not allowed:
        logger.info("Rate-limited %s (id=%s): %s", user_name, user_id, denial_msg)
        await update.effective_message.reply_text(denial_msg)
        return

    # ── Show typing indicator ──
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    # ── Query AI with multi-provider fallback ──
    prompt = f"[Group member {user_name} asks]: {user_text}"

    try:
        answer, provider_name = await provider_mgr.generate(
            system_prompt=SYSTEM_PROMPT,
            user_message=prompt,
            max_tokens=800,  # Keep concise to save rate limits
        )

        logger.info("Response from %s (%d chars)", provider_name, len(answer))

        # Record successful API call for rate limiting
        rate_limiter.record(user_id)

        # Append psychology footer if needed
        if mentions_psychology(user_text) and "/psychology" not in answer:
            answer += PSYCHOLOGY_FOOTER

        # Sanitize Markdown → HTML *before* caching so cache always stores clean HTML
        answer = sanitize_for_html(answer)

        # Cache the sanitized response
        response_cache.put(user_text, answer)

        await _send_reply(update, answer)

    except Exception as exc:
        logger.exception("All providers failed: %s", exc)
        await update.effective_message.reply_text(
            "⚠️ I'm having a temporary issue. Please try again in a moment!"
        )


async def _send_reply(update: Update, answer: str):
    """Send the answer, splitting if needed, with HTML fallback to plain text."""
    chunks = smart_split(answer, 4096)
    for chunk in chunks:
        try:
            await update.effective_message.reply_text(chunk, parse_mode="HTML")
        except Exception:
            # If HTML parsing fails, strip tags and send as plain text
            clean = re.sub(r'<[^>]+>', '', chunk)
            await update.effective_message.reply_text(clean)


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
