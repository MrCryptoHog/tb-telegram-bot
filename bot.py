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

# ── Configuration ────────────────────────────────────────────────────────────

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BOT_USERNAME = os.getenv("BOT_USERNAME", "HeyTB")  # without the @

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

8. **Telegram formatting** – Use Telegram-compatible Markdown \
(bold with *, italic with _, code with `, etc.). Do NOT use headers \
with ## or ### or complex markdown that Telegram won't render. Use bold \
text and line breaks to structure your answers instead.
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


def mentions_psychology(text: str) -> bool:
    """Return True if the message touches on trading psychology topics."""
    return bool(PSYCHOLOGY_KEYWORDS.search(text))


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

    user_name = update.effective_user.first_name or "trader"
    logger.info("Question from %s: %s", user_name, user_text[:120])

    # ── Check cache first (saves rate limits!) ──
    cached = response_cache.get(user_text)
    if cached:
        logger.info("Cache hit — serving cached response")
        await _send_reply(update, cached)
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

        # Append psychology footer if needed
        if mentions_psychology(user_text) and "/psychology" not in answer:
            answer += PSYCHOLOGY_FOOTER

        # Cache the response
        response_cache.put(user_text, answer)

        await _send_reply(update, answer)

    except Exception as exc:
        logger.exception("All providers failed: %s", exc)
        await update.effective_message.reply_text(
            "⚠️ I'm having a temporary issue. Please try again in a moment!"
        )


async def _send_reply(update: Update, answer: str):
    """Send the answer, splitting if needed, with Markdown fallback."""
    chunks = smart_split(answer, 4096)
    for chunk in chunks:
        try:
            await update.effective_message.reply_text(chunk, parse_mode="Markdown")
        except Exception:
            # If Markdown parsing fails, send as plain text
            await update.effective_message.reply_text(chunk)


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

    logger.info("TB is live and listening for @mentions. Let's go! 🚀")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
