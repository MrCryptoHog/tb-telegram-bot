"""
Multi-provider AI manager with intelligent fallback and rate-limit awareness.

Priority order:
  1. Groq       – Llama 3.3 70B  (fastest, generous free tier)
  2. Gemini     – gemini-2.0-flash (Google free tier)
  3. Cerebras   – Llama 3.3 70B  (fast, free tier)
  4. Mistral    – mistral-small-latest (free tier)
  5. SambaNova  – Llama 3.1 70B  (free tier)

Each provider has a cooldown tracker: if it returns a rate-limit error,
it's temporarily disabled for a configurable back-off period before
being tried again. This spreads load across providers automatically.
"""

import os
import time
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger("TB.providers")

# ── Base provider ────────────────────────────────────────────────────────────

@dataclass
class RateLimitState:
    """Tracks per-provider rate-limit cooldowns."""
    blocked_until: float = 0.0        # unix timestamp
    consecutive_errors: int = 0
    base_cooldown: float = 60.0       # seconds

    def mark_error(self):
        self.consecutive_errors += 1
        backoff = min(self.base_cooldown * (2 ** (self.consecutive_errors - 1)), 600)
        self.blocked_until = time.time() + backoff
        logger.warning("Provider cooldown for %.0fs (consecutive errors: %d)",
                       backoff, self.consecutive_errors)

    def mark_success(self):
        self.consecutive_errors = 0
        self.blocked_until = 0.0

    @property
    def is_available(self) -> bool:
        return time.time() >= self.blocked_until


class AIProvider(ABC):
    """Base class for all AI providers."""

    def __init__(self, name: str, api_key: str | None):
        self.name = name
        self.api_key = api_key
        self.rate_limit = RateLimitState()
        self._client: httpx.AsyncClient | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def is_available(self) -> bool:
        return self.is_configured and self.rate_limit.is_available

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    @abstractmethod
    async def _call_api(self, system_prompt: str, user_message: str,
                        max_tokens: int) -> str:
        ...

    async def generate(self, system_prompt: str, user_message: str,
                       max_tokens: int) -> str:
        """Call the provider. Raises on failure."""
        try:
            result = await self._call_api(system_prompt, user_message, max_tokens)
            self.rate_limit.mark_success()
            return result
        except Exception as exc:
            self.rate_limit.mark_error()
            raise


# ── Groq ─────────────────────────────────────────────────────────────────────

class GroqProvider(AIProvider):
    """Groq – OpenAI-compatible API, Llama 3.3 70B."""

    def __init__(self):
        super().__init__("Groq", os.getenv("GROQ_API_KEY"))
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile"

    async def _call_api(self, system_prompt, user_message, max_tokens):
        client = await self._get_client()
        resp = await client.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
        )
        if resp.status_code == 429:
            raise RateLimitError(f"Groq 429: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ── Gemini ───────────────────────────────────────────────────────────────────

class GeminiProvider(AIProvider):
    """Google Gemini – REST API, gemini-2.0-flash."""

    def __init__(self):
        super().__init__("Gemini", os.getenv("GEMINI_API_KEY"))
        self.model = "gemini-2.0-flash"

    async def _call_api(self, system_prompt, user_message, max_tokens):
        client = await self._get_client()
        url = (f"https://generativelanguage.googleapis.com/v1beta/"
               f"models/{self.model}:generateContent?key={self.api_key}")
        resp = await client.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"parts": [{"text": user_message}]}],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.7,
                },
            },
        )
        if resp.status_code == 429:
            raise RateLimitError(f"Gemini 429: {resp.text[:200]}")
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


# ── Cerebras ─────────────────────────────────────────────────────────────────

class CerebrasProvider(AIProvider):
    """Cerebras – OpenAI-compatible API, Llama 3.3 70B."""

    def __init__(self):
        super().__init__("Cerebras", os.getenv("CEREBRAS_API_KEY"))
        self.base_url = "https://api.cerebras.ai/v1/chat/completions"
        self.model = "llama-3.3-70b"

    async def _call_api(self, system_prompt, user_message, max_tokens):
        client = await self._get_client()
        resp = await client.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
        )
        if resp.status_code == 429:
            raise RateLimitError(f"Cerebras 429: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ── Mistral ──────────────────────────────────────────────────────────────────

class MistralProvider(AIProvider):
    """Mistral – OpenAI-compatible API, mistral-small-latest."""

    def __init__(self):
        super().__init__("Mistral", os.getenv("MISTRAL_API_KEY"))
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        self.model = "mistral-small-latest"

    async def _call_api(self, system_prompt, user_message, max_tokens):
        client = await self._get_client()
        resp = await client.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
        )
        if resp.status_code == 429:
            raise RateLimitError(f"Mistral 429: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ── SambaNova ────────────────────────────────────────────────────────────────

class SambaNovaProvider(AIProvider):
    """SambaNova – OpenAI-compatible API, Llama 3.1 70B."""

    def __init__(self):
        super().__init__("SambaNova", os.getenv("SAMBANOVA_API_KEY"))
        self.base_url = "https://api.sambanova.ai/v1/chat/completions"
        self.model = "Meta-Llama-3.1-70B-Instruct"

    async def _call_api(self, system_prompt, user_message, max_tokens):
        client = await self._get_client()
        resp = await client.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
        )
        if resp.status_code == 429:
            raise RateLimitError(f"SambaNova 429: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ── Custom exceptions ────────────────────────────────────────────────────────

class RateLimitError(Exception):
    pass


class AllProvidersExhausted(Exception):
    pass


# ── Provider Manager ─────────────────────────────────────────────────────────

class ProviderManager:
    """Manages multiple AI providers with automatic fallback."""

    def __init__(self):
        # Priority order – cheapest / fastest first
        self.providers: list[AIProvider] = [
            GroqProvider(),
            GeminiProvider(),
            CerebrasProvider(),
            MistralProvider(),
            SambaNovaProvider(),
        ]

        configured = [p.name for p in self.providers if p.is_configured]
        if not configured:
            raise RuntimeError(
                "No AI provider API keys found! Set at least one of: "
                "GROQ_API_KEY, GEMINI_API_KEY, CEREBRAS_API_KEY, "
                "MISTRAL_API_KEY, SAMBANOVA_API_KEY"
            )
        logger.info("Configured AI providers: %s", ", ".join(configured))

    def list_providers(self) -> str:
        """Human-readable list of configured providers and their status."""
        parts = []
        for p in self.providers:
            if p.is_configured:
                status = "✅" if p.is_available else "⏳ cooldown"
                parts.append(f"{p.name} ({status})")
        return ", ".join(parts) if parts else "none"

    async def generate(self, system_prompt: str, user_message: str,
                       max_tokens: int = 800) -> tuple[str, str]:
        """
        Try each provider in priority order. Returns (answer, provider_name).
        Raises AllProvidersExhausted if every provider fails.
        """
        errors: list[str] = []

        for provider in self.providers:
            if not provider.is_available:
                logger.debug("Skipping %s (unavailable)", provider.name)
                continue

            try:
                logger.info("Trying %s …", provider.name)
                answer = await provider.generate(system_prompt, user_message,
                                                  max_tokens)
                return answer, provider.name

            except RateLimitError as e:
                logger.warning("%s rate-limited: %s", provider.name, e)
                errors.append(f"{provider.name}: rate-limited")
                continue

            except Exception as e:
                logger.warning("%s failed: %s", provider.name, e)
                errors.append(f"{provider.name}: {e}")
                continue

        raise AllProvidersExhausted(
            f"All providers exhausted. Errors: {'; '.join(errors)}"
        )
