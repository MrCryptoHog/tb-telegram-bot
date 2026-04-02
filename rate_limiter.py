"""
Smart rate limiter for TB bot.

Designed to keep combined free-tier API usage sustainable 24/7.

┌─────────────────────────────────────────────────────────────────┐
│  FREE-TIER DAILY BUDGET (conservative estimates)                │
│                                                                 │
│  Provider    │ RPM  │ Daily safe  │ Notes                       │
│  ────────────┼──────┼─────────────┼───────────────────────────  │
│  Groq        │  30  │   6,000     │ Llama 3.3 70B, most quota   │
│  Gemini      │  15  │   1,200     │ gemini-2.0-flash             │
│  Cerebras    │  30  │   1,000     │ Free tier daily cap          │
│  Mistral     │   2  │     400     │ Smallest free tier           │
│  SambaNova   │  10  │     100     │ Conservative estimate        │
│  ────────────┼──────┼─────────────┼───────────────────────────  │
│  TOTAL       │      │   8,700/day │ ≈ 362/hour                  │
│  + 30% cache │      │  12,400/day │ ≈ 517/hour effective        │
│                                                                 │
│  LIMITS SET:                                                    │
│  • Per user:  8 questions / 2 hours (96/day if someone maxes)   │
│  • Cooldown:  45 seconds between questions per user             │
│  • Global:    150 API calls / 2 hours (1,800/day max)           │
│  • Usage:     1,800 / 8,700 = 20.7% of daily budget            │
│  • Headroom:  ~79% spare for provider outages or spikes         │
└─────────────────────────────────────────────────────────────────┘
"""

import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("TB.ratelimit")


@dataclass
class UserRecord:
    """Tracks a single user's request timestamps within the current window."""
    timestamps: list[float] = field(default_factory=list)

    def prune(self, window_start: float):
        """Remove timestamps older than the window."""
        self.timestamps = [t for t in self.timestamps if t >= window_start]

    @property
    def count(self) -> int:
        return len(self.timestamps)

    @property
    def last(self) -> float:
        return self.timestamps[-1] if self.timestamps else 0.0


class RateLimiter:
    """
    Three-layer rate limiter:
      1. Per-user cooldown   – min seconds between consecutive questions
      2. Per-user window cap – max questions per rolling window
      3. Global window cap   – max API calls across ALL users per window
    """

    def __init__(
        self,
        user_max_per_window: int = 8,
        window_seconds: int = 7200,
        user_cooldown_seconds: int = 45,
        global_max_per_window: int = 150,
    ):
        self.user_max = user_max_per_window
        self.window = window_seconds
        self.cooldown = user_cooldown_seconds
        self.global_max = global_max_per_window

        self._users: dict[int, UserRecord] = {}
        self._global_timestamps: list[float] = []

    def _window_start(self) -> float:
        return time.time() - self.window

    def _prune_global(self):
        cutoff = self._window_start()
        self._global_timestamps = [t for t in self._global_timestamps if t >= cutoff]

    def _get_user(self, user_id: int) -> UserRecord:
        if user_id not in self._users:
            self._users[user_id] = UserRecord()
        rec = self._users[user_id]
        rec.prune(self._window_start())
        return rec

    def check(self, user_id: int, user_name: str = "") -> tuple[bool, str]:
        """
        Check if a request from this user is allowed.
        Returns (allowed: bool, message: str).
        On denial, message contains a friendly explanation.
        """
        now = time.time()
        user = self._get_user(user_id)
        self._prune_global()

        # Layer 1: Per-user cooldown (anti-spam)
        if user.last and (now - user.last) < self.cooldown:
            wait = int(self.cooldown - (now - user.last)) + 1
            return False, (
                f"⏳ Easy there! Please wait {wait} more second{'s' if wait != 1 else ''} "
                f"before your next question. This helps me stay available for everyone!"
            )

        # Layer 2: Per-user window cap
        if user.count >= self.user_max:
            # Find when the oldest request in the window expires
            oldest = min(user.timestamps)
            resets_in = int(oldest + self.window - now) + 1
            mins = resets_in // 60
            return False, (
                f"🔒 You've used all {self.user_max} questions for this 2-hour window. "
                f"Your next slot opens in ~{mins} minute{'s' if mins != 1 else ''}. "
                f"In the meantime, scroll up — there might be answers to similar "
                f"questions already in the chat!"
            )

        # Layer 3: Global window cap
        if len(self._global_timestamps) >= self.global_max:
            oldest_global = min(self._global_timestamps)
            resets_in = int(oldest_global + self.window - now) + 1
            mins = resets_in // 60
            return False, (
                f"🌐 The group has been really active! We've hit our 2-hour "
                f"question limit. Next slot opens in ~{mins} minute{'s' if mins != 1 else ''}. "
                f"Try again shortly!"
            )

        return True, ""

    def record(self, user_id: int):
        """Record a successful API call (call AFTER getting the AI response)."""
        now = time.time()
        user = self._get_user(user_id)
        user.timestamps.append(now)
        self._global_timestamps.append(now)

        logger.info(
            "Rate limit status — user %s: %d/%d this window | global: %d/%d",
            user_id, user.count, self.user_max,
            len(self._global_timestamps), self.global_max,
        )

    def get_status(self, user_id: int) -> dict:
        """Get current rate limit status for a user (for debugging/info)."""
        user = self._get_user(user_id)
        self._prune_global()
        return {
            "user_used": user.count,
            "user_max": self.user_max,
            "user_remaining": max(0, self.user_max - user.count),
            "global_used": len(self._global_timestamps),
            "global_max": self.global_max,
            "global_remaining": max(0, self.global_max - len(self._global_timestamps)),
            "cooldown_seconds": self.cooldown,
            "window_seconds": self.window,
        }
