"""Token-based rate limiter for Groq's free-tier TPM (tokens-per-minute) cap.

Groq's 429 responses report a hard TPM limit (seen at 8000 for
openai/gpt-oss-20b) — this is a token budget over a rolling 60s window,
not a request-count limit, so a simple requests-per-second limiter
(e.g. langchain_core's InMemoryRateLimiter) doesn't model the actual
constraint. This tracks estimated token usage in a sliding window and
makes callers wait only as long as needed to stay under budget, instead
of the SDK's blind fixed-backoff-then-retry (which is what's currently
costing 10-15s per run across verify/synthesize).

Token counts aren't known before a call completes, so this uses a cheap
chars/4 estimate for the pre-call reservation, then corrects the ledger
with the real usage once the response comes back (if usage metadata is
available) — the estimate only needs to be good enough to avoid tripping
the actual 429, not exact.
"""
import asyncio
import itertools
import logging
import time
from collections import deque
from typing import Deque, List

logger = logging.getLogger(__name__)

_reservation_ids = itertools.count()


class TokenRateLimiter:
    """Sliding-window token budget limiter. One instance per provider/model
    that has a TPM cap — Groq free tier in this codebase's case."""

    def __init__(self, tpm_limit: int, window_seconds: float = 60.0, safety_margin: float = 0.9):
        # safety_margin: reserve a little headroom below the hard limit so
        # estimation error (chars/4 is approximate) doesn't still trip a 429.
        self.budget = int(tpm_limit * safety_margin)
        self.window_seconds = window_seconds
        # [timestamp, tokens, reservation_id] entries for everything counted
        # in the current window. Lists (not tuples) so record_actual can
        # mutate the tokens field of a specific entry in place.
        self._usage: Deque[List] = deque()
        self._lock = asyncio.Lock()

    def _prune(self, now: float) -> int:
        """Drop entries older than the window, return current window's token total."""
        cutoff = now - self.window_seconds
        while self._usage and self._usage[0][0] < cutoff:
            self._usage.popleft()
        return sum(tokens for _, tokens, _ in self._usage)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        # Rough chars/4 heuristic — good enough for reservation purposes;
        # corrected against real usage after the call via record_actual().
        return max(1, len(text) // 4)

    async def acquire(self, estimated_tokens: int) -> int:
        """Block until there's room in the current window for
        estimated_tokens. Reserves that amount immediately on return so
        concurrent callers don't all pass the check before any of them
        record usage. Returns a reservation_id — pass it to record_actual()
        so the correction lands on the right entry even when multiple
        calls are in flight at once (e.g. verifier.py's max_concurrent=2).
        Without id-tagging, concurrent calls corrected whichever entry
        happened to be last in the deque, which could belong to a
        different in-flight call — the ledger's estimates never got
        corrected against real usage and stayed inflated, causing the
        budget to look perpetually over-full and forcing near-full-window
        (~60s) waits even when actual usage was well under budget.
        """
        async with self._lock:
            while True:
                now = time.monotonic()
                used = self._prune(now)
                if used + estimated_tokens <= self.budget:
                    reservation_id = next(_reservation_ids)
                    self._usage.append([now, estimated_tokens, reservation_id])
                    return reservation_id

                # Not enough room — wait until the oldest entry ages out
                # of the window, then re-check.
                oldest_ts = self._usage[0][0]
                wait_for = (oldest_ts + self.window_seconds) - now
                wait_for = max(wait_for, 0.1)
                logger.info(
                    f"[token_rate_limiter] budget full ({used}/{self.budget} tokens "
                    f"in window), waiting {wait_for:.1f}s"
                )
                # Release the lock while sleeping so this isn't a busy-wait
                # that blocks other coroutines from checking/queueing too.
                self._lock.release()
                try:
                    await asyncio.sleep(wait_for)
                finally:
                    await self._lock.acquire()

    def record_actual(self, reservation_id: int, actual_tokens: int) -> None:
        """Correct the ledger once real usage is known (e.g. from response
        usage metadata). Finds the exact entry by reservation_id instead
        of assuming it's the last one in the deque — required for
        correctness under concurrent in-flight calls."""
        for entry in self._usage:
            if entry[2] == reservation_id:
                entry[1] = max(0, actual_tokens)
                return
        # Entry already aged out of the window (rare, harmless) — nothing
        # to correct, its tokens no longer count toward the budget anyway.


# Singleton for Groq's free-tier TPM cap. If you're on a paid tier with a
# higher limit, update tpm_limit here (or move it into config.py and read
# from there — kept as a literal for now since 8000 is what the 429s
# reported).
groq_token_limiter = TokenRateLimiter(tpm_limit=8000)