"""Per-run token accounting and sliding-window rate limiter.
Call record() after every LLM invocation with the node name and the raw usage 
object/dict from the provider response. Designed to be provider-agnostic 
across Groq and Ollama's OpenAI-compatible usage shape (prompt_tokens / completion_tokens).
"""
import logging
from dataclasses import dataclass, field
from typing import Optional
from langchain_core.callbacks.base import AsyncCallbackHandler
from metrics.token_rate_limiter import groq_token_limiter, TokenRateLimiter

logger = logging.getLogger(__name__)


@dataclass
class NodeUsage:
    node: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class TokenTracker:
    """One instance per research run — attach to state or pass through
    context, then call .summary() at the end for the response payload."""
    usage_by_node: dict = field(default_factory=dict)  # node -> NodeUsage

    def record(self, node: str, model: str, usage: Optional[object]) -> None:
        """
        usage: the .usage object/dict from an LLM response. Handles both
        attribute-style (OpenAI/Groq SDK objects) and dict-style access.
        """
        input_tokens, output_tokens = self._extract_tokens(usage)

        key = f"{node}:{model}"
        entry = self.usage_by_node.setdefault(key, NodeUsage(node=node, model=model))
        entry.input_tokens += input_tokens
        entry.output_tokens += output_tokens
        entry.calls += 1

        logger.info(
            f"[tokens] {node} ({model}): +{input_tokens} in / +{output_tokens} out "
            f"(running total: {entry.total_tokens})"
        )

    @staticmethod
    def _extract_tokens(usage) -> tuple[int, int]:
        if usage is None:
            return 0, 0
        if isinstance(usage, dict):
            return (
                usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0),
                usage.get("completion_tokens", 0) or usage.get("output_tokens", 0),
            )
        # attribute-style (Groq/OpenAI SDK response.usage)
        return (
            getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0,
        )

    def summary(self) -> dict:
        total_in = sum(e.input_tokens for e in self.usage_by_node.values())
        total_out = sum(e.output_tokens for e in self.usage_by_node.values())
        return {
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "by_node": {
                key: {
                    "model": e.model,
                    "calls": e.calls,
                    "input_tokens": e.input_tokens,
                    "output_tokens": e.output_tokens,
                    "total_tokens": e.total_tokens,
                }
                for key, e in self.usage_by_node.items()
            },
        }


class TokenTrackingCallback(AsyncCallbackHandler):
    """
    Attaches at chain.ainvoke(..., config={"callbacks": [...]}) time.
    Reads usage from the raw LLM response BEFORE StrOutputParser strips
    it down to a plain string — this must run as a callback, not be
    inferred from the chain's return value.
    """
    def __init__(self, tracker: Optional[TokenTracker], node: str, model: str, provider: str = "", reservation_id: Optional[int] = None):
        self.tracker = tracker
        self.node = node
        self.model = model
        self.provider = provider
        self.reservation_id = reservation_id

    async def on_llm_end(self, response, **kwargs) -> None:
        try:
            usage = None
            if response.llm_output:
                usage = response.llm_output.get("token_usage") or response.llm_output.get("usage")
            if usage is None and response.generations:
                # fallback: LangChain's own usage_metadata on the AIMessage
                msg = getattr(response.generations[0][0], "message", None)
                usage = getattr(msg, "usage_metadata", None)

            if self.tracker:
                self.tracker.record(self.node, self.model, usage)

            # Correct the rate limiter ledger with actual token counts if Groq was explicitly used
            if self.provider.lower() == "groq" and self.reservation_id is not None:
                actual_in, actual_out = TokenTracker._extract_tokens(usage)
                actual_total = actual_in + actual_out
                if actual_total > 0:
                    groq_token_limiter.record_actual(self.reservation_id, actual_total)
        except Exception as e:
            logger.warning(f"[tokens] failed to record usage for node={self.node}: {e}")


async def track_llm_call(chain, inputs: dict, tracker: Optional[TokenTracker], node: str, model: str, provider: str = ""):
    """
    Drop-in replacement for `await chain.ainvoke(inputs)` — rate limits Groq calls
    based on TPM sliding windows explicitly via provider matching, and records usage into `tracker` if provided.
    """
    # Estimate token count for pre-call rate limiting (using input dict serialization)
    input_str = str(inputs)
    estimated_tokens = TokenRateLimiter.estimate_tokens(input_str) + 300  # +300 buffer for expected output

    # Rate-limit only if the provider is explicitly set to groq. acquire()
    # returns a reservation_id so the eventual correction in on_llm_end
    # lands on THIS call's entry specifically, not whichever entry happens
    # to be last in the deque — required for correctness when multiple
    # calls are in flight concurrently (e.g. verifier.py's max_concurrent=2).
    reservation_id = None
    if provider.lower() == "groq":
        reservation_id = await groq_token_limiter.acquire(estimated_tokens)

    callback = TokenTrackingCallback(tracker, node, model, provider=provider, reservation_id=reservation_id)
    return await chain.ainvoke(inputs, config={"callbacks": [callback]})