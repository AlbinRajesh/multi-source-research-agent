"""
Cost estimation from token counts. Rates are per-1M-tokens, matching how
most providers publish pricing. Ollama is $0 (local inference).

⚠️ PRICING_TABLE below has a placeholder for the Groq model rate —
verify against https://groq.com/pricing before relying on cost output.
"""
import logging

logger = logging.getLogger(__name__)

# (input_$_per_1M, output_$_per_1M)
PRICING_TABLE = {
    "openai/gpt-oss-20b": (0.10, 0.30),  # PLACEHOLDER — confirm against Groq's current pricing page
    "qwen2.5:3b-instruct": (0.0, 0.0),   # Ollama, local — always free
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICING_TABLE:
        logger.warning(f"[cost] no pricing entry for model={model!r}, assuming $0")
        return 0.0

    in_rate, out_rate = PRICING_TABLE[model]
    cost = (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate
    return round(cost, 6)


def cost_breakdown_from_summary(token_summary: dict) -> dict:
    """Takes TokenTracker.summary() output, returns per-node + total cost."""
    breakdown = {}
    total_cost = 0.0

    for key, node_data in token_summary.get("by_node", {}).items():
        model = node_data["model"]
        cost = estimate_cost(model, node_data["input_tokens"], node_data["output_tokens"])
        breakdown[key] = {**node_data, "estimated_cost_usd": cost}
        total_cost += cost

    return {
        "by_node": breakdown,
        "total_estimated_cost_usd": round(total_cost, 6),
    }