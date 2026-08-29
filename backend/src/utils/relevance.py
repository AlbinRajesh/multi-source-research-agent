"""Topic-relevance filter for extracted claims — cross-encoder reranker,
rank-based selection (not absolute threshold).

Runs between extract_claims and verify, BEFORE the expensive Groq
verification call — this cuts Groq's input volume, it doesn't just
discard results after paying for them.

Why rank-based, not a fixed score cutoff:
Absolute thresholds on cross-encoder raw logits proved unstable across
runs — the same topic produced score ranges anywhere from -11/+8 to
-11/+9 depending on the document mix, so a fixed cutoff sometimes kept
8/22 claims and sometimes collapsed a batch to 0/10, cutting real
biographical facts along with genuine trivia. Relative ranking within
a batch stays meaningful even when the absolute scale drifts, which is
the standard fix for this exact instability (percentile/top-K selection
instead of absolute threshold).

keep_ratio / min_survivors tuning (raised from 0.5 / 3):
Evaluation runs showed the specific correct answer to a lookup question
(a named CEO, score -10.00) getting cut even with a min_survivors floor
in place — the floor only prevents a batch collapsing to zero, it does
not protect one specific important-but-lower-scoring claim from being
outranked by several more generically topic-relevant claims within a
0.5 keep_ratio cut. Raising keep_ratio to 0.65 and min_survivors to 5
keeps more borderline claims through to verification, which is cheap
insurance — the verifier will still reject genuinely irrelevant ones,
whereas the relevance filter dropping a correct claim is unrecoverable.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_model = None  # lazy singleton — load once per process, not per call

def _get_model():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading relevance reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 on {device}")
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    return _model


def filter_by_relevance(
    claims: List,
    topic: str,
    keep_ratio: float = 0.65,
    min_survivors: int = 5,
    global_budget: Optional[int] = None,
    min_per_source: int = 1,
) -> Tuple[List, List[float]]:
    if not claims:
        return [], []

    model = _get_model()
    pairs = [[topic, c.text] for c in claims]
    scores = model.predict(pairs).tolist()

    # tie-break: nudge high-value claim types slightly when scores are close
    TYPE_BOOST = {"stat": 0.15, "role": 0.15, "date": 0.1, "event": 0.05, "other": 0.0}
    adjusted = [
        (c, s, s + TYPE_BOOST.get(getattr(c, "claim_type", "other"), 0.0))
        for c, s in zip(claims, scores)
    ]
    ranked = sorted(adjusted, key=lambda x: x[2], reverse=True)

    n = len(ranked)
    keep_count = max(min_survivors, round(n * keep_ratio))
    keep_count = min(keep_count, n)

    kept = ranked[:keep_count]
    dropped = ranked[keep_count:]

    # apply global budget on top of the ratio cut
    if global_budget is not None and len(kept) > global_budget:
        dropped = kept[global_budget:] + dropped
        kept = kept[:global_budget]

    # per-source floor: don't let a strong source crowd out every other source entirely
    kept_sources = {c.source_url for c, _, _ in kept}
    all_sources = {c.source_url for c, _, _ in ranked}
    missing_sources = all_sources - kept_sources

    if missing_sources:
        best_dropped_per_source = {}
        for c, s, adj in dropped:
            if c.source_url in missing_sources and c.source_url not in best_dropped_per_source:
                best_dropped_per_source[c.source_url] = (c, s, adj)
        rescued = list(best_dropped_per_source.values())
        if rescued:
            logger.info(f"[relevance] rescued {len(rescued)} claim(s) to satisfy per-source floor")
            kept = kept + rescued

    for claim, score, _ in dropped:
        if claim not in [c for c, _, _ in kept]:
            logger.info(f"[relevance] dropped (score={score:.2f}): {claim.text[:80]}")

    kept_claims = [c for c, _, _ in kept]
    logger.info(
        f"[relevance] kept {len(kept_claims)}/{n} "
        f"(keep_ratio={keep_ratio}, min_survivors={min_survivors}, "
        f"global_budget={global_budget}, min_per_source={min_per_source})"
    )

    return kept_claims, scores
def warmup() -> None:
    """Force the relevance reranker model to load once. Call at server startup."""
    _get_model()
    logger.info("Relevance reranker warmed up.")