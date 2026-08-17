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
from typing import List, Tuple
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
) -> Tuple[List, List[float]]:
    """
    claims: list of Claim objects (must have .text)
    keep_ratio: fraction of the batch to keep, ranked by relevance score
        (e.g. 0.65 = keep the top 65%). Robust to score-scale drift
        since it's relative to the batch, not an absolute cutoff.
    min_survivors: safety-net floor — if keep_ratio would leave fewer
        than this many claims, keep this many instead (top-N by rank).
        Prevents a batch collapsing to near-zero survivors, which
        previously forced wasted retry cycles or empty synthesis.
    Returns (kept_claims, all_scores) — all_scores is for distribution logging.
    """
    if not claims:
        return [], []

    model = _get_model()
    pairs = [[topic, c.text] for c in claims]
    scores = model.predict(pairs).tolist()

    # rank claims by score, descending
    ranked = sorted(zip(claims, scores), key=lambda x: x[1], reverse=True)

    n = len(ranked)
    keep_count = max(min_survivors, round(n * keep_ratio))
    keep_count = min(keep_count, n)  # never try to keep more than exist

    kept_pairs = ranked[:keep_count]
    dropped_pairs = ranked[keep_count:]

    for claim, score in dropped_pairs:
        logger.info(f"[relevance] dropped (score={score:.2f}, rank cutoff): {claim.text[:80]}")

    kept = [c for c, _ in kept_pairs]
    logger.info(
        f"[relevance] kept {len(kept)}/{n} (keep_ratio={keep_ratio}, "
        f"min_survivors={min_survivors}, score range kept: "
        f"{kept_pairs[-1][1]:.2f} to {kept_pairs[0][1]:.2f})" if kept_pairs else
        f"[relevance] kept 0/{n}"
    )

    return kept, scores