"""Topic-relevance filter for extracted claims — embedding-based, no LLM call.

Runs between extract_claims and verify. Uses GPU if available, falls back
to CPU automatically so this doesn't crash on machines without CUDA.
"""
import logging
from typing import List, Tuple
import torch
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

_model = None  # lazy singleton — load once per process, not per call

def _get_model():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading relevance model: all-MiniLM-L6-v2 on {device}")
        _model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return _model


def filter_by_relevance(claims: List, topic: str, min_score: float = 0.35) -> Tuple[List, List[float]]:
    if not claims:
        return [], []

    model = _get_model()
    topic_emb = model.encode(topic, convert_to_tensor=True)
    claim_embs = model.encode([c.text for c in claims], convert_to_tensor=True)

    scores = util.cos_sim(topic_emb, claim_embs)[0].tolist()

    kept = []
    for claim, score in zip(claims, scores):
        if score >= min_score:
            kept.append(claim)
        else:
            logger.info(f"[relevance] dropped (score={score:.2f}): {claim.text[:80]}")

    return kept, scores