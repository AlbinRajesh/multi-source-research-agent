"""
Semantic cache gate.
Before running the full agent pipeline, check if the incoming query is
close enough (cosine similarity) to a recently answered query. If so,
return the cached answer immediately — skips retrieval, reranking,
groundedness check, and generation entirely.

In-memory only (resets on server restart). Swap for Redis if you need
persistence across restarts or multiple worker processes.
"""

import logging
import time
import numpy as np

from retrieval.embeddings import embed_query

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.96
MAX_CACHE_SIZE = 200          # evict oldest entries beyond this
CACHE_TTL_SECONDS = 3600      # 1 hour — avoid serving stale answers forever

# Each entry: {"query": str, "embedding": np.ndarray, "answer": str,
#              "evidence": list | None, "timestamp": float}
_cache: list[dict] = []


def _prune():
    """Drop expired entries and enforce max size (oldest first)."""
    now = time.time()
    global _cache
    _cache = [e for e in _cache if now - e["timestamp"] < CACHE_TTL_SECONDS]
    if len(_cache) > MAX_CACHE_SIZE:
        _cache = _cache[-MAX_CACHE_SIZE:]


def check_cache(query: str) -> dict | None:
    """
    Return {"answer": str, "evidence": list | None} if a sufficiently
    similar query was answered recently, else None.
    """
    if not _cache:
        return None

    try:
        query_emb = embed_query(query)
    except Exception as e:
        logger.warning(f"[semantic_cache] embedding failed, skipping cache check: {e}")
        return None

    best_score, best_entry = -1.0, None
    for entry in _cache:
        score = float(np.dot(query_emb, entry["embedding"]))  # embeddings are pre-normalized
        if score > best_score:
            best_score, best_entry = score, entry

    if best_entry is not None and best_score >= SIMILARITY_THRESHOLD:
        logger.info(f"[semantic_cache] HIT (score={best_score:.4f}) for query={query!r} "
                    f"-> matched cached query={best_entry['query']!r}")
        
        return {"answer": best_entry["answer"], "evidence": best_entry.get("evidence")}

    logger.debug(f"[semantic_cache] MISS (best_score={best_score:.4f}) for query={query!r}")
    return None


def store(query: str, answer: str, evidence: list | None = None) -> None:
    """Cache a query/answer pair for future near-duplicate hits."""
    try:
        query_emb = embed_query(query)
    except Exception as e:
        logger.warning(f"[semantic_cache] embedding failed, not caching: {e}")
        return

    _cache.append({
        "query": query,
        "embedding": query_emb,
        "answer": answer,
        "evidence": evidence,
        "timestamp": time.time(),
    })
    _prune()
    logger.info(f"[semantic_cache] stored query={query!r} ({len(_cache)} entries)")