"""Chunk-level relevance pre-filter — hybrid (dense + sparse), runs
BEFORE claim extraction to avoid paying LLM cost on off-topic chunks.

Uses reciprocal rank fusion (RRF) instead of combining raw scores directly,
because dense (cosine similarity) and sparse (BM25) scores live on
different, non-comparable scales — the same instability problem noted
in relevance.py for cross-encoder logits. Rank fusion sidesteps that:
it only cares about each chunk's relative position in each ranking,
not the raw score magnitude.
"""
import logging
import time
from typing import List, Tuple, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.rag.embeddings import embed_texts, embed_query, EmbeddingError
from src.rag.keyword_index import tokenize as _tokenize
from src.state import SearchResult

logger = logging.getLogger(__name__)

RRF_K = 60  # standard RRF constant — dampens the impact of rank 1 vs rank 2 at the top


def _rrf_scores(dense_ranks: dict, sparse_ranks: dict, n: int) -> List[float]:
    scores = []
    for i in range(n):
        d_rank = dense_ranks.get(i, n)   # missing = worst possible rank
        s_rank = sparse_ranks.get(i, n)
        scores.append(1.0 / (RRF_K + d_rank + 1) + 1.0 / (RRF_K + s_rank + 1))
    return scores


def _chunk_id(r: SearchResult) -> str:
    """Compact identifier for logging — avoids dumping full chunk text."""
    meta = getattr(r, "metadata", None) or {}
    if meta.get("doc_id"):
        return f"local://{meta['doc_id']}/{meta.get('chunk_index', '?')}"
    return getattr(r, "url", "unknown")


def filter_chunks_by_relevance(
    results: List[SearchResult],
    topic: str,
    keep_ratio: float = 0.7,
    min_survivors: int = 4,
    global_budget: Optional[int] = None,
) -> Tuple[List[SearchResult], List[float]]:
    n = len(results)
    if n <= min_survivors:
        # Nothing meaningful to filter — too few chunks to bother scoring.
        return results, [1.0] * n

    texts = [r.content or r.snippet or "" for r in results]

    # --- Dense (semantic) ranking ---
    t0 = time.perf_counter()
    try:
        topic_vec = embed_query(topic)
        t1 = time.perf_counter()
        chunk_vecs = embed_texts(texts)
        t2 = time.perf_counter()
        dense_scores = chunk_vecs @ topic_vec  # cosine sim, vectors already normalized
        logger.info(
            f"[chunk_relevance_timing] embed_query={t1-t0:.3f}s "
            f"embed_texts={t2-t1:.3f}s"
        )
    except EmbeddingError as e:
        logger.warning(f"[chunk_relevance] dense scoring failed, falling back to sparse-only: {e}")
        dense_scores = np.zeros(n)

    dense_order = np.argsort(-dense_scores)
    dense_ranks = {int(idx): rank for rank, idx in enumerate(dense_order)}

    # --- Sparse (keyword) ranking — ephemeral BM25 over THIS batch only,
    # not the persisted document index (this is candidate re-ranking, not retrieval). ---
    t3 = time.perf_counter()
    tokenized = [_tokenize(t) for t in texts]
    try:
        bm25 = BM25Okapi(tokenized)
        sparse_scores = bm25.get_scores(_tokenize(topic))
        logger.info(f"[chunk_relevance_timing] bm25={time.perf_counter()-t3:.3f}s")
    except Exception as e:
        logger.warning(f"[chunk_relevance] sparse scoring failed, falling back to dense-only: {e}")
        sparse_scores = np.zeros(n)

    sparse_order = np.argsort(-np.array(sparse_scores))
    sparse_ranks = {int(idx): rank for rank, idx in enumerate(sparse_order)}

    fused = _rrf_scores(dense_ranks, sparse_ranks, n)

    ranked = sorted(zip(results, fused), key=lambda x: x[1], reverse=True)

    keep_count = max(min_survivors, round(n * keep_ratio))
    keep_count = min(keep_count, n)
    if global_budget is not None:
        keep_count = min(keep_count, global_budget)

    kept = ranked[:keep_count]
    dropped = ranked[keep_count:]

    for r, score in dropped:
        logger.info(f"[chunk_relevance] dropped (rrf={score:.4f}): {r.url} — {(r.content or r.snippet or '')[:80]}")

    kept_results = [r for r, _ in kept]
    scores = [s for _, s in ranked]
    logger.info(
        f"[chunk_relevance] candidates={n} dense_scored={len(dense_ranks)} "
        f"sparse_scored={len(sparse_ranks)} kept={len(kept_results)} "
        f"(keep_ratio={keep_ratio}, min_survivors={min_survivors}, global_budget={global_budget})"
    )
    logger.info(f"[chunk_relevance] kept_ids={[_chunk_id(r) for r in kept_results]}")
    logger.info(f"[chunk_relevance] dropped_ids={[_chunk_id(r) for r, _ in dropped]}")
    return kept_results, scores