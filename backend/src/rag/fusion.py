"""
Hybrid retrieval fusion.
Combines dense (semantic) and sparse (keyword/BM25) search results
into a single ranked list using Reciprocal Rank Fusion (RRF).
"""

import logging
import re
from dataclasses import dataclass

from src.rag.vector_store import dense_search
from src.rag.keyword_index import sparse_search
from src.rag.reranker import rerank, RerankedResult
from src.utils.timing import time_stage

logger = logging.getLogger(__name__)

RRF_K = 60  # standard smoothing constant used in RRF, not typically tuned

_SECTION_TAG_RE = re.compile(r'^\[Section:\s*([^\]]+)\]')


def _extract_named_entities(query: str) -> set[str]:
    """Capitalized tokens (len > 3) in the query — a cheap, language-
    agnostic proxy for 'the user named a specific system/entity' (e.g.
    'Instagram', 'WhatsApp', 'Cassandra'). Conservative on purpose: this
    only needs to catch the disambiguation case, not do full NER."""
    return set(re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', query))


def _boost_named_entity_matches(query: str, candidates: list) -> list:
    """
    Distractor disambiguation fix: when a query names a specific system
    ('...Instagram's Cassandra schema...'), two chunks about structurally
    identical but different systems (WhatsApp vs Instagram Cassandra
    schemas) can score nearly identically on pure embedding similarity.
    chunker.py now tags each chunk with its originating heading
    ('[Section: System 3: Instagram]') — use that tag to boost chunks
    whose section explicitly matches a named entity in the query, so they
    rank above same-shape distractors from a different system. Chunks
    without a detected section tag are left unboosted, not discarded —
    a missed heading should degrade to "no boost", not disappear.
    """
    entities = _extract_named_entities(query)
    if not entities:
        return candidates

    boosted = []
    for c in candidates:
        match = _SECTION_TAG_RE.match(c.text)
        tag = match.group(1) if match else None
        if tag and any(e.lower() in tag.lower() for e in entities):
            boosted.append(FusedResult(text=c.text, metadata=c.metadata, fused_score=c.fused_score * 1.5))
        else:
            boosted.append(c)
    return boosted


@dataclass
class FusedResult:
    text: str
    metadata: dict
    fused_score: float


def hybrid_search(query: str, top_k: int = 3, candidate_k: int = 15, doc_ids: list[str] | None = None) -> list[RerankedResult]:
    """
    candidate_k widened from 5 -> 15: fusion now casts a bigger net,
    and the reranker (not RRF) does the final precision narrowing down
    to top_k. Passes doc_ids through to scope search to active documents.
    """
    dense_results, sparse_results = [], []

    try:
        with time_stage("retrieval_dense"):
            dense_results = dense_search(query, top_k=candidate_k, doc_ids=doc_ids)
    except Exception as e:
        logger.warning(f"Dense search unavailable: {e}")

    try:
        with time_stage("retrieval_sparse"):
            sparse_results = sparse_search(query, top_k=candidate_k, doc_ids=doc_ids)
    except Exception as e:
        logger.warning(f"Sparse search unavailable: {e}")

    if not dense_results and not sparse_results:
        return []

    scores, texts, metadatas = {}, {}, {}
    for rank, r in enumerate(dense_results):
        scores[r.text] = scores.get(r.text, 0) + 1 / (RRF_K + rank + 1)
        texts[r.text], metadatas[r.text] = r.text, r.metadata
    for rank, r in enumerate(sparse_results):
        scores[r.text] = scores.get(r.text, 0) + 1 / (RRF_K + rank + 1)
        texts[r.text], metadatas[r.text] = r.text, r.metadata

    # Take a wider fused shortlist (not just top_k) to hand to the reranker
    rerank_pool_size = min(len(scores), max(candidate_k, top_k * 3))
    ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:rerank_pool_size]

    fused_candidates = [
        FusedResult(text=texts[k], metadata=metadatas[k], fused_score=scores[k])
        for k in ranked_keys
    ]
    fused_candidates = _boost_named_entity_matches(query, fused_candidates)

    with time_stage("retrieval_rerank"):
        return rerank(query, fused_candidates, top_k=top_k)