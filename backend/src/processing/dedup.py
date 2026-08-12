"""Dedup — exact + near-duplicate removal.

Runs after search, before claim extraction. Matters especially on retry
loops where search_results accumulates across rounds (same doc can come
back from a refined query).

Exact dedup: URL match.
Near-dup: Jaccard similarity over shingled text (no embedding model
dependency — keeps this stage cheap, plain code, no LLM/embedding call).
Swap in an embedding-based version later if this proves too coarse.
"""
import logging
import re
from typing import List
from urllib.parse import urlparse, urlunparse

from src.state import SearchResult

logger = logging.getLogger(__name__)


def _normalize_url(url: str) -> str:
    """Strip query params/fragment/trailing slash so tracking params don't
    defeat exact-dedup."""
    try:
        parsed = urlparse(url)
        normalized = urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))
        return normalized.lower()
    except Exception:
        return url.lower()


def _shingles(text: str, k: int = 5) -> set:
    words = re.findall(r"\w+", text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedup_results(results: List[SearchResult], near_dup_threshold: float = 0.8) -> List[SearchResult]:
    """Remove exact URL duplicates, then near-duplicate content."""
    if not results:
        return results

    # exact dedup by normalized URL, keep first occurrence
    seen_urls = set()
    exact_deduped: List[SearchResult] = []
    for r in results:
        key = _normalize_url(r.url)
        if key in seen_urls:
            continue
        seen_urls.add(key)
        exact_deduped.append(r)

    exact_removed = len(results) - len(exact_deduped)

    # near-dup via shingled Jaccard similarity on available text
    kept: List[SearchResult] = []
    kept_shingles: List[set] = []
    near_removed = 0

    for r in exact_deduped:
        text = r.content or r.snippet or ""
        shingles = _shingles(text)
        is_dup = False
        for existing in kept_shingles:
            if _jaccard(shingles, existing) >= near_dup_threshold:
                is_dup = True
                break
        if is_dup:
            near_removed += 1
            continue
        kept.append(r)
        kept_shingles.append(shingles)

    logger.info(
        f"Dedup: {len(results)} -> {len(kept)} "
        f"(-{exact_removed} exact, -{near_removed} near-dup)"
    )
    return kept