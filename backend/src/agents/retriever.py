"""Search/Retriever Agent — fans out sub-queries across sources in parallel,
extracts full content, applies the cheap credibility pre-filter.
"""
import asyncio
import logging
from typing import Dict, Any, List
from src.processing.dedup import dedup_results
from src.state import ResearchState, SearchResult
from src.processing.credibility import CredibilityScorer
from src.search_providers.content_extractor import ContentExtractor
from src.config import config
from src.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class RetrieverAgent:
    MAX_CONCURRENT_SEARCHES = 5  # stay comfortably under the Tavily connection pool (size 10)
    SEARCH_MAX_RETRIES = int(getattr(config, "search_max_retries", 3))      # attempts per sub-query before giving up on it
    SEARCH_RETRY_BASE_DELAY = 1.5  # seconds; backoff is base * 2**attempt

    def __init__(self, search_provider=None, extractor=None, scorer=None):
        self.search_provider = search_provider or self._get_default_provider()
        self.local_provider = self._get_local_provider()
        self.extractor = extractor or ContentExtractor()
        self.scorer = scorer or CredibilityScorer()

    def _get_default_provider(self):
        if config.search_provider == "tavily":
            from src.search_providers.tavily_provider import TavilySearchProvider
            return TavilySearchProvider()
        elif config.search_provider == "searxng":
            from src.search_providers.searxng_provider import SearXNGSearchProvider
            return SearXNGSearchProvider()
        raise ConfigurationError(f"Unknown search_provider: {config.search_provider}")

    def _get_local_provider(self):
        from src.search_providers.local_rag_provider import LocalRAGProvider
        return LocalRAGProvider()

    def _score_result(self, r) -> dict:
        if getattr(r, "source_type", "web") == "local":
            return {'score': 100, 'factors': ['Local document'], 'level': 'high', 'domain': 'local'}
        return self.scorer.score_url(r.url)

    async def search(self, state: ResearchState) -> Dict[str, Any]:
        if not state.plan:
            return {"error": "No plan available"}

        if state.plan.mode == "full_web":
            web_queries = list(state.plan.search_queries)
            local_queries = []
        elif state.plan.mode == "fast_local":
            web_queries = []
            local_queries = list(state.plan.search_queries)
        else:
            web_queries = [
                q for q in state.plan.search_queries if q.source_hint in ("web", "both")
            ]
            local_queries = [
                q for q in state.plan.search_queries if q.source_hint in ("local", "both")
            ]

        logger.info(
            "[source_routing] mode=%s web_queries=%d local_queries=%d",
            state.plan.mode,
            len(web_queries),
            len(local_queries),
        )

        try:
            semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_SEARCHES)

            async def bounded_search(q):
                # Retry-with-backoff around the actual provider call. A single
                # transient failure (timeout, 429, connection reset) used to
                # silently drop that sub-query's results entirely — this gives
                # it up to SEARCH_MAX_RETRIES attempts before giving up, same
                # category of resilience as the Groq-side SDK backoff.
                last_error = None
                async with semaphore:
                    for attempt in range(self.SEARCH_MAX_RETRIES):
                        try:
                            return await self.search_provider.search(
                                q.query, max_results=config.max_search_results_per_query
                            )
                        except Exception as e:
                            last_error = e
                            if attempt < self.SEARCH_MAX_RETRIES - 1:
                                delay = self.SEARCH_RETRY_BASE_DELAY * (2 ** attempt)
                                logger.warning(
                                    f"[retry] search attempt {attempt + 1} failed for "
                                    f"query {q.query!r}: {e} — retrying in {delay:.1f}s"
                                )
                                await asyncio.sleep(delay)
                            else:
                                logger.error(
                                    f"Search failed for query {q.query!r} after "
                                    f"{self.SEARCH_MAX_RETRIES} attempts: {last_error}"
                                )
                # All retries exhausted — return empty rather than raising, so
                # gather() below doesn't need return_exceptions to survive this
                # path; other sub-queries are unaffected either way.
                return []

            tasks = [bounded_search(q) for q in web_queries]
            # return_exceptions=True so one query's unexpected (non-retried)
            # failure can't take down every other successful query's results.
            results_per_query = await asyncio.gather(*tasks, return_exceptions=True)

            all_results: List[SearchResult] = []
            for q, res in zip(web_queries, results_per_query):
                if isinstance(res, Exception):
                    logger.error(f"Search task raised for query {q.query!r}: {res}")
                    continue
                all_results.extend(res)

            # Fan out local queries (local disk, no rate-limiting semaphore needed)
            local_tasks = [
                self.local_provider.search(
                    q.query,
                    max_results=config.max_search_results_per_query,
                    doc_ids=state.selected_doc_ids or None
                )
                for q in local_queries
            ]
            local_results_per_query = await asyncio.gather(*local_tasks, return_exceptions=True)
            for q, res in zip(local_queries, local_results_per_query):
                if isinstance(res, Exception):
                    logger.error(f"Local RAG search failed for query {q.query!r}: {res}")
                    continue
                all_results.extend(res)

            # Split results by source type to avoid redundant content extraction on local chunks
            web_results = [r for r in all_results if r.source_type == "web"]
            local_results = [r for r in all_results if r.source_type == "local"]
            web_results = await self.extractor.enhance_results(web_results)
            all_results = web_results + local_results
            
            # Keep track of which URLs are brand new in this search/retry cycle
            existing_urls = {r.url for r in state.search_results}
            
            combined = dedup_results(state.search_results + all_results)
            filtered = self.scorer.filter_results(combined, min_score=config.min_credibility_score)

            # Sort all filtered results by credibility score descending using source-aware scoring
            scored = sorted(filtered, key=lambda r: self._score_result(r)["score"], reverse=True)
            
            # Reserve slots for newly found results on retries so they aren't completely starved out by global top-N scoring
            max_docs = config.max_docs_for_extraction
            newly_discovered = [r for r in scored if r.url not in existing_urls]
            older_scored = [r for r in scored if r.url in existing_urls]

            # If we have newly discovered items, reserve up to a portion (e.g., at least 30% or min 2 slots) for them
            reserved_slots = min(len(newly_discovered), max(2, int(max_docs * 0.3))) if newly_discovered else 0
            top_new = newly_discovered[:reserved_slots]
            
            remaining_slots = max_docs - len(top_new)
            # Fill the rest from the highest-scoring overall remaining items
            remaining_pool = [r for r in scored if r not in top_new]
            filtered = top_new + remaining_pool[:remaining_slots]

            credibility_scores = [self._score_result(r) for r in filtered]

            logger.info(f"Retrieved {len(all_results)} -> {len(filtered)} after dedup+credibility (capped & accumulated with retry reservation)")

            result = {
                "search_results": filtered,
                "credibility_scores": credibility_scores,
                "current_stage": "extracting_claims",
                "iterations": state.iterations + 1,
            }
            if not filtered:
                result["error"] = (
                    f"Found {len(all_results)} results but none met the credibility "
                    f"threshold for this topic."
                )
            return result

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"error": f"Search failed: {e}"}