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

    def __init__(self, search_provider=None, extractor=None, scorer=None):
        self.search_provider = search_provider or self._get_default_provider()
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

    async def search(self, state: ResearchState) -> Dict[str, Any]:
        if not state.plan:
            return {"error": "No plan available"}

        web_queries = [q for q in state.plan.search_queries if q.source_hint in ("web", "both")]

        try:
            semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_SEARCHES)

            async def bounded_search(q):
                async with semaphore:
                    return await self.search_provider.search(q.query, max_results=config.max_search_results_per_query)

            tasks = [bounded_search(q) for q in web_queries]
            results_per_query: List[List[SearchResult]] = await asyncio.gather(*tasks, return_exceptions=False)
            all_results = [r for sub in results_per_query for r in sub]

            all_results = await self.extractor.enhance_results(all_results)
            
            # Keep track of which URLs are brand new in this search/retry cycle
            existing_urls = {r.url for r in state.search_results}
            
            combined = dedup_results(state.search_results + all_results)
            filtered = self.scorer.filter_results(combined, min_score=config.min_credibility_score)

            # Sort all filtered results by credibility score descending
            scored = sorted(filtered, key=lambda r: self.scorer.score_url(r.url)["score"], reverse=True)
            
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

            credibility_scores = [self.scorer.score_url(r.url) for r in filtered]

            logger.info(f"Retrieved {len(all_results)} -> {len(filtered)} after dedup+credibility (capped & accumulated with retry reservation)")

            return {
                "search_results": filtered,
                "credibility_scores": credibility_scores,
                "current_stage": "extracting_claims",
                "iterations": state.iterations + 1,
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"error": f"Search failed: {e}"}