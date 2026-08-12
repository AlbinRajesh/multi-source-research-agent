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
            # parallel fan-out across sub-queries
            tasks = [
                self.search_provider.search(q.query, max_results=config.max_search_results_per_query)
                for q in web_queries
            ]
            results_per_query: List[List[SearchResult]] = await asyncio.gather(*tasks, return_exceptions=False)
            all_results = [r for sub in results_per_query for r in sub]

            # extract full content, bounded concurrency
            all_results = await self.extractor.enhance_results(all_results)
            combined = dedup_results(state.search_results + all_results)
            filtered = self.scorer.filter_results(combined, min_score=config.min_credibility_score)
            credibility_scores = [self.scorer.score_url(r.url) for r in filtered]

            logger.info(f"Retrieved {len(all_results)} -> {len(filtered)} after dedup+credibility (accumulated)")

            return {
                "search_results": filtered,           # full deduped accumulated set, not delta
                "credibility_scores": credibility_scores,
                "current_stage": "extracting_claims",
                "iterations": state.iterations + 1,
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"error": f"Search failed: {e}"}