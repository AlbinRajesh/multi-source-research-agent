"""Tavily search provider — uses the official tavily-python SDK."""
import logging
from typing import List, Optional

from tavily import TavilyClient

from src.search_providers.base import SearchProvider, ProviderUnavailableError
from src.state import SearchResult
from src.exceptions import SearchError
import asyncio
from src.config import config

logger = logging.getLogger(__name__)


class TavilySearchProvider(SearchProvider):
    @property
    def name(self) -> str:
        return "tavily"

    def __init__(self):
        self.api_key = config.tavily_api_key
        if not self.api_key:
            raise ProviderUnavailableError("Tavily provider not configured (set tavily_api_key in config)")
        self.client = TavilyClient(api_key=self.api_key)

    async def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        limit = max_results or 5
        try:
            response = await asyncio.to_thread(self.client.search, query=query, max_results=limit, search_depth="basic")
            results = []
            skipped = 0
            for item in response.get("results", []):
                url = item.get("url")
                if not url:
                    # Never fabricate/default a URL — an empty string here
                    # used to silently propagate through dedup, credibility
                    # scoring, and citations, ending up as href="" in the
                    # frontend, which the browser resolves to the current
                    # page instead of showing a broken/missing link.
                    skipped += 1
                    continue
                results.append(SearchResult(
                    query=query,
                    title=item.get("title") or "",
                    url=url,
                    snippet=item.get("content") or "",
                    source_type="web",
                ))
            if skipped:
                logger.warning(f"Tavily: skipped {skipped} result(s) with no URL for '{query}'")
            logger.info(f"Tavily: {len(results)} results for '{query}'")
            return results
        except Exception as e:
            raise SearchError(f"Tavily search failed for '{query}'", details=str(e))