"""SearXNG search provider — fallback for Tavily.

Adapted from local-deep-research's SearXNGSearchEngine: keeps the actual
HTML/JSON parsing logic, drops their egress-classification/rate-limiter/
LLM-relevance-filter framework (out of scope here — our own circuit
breaker in web_utils.py covers resilience).
"""
import json
import logging
from typing import List, Optional

import httpx
from bs4 import BeautifulSoup

from src.state import SearchResult
from src.search_providers.base import SearchProvider
from src.utils.http_client import CircuitBreaker, is_valid_url
from src.exceptions import SearchError

logger = logging.getLogger(__name__)


class SearXNGProvider(SearchProvider):
    def __init__(
        self,
        instance_url: str = "http://localhost:8080",
        max_results: int = 5,
        result_format: str = "json",  # prefer json if the instance supports it
        timeout: float = 15.0,
    ):
        self.instance_url = instance_url.rstrip("/")
        self.max_results = max_results
        self.result_format = result_format
        self.timeout = timeout
        self.circuit_breaker = CircuitBreaker(name="searxng", failure_threshold=3, reset_timeout=60.0)

    @property
    def name(self) -> str:
        return "searxng"

    def _is_valid_result_url(self, url: str) -> bool:
        if not url or not is_valid_url(url):
            return False
        if url.startswith(self.instance_url):  # internal SearXNG pages, not real results
            return False
        return True

    async def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        from src.exceptions import CircuitOpenError

        if not self.circuit_breaker.can_execute():
            raise CircuitOpenError("searxng", self.circuit_breaker.get_retry_after())

        limit = max_results or self.max_results
        params = {
            "q": query,
            "categories": "general",
            "language": "en",
            "format": self.result_format,
            "pageno": 1,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.instance_url}/search", params=params)
                response.raise_for_status()

            if self.result_format == "json":
                results = self._parse_json(response, query, limit)
            else:
                results = self._parse_html(response.text, query, limit)

            self.circuit_breaker.record_success()
            logger.info(f"SearXNG: {len(results)} results for '{query}'")
            return results

        except Exception as e:
            self.circuit_breaker.record_failure()
            raise SearchError(f"SearXNG search failed for '{query}'", details=str(e))

    def _parse_json(self, response, query: str, limit: int) -> List[SearchResult]:
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise SearchError("Failed to parse SearXNG JSON response", details=str(e))

        results = []
        for item in data.get("results", []):
            if len(results) >= limit:
                break
            url = item.get("url", "")
            if not self._is_valid_result_url(url):
                continue
            results.append(SearchResult(
                query=query,
                title=item.get("title") or "",
                url=url,
                snippet=item.get("content") or "",
            ))
        return results

    def _parse_html(self, html: str, query: str, limit: int) -> List[SearchResult]:
        soup = BeautifulSoup(html, "html.parser")
        results = []

        result_elements = (
            soup.select(".result-item") or soup.select(".result") or soup.select("article")
        )

        for el in result_elements:
            if len(results) >= limit:
                break
            title_el = el.select_one(".result-title") or el.select_one("h3") or el.select_one("a[href]")
            url_el = el.select_one(".result-url") or el.select_one("a[href]")
            content_el = el.select_one(".result-content") or el.select_one("p")

            title = title_el.get_text(" ", strip=True) if title_el else ""
            url = url_el.get("href", "") if url_el else ""
            content = content_el.get_text(" ", strip=True) if content_el else ""

            if self._is_valid_result_url(url):
                results.append(SearchResult(query=query, title=title, url=url, snippet=content))

        return results