"""Full-page content extraction. Plain code, no LLM.

Uses the shared pooled HTTP client + circuit breaker from utils/http_client.py.
Called by the Retriever Agent to upgrade snippet-only search results into
full document text before claim extraction runs.
"""
import re
import asyncio
import logging
from typing import Optional, List
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from src.state import SearchResult
from src.utils.http_client import HTTPClientManager, CircuitBreaker, is_valid_url
from src.exceptions import ContentExtractionError

logger = logging.getLogger(__name__)


class ContentExtractor:
    CONTENT_SELECTORS = [
        "article", "main", '[role="main"]',
        ".post-content", ".article-content", ".entry-content",
        ".content", "#content", ".post", ".article",
    ]

    REMOVE_SELECTORS = [
        "script", "style", "nav", "footer", "header", "aside",
        ".sidebar", ".navigation", ".menu", ".comments", ".comment",
        ".advertisement", ".ad", ".ads", ".social-share", ".related-posts",
        '[role="navigation"]', '[role="complementary"]',
    ]

    def __init__(self, timeout: int = 15, max_content_length: int = 8000):
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.client_manager = HTTPClientManager.get_instance()
        self._breakers: dict = {}

    def _get_breaker(self, url: str) -> CircuitBreaker:
        domain = urlparse(url).netloc
        if domain not in self._breakers:
            self._breakers[domain] = CircuitBreaker(
                name=domain, failure_threshold=5, reset_timeout=30.0
            )
        return self._breakers[domain]

    async def extract_content_async(self, url: str) -> Optional[str]:
        if not is_valid_url(url):
            logger.warning(f"Invalid/unsafe URL, skipping: {url}")
            return None

        breaker = self._get_breaker(url)
        if not breaker.can_execute():
            logger.warning(f"Content extraction circuit open for domain in URL, skipping fetch: {url}")
            return None

        try:
            client = await self.client_manager.get_client()
            response = await client.get(url, timeout=self.timeout)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if not any(ct in content_type for ct in ["text/html", "application/xhtml"]):
                logger.debug(f"Unsupported content-type for {url}: {content_type}")
                return None

            extracted = self._parse_html(response.text)
            breaker.record_success()
            return extracted

        except httpx.HTTPStatusError as e:
            breaker.record_failure()
            logger.warning(f"HTTP {e.response.status_code} fetching {url}")
            return None
        except httpx.TimeoutException:
            breaker.record_failure()
            logger.warning(f"Timeout fetching {url}")
            return None
        except Exception as e:
            breaker.record_failure()
            logger.warning(f"Failed to extract {url}: {e}")
            return None

    def _parse_html(self, html: str) -> Optional[str]:
        soup = BeautifulSoup(html, "html.parser")

        for selector in self.REMOVE_SELECTORS:
            for el in soup.select(selector):
                el.decompose()

        main = None
        for selector in self.CONTENT_SELECTORS:
            main = soup.select_one(selector)
            if main:
                break
        if not main:
            main = soup.body

        if not main:
            return None

        text = main.get_text(separator="\n", strip=True)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = re.sub(r" +", " ", text)
        return text[: self.max_content_length]

    async def enhance_results(self, results: List[SearchResult], max_concurrent: int = 5) -> List[SearchResult]:
        """Fill in `.content` for results that only have a snippet.
        Bounded concurrency so we don't hammer many hosts at once."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def enhance_one(result: SearchResult) -> SearchResult:
            async with semaphore:
                if not result.content:
                    try:
                        content = await self.extract_content_async(result.url)
                        if content:
                            result.content = content
                    except Exception as e:
                        logger.warning(f"Unexpected error enhancing {result.url}: {e}")
                return result

        return list(await asyncio.gather(*[enhance_one(r) for r in results]))