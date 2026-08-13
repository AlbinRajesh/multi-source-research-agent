"""Full-page content extraction.

Primary path: Tavily Extract API — returns clean, pre-parsed Markdown,
avoiding raw-HTML noise and sites that block generic HTTP clients (403s).
Fallback path: direct httpx + BeautifulSoup, for anything Tavily Extract
can't retrieve, so a single provider outage doesn't drop sources entirely.
"""
import re
import asyncio
import logging
from typing import Optional, List
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from tavily import TavilyClient

from src.state import SearchResult
from src.utils.http_client import HTTPClientManager, CircuitBreaker, is_valid_url
from src.exceptions import ContentExtractionError
from src.config import config

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

    # Blocklist for non-article / social / video sites that yield poor text-scraping content
    BLOCKED_DOMAINS = {
        "youtube.com", "m.youtube.com", "youtu.be",
        "linkedin.com", "m.linkedin.com",
        "twitter.com", "x.com",
        "facebook.com", "instagram.com", "reddit.com", "tiktok.com"
    }

    def __init__(self, timeout: int = 15, max_content_length: int = 8000):
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.client_manager = HTTPClientManager.get_instance()
        self._breakers: dict = {}
        self._tavily = TavilyClient(api_key=config.tavily_api_key) if config.tavily_api_key else None

    def _is_blocked(self, url: str) -> bool:
        try:
            domain = urlparse(url).netloc.lower()
            return any(blocked in domain for blocked in self.BLOCKED_DOMAINS)
        except Exception:
            return False

    def _get_breaker(self, url: str) -> CircuitBreaker:
        domain = urlparse(url).netloc
        if domain not in self._breakers:
            self._breakers[domain] = CircuitBreaker(
                name=domain, failure_threshold=5, reset_timeout=30.0
            )
        return self._breakers[domain]

    async def extract_via_tavily_batch(self, urls: List[str]) -> dict[str, Optional[str]]:
        """Batch-extract multiple URLs in chunks of 20 (Tavily API limit).
        Returns {url: content or None}."""
        valid_urls = [u for u in urls if not self._is_blocked(u)]
        if not self._tavily or not valid_urls:
            return {u: None for u in urls}

        results: dict[str, Optional[str]] = {u: None for u in urls}
        
        # Chunk URLs into groups of 20 to respect Tavily's hard limit per request
        chunk_size = 20
        url_chunks = [valid_urls[i:i + chunk_size] for i in range(0, len(valid_urls), chunk_size)]

        async def fetch_chunk(chunk: List[str]):
            try:
                response = await asyncio.to_thread(self._tavily.extract, urls=chunk)
                for item in response.get("results", []):
                    content = item.get("raw_content")
                    if content:
                        results[item["url"]] = content[: self.max_content_length]
                for failed in response.get("failed_results", []):
                    logger.warning(f"Tavily Extract failed for {failed.get('url')}: {failed.get('error')}")
            except Exception as e:
                logger.warning(f"Tavily Extract chunk call failed for {len(chunk)} URLs: {e}")

        # Run chunk requests concurrently
        await asyncio.gather(*(fetch_chunk(chunk) for chunk in url_chunks))

        return results

    async def extract_content_async(self, url: str) -> Optional[str]:
        """Fallback single-URL path — direct httpx + BeautifulSoup."""
        if self._is_blocked(url) or not is_valid_url(url):
            logger.warning(f"Blocked or unsafe URL, skipping: {url}")
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
        """Filter out blocked URLs and fill in `.content` for valid results."""
        # Pre-filter blocklisted items entirely so they don't consume compute/extraction slots
        filtered_results = []
        for r in results:
            if self._is_blocked(r.url):
                logger.info(f"Dropping blocklisted non-article domain: {r.url}")
                continue
            filtered_results.append(r)

        needing_content = [r for r in filtered_results if not r.content]
        if not needing_content:
            return filtered_results

        urls = [r.url for r in needing_content]
        tavily_results = await self.extract_via_tavily_batch(urls)

        fallback_targets: List[SearchResult] = []
        for r in needing_content:
            content = tavily_results.get(r.url)
            if content:
                r.content = content
            else:
                fallback_targets.append(r)

        if fallback_targets:
            logger.info(f"Tavily Extract missed {len(fallback_targets)} URLs, falling back to direct fetch")
            semaphore = asyncio.Semaphore(max_concurrent)

            async def enhance_one(result: SearchResult) -> SearchResult:
                async with semaphore:
                    try:
                        content = await self.extract_content_async(result.url)
                        if content:
                            result.content = content
                    except Exception as e:
                        logger.warning(f"Unexpected error enhancing {result.url}: {e}")
                    return result

            await asyncio.gather(*[enhance_one(r) for r in fallback_targets])

        return filtered_results