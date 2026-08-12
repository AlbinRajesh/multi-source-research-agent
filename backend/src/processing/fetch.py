"""Fetch/Extract step.

Actual fetch + extraction logic lives in utils/web_utils.py's
ContentExtractor (httpx async client + BeautifulSoup parsing). This
module re-exports it under the Phase 1 file-plan location so imports
can reference `src.processing.fetch` if preferred.
"""
from src.search_providers.content_extractor import ContentExtractor

__all__ = ["ContentExtractor"]