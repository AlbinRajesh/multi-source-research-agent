"""Abstract base class for search providers.

Every provider (Tavily, SearXNG fallback, and eventually the Phase 2 local
RAG adapter) implements this interface so the Retriever Agent and every
downstream stage (dedup, claim extraction, verification) can treat all
sources uniformly — they only ever see `list[SearchResult]`.
"""
from abc import ABC, abstractmethod
from typing import List

from src.state import SearchResult


class SearchProvider(ABC):
    """Common interface for anything that can turn a query into SearchResults."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, used in logs and circuit breaker naming."""
        ...

    @abstractmethod
    async def search(self, query: str, max_results: int) -> List[SearchResult]:
        """Execute the search and return normalized SearchResult objects.

        Implementations must:
          - set `source_type` correctly ("web" or "local")
          - never raise on empty results (return [] instead)
          - raise CircuitOpenError / RateLimitError / SearchError from
            src.exceptions on failure, so the caller can distinguish
            "no results" from "provider is down"
        """
        ...


class ProviderUnavailableError(Exception):
    """Raised when a provider is not configured (e.g. missing API key)."""
    pass