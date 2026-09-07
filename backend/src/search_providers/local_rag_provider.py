"""Local RAG search provider — wraps the hybrid retrieval stack in rag/
to conform to the SearchProvider interface."""
import asyncio
import logging
from typing import List, Optional

from src.search_providers.base import SearchProvider
from src.state import SearchResult
from src.exceptions import SearchError
from src.rag.fusion import hybrid_search

logger = logging.getLogger(__name__)


class LocalRAGProvider(SearchProvider):
    @property
    def name(self) -> str:
        return "local_rag"

    async def search(self, query: str, max_results: int = 3, doc_ids: Optional[List[str]] = None) -> List[SearchResult]:
        try:
            results = await asyncio.to_thread(hybrid_search, query, top_k=max_results, doc_ids=doc_ids)
        except Exception as e:
            raise SearchError(f"Local RAG search failed for '{query}'", details=str(e))

        out = []
        for r in results:
            doc_id = r.metadata.get("doc_id", "unknown")
            chunk_idx = r.metadata.get("chunk_index", 0)
            source_name = r.metadata.get("source", "uploaded document")
            out.append(SearchResult(
                query=query,
                title=source_name,
                url=f"local://{doc_id}/{chunk_idx}",
                snippet=r.text[:300],
                content=r.text,
                source_type="local",
                source_name=source_name,
            ))
        logger.info(f"LocalRAG: {len(out)} results for '{query}' (doc_ids={doc_ids})")
        return out