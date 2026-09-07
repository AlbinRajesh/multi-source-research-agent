"""
Vector store wrapper (Qdrant, local/embedded).
Handles storing document chunk embeddings and performing dense
(semantic) similarity search, filtered by document_id.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.rag.embeddings import embed_texts, embed_query

logger = logging.getLogger(__name__)

QDRANT_PERSIST_DIR = str(Path(__file__).resolve().parent.parent / "qdrant_data")
COLLECTION_NAME = "documents"
EMBED_DIM = 768  # EmbeddingGemma-300M output dim — update if you switch models


class VectorStoreError(Exception):
    pass


@dataclass
class SearchResult:
    text: str
    metadata: dict
    score: float


_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        try:
            _client = QdrantClient(path=QDRANT_PERSIST_DIR)
            existing = [c.name for c in _client.get_collections().collections]
            if COLLECTION_NAME not in existing:
                _client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=qmodels.VectorParams(
                        size=EMBED_DIM, distance=qmodels.Distance.COSINE
                    ),
                )
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize vector store: {e}")
        count = _client.count(COLLECTION_NAME).count
        logger.info(f"Vector store ready at '{QDRANT_PERSIST_DIR}' ({count} existing chunk(s))")
    return _client


def _doc_filter(doc_id: str) -> qmodels.Filter:
    return qmodels.Filter(
        must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]
    )


def add_chunks(chunks: list[str], doc_id: str, source_filename: str) -> int:
    if not chunks:
        raise VectorStoreError("No chunks provided to store.")

    client = _get_client()

    existing, _ = client.scroll(COLLECTION_NAME, scroll_filter=_doc_filter(doc_id), limit=1)
    if existing:
        logger.warning(f"doc_id '{doc_id}' already exists. Skipping re-add.")
        return 0

    try:
        embeddings = embed_texts(chunks)
    except Exception as e:
        raise VectorStoreError(f"Failed to embed chunks before storage: {e}")

    uploaded_at = datetime.now(timezone.utc).isoformat()
    points = [
        qmodels.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}_{i}")),
            vector=embeddings[i].tolist(),
            payload={
                "text": chunks[i],
                "doc_id": doc_id,
                "source": source_filename,
                "chunk_index": i,
                "uploaded_at": uploaded_at,
            },
        )
        for i in range(len(chunks))
    ]

    try:
        client.upsert(COLLECTION_NAME, points=points)
    except Exception as e:
        raise VectorStoreError(f"Failed to store chunks in vector store: {e}")

    logger.info(f"Stored {len(chunks)} chunk(s) from '{source_filename}' (doc_id={doc_id})")
    _invalidate_active_doc_id_cache()
    return len(chunks)


def dense_search(query: str, top_k: int = 5, doc_ids: list[str] | None = None) -> list[SearchResult]:
    """doc_ids=None searches across all documents; pass a list to scope to
    the active document(s) — this is the multi-doc dilution fix."""
    client = _get_client()

    try:
        query_embedding = embed_query(query)
    except Exception as e:
        raise VectorStoreError(f"Failed to embed query: {e}")

    query_filter = None
    if doc_ids:
        query_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchAny(any=doc_ids))]
        )

    try:
        hits = client.query_points(
            COLLECTION_NAME,
            query=query_embedding.tolist(),
            query_filter=query_filter,
            limit=top_k,
        ).points
    except Exception as e:
        raise VectorStoreError(f"Vector search failed: {e}")

    return [
        SearchResult(text=h.payload["text"], metadata=h.payload, score=h.score)
        for h in hits
    ]


def delete_doc(doc_id: str) -> bool:
    client = _get_client()
    existing, _ = client.scroll(COLLECTION_NAME, scroll_filter=_doc_filter(doc_id), limit=1)
    if not existing:
        return False

    client.delete(COLLECTION_NAME, points_selector=_doc_filter(doc_id))
    logger.info(f"Deleted chunks for doc_id '{doc_id}' from vector store")
    _invalidate_active_doc_id_cache()
    return True


_active_doc_id_cache: str | None = None
_active_doc_id_cache_valid = False


def get_active_doc_id() -> str | None:
    """Kept for single-doc callers/back-compat. Cached after first resolution —
    invalidated on add_chunks/delete_doc/recreate_collection since those change
    which doc_ids exist."""
    global _active_doc_id_cache, _active_doc_id_cache_valid
    if _active_doc_id_cache_valid:
        return _active_doc_id_cache

    client = _get_client()
    if client.count(COLLECTION_NAME).count == 0:
        _active_doc_id_cache, _active_doc_id_cache_valid = None, True
        return None

    points, _ = client.scroll(COLLECTION_NAME, limit=10000, with_payload=["doc_id"])
    doc_ids = {p.payload["doc_id"] for p in points}

    result = doc_ids.pop() if len(doc_ids) == 1 else None
    if result is None:
        logger.warning(f"get_active_doc_id: expected exactly 1 doc_id, found {doc_ids} — returning None")
    _active_doc_id_cache, _active_doc_id_cache_valid = result, True
    return result

def has_any_documents() -> bool:
    """Cheap existence check — True if the vector store has any indexed
    chunks at all, regardless of how many distinct doc_ids. Used to decide
    whether 'local' should be added to sources_available automatically."""
    client = _get_client()
    return client.count(COLLECTION_NAME).count > 0


def _invalidate_active_doc_id_cache() -> None:
    global _active_doc_id_cache_valid
    _active_doc_id_cache_valid = False


def get_all_chunks(doc_id: str) -> list[SearchResult]:
    client = _get_client()
    points, _ = client.scroll(
        COLLECTION_NAME, scroll_filter=_doc_filter(doc_id), limit=10000, with_payload=True
    )
    if not points:
        return []

    points.sort(key=lambda p: p.payload.get("chunk_index", 0))
    return [SearchResult(text=p.payload["text"], metadata=p.payload, score=1.0) for p in points]


def list_all_documents() -> list[dict]:
    client = _get_client()
    points, _ = client.scroll(COLLECTION_NAME, limit=10000, with_payload=True)
    docs: dict[str, dict] = {}
    for p in points:
        did = p.payload["doc_id"]
        if did not in docs:
            docs[did] = {
                "doc_id": did,
                "filename": p.payload.get("source", "unknown"),
                "uploaded_at": p.payload.get("uploaded_at", ""),
                "chunk_count": 0,
            }
        docs[did]["chunk_count"] += 1
    return list(docs.values())


def recreate_collection() -> None:
    """Drop and recreate the collection from scratch. Used by the test
    harness's reset_all() instead of scroll+delete-by-id — repeated
    add/delete-by-id cycles in Qdrant's local embedded storage leave
    tombstoned points that aren't reclaimed until segment optimization
    runs, which doesn't reliably trigger for small local collections.
    Across a multi-doc run that buildup accumulates and can exhaust
    memory. Dropping the collection avoids it entirely."""
    client = _get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        logger.warning(f"delete_collection failed (may not exist yet): {e}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE),
    )
    logger.info(f"Recreated collection '{COLLECTION_NAME}' from scratch.")
    _invalidate_active_doc_id_cache()


def close_client() -> None:
    """Close and release the current Qdrant client so the next _get_client()
    call builds a genuinely fresh local storage engine instance, instead of
    reusing the same long-lived one across an entire multi-document test run."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception as e:
            logger.warning(f"Error closing Qdrant client: {e}")
        _client = None