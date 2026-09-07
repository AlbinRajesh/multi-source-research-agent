"""
Sparse (keyword-based) search using BM25.
Complements dense/semantic search — catches exact terms, codes, and
names that embedding similarity can miss.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
import re
import threading

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

INDEX_FILE = Path(__file__).resolve().parent.parent / "bm25_data" / "index.json"

class KeywordIndexError(Exception):
    """Raised when a keyword index operation fails."""
    pass


@dataclass
class SearchResult:
    text: str
    metadata: dict
    score: float


# In-memory state, rebuilt from disk on first use.
_chunk_ids: list[str] = []
_chunk_texts: list[str] = []
_chunk_metadatas: list[dict] = []
_bm25: BM25Okapi | None = None
_loaded = False
_dirty = False
PERSIST_EVERY_N_CHUNKS = 1000


_TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _rebuild_bm25():
    global _bm25
    if _chunk_texts:
        tokenized = [_tokenize(t) for t in _chunk_texts]
        _bm25 = BM25Okapi(tokenized)
    else:
        _bm25 = None


def _ensure_loaded():
    """Load persisted index from disk on first use in this process."""
    global _loaded
    if _loaded:
        return
    if INDEX_FILE.exists():
        try:
            data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            _chunk_ids.extend(data["ids"])
            _chunk_texts.extend(data["texts"])
            _chunk_metadatas.extend(data["metadatas"])
            _rebuild_bm25()
            logger.info(f"Loaded BM25 index with {len(_chunk_texts)} chunk(s) from disk.")
        except Exception as e:
            raise KeywordIndexError(f"Failed to load BM25 index from disk: {e}")
    _loaded = True


def _persist(force: bool = False):
    global _dirty
    if not force and not _dirty:
        return
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        INDEX_FILE.write_text(json.dumps({
            "ids": _chunk_ids, "texts": _chunk_texts, "metadatas": _chunk_metadatas,
        }), encoding="utf-8")
        _dirty = False
    except Exception as e:
        raise KeywordIndexError(f"Failed to persist BM25 index to disk: {e}")


def _persist_async(force: bool = False):
    """Persist to disk on a background thread so callers (e.g. /upload) don't block on I/O."""
    def _do():
        try:
            _persist(force=force)
        except KeywordIndexError as e:
            logger.error(f"Background BM25 persist failed: {e}")
    threading.Thread(target=_do, daemon=True).start()


def flush_index() -> None:
    """Force a synchronous write of any pending changes. Call on graceful shutdown."""
    _persist(force=True)


def add_chunks(chunks: list[str], doc_id: str, source_filename: str) -> int:
    """
    Add a document's chunks to the keyword index. Uses the same
    deterministic chunk-ID scheme as the vector store, so results
    can be matched during hybrid fusion.
    """
    _ensure_loaded()

    if not chunks:
        raise KeywordIndexError("No chunks provided to index.")

    if doc_id in {mid.rsplit("_", 1)[0] for mid in _chunk_ids}:
        logger.warning(f"doc_id '{doc_id}' already exists in keyword index. Skipping re-add.")
        return 0

    global _dirty

    for i, chunk in enumerate(chunks):
        _chunk_ids.append(f"{doc_id}_{i}")
        _chunk_texts.append(chunk)
        _chunk_metadatas.append({"doc_id": doc_id, "source": source_filename, "chunk_index": i})

    _dirty = True
    _rebuild_bm25()
    _persist_async(force=True)  

    logger.info(f"Indexed {len(chunks)} chunk(s) from '{source_filename}' (doc_id={doc_id}) for keyword search")
    return len(chunks)


def sparse_search(query: str, top_k: int = 5, doc_ids: list[str] | None = None) -> list[SearchResult]:
    """
    Perform BM25 keyword search over indexed chunks, optionally filtered by document IDs.

    Raises:
        KeywordIndexError: If the index is empty.
    """
    _ensure_loaded()

    if _bm25 is None or not _chunk_texts:
        raise KeywordIndexError("Keyword index is empty — no documents have been indexed yet.")

    scores = _bm25.get_scores(_tokenize(query))

    if doc_ids:
        allowed = set(doc_ids)
        scores = [
            s if _chunk_metadatas[i]["doc_id"] in allowed else -1
            for i, s in enumerate(scores)
        ]

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        SearchResult(text=_chunk_texts[i], metadata=_chunk_metadatas[i], score=float(scores[i]))
        for i in ranked if scores[i] > 0
    ]


def delete_doc(doc_id: str) -> bool:
    """Delete all chunks belonging to a doc_id from the keyword index."""
    _ensure_loaded()

    indices_to_remove = [i for i, mid in enumerate(_chunk_ids) if mid.rsplit("_", 1)[0] == doc_id]

    if not indices_to_remove:
        return False

    for i in reversed(indices_to_remove):
        del _chunk_ids[i]
        del _chunk_texts[i]
        del _chunk_metadatas[i]

    _rebuild_bm25()
    _persist()

    logger.info(f"Deleted {len(indices_to_remove)} chunk(s) for doc_id '{doc_id}' from keyword index")
    return True