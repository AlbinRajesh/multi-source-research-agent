"""
Text embedding generation.
Wraps a Hugging Face sentence-transformers model to convert text into
dense vector embeddings for semantic search.
"""

import logging
import torch

import os
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "google/embeddinggemma-300m"

# Loaded once at module import time — reused across every embed call,
# since loading the model itself is the expensive part, not encoding text.
_model: SentenceTransformer | None = None


class EmbeddingError(Exception):
    """Raised when text embedding fails."""
    pass

import os

# Minimum free VRAM required before we'll place the embedding model on GPU.
# embeddinggemma-300m needs roughly ~1.2GB; we require some headroom on top
# so it doesn't contend with the reranker (and Ollama, on a 4GB card).
MIN_FREE_VRAM_FOR_GPU_EMBED_BYTES = 1.5 * 1024**3  # 1.5GB


def _pick_embedding_device() -> str:
    """Forces GPU usage if CUDA is available, otherwise falls back to CPU."""
    if not torch.cuda.is_available():
        logger.info("CUDA is not available — falling back to CPU.")
        return "cpu"
    
    logger.info("CUDA is available — forcing embedding model onto GPU.")
    return "cuda"

def _get_model() -> SentenceTransformer:
    """Lazily load the embedding model on first use, then reuse it.
    Priority: CPU by default; GPU only if there's spare VRAM after the
    reranker/Ollama have loaded (see _pick_embedding_device)."""
    global _model
    if _model is None:
        device = _pick_embedding_device()

        logger.info(f"Loading embedding model '{MODEL_NAME}' on device={device}...")
        try:
            _model = SentenceTransformer(MODEL_NAME, device=device)
        except Exception as e:
            if device == "cuda":
                logger.warning(f"Failed to load embedding model on GPU ({e}), falling back to CPU...")
                device = "cpu"
                try:
                    _model = SentenceTransformer(MODEL_NAME, device=device)
                except Exception as inner_e:
                    raise EmbeddingError(f"Failed to load embedding model '{MODEL_NAME}' on CPU as well: {inner_e}")
            else:
                raise EmbeddingError(f"Failed to load embedding model '{MODEL_NAME}': {e}")

        logger.info(f"Embedding model loaded successfully on {device}.")
    return _model


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Convert a list of text strings into embedding vectors.

    Args:
        texts: List of text strings to embed (e.g. document chunks).

    Returns:
        A NumPy array of shape (len(texts), embedding_dim).

    Raises:
        EmbeddingError: If texts is empty, contains no valid strings,
                         or embedding fails.
    """
    if not texts:
        raise EmbeddingError("No texts provided to embed.")

    cleaned = [t.strip() for t in texts if t and t.strip()]
    if not cleaned:
        raise EmbeddingError("All provided texts were empty after cleaning.")

    model = _get_model()
    try:
        embeddings = model.encode(
            cleaned,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
            convert_to_numpy=True,
        )
    except Exception as e:
        raise EmbeddingError(f"Embedding generation failed: {e}")

    logger.info(f"Embedded {len(cleaned)} text(s) into {embeddings.shape[1]}-dimensional vectors")
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Convert a single query string into an embedding vector.
    Separate from embed_texts for clarity at call sites (query vs. document
    embedding), even though the underlying operation is currently identical.

    Args:
        query: The user's question/search text.

    Returns:
        A NumPy array of shape (embedding_dim,).

    Raises:
        EmbeddingError: If query is empty or embedding fails.
    """
    if not query or not query.strip():
        raise EmbeddingError("Query is empty.")

    result = embed_texts([query])
    return result[0]

def warmup() -> None:
    """Force the embedding model to load and run once. Call at server startup."""
    _get_model()
    embed_texts(["warmup"])
    logger.info("Embedding model warmed up.")