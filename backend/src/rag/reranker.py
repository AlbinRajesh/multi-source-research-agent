"""
Cross-encoder reranker.
Takes a query and a list of candidate chunks and re-scores them jointly
(query + chunk seen together), which is more accurate than the bi-encoder
similarity used in dense_search — at the cost of being too slow to run
over the whole corpus, so it only ever runs on a small shortlist.
"""

import logging
from dataclasses import dataclass

from sentence_transformers import CrossEncoder
import torch

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-reranker-base"

_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading reranker model '{MODEL_NAME}' on preferred device={device}...")
        
        try:
            dtype = torch.float16 if device == "cuda" else torch.float32
            _model = CrossEncoder(MODEL_NAME, max_length=512, device=device, model_kwargs={"torch_dtype": dtype})
        except Exception as e:
            if device == "cuda":
                logger.warning(f"Failed to load reranker on GPU ({e}), falling back to CPU...")
                device = "cpu"
                dtype = torch.float32
                _model = CrossEncoder(MODEL_NAME, max_length=512, device=device, model_kwargs={"torch_dtype": dtype})
            else:
                raise e

        logger.info(f"Reranker model loaded successfully on {device}.")
    return _model


@dataclass
class RerankedResult:
    text: str
    metadata: dict
    rerank_score: float  # raw cross-encoder logit — higher = more relevant


def rerank(query: str, candidates: list, top_k: int = 3) -> list[RerankedResult]:
    """
    Args:
        query: The user's question.
        candidates: List of objects with .text and .metadata (e.g. FusedResult).
        top_k: How many to keep after reranking.

    Returns:
        List of RerankedResult, best first. Empty list if candidates is empty.
    """
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return [
        RerankedResult(text=c.text, metadata=c.metadata, rerank_score=float(s))
        for c, s in scored[:top_k]
    ]
def warmup() -> None:
    """Force the reranker model to load once. Call at server startup."""
    _get_model()
    logger.info("Reranker model warmed up.")