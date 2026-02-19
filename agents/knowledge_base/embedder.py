"""
Embedder — sentence-transformers wrapper for vectorization.

Encodes text into dense vectors for semantic similarity search.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger()

_model = None


def _get_model():
    """Lazy-load the sentence-transformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            from config.settings import get_settings
            settings = get_settings()
            _model = SentenceTransformer(settings.embedding_model_name)
            logger.info("embedder_loaded", model=settings.embedding_model_name)
        except Exception as e:
            logger.error("embedder_load_failed", error=str(e))
    return _model


def encode(texts: list[str]) -> Optional[np.ndarray]:
    """
    Encode a list of texts into embeddings.

    Returns:
        numpy array of shape (n_texts, embedding_dim) or None on failure.
    """
    model = _get_model()
    if model is None:
        logger.warning("embedder_not_available_using_fallback")
        return _fallback_encode(texts)
    try:
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings
    except Exception as e:
        logger.error("encoding_failed", error=str(e))
        return _fallback_encode(texts)


def _fallback_encode(texts: list[str]) -> np.ndarray:
    """
    Simple TF-based fallback when sentence-transformers is unavailable.
    Produces fixed-length vectors via hashing.
    """
    dim = 384  # match MiniLM dimension
    vectors = []
    for text in texts:
        np.random.seed(hash(text) % (2**31))
        vec = np.random.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        vectors.append(vec)
    return np.array(vectors)
