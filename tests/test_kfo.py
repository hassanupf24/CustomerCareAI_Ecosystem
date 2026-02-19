"""
Tests for the Knowledge Base & FAQ Optimizer Agent (KFO).
"""

import pytest

from agents.knowledge_base.embedder import encode, _fallback_encode
from agents.knowledge_base.vector_store import VectorStore
from agents.knowledge_base.kfo_agent import KFOAgent
from api.schemas import KFOInput, KFOOutput


# ── Embedder ──────────────────────────────────────────────────────────────────

class TestEmbedder:
    def test_fallback_encode_shape(self):
        texts = ["Hello world", "Test sentence"]
        result = _fallback_encode(texts)
        assert result.shape == (2, 384)

    def test_fallback_encode_normalized(self):
        import numpy as np
        result = _fallback_encode(["test"])
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 0.01

    def test_encode_returns_array(self):
        result = encode(["Hello world"])
        assert result is not None
        assert result.shape[0] == 1


# ── Vector Store ──────────────────────────────────────────────────────────────

class TestVectorStore:
    def test_add_and_search(self):
        import numpy as np
        store = VectorStore(dimension=4)
        vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
        metadata = [
            {"id": "a", "title": "Alpha"},
            {"id": "b", "title": "Beta"},
            {"id": "c", "title": "Gamma"},
        ]
        store.add(vectors, metadata)
        assert store.size == 3

        query = np.array([1, 0, 0, 0], dtype=np.float32)
        results = store.search(query, top_k=2)
        assert len(results) <= 2
        assert results[0]["id"] == "a"

    def test_empty_store_search(self):
        import numpy as np
        store = VectorStore(dimension=4)
        query = np.array([1, 0, 0, 0], dtype=np.float32)
        results = store.search(query, top_k=5)
        assert results == []


# ── KFO Agent ─────────────────────────────────────────────────────────────────

class TestKFOAgent:
    @pytest.mark.asyncio
    async def test_process_returns_articles(self):
        agent = KFOAgent()
        input_data = KFOInput(
            interaction_id="test-kfo-001",
            query_text="How do I reset my password?",
            top_k=3,
        )
        result = await agent.process(input_data)
        assert isinstance(result, KFOOutput)
        assert isinstance(result.suggested_faq_articles, list)

    @pytest.mark.asyncio
    async def test_empty_query(self):
        agent = KFOAgent()
        input_data = KFOInput(
            interaction_id="test-kfo-002",
            query_text="",
            top_k=5,
        )
        result = await agent.process(input_data)
        assert isinstance(result, KFOOutput)
        assert len(result.suggested_faq_articles) == 0
