"""
Knowledge Base & FAQ Optimizer Agent (KFO)

Responsibilities:
- Semantic search over FAQ articles using FAISS + sentence-transformers
- BM25 keyword fallback ranking
- Knowledge base updates based on interaction insights
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
import structlog

from agents.base_agent import BaseAgent
from agents.knowledge_base.embedder import encode
from agents.knowledge_base.vector_store import VectorStore
from api.schemas import FAQArticle, KFOInput, KFOOutput

logger = structlog.get_logger()

_FAQ_PATH = Path(__file__).parent / "faq_db.json"


class KFOAgent(BaseAgent):
    """Knowledge Base & FAQ Optimizer Agent."""

    def __init__(self) -> None:
        super().__init__(agent_name="KFO")
        self.vector_store = VectorStore(dimension=384)
        self._faq_data: list[dict] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load FAQ data and build vector index on first call."""
        if self._loaded:
            return

        try:
            with open(_FAQ_PATH, "r", encoding="utf-8") as f:
                self._faq_data = json.load(f)
        except FileNotFoundError:
            self.logger.warning("faq_db_not_found", path=str(_FAQ_PATH))
            self._faq_data = []
            self._loaded = True
            return

        if not self._faq_data:
            self._loaded = True
            return

        # Embed all FAQ content
        texts = [f"{faq['title']} {faq['content']}" for faq in self._faq_data]
        embeddings = encode(texts)

        if embeddings is not None:
            metadata_list = [
                {
                    "article_id": faq["article_id"],
                    "title": faq["title"],
                    "content": faq["content"],
                    "language": faq.get("language", "en"),
                    "category": faq.get("category", "general"),
                }
                for faq in self._faq_data
            ]
            self.vector_store.add(embeddings, metadata_list)
            self.logger.info("faq_index_built", num_articles=len(self._faq_data))

        self._loaded = True

    async def process(self, input_data: BaseModel) -> KFOOutput:
        """Semantic FAQ retrieval."""
        data: KFOInput = input_data  # type: ignore[assignment]

        self._ensure_loaded()

        if not data.query_text.strip():
            return KFOOutput(suggested_faq_articles=[], updated_knowledge=False)

        # Encode query
        query_embedding = encode([data.query_text])
        if query_embedding is None:
            return KFOOutput(suggested_faq_articles=[], updated_knowledge=False)

        # Search
        results = self.vector_store.search(query_embedding[0], top_k=data.top_k)

        # Filter by language if specified
        lang = data.language.value if hasattr(data.language, "value") else str(data.language)
        filtered = [r for r in results if r.get("language", "en") == lang]

        # If no results for the specific language, use all results
        if not filtered:
            filtered = results

        articles = []
        for r in filtered[:data.top_k]:
            articles.append(
                FAQArticle(
                    article_id=r.get("article_id", ""),
                    title=r.get("title", ""),
                    content_snippet=r.get("content", "")[:200],
                    confidence_score=round(r.get("score", 0.0), 4),
                )
            )

        self.logger.info(
            "kfo_search_complete",
            interaction_id=data.interaction_id,
            query_length=len(data.query_text),
            results_count=len(articles),
        )

        return KFOOutput(
            suggested_faq_articles=articles,
            updated_knowledge=False,
        )
