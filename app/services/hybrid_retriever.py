from typing import List, Optional
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import settings
from app.models.schemas import SourceCitation
from app.services.reranker import CrossEncoderReranker


class HybridFinancialRetriever:
    """Hybrid financial document retriever using Qdrant vector search and CrossEncoder reranker."""

    def __init__(self, qdrant_client: Optional[QdrantClient] = None, reranker: Optional[CrossEncoderReranker] = None):
        self.client = qdrant_client or QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.reranker = reranker or CrossEncoderReranker()
        self.genai_client = genai.Client(api_key=settings.GEMINI_API_KEY) if settings.GEMINI_API_KEY else None

    def _get_embedding(self, text: str) -> List[float]:
        """Generate text embedding vector using Google GenAI or fallback representation."""
        if self.genai_client:
            try:
                response = self.genai_client.models.embed_content(
                    model=settings.EMBEDDING_MODEL_NAME,
                    contents=text
                )
                if hasattr(response, "embedding") and response.embedding:
                    return response.embedding.values
                if hasattr(response, "embeddings") and response.embeddings:
                    return response.embeddings[0].values
            except Exception:
                pass
        # Fallback dummy vector of size 768 for offline/testing mode
        return [0.01] * 768

    def search(self, query: str, ticker: Optional[str] = None, top_k: int = 5) -> List[SourceCitation]:
        """Retrieve and rerank the most relevant source citations for the query."""
        filter_condition = None
        if ticker:
            filter_condition = models.Filter(
                must=[models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker))]
            )

        query_vec = self._get_embedding(query)

        try:
            search_results = self.client.search(
                collection_name=settings.COLLECTION_NAME,
                query_vector=query_vec,
                query_filter=filter_condition,
                limit=top_k * 3
            )
        except Exception:
            search_results = []

        if not search_results:
            return []

        passages = [hit.payload.get("text", "") for hit in search_results]
        pairs = [(query, p) for p in passages]
        scores = self.reranker.predict(pairs)

        citations = []
        for idx, score in enumerate(scores):
            hit = search_results[idx]
            citations.append(
                SourceCitation(
                    document_id=hit.payload.get("document_id", "doc_unknown"),
                    page_number=hit.payload.get("page_number", 1),
                    content=hit.payload.get("text", ""),
                    relevance_score=float(score)
                )
            )

        citations.sort(key=lambda x: x.relevance_score, reverse=True)
        return citations[:top_k]
