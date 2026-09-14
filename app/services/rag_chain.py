import time
from typing import Optional
from google import genai

from app.core.config import settings
from app.models.schemas import QueryResponse
from app.services.hybrid_retriever import HybridFinancialRetriever


class FinancialRAGEngine:
    """End-to-end RAG Engine generating citation-verified financial responses."""

    def __init__(self, retriever: Optional[HybridFinancialRetriever] = None):
        self.retriever = retriever or HybridFinancialRetriever()
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY) if settings.GEMINI_API_KEY else None

    def generate_answer(self, query: str, ticker: Optional[str] = None, top_k: int = 5) -> QueryResponse:
        start_time = time.time()
        citations = self.retriever.search(query=query, ticker=ticker, top_k=top_k)

        if not citations:
            return QueryResponse(
                answer="No relevant financial documents or disclosures were found to answer this query.",
                citations=[],
                execution_time_sec=round(time.time() - start_time, 3)
            )

        context_str = "\n\n".join(
            [f"[Doc: {c.document_id}, Page: {c.page_number}] {c.content}" for c in citations]
        )

        system_instruction = (
            "You are an elite quantitative analyst and audit supervisor. "
            "Answer the financial query strictly based on the provided context. "
            "Include inline citations [Doc ID, Page] for every quantitative claim made."
        )

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

        if self.client:
            try:
                response = self.client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=user_prompt,
                    config={"system_instruction": system_instruction, "temperature": 0.1}
                )
                answer = response.text
            except Exception as e:
                answer = f"Error communicating with Gemini model: {str(e)}"
        else:
            # Fallback mock answer if API key is not configured
            answer = f"Based on available disclosures for {ticker or 'the query'}: {citations[0].content[:200]}... [Doc: {citations[0].document_id}, Page: {citations[0].page_number}]"

        return QueryResponse(
            answer=answer,
            citations=citations,
            execution_time_sec=round(time.time() - start_time, 3)
        )
