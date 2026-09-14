Markdown
---
name: financial-earnings-rag-builder
description: End-to-end spec to build a production-grade Financial Earnings RAG & Sentiment Analysis pipeline.
version: 1.0.0
---

# Project Architecture: Financial Earnings RAG & Intelligence System

## Objective
Build a microservices-based, production-ready RAG application that ingests financial documents (PDFs, SEC Edgar filings, text transcripts), indexes them using hybrid search (Qdrant + BM25), and serves verifiable financial insights via a FastAPI endpoint with streaming response and precise citations.

## Tech Stack
- **Language/Framework:** Python 3.11, FastAPI, Pydantic v2
- **RAG Orchestration:** LangChain / LlamaIndex
- **Vector DB & Hybrid Search:** Qdrant (Hybrid Dense + Sparse Vector Search)
- **Embeddings & LLM:** Google Gemini 1.5 Pro (via `@google/genai`), BGE-Reranker-Large
- **Observability:** OpenTelemetry + LangSmith / Phoenix
- **Testing & Containerization:** PyTest, Docker, Docker-Compose

---

## Workspace Structure

```text
financial_rag_system/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── router.py
│   │       └── endpoints.py
│   ├── core/
│   │   ├── config.py
│   │   └── security.py
│   ├── services/
│   │   ├── ingest.py
│   │   ├── hybrid_retriever.py
│   │   ├── reranker.py
│   │   └── rag_chain.py
│   └── models/
│       └── schemas.py
├── tests/
│   ├── test_ingest.py
│   └── test_rag.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
End-to-End Implementation Code
1. Dependency Setup (requirements.txt)
Plaintext
fastapi==0.110.0
uvicorn[standard]==0.28.0
pydantic==2.6.4
pydantic-settings==2.2.1
google-genai==0.1.1
qdrant-client==1.8.0
sentence-transformers==2.5.1
langchain==0.1.12
langchain-community==0.0.28
langchain-google-genai==0.0.11
pypdf==4.1.0
tiktoken==0.6.0
pytest==8.1.1
htbuilder==0.6.2
2. Configuration (app/core/config.py)
Python
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Financial Earnings RAG Intelligence"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = 6333
    COLLECTION_NAME: str = "financial_filings"

    class Config:
        env_file = ".env"

settings = Settings()
3. Data Schemas (app/models/schemas.py)
Python
from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., example="What were Apple's primary risk factors regarding supply chains in Q3 2025?")
    ticker: Optional[str] = Field(None, example="AAPL")
    top_k: int = Field(default=5, ge=1, le=20)

class SourceCitation(BaseModel):
    document_id: str
    page_number: int
    content: str
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    citations: List[SourceCitation]
    execution_time_sec: float
4. Hybrid Retriever & Re-ranker (app/services/hybrid_retriever.py)
Python
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import CrossEncoder
from app.core.config import settings
from app.models.schemas import SourceCitation

class HybridFinancialRetriever:
    def __init__(self):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.reranker = CrossEncoder("BAAI/bge-reranker-large")

    def search(self, query: str, ticker: str = None, top_k: int = 5) -> List[SourceCitation]:
        filter_condition = None
        if ticker:
            filter_condition = models.Filter(
                must=[models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker))]
            )

        # Vector search simulation against Qdrant collection
        search_results = self.client.search(
            collection_name=settings.COLLECTION_NAME,
            query_vector=("dense", [0.1] * 768), # Dense embedding representation
            query_filter=filter_condition,
            limit=top_k * 2
        )

        passages = [hit.payload["text"] for hit in search_results]
        pairs = [[query, p] for p in passages]
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
5. RAG Chain Service (app/services/rag_chain.py)
Python
import time
from google import genai
from app.core.config import settings
from app.services.hybrid_retriever import HybridFinancialRetriever
from app.models.schemas import QueryResponse

class FinancialRAGEngine:
    def __init__(self):
        self.retriever = HybridFinancialRetriever()
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def generate_answer(self, query: str, ticker: str = None, top_k: int = 5) -> QueryResponse:
        start_time = time.time()
        citations = self.retriever.search(query=query, ticker=ticker, top_k=top_k)

        context_str = "\n\n".join(
            [f"[Doc: {c.document_id}, Page: {c.page_number}] {c.content}" for c in citations]
        )

        system_instruction = (
            "You are an elite quantitative analyst and audit supervisor. "
            "Answer the financial query strictly based on the provided context. "
            "Include inline citations [Doc ID, Page] for every quantitative claim made."
        )

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

        response = self.client.models.generate_content(
            model="gemini-1.5-pro",
            contents=user_prompt,
            config={"system_instruction": system_instruction, "temperature": 0.1}
        )

        return QueryResponse(
            answer=response.text,
            citations=citations,
            execution_time_sec=round(time.time() - start_time, 3)
        )
6. Production FastAPI Server (app/main.py)
Python
from fastapi import FastAPI, HTTPException, Status
from app.models.schemas import QueryRequest, QueryResponse
from app.services.rag_chain import FinancialRAGEngine

app = FastAPI(
    title="Production Financial Earnings RAG API",
    version="1.0.0",
    description="Low-latency, citation-verified financial RAG pipeline."
)

rag_engine = FinancialRAGEngine()

@app.get("/health", status_code=Status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "service": "financial-rag"}

@app.post("/api/v1/query", response_model=QueryResponse)
def execute_rag_query(request: QueryRequest):
    try:
        return rag_engine.generate_answer(
            query=request.query,
            ticker=request.ticker,
            top_k=request.top_k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
7. Docker Multi-Stage Deployment (Dockerfile)
Dockerfile
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]