# Financial Earnings RAG Intelligence System

Production-grade, low-latency Financial Earnings RAG and disclosure analysis pipeline built with FastAPI, Qdrant vector database, CrossEncoder reranking, and Google Gemini.

---

## 🏗️ Architecture Overview

- **FastAPI**: Asynchronous high-performance REST API.
- **Qdrant**: Scalable vector search engine for dense/sparse embedding retrieval.
- **CrossEncoder (`bge-reranker-large`)**: Precision passage re-ranking.
- **Google Gemini 1.5 Pro**: Context-grounded response generation with inline citations.
- **Document Ingestion**: Parsing PDF and text documents into structured vector chunks.

---

## 📁 Project Structure

```text
financial_rag_system/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI entry point & health check
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── router.py        # API router
│   │       └── endpoints.py     # Endpoints for query & ingestion
│   ├── core/
│   │   ├── config.py            # Pydantic Settings
│   │   └── security.py          # API key security
│   ├── services/
│   │   ├── ingest.py            # PDF / text ingestion and chunking
│   │   ├── hybrid_retriever.py  # Qdrant retrieval + reranking
│   │   ├── reranker.py          # BGE CrossEncoder wrapper
│   │   └── rag_chain.py         # Gemini RAG generation chain
│   └── models/
│       └── schemas.py           # Pydantic schemas
├── tests/
│   ├── test_ingest.py
│   └── test_rag.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Getting Started

### 1. Environment Setup

Copy `.env.example` to `.env` and set your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=financial_filings
```

### 2. Local Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run with Docker Compose

To spin up both Qdrant and the FastAPI application:

```bash
docker-compose up --build
```

### 4. Run API Locally

Ensure Qdrant is running on port 6333, then start the FastAPI server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 🧪 Testing

Run unit tests with pytest:

```bash
pytest tests/ -v
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check endpoint |
| `POST` | `/api/v1/query` | Execute financial question answering with citations |
| `POST` | `/api/v1/ingest/text` | Ingest raw financial text |
| `POST` | `/api/v1/ingest/pdf` | Upload and ingest PDF report (10-K, 10-Q) |
