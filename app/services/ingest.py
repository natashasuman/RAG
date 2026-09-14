import io
import uuid
from typing import List, Optional
from google import genai
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import settings
from app.models.schemas import IngestionResponse


class DocumentIngestionService:
    """Service to process PDFs / financial documents and index them in Qdrant."""

    def __init__(self, qdrant_client: Optional[QdrantClient] = None):
        self.client = qdrant_client or QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.genai_client = genai.Client(api_key=settings.GEMINI_API_KEY) if settings.GEMINI_API_KEY else None
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the target Qdrant collection exists."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == settings.COLLECTION_NAME for c in collections)
            if not exists:
                self.client.create_collection(
                    collection_name=settings.COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=768,
                        distance=models.Distance.COSINE
                    )
                )
        except Exception:
            pass

    def _get_embedding(self, text: str) -> List[float]:
        """Generate vector embedding."""
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
        return [0.01] * 768

    def _chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start += chunk_size - chunk_overlap
        return [c for c in chunks if c]

    def ingest_text(self, document_id: str, text: str, ticker: Optional[str] = None, page_number: int = 1) -> IngestionResponse:
        """Ingest raw text content into Qdrant."""
        chunks = self._chunk_text(text)
        points = []

        for idx, chunk in enumerate(chunks):
            embedding = self._get_embedding(chunk)
            point_id = str(uuid.uuid4())
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "document_id": document_id,
                        "ticker": ticker,
                        "page_number": page_number,
                        "chunk_index": idx,
                        "text": chunk
                    }
                )
            )

        if points:
            self.client.upsert(
                collection_name=settings.COLLECTION_NAME,
                points=points
            )

        return IngestionResponse(
            document_id=document_id,
            chunks_indexed=len(points),
            status="success"
        )

    def ingest_pdf(self, document_id: str, file_bytes: bytes, ticker: Optional[str] = None) -> IngestionResponse:
        """Parse PDF document and ingest page by page."""
        reader = PdfReader(io.BytesIO(file_bytes))
        total_chunks = 0

        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                res = self.ingest_text(
                    document_id=document_id,
                    text=page_text,
                    ticker=ticker,
                    page_number=page_idx + 1
                )
                total_chunks += res.chunks_indexed

        return IngestionResponse(
            document_id=document_id,
            chunks_indexed=total_chunks,
            status="success"
        )
