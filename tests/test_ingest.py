from unittest.mock import MagicMock
from app.services.ingest import DocumentIngestionService


def test_chunk_text():
    mock_client = MagicMock()
    service = DocumentIngestionService(qdrant_client=mock_client)
    
    sample_text = "Apple Inc. reports record Q3 revenue driven by iPhone sales and Services momentum. " * 30
    chunks = service._chunk_text(sample_text, chunk_size=200, chunk_overlap=50)
    
    assert len(chunks) > 1
    assert all(len(c) > 0 for c in chunks)


def test_ingest_text():
    mock_client = MagicMock()
    service = DocumentIngestionService(qdrant_client=mock_client)
    service.genai_client = None  # Use fallback embedding

    res = service.ingest_text(
        document_id="AAPL_Q3_2025",
        text="Total revenue grew 8% year-over-year to $94.9 billion.",
        ticker="AAPL"
    )

    assert res.status == "success"
    assert res.document_id == "AAPL_Q3_2025"
    assert res.chunks_indexed >= 1
    assert mock_client.upsert.called
