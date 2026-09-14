from unittest.mock import MagicMock
from app.models.schemas import SourceCitation
from app.services.rag_chain import FinancialRAGEngine


def test_rag_generation_no_citations():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    
    engine = FinancialRAGEngine(retriever=mock_retriever)
    response = engine.generate_answer(query="What was Apple's gross margin?")
    
    assert "No relevant financial documents" in response.answer
    assert response.citations == []
    assert response.execution_time_sec >= 0


def test_rag_generation_with_citations_fallback():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        SourceCitation(
            document_id="AAPL_10Q",
            page_number=12,
            content="Gross margin was 46.2 percent compared to 44.5 percent.",
            relevance_score=0.95
        )
    ]
    
    engine = FinancialRAGEngine(retriever=mock_retriever)
    engine.client = None  # test fallback logic
    
    response = engine.generate_answer(query="What was Apple's gross margin?", ticker="AAPL")
    
    assert len(response.citations) == 1
    assert response.citations[0].document_id == "AAPL_10Q"
    assert "Gross margin was 46.2 percent" in response.answer
    assert response.execution_time_sec >= 0
