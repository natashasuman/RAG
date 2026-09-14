from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from typing import Optional

from app.core.security import verify_api_key
from app.models.schemas import (
    IngestionRequest,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.ingest import DocumentIngestionService
from app.services.rag_chain import FinancialRAGEngine

router = APIRouter()
rag_engine = FinancialRAGEngine()
ingestion_service = DocumentIngestionService()


@router.post("/query", response_model=QueryResponse, summary="Execute Financial RAG Query")
def execute_rag_query(request: QueryRequest, authorized: bool = Depends(verify_api_key)):
    """Query the indexed financial earnings and disclosures with citations."""
    try:
        return rag_engine.generate_answer(
            query=request.query,
            ticker=request.ticker,
            top_k=request.top_k
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )


@router.post("/ingest/text", response_model=IngestionResponse, summary="Ingest Plain Financial Text")
def ingest_text_document(request: IngestionRequest, authorized: bool = Depends(verify_api_key)):
    """Ingest raw financial text into vector store."""
    if not request.text_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="text_content must not be empty"
        )
    try:
        return ingestion_service.ingest_text(
            document_id=request.document_id,
            text=request.text_content,
            ticker=request.ticker
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text ingestion failed: {str(e)}"
        )


@router.post("/ingest/pdf", response_model=IngestionResponse, summary="Ingest Financial PDF")
async def ingest_pdf_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    ticker: Optional[str] = Form(None),
    authorized: bool = Depends(verify_api_key)
):
    """Upload and ingest a financial report PDF (10-K, 10-Q, transcript)."""
    try:
        contents = await file.read()
        doc_id = document_id or file.filename or "unknown_doc"
        return ingestion_service.ingest_pdf(
            document_id=doc_id,
            file_bytes=contents,
            ticker=ticker
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF ingestion failed: {str(e)}"
        )
