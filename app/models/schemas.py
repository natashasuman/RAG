from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="The natural language financial question.",
        example="What were Apple's primary risk factors regarding supply chains in Q3 2025?"
    )
    ticker: Optional[str] = Field(
        None,
        description="Stock ticker symbol filter (e.g., AAPL, MSFT, TSLA).",
        example="AAPL"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top relevant chunks to retrieve after reranking."
    )


class SourceCitation(BaseModel):
    document_id: str = Field(..., description="ID or title of the source document.")
    page_number: int = Field(..., description="Page number where the content appears.")
    content: str = Field(..., description="The excerpted text passage.")
    relevance_score: float = Field(..., description="Cross-encoder relevance score.")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer with inline citations.")
    citations: List[SourceCitation] = Field(default_factory=list, description="List of source citations used.")
    execution_time_sec: float = Field(..., description="Total execution time in seconds.")


class IngestionRequest(BaseModel):
    document_id: str = Field(..., description="Unique document identifier or filename.")
    ticker: Optional[str] = Field(None, description="Stock ticker symbol if applicable.")
    text_content: Optional[str] = Field(None, description="Raw text to ingest.")


class IngestionResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    status: str
