import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Financial Earnings RAG Intelligence"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "financial_filings")
    EMBEDDING_MODEL_NAME: str = "models/text-embedding-004"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-large"
    GEMINI_MODEL: str = "gemini-1.5-pro"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
