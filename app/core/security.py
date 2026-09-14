"""Security and authentication utilities for API endpoints."""
import os
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> bool:
    """Validate API key if API_SECRET_KEY is configured."""
    if not API_SECRET_KEY:
        return True  # If no API secret is configured, bypass check in dev mode
    if api_key != API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key."
        )
    return True
