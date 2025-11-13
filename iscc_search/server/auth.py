"""Authentication utilities for ISCC-Search API."""

import secrets
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from iscc_search.settings import search_settings


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key=Security(api_key_header)):
    # type: (str | None) -> None
    """
    Verify API key if API_SECRET is configured.

    When `api_secret` is None (default), no authentication is required and all
    requests are allowed (public mode).

    When `api_secret` is set, requests must include a matching `X-API-Key` header.
    Uses constant-time comparison to prevent timing attacks.

    :param api_key: API key from X-API-Key header (None if not provided)
    :raises HTTPException: 401 Unauthorized if key is invalid or missing
    """
    # Public mode - no authentication required
    if search_settings.api_secret is None:
        return

    # Protected mode - require valid API key
    if api_key is None or not secrets.compare_digest(api_key, search_settings.api_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )
