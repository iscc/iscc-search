"""Authentication and endpoint-gating utilities for ISCC-Search API."""

import secrets
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from iscc_search.options import search_opts


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def block_if_aggregator():
    # type: () -> None
    """
    Suppress an endpoint in aggregator mode with a route-hiding 404.

    Attached via route-level `dependencies=[Depends(block_if_aggregator)]` so it
    resolves before parameter-level dependencies like verify_api_key — a key-less
    request to a suppressed endpoint gets 404, never 401. Reads the mode flag
    from search_opts (like verify_api_key), so gating is enforced regardless of
    whether the lifespan ran.

    :raises HTTPException: 404 Not Found when aggregator mode is active
    """
    if search_opts.aggregator_mode:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")


def block_foreign_index_if_aggregator(name: str):
    # type: (...) -> None
    """
    Restrict the read surface to the aggregator index in aggregator mode.

    In aggregator mode, reads against any index other than the derived
    aggregator index (idp/idptest) get the same route-hiding 404 as the
    suppressed endpoints. Normal mode is unaffected.

    :param name: Index name from the route path
    :raises HTTPException: 404 Not Found for a foreign index in aggregator mode
    """
    if search_opts.aggregator_mode and name != search_opts.aggregator_index_name:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")


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
    if search_opts.api_secret is None:
        return

    # Protected mode - require valid API key
    if api_key is None or not secrets.compare_digest(api_key, search_opts.api_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )
