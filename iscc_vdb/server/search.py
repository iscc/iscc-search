"""Search endpoints for ISCC-VDB API."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from iscc_vdb.protocol import IsccIndexProtocol
from iscc_vdb.schema import IsccAsset, IsccSearchResult
from iscc_vdb.server import get_index_from_state


router = APIRouter(tags=["search"])


@router.post("/indexes/{name}/search", response_model=IsccSearchResult)
def search_post(
    name: str,
    query: IsccAsset,
    limit: int = 100,
    index: IsccIndexProtocol = Depends(get_index_from_state),
):
    # type: (...) -> IsccSearchResult
    """
    Search for similar assets using POST with full query asset.

    Performs similarity search using the query asset's ISCC-CODE or units.
    Results are aggregated across all unit types and returned sorted by
    relevance (highest scores first).

    :param name: Index name
    :param query: IsccAsset to search for (iscc_code or units required)
    :param limit: Maximum number of results to return (default: 100)
    :param index: Index instance injected from app state
    :return: IsccSearchResult with matches
    :raises HTTPException: 404 if index not found, 400 for invalid query
    """
    try:
        return index.search_assets(name, query, limit)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/indexes/{name}/search", response_model=IsccSearchResult)
def search_get(
    name: str,
    iscc_code: str = Query(..., description="ISCC-CODE to search for"),
    limit: int = 100,
    index: IsccIndexProtocol = Depends(get_index_from_state),
):
    # type: (...) -> IsccSearchResult
    """
    Search for similar assets using GET with ISCC-CODE query parameter.

    Convenience endpoint for searching by ISCC-CODE. The ISCC-CODE is wrapped
    in an IsccAsset and passed to the backend, which handles decomposition
    into units internally.

    :param name: Index name
    :param iscc_code: ISCC-CODE to search for (query parameter)
    :param limit: Maximum number of results to return (default: 100)
    :param index: Index instance injected from app state
    :return: IsccSearchResult with matches
    :raises HTTPException: 404 if index not found, 400 for invalid ISCC-CODE
    """
    try:
        # Create query asset with iscc_code - backend handles decomposition
        query = IsccAsset(iscc_code=iscc_code)
        return index.search_assets(name, query, limit)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
