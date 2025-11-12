"""Search endpoints for ISCC-Search API."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from iscc_search.processing import text_simprints
from iscc_search.protocols.index import IsccIndexProtocol
from iscc_search.schema import IsccQuery, IsccSearchResult, TextQuery
from iscc_search.server import get_index_from_state


router = APIRouter(tags=["search"])


@router.post("/indexes/{name}/search", response_model=IsccSearchResult)
def search_post(
    name: str,
    query: IsccQuery,
    limit: int = 100,
    index: IsccIndexProtocol = Depends(get_index_from_state),
):
    # type: (...) -> IsccSearchResult
    """
    Search for similar assets using POST with full query.

    Performs similarity search using the query's ISCC-CODE or units.
    Results are aggregated across all unit types and returned sorted by
    relevance (highest scores first).

    :param name: Index name
    :param query: IsccQuery to search for (iscc_code or units required)
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
    in an IsccQuery and passed to the backend, which handles decomposition
    into units internally.

    :param name: Index name
    :param iscc_code: ISCC-CODE to search for (query parameter)
    :param limit: Maximum number of results to return (default: 100)
    :param index: Index instance injected from app state
    :return: IsccSearchResult with matches
    :raises HTTPException: 404 if index not found, 400 for invalid ISCC-CODE
    """
    try:
        # Create query with iscc_code - backend handles decomposition
        query = IsccQuery(iscc_code=iscc_code)
        return index.search_assets(name, query, limit)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/indexes/{name}/search/text", response_model=IsccSearchResult)
def search_text(
    name: str,
    text_query: TextQuery,
    limit: int = 100,
    index: IsccIndexProtocol = Depends(get_index_from_state),
):
    # type: (...) -> IsccSearchResult
    """
    Search for similar content by plain text.

    Generates CONTENT_TEXT_V0 simprints from the input text and searches for
    similar content segments in the index. Returns chunk-level matches based on
    text similarity.

    :param name: Index name
    :param text_query: TextQuery with plain text content
    :param limit: Maximum number of results to return (default: 100)
    :param index: Index instance injected from app state
    :return: IsccSearchResult with chunk_matches
    :raises HTTPException: 404 if index not found, 400 for invalid text
    """
    try:
        # Generate simprints from text
        simprints = text_simprints(text_query.text)

        # Create query with generated simprints
        query = IsccQuery(simprints={"CONTENT_TEXT_V0": simprints})

        # Search using the simprint query
        return index.search_assets(name, query, limit)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
