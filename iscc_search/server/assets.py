"""Asset management endpoints for ISCC-Search API."""

from fastapi import APIRouter, Depends, HTTPException, status
from iscc_search.protocols.index import IsccIndexProtocol
from iscc_search.schema import IsccAddResult, IsccEntry
from iscc_search.server import get_index_from_state


router = APIRouter(tags=["assets"])


@router.post(
    "/indexes/{name}/assets",
    response_model=list[IsccAddResult],
    status_code=status.HTTP_201_CREATED,
)
def add_assets(
    name: str,
    assets: list[IsccEntry],
    index: IsccIndexProtocol = Depends(get_index_from_state),
):
    # type: (...) -> list[IsccAddResult]
    """
    Add multiple assets to an index.

    Adds ISCC assets to the specified index. Each asset must contain an
    iscc_id field and at least one of iscc_code or units. Returns status
    for each asset indicating whether it was created or updated.

    :param name: Index name
    :param assets: List of IsccEntry objects to add
    :param index: Index instance injected from app state
    :return: List of IsccAddResult with status for each asset
    :raises HTTPException: 404 if index not found, 400 for invalid assets
    """
    try:
        return index.add_assets(name, assets)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/indexes/{name}/assets/{iscc_id}", response_model=IsccEntry)
def get_asset(
    name: str,
    iscc_id: str,
    index: IsccIndexProtocol = Depends(get_index_from_state),
):
    # type: (...) -> IsccEntry
    """
    Get a specific asset by ISCC-ID.

    Retrieves the full asset details for a given ISCC-ID from the specified
    index. This is useful for fetching complete metadata after a search
    operation, which returns only ISCC-IDs and scores.

    :param name: Index name
    :param iscc_id: ISCC-ID of the asset to retrieve
    :param index: Index instance injected from app state
    :return: IsccEntry with all stored data
    :raises HTTPException: 404 if index or asset not found, 400 for invalid ISCC-ID
    """
    try:
        return index.get_asset(name, iscc_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
