"""Index management endpoints for ISCC-VDB API."""

from fastapi import APIRouter, Depends, HTTPException, status
from iscc_search.protocol import IsccIndexProtocol
from iscc_search.schema import IsccIndex
from iscc_search.server import get_index_from_state


router = APIRouter(tags=["indexes"])


@router.get("/indexes", response_model=list[IsccIndex])
def list_indexes(index: IsccIndexProtocol = Depends(get_index_from_state)):
    # type: (...) -> list[IsccIndex]
    """
    List all available indexes with metadata.

    Returns information about all existing indexes including name, asset count,
    and storage size.

    :param index: Index instance injected from app state
    :return: List of IsccIndex objects
    """
    return index.list_indexes()


@router.post("/indexes", response_model=IsccIndex, status_code=status.HTTP_201_CREATED)
def create_index(
    index_data: IsccIndex,
    index: IsccIndexProtocol = Depends(get_index_from_state),
):
    # type: (...) -> IsccIndex
    """
    Create a new ISCC index.

    Initializes a new index with the specified name. The index starts empty
    with 0 assets.

    :param index_data: Index configuration with name
    :param index: Index instance injected from app state
    :return: Created index with initial metadata
    :raises HTTPException: 400 for invalid name, 409 if index already exists
    """
    try:
        return index.create_index(index_data)
    except ValueError as e:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except FileExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get("/indexes/{name}", response_model=IsccIndex)
def get_index(name: str, index: IsccIndexProtocol = Depends(get_index_from_state)):
    # type: (...) -> IsccIndex
    """
    Get metadata for a specific index.

    Returns current information about the specified index including asset count
    and storage size.

    :param name: Index name
    :param index: Index instance injected from app state
    :return: IsccIndex with current metadata
    :raises HTTPException: 404 if index not found
    """
    try:
        return index.get_index(name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete("/indexes/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_index(name: str, index: IsccIndexProtocol = Depends(get_index_from_state)):
    # type: (...) -> None
    """
    Delete an index and all its data.

    Permanently removes the index and all associated assets. This operation
    cannot be undone.

    :param name: Index name
    :param index: Index instance injected from app state
    :raises HTTPException: 404 if index not found
    """
    try:
        index.delete_index(name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
