"""
ISCC Index Protocol Definition

Defines the protocol interface that all ISCC index implementations must satisfy.
This protocol-based abstraction enables multiple backend implementations (usearch,
postgres, memory) to be used interchangeably through a unified interface.
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from iscc_search.schema import IsccAddResult  # noqa: F401
    from iscc_search.schema import IsccEntry  # noqa: F401
    from iscc_search.schema import IsccIndex  # noqa: F401
    from iscc_search.schema import IsccSearchResult  # noqa: F401


@runtime_checkable
class IsccIndexProtocol(Protocol):
    """
    Protocol for ISCC index backends.

    All methods are synchronous. Backends are free to use
    threading, connection pools, etc. internally.

    This protocol defines the core operations that all ISCC index implementations
    must support:
    - Index lifecycle: create, get, list, delete
    - Asset operations: add, search
    - Resource cleanup: close

    All index implementations should handle the protocol's exception contract:
    - ValueError: Invalid parameters or validation failures
    - FileExistsError: Attempting to create an existing index
    - FileNotFoundError: Attempting to access a non-existent index
    """

    def list_indexes(self):
        # type: () -> list[IsccIndex]
        """
        List all available indexes with metadata.

        Scans the backend storage and returns metadata for all existing indexes.
        The metadata includes index name, asset count, and storage size.

        :return: List of IsccIndex objects with name, assets, and size
        """
        ...

    def create_index(self, index):
        # type: (IsccIndex) -> IsccIndex
        """
        Create a new named index.

        Initializes a new index with the specified name. The index starts empty
        with 0 assets. If the backend requires initialization (creating directories,
        database tables, etc.), this method handles it.

        :param index: IsccIndex with name (assets and size fields are ignored)
        :return: Created IsccIndex with initial metadata (assets=0, size=0)
        :raises ValueError: If name is invalid (doesn't match pattern ^[a-z][a-z0-9]*$)
        :raises FileExistsError: If index with this name already exists
        """
        ...

    def get_index(self, name):
        # type: (str) -> IsccIndex
        """
        Get index metadata by name.

        Retrieves current metadata for the specified index, including
        the number of assets and storage size. This is useful for monitoring
        index growth and health.

        :param name: Index name (must match pattern ^[a-z][a-z0-9]*$)
        :return: IsccIndex with current metadata
        :raises FileNotFoundError: If index doesn't exist
        """
        ...

    def delete_index(self, name):
        # type: (str) -> None
        """
        Delete an index and all its data.

        Permanently removes the index and all associated data. This operation
        cannot be undone. Implementations should clean up all resources
        (files, database tables, etc.).

        :param name: Index name
        :raises FileNotFoundError: If index doesn't exist
        """
        ...

    def add_assets(self, index_name, assets):
        # type: (str, list[IsccEntry]) -> list[IsccAddResult]
        """
        Add assets to index.

        Adds multiple ISCC assets to the specified index. Each asset contains
        an ISCC-ID and ISCC-UNITs for similarity indexing. Assets with missing
        ISCC-IDs will have them auto-generated.

        Implementations should:
        - Store asset metadata for later retrieval
        - Index ISCC-UNITs by type for similarity search
        - Handle duplicates gracefully (update vs create)
        - Return status for each asset

        :param index_name: Target index name
        :param assets: List of IsccEntry objects to add
        :return: List of IsccAddResult with status for each asset
        :raises FileNotFoundError: If index doesn't exist
        :raises ValueError: If assets contain invalid ISCC codes
        """
        ...

    def get_asset(self, index_name, iscc_id):
        # type: (str, str) -> IsccEntry
        """
        Get a specific asset by ISCC-ID.

        Retrieves the full asset details for a given ISCC-ID from the specified
        index. This is useful for fetching complete asset metadata after performing
        a search, which returns only ISCC-IDs and scores.

        :param index_name: Target index name
        :param iscc_id: ISCC-ID of the asset to retrieve
        :return: IsccEntry with all stored metadata
        :raises FileNotFoundError: If index doesn't exist or asset not found
        :raises ValueError: If ISCC-ID format is invalid
        """
        ...

    def search_assets(self, index_name, query, limit=100):
        # type: (str, IsccEntry, int) -> IsccSearchResult
        """
        Search for similar assets in index.

        Performs similarity search using the query asset's ISCC-UNITs.
        Results are aggregated across all unit types and returned sorted
        by relevance (highest scores first).

        The returned IsccSearchResult includes:
        - query: The original query asset (may have auto-generated iscc_id)
        - metric: The distance metric used (nphd, hamming, bitlength)
        - global_matches: List of IsccGlobalMatch objects with scores and per-unit breakdowns

        :param index_name: Target index name
        :param query: IsccEntry to search for (either iscc_code or units required)
        :param limit: Maximum number of results to return (default: 100)
        :return: IsccSearchResult with query, metric, and list of matches
        :raises FileNotFoundError: If index doesn't exist
        :raises ValueError: If query asset is invalid
        """
        ...

    def close(self):
        # type: () -> None
        """
        Close connections and cleanup resources.

        Should be called when the backend is no longer needed. Implementations
        should clean up resources like database connections, file handles, and
        memory caches. This method should be idempotent (safe to call multiple times).

        After calling close(), the index instance should not be used for further
        operations.
        """
        ...
