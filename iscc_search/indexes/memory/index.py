"""
In-memory ISCC index implementation for testing and development.

This module provides a simple, non-persistent index backend that stores all
data in memory using dictionaries. It's ideal for testing, development, and
scenarios where persistence isn't needed.
"""

import re
from iscc_search.schema import IsccAddResult, IsccGlobalMatch, IsccIndex, IsccSearchResult, Status
from iscc_search.indexes import common


INDEX_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*$")


class MemoryIndex:
    """
    In-memory index implementing IsccIndexProtocol.

    Stores all data in memory using dictionaries. No persistence.
    Useful for testing and development.

    Storage structure:
        _indexes = {
            "index_name": {
                "assets": {iscc_id: IsccEntry, ...},
                "metadata": {}
            },
            ...
        }
    """

    def __init__(self):
        # type: () -> None
        """
        Initialize MemoryIndex.

        Creates empty in-memory storage for indexes and assets.
        """
        self._indexes = {}  # type: dict[str, dict]

    def list_indexes(self):
        # type: () -> list[IsccIndex]
        """
        List all in-memory indexes.

        :return: List of IsccIndex objects with metadata
        """
        indexes = []
        for name, data in self._indexes.items():
            indexes.append(
                IsccIndex(
                    name=name,
                    assets=len(data["assets"]),
                    size=0,  # Memory indexes don't track size
                )
            )
        return indexes

    def create_index(self, index):
        # type: (IsccIndex) -> IsccIndex
        """
        Create new in-memory index.

        :param index: IsccIndex with name (assets and size ignored)
        :return: Created IsccIndex with initial metadata
        :raises ValueError: If name is invalid
        :raises FileExistsError: If index already exists
        """
        # Validate index name (Pydantic already validates, this is defensive)
        if not INDEX_NAME_PATTERN.match(index.name):  # pragma: no cover
            raise ValueError(
                f"Invalid index name: '{index.name}'. Must match pattern ^[a-z][a-z0-9]*$ "
                f"(lowercase letters, digits, no special chars)"
            )

        if index.name in self._indexes:
            raise FileExistsError(f"Index '{index.name}' already exists")

        self._indexes[index.name] = {"assets": {}, "metadata": {}}
        return IsccIndex(name=index.name, assets=0, size=0)

    def get_index(self, name):
        # type: (str) -> IsccIndex
        """
        Get index metadata.

        :param name: Index name
        :return: IsccIndex with current metadata
        :raises FileNotFoundError: If index doesn't exist
        """
        if name not in self._indexes:
            raise FileNotFoundError(f"Index '{name}' not found")

        data = self._indexes[name]
        return IsccIndex(name=name, assets=len(data["assets"]), size=0)

    def delete_index(self, name):
        # type: (str) -> None
        """
        Delete in-memory index.

        :param name: Index name
        :raises FileNotFoundError: If index doesn't exist
        """
        if name not in self._indexes:
            raise FileNotFoundError(f"Index '{name}' not found")

        del self._indexes[name]

    def add_assets(self, index_name, assets):
        # type: (str, list[IsccEntry]) -> list[IsccAddResult]
        """
        Add assets to in-memory index.

        Assets are stored by iscc_id. If an asset with the same iscc_id already
        exists, it's updated (not duplicated). The iscc_id field must be provided
        by the client when adding assets.

        :param index_name: Target index name
        :param assets: List of IsccEntry objects to add (must include iscc_id)
        :return: List of IsccAddResult with status for each asset
        :raises FileNotFoundError: If index doesn't exist
        :raises ValueError: If asset is missing required iscc_id field
        """
        if index_name not in self._indexes:
            raise FileNotFoundError(f"Index '{index_name}' not found")

        results = []
        index_data = self._indexes[index_name]

        for asset in assets:
            # Validate that asset has required iscc_id for add operations
            if asset.iscc_id is None:
                raise ValueError("Asset must have iscc_id field when adding to index")

            # Check if asset already exists
            status = Status.updated if asset.iscc_id in index_data["assets"] else Status.created

            # Store/update asset
            index_data["assets"][asset.iscc_id] = asset

            results.append(IsccAddResult(iscc_id=asset.iscc_id, status=status))

        return results

    def get_asset(self, index_name, iscc_id):
        # type: (str, str) -> IsccEntry
        """
        Get a specific asset by ISCC-ID.

        :param index_name: Target index name
        :param iscc_id: ISCC-ID of the asset to retrieve
        :return: IsccEntry with all stored metadata
        :raises FileNotFoundError: If index doesn't exist or asset not found
        :raises ValueError: If ISCC-ID format is invalid
        """
        if index_name not in self._indexes:
            raise FileNotFoundError(f"Index '{index_name}' not found")

        index_data = self._indexes[index_name]

        if iscc_id not in index_data["assets"]:
            raise FileNotFoundError(f"Asset '{iscc_id}' not found in index '{index_name}'")

        return index_data["assets"][iscc_id]

    def search_assets(self, index_name, query, limit=100):
        # type: (str, IsccEntry, int) -> IsccSearchResult
        """
        Search for similar assets (simple exact match for testing).

        This is a simplified implementation that performs exact matching
        on iscc_code. For production use, a real similarity search backend
        like usearch should be used.

        Accepts query with either iscc_code or units (or both). If only units
        are provided, iscc_code is automatically derived for matching (when
        units form a valid code). Units-only queries that don't form valid
        ISCC-CODEs will return no matches.

        :param index_name: Target index name
        :param query: IsccEntry with iscc_code or units (or both)
        :param limit: Maximum number of results
        :return: IsccSearchResult with matches
        :raises FileNotFoundError: If index doesn't exist
        :raises ValueError: If query has neither iscc_code nor units
        """
        if index_name not in self._indexes:
            raise FileNotFoundError(f"Index '{index_name}' not found")

        # Normalize query to ensure it has units (derive from iscc_code if needed)
        # This ensures consistent behavior across backends
        query = common.normalize_query_asset(query)

        # Simple implementation: exact match on iscc_code if available
        match_list = []
        index_data = self._indexes[index_name]

        for asset in index_data["assets"].values():
            # Match by iscc_code (query always has iscc_code after normalization)
            if query.iscc_code and asset.iscc_code:
                if asset.iscc_code == query.iscc_code:
                    match_list.append(
                        IsccGlobalMatch(
                            iscc_id=asset.iscc_id,  # type: ignore
                            score=1.0,
                            types={},
                            metadata=asset.metadata,
                        )
                    )

        return IsccSearchResult(
            query=query,
            global_matches=match_list[:limit],
        )

    def close(self):
        # type: () -> None
        """
        No-op for in-memory index.

        Since there are no external resources to clean up (no file handles,
        database connections, etc.), this method does nothing. It's provided
        for protocol compliance.
        """
        pass
