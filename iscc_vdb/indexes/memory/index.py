"""
In-memory ISCC index implementation for testing and development.

This module provides a simple, non-persistent index backend that stores all
data in memory using dictionaries. It's ideal for testing, development, and
scenarios where persistence isn't needed.
"""

import re
from iscc_vdb.schema import IsccAddResult, IsccIndex, IsccMatch, IsccSearchResult, Metric, Status


INDEX_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*$")


class MemoryIndex:
    """
    In-memory index implementing IsccIndexProtocol.

    Stores all data in memory using dictionaries. No persistence.
    Useful for testing and development.

    Storage structure:
        _indexes = {
            "index_name": {
                "items": {iscc_id: IsccItem, ...},
                "metadata": {}
            },
            ...
        }
    """

    def __init__(self):
        # type: () -> None
        """
        Initialize MemoryIndex.

        Creates empty in-memory storage for indexes and items.
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
                    items=len(data["items"]),
                    size=0,  # Memory indexes don't track size
                )
            )
        return indexes

    def create_index(self, index):
        # type: (IsccIndex) -> IsccIndex
        """
        Create new in-memory index.

        :param index: IsccIndex with name (items and size ignored)
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

        self._indexes[index.name] = {"items": {}, "metadata": {}}
        return IsccIndex(name=index.name, items=0, size=0)

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
        return IsccIndex(name=name, items=len(data["items"]), size=0)

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

    def add_items(self, index_name, items):
        # type: (str, list[IsccItem]) -> list[IsccAddResult]
        """
        Add items to in-memory index.

        Items are stored by iscc_id. If an item with the same iscc_id already
        exists, it's updated (not duplicated).

        :param index_name: Target index name
        :param items: List of IsccItem objects to add
        :return: List of IsccAddResult with status for each item
        :raises FileNotFoundError: If index doesn't exist
        :raises ValueError: If items contain invalid ISCC codes
        """
        if index_name not in self._indexes:
            raise FileNotFoundError(f"Index '{index_name}' not found")

        results = []
        index_data = self._indexes[index_name]

        for item in items:
            # Validate that item has required fields
            if item.iscc_id is None:
                raise ValueError("Item must have iscc_id field")

            # Check if item already exists
            status = Status.updated if item.iscc_id in index_data["items"] else Status.created

            # Store/update item
            index_data["items"][item.iscc_id] = item

            results.append(IsccAddResult(iscc_id=item.iscc_id, status=status))

        return results

    def search_items(self, index_name, query, limit=100):
        # type: (str, IsccItem, int) -> IsccSearchResult
        """
        Search for similar items (simple exact match for testing).

        This is a simplified implementation that performs exact matching
        on iscc_code. For production use, a real similarity search backend
        like usearch should be used.

        :param index_name: Target index name
        :param query: IsccItem to search for
        :param limit: Maximum number of results
        :return: IsccSearchResult with matches
        :raises FileNotFoundError: If index doesn't exist
        :raises ValueError: If query item is invalid
        """
        if index_name not in self._indexes:
            raise FileNotFoundError(f"Index '{index_name}' not found")

        # Simple implementation: exact match on iscc_code if available
        match_list = []
        index_data = self._indexes[index_name]

        for item in index_data["items"].values():
            # Match by iscc_code if both query and item have it
            if query.iscc_code and item.iscc_code:
                if item.iscc_code == query.iscc_code:
                    match_list.append(
                        IsccMatch(
                            iscc_id=item.iscc_id,  # type: ignore
                            score=1.0,
                            matches={},
                        )
                    )
            # Match by iscc_id if query has it
            elif query.iscc_id and item.iscc_id == query.iscc_id:
                match_list.append(
                    IsccMatch(
                        iscc_id=item.iscc_id,  # type: ignore
                        score=1.0,
                        matches={},
                    )
                )

        return IsccSearchResult(
            query=query,
            metric=Metric.bitlength,
            matches=match_list[:limit],
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
