"""
LMDB Index Manager - Protocol Implementation.

Manages multiple LMDB-backed indexes in a base directory.
Each index is stored as a separate .lmdb file (e.g., myindex.lmdb).

Implements IsccIndexProtocol for use as backend in CLI and server.
"""

import os
from pathlib import Path
from iscc_search.schema import IsccIndex
from iscc_search.indexes.lmdb.index import LmdbIndex
from iscc_search.indexes import common


class LmdbIndexManager:
    """
    Protocol implementation managing multiple LMDB indexes.

    Directory structure:
    base_path/
    ├── index1.lmdb
    ├── index2.lmdb
    └── ...

    Each .lmdb file is managed by a separate LmdbIndex instance.
    Instances are cached for performance.

    CONCURRENCY: LMDB-only indexes support multi-reader/single-writer with built-in locking.
    However, the instance cache does not synchronize between processes. For production use,
    consider single-process deployment with async/await for concurrent connections.
    """

    def __init__(self, base_path):
        # type: (os.PathLike) -> None
        """
        Initialize LmdbIndexManager.

        Creates base directory if it doesn't exist.

        :param base_path: Directory containing .lmdb index files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._index_cache = {}  # type: dict[str, LmdbIndex]

    def list_indexes(self):
        # type: () -> list[IsccIndex]
        """
        List all indexes by scanning for *.lmdb files.

        :return: List of IsccIndex objects with metadata
        """
        indexes = []

        for lmdb_file in self.base_path.glob("*.lmdb"):
            # Extract index name from filename
            name = lmdb_file.stem

            # Get metadata
            try:
                idx = self._get_or_load_index(name)
                asset_count = idx.get_asset_count()
                size_mb = self._get_file_size_mb(lmdb_file)

                indexes.append(IsccIndex(name=name, assets=asset_count, size=size_mb))
            except Exception:
                # Skip corrupted or inaccessible indexes
                continue

        # Sort by name for consistent ordering
        indexes.sort(key=lambda x: x.name)
        return indexes

    def create_index(self, index):
        # type: (IsccIndex) -> IsccIndex
        """
        Create new index.

        :param index: IsccIndex with name (assets and size ignored)
        :return: Created IsccIndex with initial metadata (assets=0, size=0)
        :raises ValueError: If name is invalid
        :raises FileExistsError: If index already exists
        """
        # Validate name
        common.validate_index_name(index.name)

        # Check if exists
        index_path = self.base_path / f"{index.name}.lmdb"
        if index_path.exists():
            raise FileExistsError(f"Index '{index.name}' already exists")

        # Create new LmdbIndex
        idx = LmdbIndex(index_path)
        self._index_cache[index.name] = idx

        return IsccIndex(name=index.name, assets=0, size=0)

    def get_index(self, name):
        # type: (str) -> IsccIndex
        """
        Get index metadata by name.

        :param name: Index name
        :return: IsccIndex with current metadata
        :raises FileNotFoundError: If index doesn't exist
        """
        self._validate_index_exists(name)

        # Load index and get metadata
        idx = self._get_or_load_index(name)
        asset_count = idx.get_asset_count()
        index_path = self.base_path / f"{name}.lmdb"
        size_mb = self._get_file_size_mb(index_path)

        return IsccIndex(name=name, assets=asset_count, size=size_mb)

    def delete_index(self, name):
        # type: (str) -> None
        """
        Delete index and all its data.

        :param name: Index name
        :raises FileNotFoundError: If index doesn't exist
        """
        self._validate_index_exists(name)

        # Close cached instance if open
        if name in self._index_cache:
            self._index_cache[name].close()
            del self._index_cache[name]

        # Delete file
        index_path = self.base_path / f"{name}.lmdb"
        os.remove(index_path)

    def add_assets(self, index_name, assets):
        # type: (str, list[IsccAsset]) -> list[IsccAddResult]
        """
        Add assets to index.

        :param index_name: Target index name
        :param assets: List of IsccAsset objects to add
        :return: List of IsccAddResult with status for each asset
        :raises FileNotFoundError: If index doesn't exist
        :raises ValueError: If asset validation fails
        """
        self._validate_index_exists(index_name)

        # Delegate to LmdbIndex
        idx = self._get_or_load_index(index_name)
        return idx.add_assets(assets)

    def get_asset(self, index_name, iscc_id):
        # type: (str, str) -> IsccAsset
        """
        Get a specific asset by ISCC-ID.

        :param index_name: Target index name
        :param iscc_id: ISCC-ID of the asset to retrieve
        :return: IsccAsset with all stored metadata
        :raises FileNotFoundError: If index doesn't exist or asset not found
        :raises ValueError: If ISCC-ID format is invalid
        """
        self._validate_index_exists(index_name)

        # Delegate to LmdbIndex
        idx = self._get_or_load_index(index_name)
        return idx.get_asset(iscc_id)

    def search_assets(self, index_name, query, limit=100):
        # type: (str, IsccAsset, int) -> IsccSearchResult
        """
        Search for similar assets in index.

        :param index_name: Target index name
        :param query: IsccAsset to search for
        :param limit: Maximum number of results
        :return: IsccSearchResult with query, metric, and list of matches
        :raises FileNotFoundError: If index doesn't exist
        :raises ValueError: If query validation fails
        """
        self._validate_index_exists(index_name)

        # Delegate to LmdbIndex
        idx = self._get_or_load_index(index_name)
        return idx.search_assets(query, limit)

    def close(self):
        # type: () -> None
        """
        Close all cached indexes and cleanup resources.

        Safe to call multiple times.
        """
        for idx in self._index_cache.values():
            idx.close()
        self._index_cache = {}

    # Helper methods

    def _get_or_load_index(self, name):
        # type: (str) -> LmdbIndex
        """
        Get cached index or load from disk.

        :param name: Index name
        :return: LmdbIndex instance
        """
        if name in self._index_cache:
            return self._index_cache[name]

        index_path = self.base_path / f"{name}.lmdb"
        idx = LmdbIndex(index_path)
        self._index_cache[name] = idx
        return idx

    def _validate_index_exists(self, name):
        # type: (str) -> None
        """
        Validate that an index exists.

        :param name: Index name
        :raises FileNotFoundError: If index doesn't exist
        """
        index_path = self.base_path / f"{name}.lmdb"
        if not index_path.exists():
            raise FileNotFoundError(f"Index '{name}' not found")

    def _get_file_size_mb(self, path):
        # type: (Path) -> int
        """
        Get file size in megabytes.

        :param path: Path to file
        :return: Size in MB (rounded down)
        """
        size_bytes = os.path.getsize(path)
        return size_bytes // (1024 * 1024)
