"""
Usearch Index Manager - Protocol Implementation.

Manages multiple usearch-backed indexes in a base directory.
Each index is stored as a separate directory containing index.lmdb + .usearch files.

Implements IsccIndexProtocol for use as backend in CLI and server.
"""

import shutil
from pathlib import Path
from iscc_search.schema import IsccIndex
from iscc_search.indexes.usearch.index import UsearchIndex
from iscc_search.indexes import common


class UsearchIndexManager:
    """
    Protocol implementation managing multiple usearch indexes.

    Directory structure:
    base_path/
    ├── index1/
    │   ├── index.lmdb
    │   ├── CONTENT_TEXT_V0.usearch
    │   └── DATA_NONE_V0.usearch
    ├── index2/
    │   ├── index.lmdb
    │   └── ...
    └── ...

    Each subdirectory with index.lmdb is managed by a UsearchIndex instance.
    Instances are cached for performance.
    """

    def __init__(self, base_path, max_dim=256):
        # type: (os.PathLike, int) -> None
        """
        Initialize UsearchIndexManager.

        Creates base directory if it doesn't exist.

        :param base_path: Directory containing index subdirectories
        :param max_dim: Default max dimensions for new indexes (64, 128, 192, or 256)
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.max_dim = max_dim
        self._index_cache = {}  # type: dict[str, UsearchIndex]

    def list_indexes(self):
        # type: () -> list[IsccIndex]
        """
        List all indexes by scanning for subdirectories with index.lmdb.

        :return: List of IsccIndex objects with metadata
        """
        indexes = []

        for index_dir in self.base_path.iterdir():
            if not index_dir.is_dir():
                continue

            # Check for index.lmdb to identify valid index
            lmdb_file = index_dir / "index.lmdb"
            if not lmdb_file.exists():
                continue

            # Extract index name from directory name
            name = index_dir.name

            # Get metadata
            try:
                idx = self._get_or_load_index(name)
                asset_count = len(idx)
                size_mb = self._get_directory_size_mb(index_dir)

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
        index_path = self.base_path / index.name
        if index_path.exists():
            raise FileExistsError(f"Index '{index.name}' already exists")

        # Create new UsearchIndex (creates directory and index.lmdb)
        idx = UsearchIndex(index_path, realm_id=0, max_dim=self.max_dim)
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
        index_path = self.base_path / name
        lmdb_file = index_path / "index.lmdb"

        if not lmdb_file.exists():
            raise FileNotFoundError(f"Index '{name}' not found")

        # Load index and get metadata
        idx = self._get_or_load_index(name)
        asset_count = len(idx)
        size_mb = self._get_directory_size_mb(index_path)

        return IsccIndex(name=name, assets=asset_count, size=size_mb)

    def delete_index(self, name):
        # type: (str) -> None
        """
        Delete index and all its data (directory and all files).

        :param name: Index name
        :raises FileNotFoundError: If index doesn't exist
        """
        index_path = self.base_path / name
        lmdb_file = index_path / "index.lmdb"

        if not lmdb_file.exists():
            raise FileNotFoundError(f"Index '{name}' not found")

        # Close cached instance if open
        if name in self._index_cache:
            self._index_cache[name].close()
            del self._index_cache[name]

        # Delete entire directory
        shutil.rmtree(index_path)

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
        # Check existence
        index_path = self.base_path / index_name
        lmdb_file = index_path / "index.lmdb"

        if not lmdb_file.exists():
            raise FileNotFoundError(f"Index '{index_name}' not found")

        # Delegate to UsearchIndex
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
        # Check index existence
        index_path = self.base_path / index_name
        lmdb_file = index_path / "index.lmdb"

        if not lmdb_file.exists():
            raise FileNotFoundError(f"Index '{index_name}' not found")

        # Delegate to UsearchIndex
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
        # Check existence
        index_path = self.base_path / index_name
        lmdb_file = index_path / "index.lmdb"

        if not lmdb_file.exists():
            raise FileNotFoundError(f"Index '{index_name}' not found")

        # Delegate to UsearchIndex
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
        # type: (str) -> UsearchIndex
        """
        Get cached index or load from disk.

        :param name: Index name
        :return: UsearchIndex instance
        """
        if name in self._index_cache:
            return self._index_cache[name]

        index_path = self.base_path / name
        idx = UsearchIndex(index_path, max_dim=self.max_dim)
        self._index_cache[name] = idx
        return idx

    def _get_directory_size_mb(self, path):
        # type: (Path) -> int
        """
        Get total size of all files in directory in megabytes.

        :param path: Path to directory
        :return: Total size in MB (rounded down)
        """
        total_bytes = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_bytes += file_path.stat().st_size

        return total_bytes // (1024 * 1024)
