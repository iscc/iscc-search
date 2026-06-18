"""
Usearch Index Manager - Protocol Implementation.

Manages multiple usearch-backed indexes in a base directory.
Each index is stored as a separate directory containing index.lmdb + .usearch files.

Implements IsccIndexProtocol for use as backend in CLI and server.
"""

import shutil
import threading
from pathlib import Path
from typing import TYPE_CHECKING
from loguru import logger
from iscc_search.schema import IsccIndex
from iscc_search.indexes.usearch.index import UsearchIndex
from iscc_search.indexes import common

if TYPE_CHECKING:
    from iscc_search.schema import IsccAddResult  # noqa: F401
    from iscc_search.schema import IsccEntry  # noqa: F401
    from iscc_search.schema import IsccSearchResult  # noqa: F401


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

    CONCURRENCY: Single-process only. The .usearch files have no file locking or multi-process
    coordination. The instance cache does not synchronize between processes. Running multiple
    processes against the same indexes may corrupt data. Use a single process with async/await
    for concurrent connections (e.g., FastAPI with Uvicorn).
    """

    def __init__(self, base_path, max_dim=256):
        # type: (str | Path, int) -> None
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
        # Serialize first-load construction so concurrent requests don't race on lmdb.open()
        # (which raises "already open in this process" if two threads call it simultaneously).
        self._cache_lock = threading.Lock()

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
                size_mb, sizes_mb = self._get_index_sizes_mb(index_dir, idx)

                indexes.append(IsccIndex(name=name, assets=asset_count, size=size_mb, sizes=sizes_mb))
            except Exception as e:
                # Log and skip corrupted or inaccessible indexes
                logger.warning(f"Failed to load index '{name}': {type(e).__name__}: {e}")
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
        # realm_id is None - will be inferred from first asset
        idx = UsearchIndex(index_path, realm_id=None, max_dim=self.max_dim)
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
        asset_count = len(idx)
        index_path = self.base_path / name
        size_mb, sizes_mb = self._get_index_sizes_mb(index_path, idx)

        return IsccIndex(name=name, assets=asset_count, size=size_mb, sizes=sizes_mb)

    def delete_index(self, name):
        # type: (str) -> None
        """
        Delete index and all its data (directory and all files).

        :param name: Index name
        :raises FileNotFoundError: If index doesn't exist
        """
        self._validate_index_exists(name)

        # Close cached instance if open
        if name in self._index_cache:  # pragma: no branch
            self._index_cache[name].close()
            del self._index_cache[name]

        # Delete entire directory
        index_path = self.base_path / name
        shutil.rmtree(index_path)

    def add_assets(self, index_name, assets):
        # type: (str, list[IsccEntry]) -> list[IsccAddResult]
        """
        Add assets to index.

        :param index_name: Target index name
        :param assets: List of IsccEntry objects to add
        :return: List of IsccAddResult with status for each asset
        :raises FileNotFoundError: If index doesn't exist
        :raises ValueError: If asset validation fails
        """
        self._validate_index_exists(index_name)

        # Delegate to UsearchIndex
        idx = self._get_or_load_index(index_name)
        return idx.add_assets(assets)

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
        self._validate_index_exists(index_name)

        # Delegate to UsearchIndex
        idx = self._get_or_load_index(index_name)
        return idx.get_asset(iscc_id)

    def search_assets(self, index_name, query, limit=100):
        # type: (str, IsccQuery, int) -> IsccSearchResult
        """
        Search for similar assets in index.

        :param index_name: Target index name
        :param query: IsccQuery to search for
        :param limit: Maximum number of results
        :return: IsccSearchResult with query and list of matches
        :raises FileNotFoundError: If index doesn't exist
        :raises ValueError: If query validation fails
        """
        self._validate_index_exists(index_name)

        # Delegate to UsearchIndex
        idx = self._get_or_load_index(index_name)
        return idx.search_assets(query, limit)

    def rebuild(self, name, unit_types=None, simprint_types=None):
        # type: (str, list[str] | None, list[str] | None) -> dict
        """
        Rebuild derived NPHD/simprint indexes for the named index from LMDB source.

        ``None`` for ``unit_types`` or ``simprint_types`` means "rebuild every type
        of that kind currently tracked in LMDB metadata".

        :param name: Target index name
        :param unit_types: NPHD unit_types to rebuild, or None for all tracked
        :param simprint_types: Simprint types to rebuild, or None for all tracked
        :return: Dict with ``unit_types`` and ``simprint_types`` lists actually rebuilt
        :raises FileNotFoundError: If index doesn't exist
        """
        self._validate_index_exists(name)
        idx = self._get_or_load_index(name)
        if unit_types is None:
            unit_types = idx.tracked_unit_types
        if simprint_types is None:
            simprint_types = idx.tracked_simprint_types
        return idx.rebuild(unit_types, simprint_types)

    def close(self):
        # type: () -> None
        """
        Close all cached indexes and cleanup resources.

        Exception-safe: each index is closed independently so a failure in one
        does not prevent the others from being saved. Safe to call multiple times.
        """
        for name, idx in list(self._index_cache.items()):
            try:
                idx.close()
            except Exception:  # pragma: no cover
                logger.exception(f"Failed to close index '{name}'")
        self._index_cache = {}

    # Helper methods

    def _get_or_load_index(self, name):
        # type: (str) -> UsearchIndex
        """
        Get cached index or load from disk.

        Thread-safe: a lock guards the cache-miss construction path so concurrent
        first-burst requests don't race on lmdb.open() (which raises
        "already open in this process" if two threads call it simultaneously).

        :param name: Index name
        :return: UsearchIndex instance
        """
        if name in self._index_cache:
            return self._index_cache[name]

        with self._cache_lock:
            if name in self._index_cache:  # pragma: no cover - race condition guard
                return self._index_cache[name]
            index_path = self.base_path / name
            idx = UsearchIndex(index_path, max_dim=self.max_dim)
            self._index_cache[name] = idx
            return idx

    def _validate_index_exists(self, name):
        # type: (str) -> None
        """
        Validate that an index exists.

        :param name: Index name
        :raises FileNotFoundError: If index doesn't exist
        """
        index_path = self.base_path / name
        lmdb_file = index_path / "index.lmdb"
        if not lmdb_file.exists():
            raise FileNotFoundError(f"Index '{name}' not found")

    def _get_index_sizes_mb(self, path, idx):
        # type: (Path, UsearchIndex) -> tuple[int, dict[str, int]]
        """
        Get index size in megabytes, total and per component.

        Component "lmdb" covers the LMDB environment via page accounting (the
        sparse index.lmdb data file reports the nominal 1 TiB map size as
        st_size on some platforms) plus auxiliary top-level files like the lock
        file. Derived components (NPHD unit types, SIMPRINT_* types) report
        their serialized data size measured from the live index, so unflushed
        vectors are included. Shard directories that are NOT loaded (failed to
        load or left over from an interrupted rebuild) are measured raw from
        disk so the reported size never silently understates disk usage.

        :param path: Path to index directory
        :param idx: Loaded UsearchIndex providing the LMDB environment
        :return: (total_mb, component_mb) with values rounded down to MB
        """
        component_bytes = {"lmdb": common.lmdb_used_bytes(idx.env)}
        derived = idx.derived_sizes
        for entry in path.iterdir():
            if entry.is_file() and entry.name != "index.lmdb":
                try:
                    component_bytes["lmdb"] += entry.stat().st_size
                except OSError:  # pragma: no cover - race with concurrent flush/rotation file ops
                    continue
            elif entry.is_dir() and entry.name not in derived:
                dir_bytes = 0
                for f in entry.rglob("*"):
                    if f.is_file():  # pragma: no branch
                        try:
                            dir_bytes += f.stat().st_size
                        except OSError:  # pragma: no cover - race with concurrent flush/rotation file ops
                            continue
                component_bytes[entry.name] = dir_bytes
        component_bytes.update(derived)

        mb = 1024 * 1024
        component_mb = {name: size // mb for name, size in component_bytes.items()}
        # Floor the byte total once so the reported size never under-reports vs the
        # per-component breakdown (which loses each component's sub-MB remainder).
        return sum(component_bytes.values()) // mb, component_mb
