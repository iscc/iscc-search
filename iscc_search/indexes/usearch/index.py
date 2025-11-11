"""
Usearch-backed single index implementation with LMDB metadata storage.

Hybrid architecture combining:
- LMDB: Asset storage, metadata, INSTANCE exact-matching (dupsort)
- NphdIndex: Similarity search for META, CONTENT, DATA units

Directory structure:
- index.lmdb: LMDB environment with __metadata__, __assets__, __instance__ databases
- {unit_type}.usearch: NphdIndex files for similarity units (lazy-created)

Key strategy: ISCC-ID body as uint64 consistently across LMDB and NphdIndex.
"""

import struct
import time
from pathlib import Path
from typing import TYPE_CHECKING
import lmdb
from loguru import logger
from iscc_search.schema import IsccAddResult, IsccGlobalMatch, IsccSearchResult, Status
from iscc_search.models import IsccUnit, IsccID
from iscc_search.indexes import common
from iscc_search.nphd import NphdIndex

if TYPE_CHECKING:
    from iscc_search.schema import IsccEntry  # noqa: F401


class UsearchIndex:
    """
    Single usearch-backed index with LMDB for metadata and INSTANCE matching.

    Storage structure:
    LMDB databases (in index.lmdb file):
    - __metadata__: realm_id (int), max_dim (int), created_at (float)
    - __assets__: uint64 key → IsccEntry JSON bytes
    - __instance__: instance_code digest → [iscc_id_body uint64, ...] (dupsort/dupfixed/integerdup)

    NphdIndex files:
    - {unit_type}.usearch: One file per similarity unit type (META, CONTENT, DATA)

    All keys use ISCC-ID body as uint64 for consistency between LMDB and usearch.
    """

    DEFAULT_LMDB_OPTIONS = {
        "readonly": False,
        "metasync": False,
        "sync": False,
        "mode": 0o644,
        "create": True,
        "readahead": False,
        "writemap": True,
        "meminit": True,
        "map_async": False,
        "max_readers": 126,
        "max_spare_txns": 16,
        "lock": True,
    }

    # MapFullError retry limits
    MAX_RESIZE_RETRIES = 10  # Maximum number of resize attempts
    MAX_MAP_SIZE = 1024 * 1024 * 1024 * 1024  # 1 TB maximum map size

    def __init__(self, path, realm_id=None, max_dim=256, lmdb_options=None):
        # type: (str | Path, int | None, int, dict | None) -> None
        """
        Create or open usearch index at directory path.

        :param path: Path to index directory (contains index.lmdb + .usearch files)
        :param realm_id: ISCC realm ID for new indexes (0 or 1). If None, inferred from first asset.
        :param max_dim: Maximum dimensions for NphdIndex (any multiple of 8 bits up to 256)
        :param lmdb_options: Custom LMDB options (max_dbs and subdir are forced)
        """
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.max_dim = max_dim
        self._realm_id = None  # type: int | None
        self._nphd_indexes = {}  # type: dict[str, NphdIndex]

        # Setup LMDB
        lmdb_path = self.path / "index.lmdb"
        options = self.DEFAULT_LMDB_OPTIONS.copy()
        if lmdb_options:
            options.update(lmdb_options)

        # Force critical parameters
        options["max_dbs"] = 3  # __metadata__, __assets__, __instance__
        options["subdir"] = False  # Path points to file

        self.env = lmdb.open(str(lmdb_path), **options)

        # Initialize or load metadata
        self._init_metadata(realm_id)

        # Load existing NphdIndex files
        self._load_nphd_indexes()

    def add_assets(self, assets):
        # type: (list[IsccEntry]) -> list[IsccAddResult]
        """
        Add assets to index.

        Stores assets in LMDB, INSTANCE units in dupsort database,
        and similarity units in NphdIndex files.

        Consistency model: LMDB commits before NphdIndex operations. If NphdIndex
        operations fail, assets are in LMDB (source of truth) but not in similarity
        search. This is acceptable as NphdIndex can be rebuilt from LMDB. True
        two-phase commit would add significant complexity for rare failure scenarios.

        :param assets: List of IsccEntry instances to add
        :return: List of IsccAddResult with created/updated status
        :raises ValueError: If realm_id inconsistent or missing iscc_id
        """
        if not assets:
            return []

        results = []
        retry_count = 0

        while retry_count <= self.MAX_RESIZE_RETRIES:  # pragma: no branch
            try:
                with self.env.begin(write=True) as txn:
                    # Get database handles
                    metadata_db = self.env.open_db(b"__metadata__", txn=txn)
                    assets_db = self.env.open_db(b"__assets__", txn=txn)
                    instance_db = self.env.open_db(
                        b"__instance__",
                        txn=txn,
                        dupsort=True,
                        dupfixed=True,
                        integerdup=True,
                    )

                    # Infer realm_id from first asset if not yet persisted
                    # Check database, not just self._realm_id (persists across retries)
                    realm_bytes = txn.get(b"realm_id", db=metadata_db)
                    if realm_bytes is None:
                        # Realm not in database - infer from first asset
                        if self._realm_id is None:  # pragma: no branch
                            # Not in memory either - validate and infer
                            if assets[0].iscc_id is None:
                                raise ValueError("Asset must have iscc_id field when adding to index")
                            self._realm_id = common.extract_realm_id(assets[0].iscc_id)

                        # Store in metadata (handles both initial write and retry after rollback)
                        txn.put(b"realm_id", struct.pack(">I", self._realm_id), db=metadata_db)
                        logger.info(f"Inferred realm_id={self._realm_id} from first asset")

                    # Prepare vectors for batch add to NphdIndex
                    nphd_batches = {}  # type: dict[str, tuple[list[int], list[bytes]]]

                    for asset in assets:
                        # Validate iscc_id present
                        if asset.iscc_id is None:
                            raise ValueError("Asset must have iscc_id field when adding to index")

                        # Extract and validate realm_id
                        asset_realm = common.extract_realm_id(asset.iscc_id)
                        if self._realm_id != asset_realm:
                            raise ValueError(
                                f"Realm ID mismatch: index has realm={self._realm_id}, "
                                f"but asset '{asset.iscc_id}' has realm={asset_realm}. "
                                f"All assets in an index must have the same realm ID."
                            )

                        # Convert ISCC-ID to integer key
                        iscc_id_obj = IsccID(asset.iscc_id)
                        key = int(iscc_id_obj)
                        key_bytes = struct.pack(">Q", key)  # Big-endian uint64

                        # Check if asset exists (for status)
                        existing = txn.get(key_bytes, db=assets_db)
                        status = Status.updated if existing else Status.created

                        # Store asset
                        asset_bytes = common.serialize_asset(asset)
                        txn.put(key_bytes, asset_bytes, db=assets_db)

                        # Index units
                        if asset.units:  # pragma: no branch
                            for unit_str in asset.units:
                                unit = IsccUnit(unit_str)
                                unit_type = unit.unit_type
                                unit_body = unit.body

                                # Check if INSTANCE unit
                                if unit_type.startswith("INSTANCE_"):
                                    # Add to LMDB dupsort (exact matching)
                                    cursor = txn.cursor(instance_db)
                                    cursor.put(unit_body, key_bytes, dupdata=False)
                                else:
                                    # Batch for NphdIndex (similarity matching)
                                    if unit_type not in nphd_batches:
                                        nphd_batches[unit_type] = ([], [])
                                    nphd_batches[unit_type][0].append(key)
                                    nphd_batches[unit_type][1].append(unit_body)

                        results.append(IsccAddResult(iscc_id=asset.iscc_id, status=status))

                # LMDB transaction commits here (exits context manager)
                # NphdIndex operations below are NOT atomic with LMDB - see docstring
                # Batch add to NphdIndex (outside transaction)
                for unit_type, (keys, vectors) in nphd_batches.items():
                    nphd_index = self._get_or_create_nphd_index(unit_type)

                    # Deduplicate keys (keep last occurrence for each duplicate)
                    # This handles cases where multiple assets share the same ISCC-ID
                    # TODO Review Duplicate Key Handling
                    if len(keys) != len(set(keys)):
                        # Build dict with key -> (last) vector mapping
                        unique_items = {}  # type: dict[int, bytes]
                        for key, vector in zip(keys, vectors):
                            unique_items[key] = vector
                        # Rebuild lists from deduplicated items
                        keys = list(unique_items.keys())
                        vectors = list(unique_items.values())

                    # Remove existing keys first (for updates)
                    # remove() handles non-existent keys gracefully (returns 0)
                    nphd_index.remove(keys)

                    nphd_index.add(keys, vectors)
                    # Update metadata with new vector count
                    self._update_nphd_metadata(unit_type, nphd_index.size)

                break  # Success

            except lmdb.MapFullError:  # pragma: no cover
                retry_count += 1

                # Check if we've exceeded retry limit
                if retry_count > self.MAX_RESIZE_RETRIES:
                    raise RuntimeError(
                        f"Failed to add assets after {self.MAX_RESIZE_RETRIES} resize attempts. "
                        f"Current map_size: {self.map_size:,} bytes. "
                        f"This may indicate disk space issues, permissions problems, or filesystem limits."
                    )

                # Clear state for retry
                results = []
                # Reset NphdIndexes - they have vectors from failed transaction
                self._nphd_indexes = {}
                old_size = self.map_size
                new_size = old_size * 2

                # Check if new size would exceed maximum
                if new_size > self.MAX_MAP_SIZE:
                    raise RuntimeError(
                        f"Cannot resize LMDB map beyond MAX_MAP_SIZE ({self.MAX_MAP_SIZE:,} bytes). "
                        f"Current size: {old_size:,}, attempted size: {new_size:,}. "
                        f"Consider splitting data across multiple indexes or increasing MAX_MAP_SIZE."
                    )

                logger.info(
                    f"UsearchIndex map_size increased from {old_size:,} to {new_size:,} bytes "
                    f"(retry {retry_count}/{self.MAX_RESIZE_RETRIES})"
                )
                self.env.set_mapsize(new_size)

        return results

    def get_asset(self, iscc_id):
        # type: (str) -> IsccEntry
        """
        Retrieve asset by ISCC-ID.

        :param iscc_id: ISCC-ID string
        :return: IsccEntry instance
        :raises ValueError: If ISCC-ID realm doesn't match index realm or format invalid
        :raises FileNotFoundError: If asset not found
        """
        # Validate format and realm in single decode operation
        common.validate_iscc_id(iscc_id, expected_realm=self._realm_id)

        # Convert ISCC-ID to integer key
        iscc_id_obj = IsccID(iscc_id)
        key = int(iscc_id_obj)
        key_bytes = struct.pack(">Q", key)

        try:
            with self.env.begin() as txn:
                assets_db = self.env.open_db(b"__assets__", txn=txn)
                asset_bytes = txn.get(key_bytes, db=assets_db)
        except lmdb.ReadonlyError:  # pragma: no cover
            # Database doesn't exist yet (empty index)
            raise FileNotFoundError(f"Asset '{iscc_id}' not found in index")

        if asset_bytes is None:  # pragma: no cover
            raise FileNotFoundError(f"Asset '{iscc_id}' not found in index")

        return common.deserialize_asset(asset_bytes)

    def search_assets(self, query, limit=100):
        # type: (IsccQuery, int) -> IsccSearchResult
        """
        Search for similar assets using NPHD metric.

        Combines exact INSTANCE matching (LMDB) with similarity matching (NphdIndex).
        Scores: INSTANCE = proportional to match length (0.25/0.5/1.0), similarity = 1.0 - nphd_distance.

        :param query: Query with units
        :param limit: Maximum number of results
        :return: IsccSearchResult with query and list of matches
        """
        # Normalize query
        query = common.normalize_query(query)
        if not query.units:  # pragma: no cover
            # No units to search
            return IsccSearchResult(query=query, global_matches=[])

        # Aggregation: {key (int): {unit_type: score}}
        aggregated = {}  # type: dict[int, dict[str, float]]

        for unit_str in query.units:
            unit = IsccUnit(unit_str)
            unit_type = unit.unit_type
            unit_body = unit.body

            if unit_type.startswith("INSTANCE_"):
                # Exact matching via LMDB dupsort
                matches = self._search_instance_unit(unit_body)
                for key, score in matches.items():
                    if key not in aggregated:
                        aggregated[key] = {}
                    aggregated[key][unit_type] = score
            else:
                # Similarity matching via NphdIndex
                if unit_type in self._nphd_indexes:
                    matches = self._search_similarity_unit(unit_type, unit_body, limit)
                    for key, score in matches.items():
                        if key not in aggregated:
                            aggregated[key] = {}
                        # Store max score if multiple matches for same unit_type
                        aggregated[key][unit_type] = max(aggregated[key].get(unit_type, 0.0), score)

        # Calculate total scores and sort
        scored_results = []  # type: list[tuple[int, float, dict[str, float]]]
        for key, unit_scores in aggregated.items():
            total_score = sum(unit_scores.values())
            scored_results.append((key, total_score, unit_scores))

        # Sort by total score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Take top limit results
        scored_results = scored_results[:limit]

        # Build IsccGlobalMatch objects with metadata
        matches = []
        try:
            with self.env.begin() as txn:
                assets_db = self.env.open_db(b"__assets__", txn=txn)
                for key, total_score, unit_scores in scored_results:
                    # Reconstruct ISCC-ID from key
                    iscc_id = str(IsccID.from_int(key, self._realm_id))

                    # Fetch asset metadata
                    metadata = None
                    key_bytes = struct.pack(">Q", key)
                    asset_bytes = txn.get(key_bytes, db=assets_db)
                    if asset_bytes is not None:
                        asset = common.deserialize_asset(asset_bytes)
                        metadata = asset.metadata

                    matches.append(
                        IsccGlobalMatch(iscc_id=iscc_id, score=total_score, types=unit_scores, metadata=metadata)
                    )
        except lmdb.ReadonlyError:  # pragma: no cover
            # Database doesn't exist yet (empty index) - return matches without metadata
            for key, total_score, unit_scores in scored_results:
                iscc_id = str(IsccID.from_int(key, self._realm_id))
                matches.append(IsccGlobalMatch(iscc_id=iscc_id, score=total_score, types=unit_scores))

        return IsccSearchResult(query=query, global_matches=matches)

    def flush(self):
        # type: () -> None
        """
        Save all NphdIndex files to disk.

        Explicit flush for power users who want control over persistence.
        NphdIndexes are automatically saved on close(), so flush() is only
        needed for durability guarantees during long-running sessions.
        """
        for unit_type, nphd_index in self._nphd_indexes.items():
            usearch_file = self.path / f"{unit_type}.usearch"
            nphd_index.save(str(usearch_file))
            # Update metadata with current vector count
            self._update_nphd_metadata(unit_type, nphd_index.size)
            logger.debug(f"Flushed NphdIndex for unit_type '{unit_type}'")

    def close(self):
        # type: () -> None
        """Close LMDB environment and NphdIndex files, saving all indexes."""
        # Save all NphdIndex instances before closing
        for unit_type, nphd_index in self._nphd_indexes.items():
            usearch_file = self.path / f"{unit_type}.usearch"
            nphd_index.save(str(usearch_file))
            # Update metadata with current vector count
            self._update_nphd_metadata(unit_type, nphd_index.size)
            logger.debug(f"Saved NphdIndex for unit_type '{unit_type}'")
            # Release file handles for memory-mapped indexes
            nphd_index.reset()

        self._nphd_indexes.clear()

        # Close LMDB
        self.env.close()

    def __len__(self):  # pragma: no cover
        # type: () -> int
        """Return number of assets in index."""
        try:
            with self.env.begin() as txn:
                assets_db = self.env.open_db(b"__assets__", txn=txn)
                return txn.stat(db=assets_db)["entries"]
        except lmdb.ReadonlyError:
            # Database doesn't exist yet (empty index)
            return 0

    @property
    def map_size(self):
        # type: () -> int
        """Get current LMDB map_size."""
        return self.env.info()["map_size"]

    def __del__(self):  # pragma: no cover
        # type: () -> None
        """Cleanup on deletion."""
        if hasattr(self, "env"):
            self.env.close()

    # Helper methods

    def _init_metadata(self, realm_id):
        # type: (int | None) -> None
        """Initialize or load metadata from LMDB."""
        with self.env.begin(write=True) as txn:
            metadata_db = self.env.open_db(b"__metadata__", txn=txn)

            # Try to load existing realm_id
            realm_bytes = txn.get(b"realm_id", db=metadata_db)
            if realm_bytes is not None:
                # Existing index - load stored configuration
                self._realm_id = struct.unpack(">I", realm_bytes)[0]
                max_dim_bytes = txn.get(b"max_dim", db=metadata_db)
                self.max_dim = struct.unpack(">I", max_dim_bytes)[0]
            else:
                # No realm_id in metadata - check if this is truly a new index or needs migration
                assets_db = self.env.open_db(b"__assets__", txn=txn)
                asset_count = txn.stat(db=assets_db)["entries"]

                if asset_count > 0:
                    # Existing index from before realm_id was stored - infer from first asset
                    cursor = txn.cursor(assets_db)
                    if cursor.first():  # pragma: no branch
                        _, asset_bytes = cursor.item()
                        asset = common.deserialize_asset(asset_bytes)
                        if asset.iscc_id:
                            inferred_realm = common.extract_realm_id(asset.iscc_id)
                            self._realm_id = inferred_realm
                            txn.put(b"realm_id", struct.pack(">I", inferred_realm), db=metadata_db)
                            logger.info(f"Migrated existing index: inferred realm_id={inferred_realm} from first asset")
                        else:
                            raise ValueError("Cannot infer realm_id: first asset has no iscc_id")
                    cursor.close()

                    # Load max_dim if exists, otherwise use default
                    max_dim_bytes = txn.get(b"max_dim", db=metadata_db)
                    if max_dim_bytes:
                        self.max_dim = struct.unpack(">I", max_dim_bytes)[0]
                    else:
                        txn.put(b"max_dim", struct.pack(">I", self.max_dim), db=metadata_db)

                    # Add created_at if missing
                    if not txn.get(b"created_at", db=metadata_db):
                        txn.put(b"created_at", struct.pack(">d", time.time()), db=metadata_db)
                else:
                    # Truly new index
                    if realm_id is None:
                        # Defer realm_id assignment until first asset is added
                        self._realm_id = None
                    else:
                        # Explicit realm_id provided - store immediately
                        self._realm_id = realm_id
                        txn.put(b"realm_id", struct.pack(">I", realm_id), db=metadata_db)

                    # Always store max_dim and created_at for new indexes
                    txn.put(b"max_dim", struct.pack(">I", self.max_dim), db=metadata_db)
                    txn.put(b"created_at", struct.pack(">d", time.time()), db=metadata_db)

    def _update_nphd_metadata(self, unit_type, vector_count):
        # type: (str, int) -> None
        """
        Update NphdIndex metadata in LMDB.

        Tracks expected vector count for sync detection on next load.

        :param unit_type: Unit type identifier
        :param vector_count: Current number of vectors in NphdIndex
        """
        with self.env.begin(write=True) as txn:
            metadata_db = self.env.open_db(b"__metadata__", txn=txn)
            key = f"nphd_count:{unit_type}".encode()
            txn.put(key, struct.pack(">Q", vector_count), db=metadata_db)

    def _get_nphd_metadata(self, unit_type):
        # type: (str) -> int | None
        """
        Get expected NphdIndex vector count from LMDB metadata.

        :param unit_type: Unit type identifier
        :return: Expected vector count, or None if not tracked
        """
        try:
            with self.env.begin() as txn:
                metadata_db = self.env.open_db(b"__metadata__", txn=txn)
                key = f"nphd_count:{unit_type}".encode()
                value = txn.get(key, db=metadata_db)
                if value is None:
                    return None
                return struct.unpack(">Q", value)[0]
        except lmdb.ReadonlyError:  # pragma: no cover
            return None

    def _get_all_tracked_unit_types(self):
        # type: () -> set[str]
        """
        Get all unit_types tracked in LMDB metadata.

        Scans metadata database for all 'nphd_count:*' keys and extracts unit_types.

        :return: Set of unit_type identifiers that have been indexed
        """
        unit_types = set()  # type: set[str]
        prefix = b"nphd_count:"

        try:
            with self.env.begin() as txn:
                metadata_db = self.env.open_db(b"__metadata__", txn=txn)
                cursor = txn.cursor(metadata_db)

                # Seek to first key matching prefix
                if cursor.set_range(prefix):
                    for key_bytes, _ in cursor:
                        if not key_bytes.startswith(prefix):
                            break
                        # Extract unit_type from key (format: "nphd_count:UNIT_TYPE")
                        unit_type = key_bytes[len(prefix) :].decode()
                        unit_types.add(unit_type)

        except lmdb.ReadonlyError:  # pragma: no cover
            # Database doesn't exist yet (empty index)
            pass

        return unit_types

    def _load_nphd_indexes(self):
        # type: () -> None
        """
        Load existing NphdIndex files with auto-rebuild on sync mismatch.

        Compares actual vector count in .usearch file with expected count
        in LMDB metadata. Triggers full rebuild from LMDB if out of sync.
        Also rebuilds missing .usearch files for unit_types tracked in metadata
        (crash recovery for unflushed indexes).
        """
        # Track which unit_types we've loaded from disk
        loaded_unit_types = set()  # type: set[str]

        for usearch_file in self.path.glob("*.usearch"):
            unit_type = usearch_file.stem  # Filename without extension
            loaded_unit_types.add(unit_type)
            try:
                # Note: restore() gets max_dim from saved metadata, don't pass it
                nphd_index = NphdIndex.restore(str(usearch_file))
                if nphd_index is None:  # pragma: no cover
                    logger.warning(f"Failed to load NphdIndex '{usearch_file}': invalid metadata")
                    self._rebuild_nphd_index(unit_type)
                    continue

                # Check if index is in sync with LMDB
                expected_count = self._get_nphd_metadata(unit_type)
                actual_count = nphd_index.size

                if expected_count is not None and expected_count != actual_count:
                    logger.warning(
                        f"NphdIndex '{unit_type}' out of sync: "
                        f"expected {expected_count} vectors, found {actual_count}. "
                        f"Rebuilding from LMDB..."
                    )
                    # Close the loaded index and rebuild
                    nphd_index.reset()
                    self._rebuild_nphd_index(unit_type)
                else:
                    self._nphd_indexes[unit_type] = nphd_index
                    logger.debug(f"Loaded NphdIndex for unit_type '{unit_type}' ({actual_count} vectors)")

            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to load NphdIndex '{usearch_file}': {e}. Rebuilding...")
                self._rebuild_nphd_index(unit_type)

        # Check for orphaned metadata (tracked unit_types without .usearch files)
        # This handles crash recovery when vectors were added but never flushed
        tracked_unit_types = self._get_all_tracked_unit_types()
        missing_unit_types = tracked_unit_types - loaded_unit_types

        if missing_unit_types:
            logger.warning(
                f"Found {len(missing_unit_types)} unit_type(s) in metadata without .usearch files: "
                f"{sorted(missing_unit_types)}. Rebuilding from LMDB (crash recovery)..."
            )
            for unit_type in missing_unit_types:
                self._rebuild_nphd_index(unit_type)

    def _rebuild_nphd_index(self, unit_type):
        # type: (str) -> None
        """
        Rebuild NphdIndex from LMDB asset data.

        Full rebuild: iterates all assets, extracts vectors for unit_type,
        and creates fresh NphdIndex. Automatically saves and updates metadata.

        :param unit_type: Unit type identifier to rebuild
        """
        import time as time_module

        start_time = time_module.time()
        logger.info(f"Rebuilding NphdIndex for unit_type '{unit_type}'...")

        # Collect all vectors for this unit_type from LMDB
        keys = []  # type: list[int]
        vectors = []  # type: list[bytes]

        with self.env.begin() as txn:
            assets_db = self.env.open_db(b"__assets__", txn=txn)
            cursor = txn.cursor(assets_db)

            for key_bytes, asset_bytes in cursor:
                asset = common.deserialize_asset(asset_bytes)
                if not asset.units:  # pragma: no cover
                    continue

                # Extract vectors for this unit_type
                for unit_str in asset.units:
                    unit = IsccUnit(unit_str)
                    if unit.unit_type == unit_type:
                        key = struct.unpack(">Q", key_bytes)[0]
                        keys.append(key)
                        vectors.append(unit.body)

        if not keys:
            logger.info(f"No vectors found for unit_type '{unit_type}' - skipping rebuild")
            return

        # Create new NphdIndex and add all vectors
        nphd_index = NphdIndex(max_dim=self.max_dim)
        nphd_index.add(keys, vectors)

        # Save to disk
        usearch_file = self.path / f"{unit_type}.usearch"
        nphd_index.save(str(usearch_file))

        # Update metadata
        self._update_nphd_metadata(unit_type, nphd_index.size)

        # Store in memory
        self._nphd_indexes[unit_type] = nphd_index

        elapsed = time_module.time() - start_time
        logger.info(f"Rebuilt NphdIndex for unit_type '{unit_type}': {len(keys)} vectors in {elapsed:.2f}s")

    def _get_or_create_nphd_index(self, unit_type):
        # type: (str) -> NphdIndex
        """Get or create NphdIndex for unit_type."""
        if unit_type not in self._nphd_indexes:  # pragma: no branch
            nphd_index = NphdIndex(max_dim=self.max_dim)
            self._nphd_indexes[unit_type] = nphd_index
            logger.debug(f"Created new NphdIndex for unit_type '{unit_type}'")

        return self._nphd_indexes[unit_type]

    def _search_instance_unit(self, instance_code):
        # type: (bytes) -> dict[int, float]
        """
        Search INSTANCE unit in LMDB dupsort for exact matches.

        Scores are proportional to match length:
        - 64-bit match: 0.25 (64/256)
        - 128-bit match: 0.5 (128/256)
        - 256-bit match: 1.0 (256/256)

        :param instance_code: Instance-Code digest bytes
        :return: Dict mapping integer keys to score (proportional to match length)
        """
        results = {}  # type: dict[int, float]

        with self.env.begin() as txn:
            instance_db = self.env.open_db(
                b"__instance__",
                txn=txn,
                dupsort=True,
                dupfixed=True,
                integerdup=True,
            )

            cursor = txn.cursor(instance_db)
            # Bidirectional prefix matching (like LmdbIndex)
            # For INSTANCE, we support variable lengths: 64, 128, 256 bits

            # Forward search: Find stored codes starting with query prefix
            if cursor.set_range(instance_code):
                for key_bytes, value_bytes in cursor:
                    if not key_bytes.startswith(instance_code):
                        break
                    # Score proportional to match length (shorter of query and stored)
                    match_bytes = min(len(instance_code), len(key_bytes))
                    score = match_bytes / 32  # 32 bytes = 256 bits max
                    key = struct.unpack(">Q", value_bytes)[0]
                    results[key] = max(results.get(key, 0.0), score)

            # Reverse search: Find stored codes that are prefixes of query
            # Check shorter versions (for 256-bit query, check 128-bit and 64-bit prefixes)
            query_len = len(instance_code)
            if query_len == 32:  # 256-bit query
                # Check 128-bit prefix
                prefix_128 = instance_code[:16]
                if cursor.set_key(prefix_128):
                    for value_bytes in cursor.iternext_dup(keys=False, values=True):
                        key = struct.unpack(">Q", value_bytes)[0]
                        score = 0.5  # 128-bit match (16 bytes / 32 bytes)
                        results[key] = max(results.get(key, 0.0), score)

            if query_len >= 16:  # 128-bit or 256-bit query
                # Check 64-bit prefix
                prefix_64 = instance_code[:8]
                if cursor.set_key(prefix_64):
                    for value_bytes in cursor.iternext_dup(keys=False, values=True):
                        key = struct.unpack(">Q", value_bytes)[0]
                        score = 0.25  # 64-bit match (8 bytes / 32 bytes)
                        results[key] = max(results.get(key, 0.0), score)

        return results

    def _search_similarity_unit(self, unit_type, vector, limit):
        # type: (str, bytes, int) -> dict[int, float]
        """
        Search similarity unit in NphdIndex.

        :param unit_type: Unit type identifier
        :param vector: Unit body bytes
        :param limit: Maximum number of results
        :return: Dict mapping integer keys to scores (1.0 - nphd_distance)
        """
        nphd_index = self._nphd_indexes[unit_type]

        # Search usearch - NphdIndex.search expects single vector or list
        # For single vector, wrap in list if needed
        matches = nphd_index.search([vector], count=limit)

        # Convert distances to scores: score = 1.0 - distance
        results = {}  # type: dict[int, float]
        for key, distance in zip(matches.keys, matches.distances):
            score = 1.0 - float(distance)
            results[int(key)] = max(0.0, score)  # Clamp to non-negative

        return results
