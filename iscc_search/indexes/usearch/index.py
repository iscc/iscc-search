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
import lmdb
from loguru import logger
from iscc_search.schema import IsccAddResult, IsccSearchResult, IsccMatch, Status, Metric
from iscc_search.models import IsccUnit, IsccID
from iscc_search.indexes import common
from iscc_search.nphd import NphdIndex


class UsearchIndex:
    """
    Single usearch-backed index with LMDB for metadata and INSTANCE matching.

    Storage structure:
    LMDB databases (in index.lmdb file):
    - __metadata__: realm_id (int), max_dim (int), created_at (float)
    - __assets__: uint64 key → IsccAsset JSON bytes
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

    def __init__(self, path, realm_id=0, max_dim=256, lmdb_options=None):
        # type: (os.PathLike, int, int, dict | None) -> None
        """
        Create or open usearch index at directory path.

        :param path: Path to index directory (contains index.lmdb + .usearch files)
        :param realm_id: ISCC realm ID for new indexes (0 or 1)
        :param max_dim: Maximum dimensions for NphdIndex (64, 128, 192, or 256)
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
        # type: (list[IsccAsset]) -> list[IsccAddResult]
        """
        Add assets to index.

        Stores assets in LMDB, INSTANCE units in dupsort database,
        and similarity units in NphdIndex files.

        :param assets: List of IsccAsset instances to add
        :return: List of IsccAddResult with created/updated status
        :raises ValueError: If realm_id inconsistent or missing iscc_id
        """
        if not assets:
            return []

        results = []

        while True:  # Auto-retry loop for MapFullError
            try:
                with self.env.begin(write=True) as txn:
                    # Get database handles
                    self.env.open_db(b"__metadata__", txn=txn)
                    assets_db = self.env.open_db(b"__assets__", txn=txn)
                    instance_db = self.env.open_db(
                        b"__instance__",
                        txn=txn,
                        dupsort=True,
                        dupfixed=True,
                        integerdup=True,
                    )

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

                # Batch add to NphdIndex (outside transaction)
                for unit_type, (keys, vectors) in nphd_batches.items():
                    nphd_index = self._get_or_create_nphd_index(unit_type)

                    # Check for existing keys and remove them first (for updates)
                    existing_keys = [k for k in keys if nphd_index.contains(k)]
                    if existing_keys:
                        nphd_index.remove(existing_keys)

                    nphd_index.add(keys, vectors)
                    # Save after add for durability
                    usearch_file = self.path / f"{unit_type}.usearch"
                    nphd_index.save(str(usearch_file))

                break  # Success

            except lmdb.MapFullError:  # pragma: no cover
                results = []  # Clear for retry
                old_size = self.map_size
                new_size = old_size * 2
                logger.info(f"UsearchIndex map_size increased from {old_size:,} to {new_size:,} bytes")
                self.env.set_mapsize(new_size)

        return results

    def get_asset(self, iscc_id):
        # type: (str) -> IsccAsset
        """
        Retrieve asset by ISCC-ID.

        :param iscc_id: ISCC-ID string
        :return: IsccAsset instance
        :raises FileNotFoundError: If asset not found
        """
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
        # type: (IsccAsset, int) -> IsccSearchResult
        """
        Search for similar assets using NPHD metric.

        Combines exact INSTANCE matching (LMDB) with similarity matching (NphdIndex).
        Scores: INSTANCE = 1.0 (exact), similarity = 1.0 - nphd_distance.

        :param query: Query asset with units
        :param limit: Maximum number of results
        :return: IsccSearchResult with NPHD metric
        """
        # Normalize query
        query = common.normalize_query_asset(query)
        if not query.units:  # pragma: no cover
            # No units to search
            return IsccSearchResult(query=query, metric=Metric.nphd, matches=[])

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

        # Build IsccMatch objects
        matches = []
        for key, total_score, unit_scores in scored_results:
            # Reconstruct ISCC-ID from key
            iscc_id = str(IsccID.from_int(key, self._realm_id))
            matches.append(IsccMatch(iscc_id=iscc_id, score=total_score, matches=unit_scores))

        return IsccSearchResult(query=query, metric=Metric.nphd, matches=matches)

    def close(self):
        # type: () -> None
        """Close LMDB environment and NphdIndex files."""
        # Close all NphdIndex instances
        for nphd_index in self._nphd_indexes.values():
            # NphdIndex doesn't have explicit close, but we can clear references
            pass

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
        # type: (int) -> None
        """Initialize or load metadata from LMDB."""
        with self.env.begin(write=True) as txn:
            metadata_db = self.env.open_db(b"__metadata__", txn=txn)

            # Try to load existing realm_id
            realm_bytes = txn.get(b"realm_id", db=metadata_db)
            if realm_bytes is not None:
                self._realm_id = struct.unpack(">I", realm_bytes)[0]
            else:
                # New index - store initial metadata
                self._realm_id = realm_id
                txn.put(b"realm_id", struct.pack(">I", realm_id), db=metadata_db)
                txn.put(b"max_dim", struct.pack(">I", self.max_dim), db=metadata_db)
                txn.put(b"created_at", struct.pack(">d", time.time()), db=metadata_db)

    def _load_nphd_indexes(self):
        # type: () -> None
        """Load existing NphdIndex files from directory."""
        for usearch_file in self.path.glob("*.usearch"):
            unit_type = usearch_file.stem  # Filename without extension
            try:
                # Note: restore() gets max_dim from saved metadata, don't pass it
                nphd_index = NphdIndex.restore(str(usearch_file))
                self._nphd_indexes[unit_type] = nphd_index
                logger.debug(f"Loaded NphdIndex for unit_type '{unit_type}'")
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to load NphdIndex '{usearch_file}': {e}")

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

        :param instance_code: Instance-Code digest bytes
        :return: Dict mapping integer keys to score (1.0 for exact match)
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
            # For INSTANCE, we support variable lengths: 64, 128, 192, 256 bits

            # Forward search: Find stored codes starting with query prefix
            if cursor.set_range(instance_code):
                for key_bytes, value_bytes in cursor:
                    if not key_bytes.startswith(instance_code):
                        break
                    # Exact or prefix match
                    key = struct.unpack(">Q", value_bytes)[0]
                    results[key] = 1.0  # Exact match score

            # Reverse search: Find stored codes that are prefixes of query
            # Check shorter versions (for 256-bit query, check 128-bit and 64-bit prefixes)
            query_len = len(instance_code)
            if query_len == 32:  # pragma: no cover - 256-bit query
                # Check 128-bit prefix
                prefix_128 = instance_code[:16]
                if cursor.set_key(prefix_128):
                    for _, value_bytes in cursor.iternext_dup():
                        key = struct.unpack(">Q", value_bytes)[0]
                        results[key] = 1.0

            if query_len >= 16:  # pragma: no cover - 128-bit or 256-bit query
                # Check 64-bit prefix
                prefix_64 = instance_code[:8]
                if cursor.set_key(prefix_64):
                    for _, value_bytes in cursor.iternext_dup():
                        key = struct.unpack(">Q", value_bytes)[0]
                        results[key] = 1.0

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
