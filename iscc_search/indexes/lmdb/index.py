"""
LMDB-backed single index implementation.

Manages a single LMDB file containing:
- Asset storage (full IsccAsset metadata)
- Inverted indexes per ISCC-UNIT type for similarity search
- Index metadata (realm_id, timestamps)

Uses bidirectional prefix matching for variable-length ISCC search.
"""

import os
import time
import lmdb
from loguru import logger
from iscc_search.schema import IsccAddResult, IsccSearchResult, IsccMatch, Status, Metric
from iscc_search.models import IsccUnit
from iscc_search.indexes import common


class LmdbIndex:
    """
    Single LMDB-backed index with inverted unit-type indexes.

    Storage structure:
    - __assets__ database: iscc_id → IsccAsset JSON
    - __metadata__ database: realm_id, created_at
    - Per unit-type databases: unit_body → [iscc_id_body, ...] (dupsort)

    Supports variable-length ISCC matching via bidirectional prefix search.
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

    def __init__(self, path, lmdb_options=None):
        # type: (os.PathLike, dict | None) -> None
        """
        Create or open LMDB index at path.

        :param path: Path to LMDB file (subdir=False)
        :param lmdb_options: Custom LMDB options (max_dbs and subdir are forced)
        """
        self.path = os.fspath(path)
        self._realm_id = None  # type: int | None

        # Merge options
        options = self.DEFAULT_LMDB_OPTIONS.copy()
        if lmdb_options:
            options.update(lmdb_options)

        # Force critical parameters
        options["max_dbs"] = 16  # Special dbs + unit types
        options["subdir"] = False  # Path points to file, not directory

        self.env = lmdb.open(self.path, **options)
        self._db_cache = {}  # type: dict[str, object]

        # Pre-open common databases to cache handles
        self._init_common_databases()

        # Load realm_id from metadata if exists
        self._load_metadata()

    def add_assets(self, assets):
        # type: (list[IsccAsset]) -> list[IsccAddResult]
        """
        Add assets to index with inverted unit indexing.

        Validates realm_id consistency across all assets.
        Auto-retries on MapFullError by doubling map_size.

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
                    assets_db = self._get_or_create_db("__assets__", txn, dupsort=False)

                    for asset in assets:
                        # Validate iscc_id present
                        if asset.iscc_id is None:
                            raise ValueError("Asset must have iscc_id field when adding to index")

                        # Extract and validate realm_id
                        asset_realm = common.extract_realm_id(asset.iscc_id)
                        if self._realm_id is None:
                            # First asset sets realm for index
                            self._realm_id = asset_realm
                            self._save_realm_id(txn, asset_realm)
                        elif self._realm_id != asset_realm:
                            raise ValueError(
                                f"Realm ID mismatch: index has realm={self._realm_id}, "
                                f"but asset '{asset.iscc_id}' has realm={asset_realm}. "
                                f"All assets in an index must have the same realm ID."
                            )

                        # Check if asset exists (for status)
                        iscc_id_key = asset.iscc_id.encode("utf-8")
                        existing = txn.get(iscc_id_key, db=assets_db)
                        status = Status.updated if existing else Status.created

                        # Store asset
                        asset_bytes = common.serialize_asset(asset)
                        txn.put(iscc_id_key, asset_bytes, db=assets_db)

                        # Index units
                        if asset.units:
                            iscc_id_body = common.extract_iscc_id_body(asset.iscc_id)

                            for unit_str in asset.units:
                                # Units are plain ISCC strings
                                unit = IsccUnit(unit_str)
                                unit_type = unit.unit_type
                                unit_body = unit.body

                                # Get or create unit-type database
                                unit_db = self._get_or_create_db(unit_type, txn, dupsort=True)

                                # Add to inverted index (dupdata=False: don't duplicate same key-value)
                                cursor = txn.cursor(unit_db)
                                cursor.put(unit_body, iscc_id_body, dupdata=False)

                        results.append(IsccAddResult(iscc_id=asset.iscc_id, status=status))

                break  # Success

            except lmdb.MapFullError:
                results = []  # Clear for retry
                old_size = self.map_size
                new_size = old_size * 2
                logger.info(f"LmdbIndex map_size increased from {old_size:,} to {new_size:,} bytes")
                self.env.set_mapsize(new_size)
                self._db_cache = {}  # DB handles invalid after resize

        return results

    def get_asset(self, iscc_id):
        # type: (str) -> IsccAsset
        """
        Get asset by ISCC-ID.

        :param iscc_id: ISCC-ID to retrieve
        :return: IsccAsset with full metadata
        :raises ValueError: If ISCC-ID realm doesn't match index realm or format invalid
        :raises FileNotFoundError: If asset not found
        """
        # Validate format and realm in single decode operation
        common.validate_iscc_id(iscc_id, expected_realm=self._realm_id)

        with self.env.begin() as txn:
            assets_db = self._get_db("__assets__", txn)
            if assets_db is None:
                raise FileNotFoundError(f"Asset '{iscc_id}' not found (index empty)")

            iscc_id_key = iscc_id.encode("utf-8")
            asset_bytes = txn.get(iscc_id_key, db=assets_db)

            if asset_bytes is None:
                raise FileNotFoundError(f"Asset '{iscc_id}' not found")

            return common.deserialize_asset(asset_bytes)

    def search_assets(self, query, limit=100):
        # type: (IsccAsset, int) -> IsccSearchResult
        """
        Search for similar assets using bidirectional prefix matching.

        Searches across all unit types and aggregates scores.
        Each ISCC-ID appears once with max bits per unit_type, summed across types.

        Accepts query with either iscc_code or units (or both). If only iscc_code
        is provided, units are automatically derived for search.

        :param query: IsccAsset with iscc_code or units (or both)
        :param limit: Maximum number of results
        :return: IsccSearchResult with matches sorted by score (descending)
        :raises ValueError: If query has neither iscc_code nor units
        """
        # Normalize query to ensure it has units (derive from iscc_code if needed)
        query = common.normalize_query_asset(query)

        with self.env.begin() as txn:
            # Aggregate matches: iscc_id → {unit_type → max_bits}
            matches = {}  # type: dict[str, dict[str, int]]

            for unit_str in query.units:
                # Units are plain ISCC strings
                unit = IsccUnit(unit_str)
                unit_type = unit.unit_type

                # Get unit-type database (skip if not indexed)
                unit_db = self._get_db(unit_type, txn)
                if unit_db is None:
                    continue

                # Bidirectional prefix search
                unit_matches = self._search_unit(txn, unit_db, unit)

                # Aggregate results
                for iscc_id, matched_bits in unit_matches.items():
                    if iscc_id not in matches:
                        matches[iscc_id] = {}

                    # Max per unit_type (prevents double-counting same unit at different lengths)
                    matches[iscc_id][unit_type] = max(matches[iscc_id].get(unit_type, 0), matched_bits)

            # Build match list
            match_list = []
            for iscc_id, unit_type_scores in matches.items():
                total_score = sum(unit_type_scores.values())
                match_list.append(
                    IsccMatch(
                        iscc_id=iscc_id,
                        score=total_score,
                        matches=unit_type_scores,
                    )
                )

            # Sort by score descending
            match_list.sort(key=lambda x: x.score, reverse=True)

            return IsccSearchResult(query=query, metric=Metric.bitlength, matches=match_list[:limit])

    def get_asset_count(self):
        # type: () -> int
        """
        Get number of assets in index.

        :return: Asset count
        """
        with self.env.begin() as txn:
            assets_db = self._get_db("__assets__", txn)
            if assets_db is None:
                return 0
            return txn.stat(assets_db)["entries"]

    def get_realm_id(self):
        # type: () -> int | None
        """
        Get index realm ID.

        :return: Realm ID (0 or 1) or None if no assets added yet
        """
        return self._realm_id

    def close(self):
        # type: () -> None
        """Close LMDB environment and release resources."""
        self.env.close()
        self._db_cache = {}

    @property
    def map_size(self):
        # type: () -> int
        """Current LMDB map_size in bytes."""
        return self.env.info()["map_size"]

    def __del__(self):
        # type: () -> None
        """Ensure LMDB environment is closed on deletion."""
        if hasattr(self, "env"):
            self.env.close()

    # Helper methods

    def _init_common_databases(self):
        # type: () -> None
        """
        Initialize common database handles on startup.

        Opens __assets__ and __metadata__ databases if they exist, caching
        handles for reuse across transactions. This ensures handles are valid
        for the environment lifetime.
        """
        try:
            # Try to open existing databases without creating them
            # Use a read transaction to check for existing databases
            with self.env.begin() as txn:
                # List all databases in the environment
                cursor = txn.cursor()
                if cursor.first():
                    # Environment has databases, try to open common ones
                    for db_name in ["__assets__", "__metadata__"]:
                        try:
                            # Open without transaction for environment-lifetime handle
                            db = self.env.open_db(db_name.encode("utf-8"), create=False)
                            self._db_cache[db_name] = db
                        except lmdb.NotFoundError:  # pragma: no cover
                            # Database doesn't exist yet, will be created on first add
                            pass
        except Exception:  # pragma: no cover
            # Defensive: ignore errors during initialization (index might be brand new)
            pass

    def _load_metadata(self):
        # type: () -> None
        """Load realm_id from __metadata__ database if exists."""
        try:
            with self.env.begin() as txn:
                metadata_db = self._get_db("__metadata__", txn)
                if metadata_db is not None:
                    realm_bytes = txn.get(b"realm_id", db=metadata_db)
                    if realm_bytes is not None:  # pragma: no branch
                        # Defensive: realm_id should always exist if __metadata__ exists
                        self._realm_id = int(realm_bytes.decode("utf-8"))
        except Exception:  # pragma: no cover
            # Defensive: ignore errors during metadata loading (index might be empty)
            pass

    def _save_realm_id(self, txn, realm_id):
        # type: (lmdb.Transaction, int) -> None
        """
        Save realm_id to __metadata__ database.

        :param txn: Write transaction
        :param realm_id: Realm ID to save
        """
        metadata_db = self._get_or_create_db("__metadata__", txn, dupsort=False)
        txn.put(b"realm_id", str(realm_id).encode("utf-8"), db=metadata_db)

        # Also save created_at if not exists
        if txn.get(b"created_at", db=metadata_db) is None:  # pragma: no branch
            # Defensive: created_at should only be set once (on first add_assets)
            txn.put(b"created_at", str(time.time()).encode("utf-8"), db=metadata_db)

    def _get_or_create_db(self, name, txn, dupsort=False):
        # type: (str, lmdb.Transaction, bool) -> object
        """
        Get or create named database.

        :param name: Database name
        :param txn: Write transaction for database creation
        :param dupsort: Enable duplicate keys (for inverted indexes)
        :return: Database handle valid for environment lifetime
        """
        if name in self._db_cache:
            return self._db_cache[name]

        if dupsort:
            # Inverted index database (unit_body → multiple iscc_id_body)
            db = self.env.open_db(
                name.encode("utf-8"),
                txn=txn,
                dupsort=True,  # Enable duplicate keys
                dupfixed=True,  # All dup values same size (8-byte ISCC-ID bodies)
                integerdup=True,  # Values are integers (enables efficient sorted storage)
            )
        else:
            # Regular database (assets, metadata)
            db = self.env.open_db(name.encode("utf-8"), txn=txn)

        self._db_cache[name] = db
        return db

    def _get_db(self, name, txn):
        # type: (str, lmdb.Transaction) -> object | None
        """
        Get database without creating.

        Opens database handle without transaction context to ensure it's valid
        for the environment lifetime, not just the current transaction.

        :param name: Database name
        :param txn: Transaction (unused, kept for API compatibility)
        :return: Database handle or None if not exists
        """
        if name in self._db_cache:
            return self._db_cache[name]

        try:
            # Open without transaction context for environment-lifetime handle
            db = self.env.open_db(name.encode("utf-8"), create=False)
            self._db_cache[name] = db
            return db
        except lmdb.NotFoundError:
            return None

    def _search_unit(self, txn, db, unit):
        # type: (lmdb.Transaction, object, IsccUnit) -> dict[str, int]
        """
        Bidirectional prefix search for variable-length ISCC matching.

        Forward: Finds stored units starting with query (query shorter than stored).
        Reverse: Finds stored units that are prefixes of query (query longer than stored).

        :param txn: Read transaction
        :param db: Unit-type database
        :param unit: Query unit
        :return: ISCC-ID → matched_bits (max bits for duplicate findings)
        """
        matches = {}  # type: dict[str, int]
        cursor = txn.cursor(db)
        query_body = unit.body
        query_bits = len(unit)

        # Forward search: stored units starting with query
        # Example: query=64bit finds stored 128bit/192bit/256bit with same prefix
        if cursor.set_range(query_body):  # pragma: no branch
            # Defensive: cursor.set_range can return False if no matching position found
            for key, value in cursor:
                if not key.startswith(query_body):
                    break
                iscc_id = self._bytes_to_iscc_id(value)
                matched_bits = query_bits  # Full query matched
                matches[iscc_id] = max(matches.get(iscc_id, 0), matched_bits)

        # Reverse search: stored units that are prefixes of query
        # Example: query=256bit finds stored 64bit/128bit/192bit units
        for bit_length in [64, 128, 192]:
            if bit_length >= query_bits:  # Skip lengths >= query
                continue

            prefix = query_body[: bit_length // 8]
            if cursor.set_range(prefix):  # pragma: no branch
                # Defensive: cursor.set_range can return False if no matching position found
                for key, value in cursor:
                    if not key.startswith(prefix):
                        break
                    if query_body.startswith(key):  # pragma: no branch
                        # Stored is prefix of query
                        iscc_id = self._bytes_to_iscc_id(value)
                        matched_bits = len(key) * 8  # Stored unit length
                        matches[iscc_id] = max(matches.get(iscc_id, 0), matched_bits)

        return matches

    def _bytes_to_iscc_id(self, digest):
        # type: (bytes) -> str
        """
        Reconstruct ISCC-ID string from 8-byte body using realm_id.

        :param digest: 8-byte ISCC-ID body from LMDB
        :return: Canonical ISCC-ID string with reconstructed header
        """
        if self._realm_id is None:  # pragma: no cover
            # Defensive: should never happen (realm_id set on first add_assets)
            raise ValueError("Cannot reconstruct ISCC-ID: realm_id not set")
        return common.reconstruct_iscc_id(digest, self._realm_id)
