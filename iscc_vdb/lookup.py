"""
LMDB-backed inverted index for fast ISCC-UNIT prefix lookups.

## Architecture

- One LMDB named database per unit_type (e.g., "CONTENT_TEXT_V0", "DATA_NONE_V0")
- Inverted index: unit_body → [iscc_id_body, ...]
- Supports variable-length ISCCs via bidirectional prefix matching
- Only stores body bytes (headers reconstructed using unit_type + realm_id)

## Search Algorithm

Bidirectional prefix matching handles variable-length ISCCs:
- Forward: Find stored units starting with query (query=64bit finds stored 128bit)
- Reverse: Find stored units that are prefixes of query (query=256bit finds stored 64bit)

Results are aggregated: each ISCC-ID appears once with max bits per unit_type, summed across types.
"""

import os
from typing import TypedDict
import lmdb
import iscc_core as ic
from loguru import logger
from iscc_vdb.models import IsccItem, IsccUnit


class IsccLookupMatchDict(TypedDict):
    """Match result: ISCC-ID with aggregated bit scores per unit_type."""

    iscc_id: str
    score: int  # Sum of all matched bits across unit_types
    matches: dict[str, int]  # unit_type -> matched_bits (max per type)


class IsccLookupResultDict(TypedDict):
    """Search result for one query item."""

    lookup_matches: list[IsccLookupMatchDict]


class IsccLookupIndex:
    """
    LMDB-backed inverted index for variable-length ISCC-UNIT prefix search.

    Each unit_type gets its own named database with dupsort enabled (unit → multiple ISCC-IDs).
    Bidirectional prefix matching ensures all variable-length matches are found.
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

    def __init__(self, path, realm_id=0, lmdb_options=None):
        # type: (os.PathLike, int, dict[str, Any] | None) -> None
        """
        Create or open LMDB lookup index.

        :param path: Path to LMDB file (subdir=False)
        :param realm_id: ISCC-ID realm (0 or 1) for header reconstruction during search
        :param lmdb_options: Custom LMDB options (max_dbs and subdir are forced internally)
        """
        self.path = os.fspath(path)
        self.realm_id = realm_id

        options = self.DEFAULT_LMDB_OPTIONS.copy()
        if lmdb_options:
            options.update(lmdb_options)

        # Force critical parameters (cannot be overridden by user)
        options["max_dbs"] = 32  # Each unit_type gets its own named database
        options["subdir"] = False  # Path points to file, not directory

        self.env = lmdb.open(self.path, **options)
        self._db_cache = {}  # type: dict[str, Any]  # Opened db handles (invalidated on resize)

    def add(self, iscc_items):
        # type: (IsccItemDict | list[IsccItemDict]) -> list[str]
        """
        Add ISCC items to index with auto-generated ISCC-IDs if missing.

        Stores inverted mappings: unit_body → iscc_id_body in unit_type-specific databases.
        Same unit can map to multiple ISCC-IDs (dupsort). Auto-retries on MapFullError.

        :param iscc_items: Single IsccItemDict or list
        :return: ISCC-IDs (generated or provided), one per input item
        """
        items = self._normalize_input(iscc_items)
        iscc_ids = []

        while True:  # Auto-retry loop for MapFullError
            try:
                with self.env.begin(write=True) as txn:
                    for item in items:
                        iscc_item = IsccItem.from_dict(item)  # Generates random ISCC-ID if missing
                        iscc_ids.append(iscc_item.iscc_id)

                        iscc_id_body = iscc_item.id_data[2:]  # Skip 2-byte header

                        for unit_str in iscc_item.units:
                            unit = IsccUnit(unit_str)
                            unit_type = unit.unit_type

                            db = self._get_or_create_db(unit_type, txn)

                            # dupdata=False: Don't add same key-value pair twice
                            # (but same key with different value is allowed due to dupsort=True)
                            cursor = txn.cursor(db)
                            cursor.put(unit.body, iscc_id_body, dupdata=False)

                break  # Success

            except lmdb.MapFullError:
                iscc_ids = []  # Clear for retry
                old_size = self.map_size
                new_size = old_size * 2
                logger.info(f"IsccLookupIndex map_size increased from {old_size:,} to {new_size:,} bytes")
                self.env.set_mapsize(new_size)
                self._db_cache = {}  # DB handles invalid after resize

        return iscc_ids

    def search(self, iscc_items, limit=100):
        # type: (IsccItemDict | list[IsccItemDict], int) -> list[IsccLookupResultDict]
        """
        Search via bidirectional prefix matching with aggregated scoring.

        Each ISCC-ID appears once per result (distinct). Scores: max bits per unit_type, summed.
        Handles variable-length ISCCs by searching both directions (forward + reverse).

        :param iscc_items: Single IsccItemDict or list
        :param limit: Max matches per query (default: 100)
        :return: One IsccLookupResultDict per query item
        """
        items = self._normalize_input(iscc_items)
        results = []

        with self.env.begin() as txn:  # Read-only transaction
            for item in items:
                iscc_item = IsccItem.from_dict(item)

                # Dict key = ISCC-ID ensures distinctness
                matches = {}  # type: dict[str, dict[str, int]]  # iscc_id -> {unit_type -> max_bits}

                for unit_str in iscc_item.units:
                    unit = IsccUnit(unit_str)
                    unit_type = unit.unit_type

                    db = self._get_db(unit_type, txn)
                    if db is None:  # Unit type not indexed
                        continue

                    unit_matches = self._search_unit(txn, db, unit)  # Bidirectional prefix search

                    # Aggregate: same ISCC-ID found via multiple units/prefixes
                    for iscc_id, matched_bits in unit_matches.items():
                        if iscc_id not in matches:
                            matches[iscc_id] = {}

                        # Max per unit_type (prevents double-counting same unit at different lengths)
                        matches[iscc_id][unit_type] = max(matches[iscc_id].get(unit_type, 0), matched_bits)

                # Build result list
                lookup_matches = []
                for iscc_id, unit_type_scores in matches.items():
                    total_score = sum(unit_type_scores.values())  # Sum across unit_types
                    match_dict = IsccLookupMatchDict(
                        iscc_id=iscc_id,
                        score=total_score,
                        matches=unit_type_scores,
                    )
                    lookup_matches.append(match_dict)

                lookup_matches.sort(key=lambda x: x["score"], reverse=True)
                lookup_matches = lookup_matches[:limit]

                results.append(IsccLookupResultDict(lookup_matches=lookup_matches))

        return results

    def close(self):
        # type: () -> None
        """Close LMDB environment and release resources."""
        self.env.close()

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

    def _normalize_input(self, iscc_items):
        # type: (IsccItemDict | list[IsccItemDict]) -> list[IsccItemDict]
        """Convert single dict to list for uniform processing."""
        if isinstance(iscc_items, dict):
            return [iscc_items]
        return iscc_items

    def _get_or_create_db(self, unit_type, txn):
        # type: (str, lmdb.Transaction) -> Any
        """
        Get or create named database for unit_type with optimized flags.

        dupsort=True: Same key can have multiple values (unit → many ISCC-IDs)
        dupfixed=True: All values are same size (8 bytes for ISCC-ID body)
        integerdup=True: Values are binary integers, enables sorted duplicates

        :param unit_type: ISCC unit type (e.g., "CONTENT_TEXT_V0")
        :param txn: Write transaction
        :return: Cached database handle
        """
        if unit_type in self._db_cache:
            return self._db_cache[unit_type]

        db = self.env.open_db(
            unit_type.encode("utf-8"),
            txn=txn,
            dupsort=True,  # Enable duplicate keys (one unit → multiple ISCC-IDs)
            dupfixed=True,  # All dup values same size (8-byte ISCC-ID bodies)
            integerdup=True,  # Values are integers (enables efficient sorted storage)
        )
        self._db_cache[unit_type] = db
        return db

    def _get_db(self, unit_type, txn):
        # type: (str, lmdb.Transaction) -> Any | None
        """
        Get database for unit_type without creating (None if missing).

        Used during search to skip unit_types that were never indexed.

        :param unit_type: ISCC unit type
        :param txn: Read transaction
        :return: Cached database handle or None
        """
        if unit_type in self._db_cache:
            return self._db_cache[unit_type]

        try:
            db = self.env.open_db(unit_type.encode("utf-8"), txn=txn, create=False)
            self._db_cache[unit_type] = db
            return db
        except lmdb.NotFoundError:
            return None

    def _search_unit(self, txn, db, unit):
        # type: (lmdb.Transaction, Any, IsccUnit) -> dict[str, int]
        """
        Bidirectional prefix search for variable-length ISCC matching.

        Forward: Finds stored units starting with query (query shorter than stored).
        Reverse: Finds stored units that are prefixes of query (query longer than stored).

        :param txn: Read transaction
        :param db: Unit_type database
        :param unit: Query unit
        :return: ISCC-ID → matched_bits (max bits for duplicate findings)
        """
        matches = {}  # type: dict[str, int]
        cursor = txn.cursor(db)
        query_body = unit.body
        query_bits = len(unit)

        # Forward search: stored units starting with query
        # Example: query=64bit finds stored 128bit/192bit/256bit with same prefix
        if cursor.set_range(query_body):
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
            if cursor.set_range(prefix):
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

        Only body is stored; header is reconstructed from realm_id.

        :param digest: 8-byte ISCC-ID body from LMDB
        :return: Canonical ISCC-ID string with reconstructed header
        """
        if self.realm_id == 0:
            header = ic.encode_header(ic.MT.ID, 0, ic.VS.V1, 0)
        elif self.realm_id == 1:
            header = ic.encode_header(ic.MT.ID, 1, ic.VS.V1, 0)
        else:
            raise ValueError(f"Invalid realm_id {self.realm_id}, must be 0 or 1")
        return "ISCC:" + ic.encode_base32(header + digest)
