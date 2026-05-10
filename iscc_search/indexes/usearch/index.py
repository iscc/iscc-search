"""
Usearch-backed single index implementation with LMDB metadata storage.

Hybrid architecture combining:
- LMDB: Asset storage, metadata, INSTANCE exact-matching (dupsort), simprint source of truth
- ShardedNphdIndex: Similarity search for META, CONTENT, DATA units (bounded RAM, auto-sharding)
- UsearchSimprintIndex: Derived approximate simprint search (ShardedIndex128 per type)

Directory structure:
- index.lmdb: LMDB environment with __metadata__, __assets__, __instance__, __sp_*__ databases
- {unit_type}/: ShardedNphdIndex directories for similarity units (lazy-created)
- SIMPRINT_{sp_type}/: ShardedIndex128 directories for simprint approximate search (derived)

Key strategy: ISCC-ID body as uint64 for LMDB and ShardedNphdIndex. 128-bit composite
keys (iscc_id_body + offset + size) for ShardedIndex128 simprint indexes.
"""

import json
import shutil
import struct
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import lmdb
from loguru import logger
from iscc_search.schema import IsccAddResult, IsccGlobalMatch, IsccSearchResult, Status
from iscc_search.models import IsccUnit, IsccID
from iscc_search.indexes import common
from iscc_usearch import ShardedNphdIndex
from iscc_search.indexes.simprint.usearch_core import UsearchSimprintIndex
from iscc_search.indexes.simprint import lmdb_ops
import iscc_core as ic

if TYPE_CHECKING:
    from iscc_search.schema import IsccEntry, IsccQuery, IsccChunkMatch  # noqa: F401


class _RWLock:
    """Readers-writer lock with writer preference for LMDB resize safety."""

    def __init__(self):
        # type: () -> None
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._writer_waiting = False

    def acquire_read(self):
        # type: () -> None
        """Acquire shared read access (blocks while a writer holds or waits for the lock)."""
        with self._cond:
            while self._writer or self._writer_waiting:
                self._cond.wait()
            self._readers += 1

    def release_read(self):
        # type: () -> None
        """Release shared read access."""
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self):
        # type: () -> None
        """Acquire exclusive write access (drains active readers, blocks new ones)."""
        with self._cond:
            self._writer_waiting = True
            while self._writer or self._readers > 0:
                self._cond.wait()
            self._writer_waiting = False
            self._writer = True

    def release_write(self):
        # type: () -> None
        """Release exclusive write access."""
        with self._cond:
            self._writer = False
            self._cond.notify_all()


class UsearchIndex:
    """
    Single usearch-backed index with LMDB for metadata and INSTANCE matching.

    Storage structure:
    LMDB databases (in index.lmdb file):
    - __metadata__: realm_id (int), max_dim (int), created_at (float)
    - __assets__: uint64 key → IsccEntry JSON bytes
    - __instance__: instance_code digest → [iscc_id_body uint64, ...] (dupsort/dupfixed/integerdup)

    ShardedNphdIndex directories:
    - {unit_type}/: One directory per similarity unit type (META, CONTENT, DATA)
      Contains shard files and bloom filter managed by ShardedNphdIndex.

    All keys use ISCC-ID body as uint64 for consistency between LMDB and usearch.
    """

    DEFAULT_LMDB_OPTIONS = {
        "readonly": False,
        "metasync": False,
        "sync": False,
        "mode": 0o644,
        "create": True,
        "readahead": False,
        "writemap": False,
        "meminit": False,
        "map_async": False,
        "map_size": 1024 * 1024 * 1024 * 1024,  # 1 TB (sparse — no disk cost on 64-bit)
        "max_readers": 126,
        "max_spare_txns": 16,
        "lock": True,
    }

    # MapFullError retry limits
    MAX_RESIZE_RETRIES = 10  # Maximum number of resize attempts
    MAX_MAP_SIZE = 1024 * 1024 * 1024 * 1024  # 1 TB maximum map size

    def __init__(self, path, realm_id=None, max_dim=256, lmdb_options=None, **options):
        # type: (str | Path, int | None, int, dict | None, Any) -> None
        """
        Create or open usearch index at directory path.

        :param path: Path to index directory (contains index.lmdb + per-type shard directories)
        :param realm_id: ISCC realm ID for new indexes (0 or 1). If None, inferred from first asset.
        :param max_dim: Maximum dimensions for ShardedNphdIndex (any multiple of 8 bits up to 256)
        :param lmdb_options: Custom LMDB options (max_dbs and subdir are forced)
        :param options: Override options from search_opts (e.g. match_threshold_units=0.8)
        """
        from iscc_search.options import search_opts

        opts = search_opts.override(options)
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.max_dim = max_dim
        self._opts = opts
        self._realm_id = None  # type: int | None
        self._nphd_indexes = {}  # type: dict[str, ShardedNphdIndex]
        self._simprint_indexes = {}  # type: dict[str, UsearchSimprintIndex]
        self._sp_data_dbs = {}  # type: dict[str, lmdb._Database]
        self._sp_assets_dbs = {}  # type: dict[str, lmdb._Database]
        self._closed = False

        # Profiling counters (cumulative across batches)
        self._batch_counter = 0  # type: int
        self._cumulative_nphd_vectors = 0  # type: int
        self._cumulative_sp_vectors = 0  # type: int

        # Serialize LMDB write operations to prevent EINVAL from concurrent set_mapsize.
        # RLock allows reentrant acquisition (e.g., _rebuild_simprint_index → _update_sp_metadata
        # both need the lock, and _update_sp_metadata is also called from within add_assets/flush/close).
        self._write_lock = threading.RLock()

        # Coordinate set_mapsize with concurrent read transactions.
        # LMDB requires no active transactions during set_mapsize (undefined behavior otherwise).
        # Read operations acquire shared access; set_mapsize acquires exclusive access.
        self._env_rwlock = _RWLock()

        # Setup LMDB
        lmdb_path = self.path / "index.lmdb"
        options = self.DEFAULT_LMDB_OPTIONS.copy()
        if lmdb_options:
            options.update(lmdb_options)

        # Force critical parameters
        min_max_dbs = 32  # __metadata__, __assets__, __instance__ + up to ~14 simprint types (2 DBs each)
        options["max_dbs"] = max(options.get("max_dbs", 0), min_max_dbs)
        options["subdir"] = False  # Path points to file

        self.env = lmdb.open(str(lmdb_path), **options)

        # Initialize or load metadata
        self._init_metadata(realm_id)

        # Load existing ShardedNphdIndex directories
        self._load_nphd_indexes()

        # Load existing LMDB simprint databases
        self._load_sp_databases()

        # Load derived simprint indexes (ShardedIndex128 per type)
        self._load_simprint_indexes()

    def add_assets(self, assets):
        # type: (list[IsccEntry]) -> list[IsccAddResult]
        """
        Add assets to index.

        Stores assets in LMDB, INSTANCE units in dupsort database,
        and similarity units in ShardedNphdIndex directories.

        Consistency model: LMDB commits before ShardedNphdIndex operations. If derived
        index operations fail, assets are in LMDB (source of truth) but not in similarity
        search. This is acceptable as derived indexes can be rebuilt from LMDB. True
        two-phase commit would add significant complexity for rare failure scenarios.

        :param assets: List of IsccEntry instances to add
        :return: List of IsccAddResult with created/updated status
        :raises ValueError: If realm_id inconsistent or missing iscc_id
        """
        if not assets:
            return []

        self._batch_counter += 1
        batch_num = self._batch_counter
        batch_t0 = time.perf_counter()

        results = []
        retry_count = 0

        while retry_count <= self.MAX_RESIZE_RETRIES:  # pragma: no branch
            self._write_lock.acquire()
            try:
                lmdb_t0 = time.perf_counter()
                lmdb_serialize_t = 0.0
                lmdb_simprint_t = 0.0
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

                    # Prepare vectors for batch add to ShardedNphdIndex
                    nphd_batches = {}  # type: dict[str, tuple[list[int], list[bytes]]]
                    # Prepare vectors for batch add to derived ShardedIndex128 simprint indexes
                    sp_batches = {}  # type: dict[str, tuple[list[bytes], list[np.ndarray]]]
                    sp_deleted_keys = {}  # type: dict[str, list[bytes]]
                    # Accumulate simprint LMDB pairs for batched putmulti
                    sp_lmdb_pairs = {}  # type: dict[str, list[tuple[bytes, bytes]]]

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
                        _t = time.perf_counter()
                        asset_bytes = common.serialize_asset(asset)
                        txn.put(key_bytes, asset_bytes, db=assets_db)
                        lmdb_serialize_t += time.perf_counter() - _t

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
                                    # Batch for ShardedNphdIndex (similarity matching)
                                    if unit_type not in nphd_batches:
                                        nphd_batches[unit_type] = ([], [])
                                    nphd_batches[unit_type][0].append(key)
                                    nphd_batches[unit_type][1].append(unit_body)

                        # Prepare simprints for batched LMDB write
                        if asset.simprints and asset.iscc_id:
                            _t = time.perf_counter()
                            iscc_id_body = IsccID(asset.iscc_id).body
                            for sp_type, sp_list in asset.simprints.items():
                                data_db, sp_assets_db = self._open_sp_databases_in_txn(txn, sp_type)
                                if txn.get(iscc_id_body, db=sp_assets_db) is not None:
                                    # Update: delete old simprint entries before re-indexing
                                    deleted = lmdb_ops.delete_asset_simprints(txn, data_db, iscc_id_body)
                                    sp_deleted_keys.setdefault(sp_type, []).extend(deleted)
                                txn.put(iscc_id_body, b"", db=sp_assets_db)
                                for sp_obj in sp_list:
                                    sp_bytes = ic.decode_base64(sp_obj.simprint)
                                    chunk_ptr = lmdb_ops.pack_chunk_pointer(iscc_id_body, sp_obj.offset, sp_obj.size)
                                    # Accumulate for batched putmulti
                                    sp_lmdb_pairs.setdefault(sp_type, []).append((sp_bytes, chunk_ptr))
                                    # Batch for derived ShardedIndex128
                                    if sp_type not in sp_batches:
                                        sp_batches[sp_type] = ([], [])
                                    sp_batches[sp_type][0].append(chunk_ptr)
                                    sp_batches[sp_type][1].append(np.frombuffer(sp_bytes, dtype=np.uint8))
                            lmdb_simprint_t += time.perf_counter() - _t

                        results.append(IsccAddResult(iscc_id=asset.iscc_id, status=status))

                    # Batch write all simprint data to LMDB using putmulti (C loop)
                    _t = time.perf_counter()
                    for sp_type, pairs in sp_lmdb_pairs.items():
                        data_db = self._sp_data_dbs[sp_type]
                        cursor = txn.cursor(data_db)
                        cursor.putmulti(pairs, dupdata=False)
                    lmdb_simprint_t += time.perf_counter() - _t

                # LMDB transaction commits here (exits context manager)
                lmdb_elapsed = time.perf_counter() - lmdb_t0

                # Derived index operations below are NOT atomic with LMDB - see docstring
                # Batch add to ShardedNphdIndex (outside transaction)
                nphd_t0 = time.perf_counter()
                nphd_remove_t = 0.0
                nphd_add_t = 0.0
                batch_nphd_vectors = 0
                for unit_type, (keys, vectors) in nphd_batches.items():
                    nphd_index = self._get_or_create_nphd_index(unit_type)

                    # Deduplicate keys within batch (keep last occurrence for each duplicate).
                    # Remove-before-add at line 314 is still needed for update semantics.
                    if len(keys) != len(set(keys)):
                        # Build dict with key -> (last) vector mapping
                        unique_items = {}  # type: dict[int, np.ndarray]
                        for key, vector in zip(keys, vectors):
                            unique_items[key] = vector
                        # Rebuild lists from deduplicated items
                        keys = list(unique_items.keys())
                        vectors = list(unique_items.values())

                    # Remove existing keys first (for updates)
                    # remove() handles non-existent keys gracefully (returns 0)
                    _t = time.perf_counter()
                    nphd_index.remove(keys)
                    nphd_remove_t += time.perf_counter() - _t

                    _t = time.perf_counter()
                    nphd_index.add(keys, vectors)
                    nphd_add_t += time.perf_counter() - _t
                    batch_nphd_vectors += len(keys)

                    # Drain pending rotations so .size reflects all vectors
                    nphd_index.drain_rotations()
                    self._update_nphd_metadata(unit_type, nphd_index.size)
                nphd_elapsed = time.perf_counter() - nphd_t0

                # Update derived ShardedIndex128 simprint indexes
                # Remove stale keys first (from updated assets), then add new vectors
                sp_t0 = time.perf_counter()
                sp_remove_t = 0.0
                sp_add_t = 0.0
                batch_sp_vectors = 0
                for sp_type, (composite_keys, sp_vectors) in sp_batches.items():
                    sp_index = self._get_or_create_simprint_index(sp_type, len(sp_vectors[0]) * 8)
                    if sp_type in sp_deleted_keys:
                        _t = time.perf_counter()
                        sp_index.remove(sp_deleted_keys[sp_type])
                        sp_remove_t += time.perf_counter() - _t
                    _t = time.perf_counter()
                    sp_index.add_raw(composite_keys, sp_vectors)
                    sp_add_t += time.perf_counter() - _t
                    batch_sp_vectors += len(composite_keys)
                    sp_index.drain_rotations()
                    self._update_sp_metadata(sp_type, sp_index.size)

                # Remove stale vectors for types with only deletions (no new vectors)
                for sp_type, deleted_keys in sp_deleted_keys.items():
                    if sp_type not in sp_batches and sp_type in self._simprint_indexes:
                        _t = time.perf_counter()
                        self._simprint_indexes[sp_type].remove(deleted_keys)
                        sp_remove_t += time.perf_counter() - _t
                        self._update_sp_metadata(sp_type, self._simprint_indexes[sp_type].size)
                sp_elapsed = time.perf_counter() - sp_t0

                # Auto-flush sub-indexes that exceed flush_interval
                flush_t0 = time.perf_counter()
                flush_interval = self._opts.flush_interval
                if flush_interval > 0:
                    for nphd_index in self._nphd_indexes.values():
                        if nphd_index.dirty >= flush_interval:
                            nphd_index.save()
                    for sp_index in self._simprint_indexes.values():
                        if sp_index.dirty >= flush_interval:
                            sp_index.save()
                flush_elapsed = time.perf_counter() - flush_t0

                # Log profiling summary
                self._cumulative_nphd_vectors += batch_nphd_vectors
                self._cumulative_sp_vectors += batch_sp_vectors
                batch_elapsed = time.perf_counter() - batch_t0
                logger.debug(
                    f"add_assets batch={batch_num} assets={len(assets)} "
                    f"nphd={batch_nphd_vectors} sp={batch_sp_vectors} "
                    f"total_nphd={self._cumulative_nphd_vectors} total_sp={self._cumulative_sp_vectors} | "
                    f"lmdb={lmdb_elapsed:.3f}s (ser={lmdb_serialize_t:.3f} sp_w={lmdb_simprint_t:.3f}) "
                    f"nphd={nphd_elapsed:.3f}s (rm={nphd_remove_t:.3f} add={nphd_add_t:.3f}) "
                    f"sp={sp_elapsed:.3f}s (rm={sp_remove_t:.3f} add={sp_add_t:.3f}) "
                    f"flush={flush_elapsed:.3f}s TOTAL={batch_elapsed:.3f}s"
                )

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

                # Clear result list for retry (LMDB transaction rolled back).
                # NPHD indexes are NOT reset: MapFullError occurs inside the LMDB
                # transaction, before NPHD additions run (line ~292), so they still
                # hold all vectors from prior successful batches.
                results = []
                old_size = self.map_size

                # Limit resize increment to max 1GB to avoid wasting space
                increase = min(old_size, 1024 * 1024 * 1024)
                new_size = min(old_size + increase, self.MAX_MAP_SIZE)

                # Check if we've hit the limit
                if new_size == old_size:
                    raise RuntimeError(f"Cannot resize beyond {self.MAX_MAP_SIZE} bytes")

                logger.info(
                    f"Resizing LMDB from {old_size:,} to {new_size:,} bytes (increase: {increase:,}) "
                    f"(retry {retry_count}/{self.MAX_RESIZE_RETRIES})"
                )
                self._safe_resize(new_size)
                # py-lmdb 2.2.0 invalidates cached named-db handles on set_mapsize.
                # Repopulate simprint db caches from metadata before retry.
                self._sp_data_dbs = {}
                self._sp_assets_dbs = {}
                self._load_sp_databases()

            finally:
                self._write_lock.release()

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
            with self._read_txn() as txn:
                assets_db = self.env.open_db(b"__assets__", txn=txn)
                asset_bytes = txn.get(key_bytes, db=assets_db)
        except lmdb.ReadonlyError:  # pragma: no cover
            # Database doesn't exist yet (empty index)
            raise FileNotFoundError(f"Asset '{iscc_id}' not found in index")

        if asset_bytes is None:  # pragma: no cover
            raise FileNotFoundError(f"Asset '{iscc_id}' not found in index")

        return common.deserialize_asset(asset_bytes)

    def search_assets(self, query, limit=100, exact=False):
        # type: (IsccQuery, int, bool) -> IsccSearchResult
        """
        Search for similar assets using NPHD metric.

        Combines exact INSTANCE matching (LMDB) with similarity matching (ShardedNphdIndex).

        **Per-unit scoring** (normalized 0.0-1.0):
        - INSTANCE units: Binary (1.0 = match, 0.0 = no match). Any prefix match scores 1.0
          since INSTANCE codes are identity codes (checksums/hashes).
        - Similarity units (CONTENT/META/DATA): score = 1.0 - nphd_distance. Perfect match
          of query bits scores 1.0 regardless of unit length (64/128/192/256 bits).

        **Score aggregation**: Confidence-weighted averaging with noise filtering:
        1. Filter out low-confidence matches below match_threshold_units (default 0.75)
        2. Apply confidence weighting: score^confidence_exponent (default 4)
        3. Calculate weighted average: sum(score^2) / sum(score)
        This emphasizes high-confidence matches and filters noise. A perfect match (1.0)
        on one type ranks higher than multiple mediocre matches.

        :param query: Query with units
        :param limit: Maximum number of results
        :param exact: If True, use LMDB exact search for simprints instead of approximate search
        :return: IsccSearchResult with query and list of matches (scores normalized 0.0-1.0)
        """
        # Handle iscc_id lookup if provided (takes precedence over other fields)
        query_iscc_id = None  # Track original query iscc_id for self-exclusion
        if query.iscc_id:
            query_iscc_id = query.iscc_id
            # Look up asset by iscc_id (raises FileNotFoundError if not found -> HTTP 404)
            asset = self.get_asset(query.iscc_id)
            # Create new query with extracted iscc_code, units and simprints
            from iscc_search.schema import IsccQuery

            query = IsccQuery(iscc_code=asset.iscc_code, units=asset.units, simprints=asset.simprints)

        # Normalize query
        query = common.normalize_query(query)

        # Search simprints for chunk-level matches (can work without units)
        chunk_matches = []  # type: list[IsccChunkMatch]
        has_simprint_index = bool(self._simprint_indexes) or bool(self._sp_data_dbs)
        if has_simprint_index and query.simprints:
            chunk_matches = self._search_simprints(query, limit, exact=exact)

        # Search units for global matches (only if units present)
        matches = []  # type: list[IsccGlobalMatch]
        if query.units:
            # Aggregation: {key (int): {unit_type: score}}
            aggregated = {}  # type: dict[int, dict[str, float]]

            for unit_str in query.units:
                unit = IsccUnit(unit_str)
                unit_type = unit.unit_type
                unit_body = unit.body

                if unit_type.startswith("INSTANCE_"):
                    # Exact matching via LMDB dupsort
                    matches_dict = self._search_instance_unit(unit_body)
                    for key, score in matches_dict.items():
                        if key not in aggregated:
                            aggregated[key] = {}
                        aggregated[key][unit_type] = score
                else:
                    # Similarity matching via ShardedNphdIndex
                    if unit_type in self._nphd_indexes:
                        matches_dict = self._search_similarity_unit(unit_type, unit_body, limit)
                        for key, score in matches_dict.items():
                            if key not in aggregated:
                                aggregated[key] = {}
                            # Store max score if multiple matches for same unit_type
                            aggregated[key][unit_type] = max(aggregated[key].get(unit_type, 0.0), score)

            # Calculate confidence-weighted scores and sort
            scored_results = []  # type: list[tuple[int, float, dict[str, float]]]
            for key, unit_scores in aggregated.items():
                # Filter out low-confidence matches (below threshold)
                confident_matches = {
                    unit_type: score
                    for unit_type, score in unit_scores.items()
                    if score >= self._opts.match_threshold_units
                }

                # Skip if no confident matches
                if not confident_matches:
                    continue

                # Confidence-weighted average (high scores count more)
                # score^2 / score = score, so this amplifies differences
                weighted_sum = sum(score**self._opts.confidence_exponent for score in confident_matches.values())
                weight_sum = sum(score for score in confident_matches.values())
                total_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0

                scored_results.append((key, total_score, unit_scores))

            # Exclude query asset from results (self-exclusion for iscc_id queries)
            if query_iscc_id:
                query_key = int(IsccID(query_iscc_id))
                scored_results = [result for result in scored_results if result[0] != query_key]

            # Sort by total score descending
            scored_results.sort(key=lambda x: x[1], reverse=True)

            # Take top limit results
            scored_results = scored_results[:limit]

            # Enrich with metadata
            try:
                with self._read_txn() as txn:
                    assets_db = self.env.open_db(b"__assets__", txn=txn)

                    for key, total_score, unit_scores in scored_results:
                        # Reconstruct ISCC-ID from key
                        iscc_id = str(IsccID.from_int(key, self._realm_id))

                        # Fetch asset metadata
                        source = None
                        metadata = None
                        key_bytes = struct.pack(">Q", key)
                        asset_bytes = txn.get(key_bytes, db=assets_db)

                        if asset_bytes is not None:
                            asset = common.deserialize_asset(asset_bytes)
                            if asset.metadata:
                                source = asset.metadata.get("source")
                                metadata = asset.metadata

                        matches.append(
                            IsccGlobalMatch(
                                iscc_id=iscc_id,
                                score=total_score,
                                types=unit_scores,
                                source=source,
                                metadata=metadata,
                            )
                        )
            except lmdb.ReadonlyError:  # pragma: no cover
                # Database doesn't exist yet (empty index) - return matches without metadata
                for key, total_score, unit_scores in scored_results:
                    iscc_id = str(IsccID.from_int(key, self._realm_id))
                    matches.append(IsccGlobalMatch(iscc_id=iscc_id, score=total_score, types=unit_scores))

        # Exclude query asset from chunk matches (self-exclusion for iscc_id queries)
        if query_iscc_id:
            chunk_matches = [match for match in chunk_matches if match.iscc_id != query_iscc_id]

        return IsccSearchResult(query=query, global_matches=matches, chunk_matches=chunk_matches)

    def flush(self):
        # type: () -> None
        """
        Save dirty derived indexes (ShardedNphdIndex and UsearchSimprintIndex) to disk.

        Exception-safe: each sub-index is saved independently so a failure in one
        does not prevent the others from being saved.
        Skips sub-indexes with dirty == 0 to avoid unnecessary I/O.
        Indexes are automatically saved on close(), so flush() is only
        needed for durability guarantees during long-running sessions.
        """
        with self._write_lock:
            for unit_type, nphd_index in self._nphd_indexes.items():
                if nphd_index.dirty == 0:
                    continue
                try:
                    nphd_index.save()
                    self._update_nphd_metadata(unit_type, nphd_index.size)
                    logger.debug(f"Flushed ShardedNphdIndex for unit_type '{unit_type}'")
                except Exception:  # pragma: no cover
                    logger.exception(f"Failed to flush ShardedNphdIndex '{unit_type}'")

            for sp_type, sp_index in self._simprint_indexes.items():
                if sp_index.dirty == 0:
                    continue
                try:
                    sp_index.save()
                    self._update_sp_metadata(sp_type, sp_index.size)
                    logger.debug(f"Flushed UsearchSimprintIndex for type '{sp_type}'")
                except Exception:  # pragma: no cover
                    logger.exception(f"Failed to flush UsearchSimprintIndex '{sp_type}'")

    def close(self):
        # type: () -> None
        """
        Close LMDB environment and all derived indexes, saving dirty state.

        Idempotent and exception-safe: each sub-index is saved independently so a
        failure in one does not prevent the others from being saved.
        """
        if self._closed:
            return

        with self._write_lock:
            if self._closed:  # pragma: no cover - race condition guard
                return

            # Close ShardedNphdIndex instances (saves if dirty, releases resources)
            for unit_type, nphd_index in list(self._nphd_indexes.items()):
                try:
                    dirty = nphd_index.dirty > 0
                    nphd_index.drain_rotations()
                    size = nphd_index.size
                    nphd_index.close()
                    if dirty:
                        self._update_nphd_metadata(unit_type, size)
                        logger.info(f"Saved ShardedNphdIndex '{unit_type}' ({size} vectors)")
                except Exception:  # pragma: no cover
                    logger.exception(f"Failed to close ShardedNphdIndex '{unit_type}'")

            self._nphd_indexes.clear()

            # Close derived simprint indexes (ShardedIndex128)
            for sp_type, sp_index in list(self._simprint_indexes.items()):
                try:
                    dirty = sp_index.dirty > 0
                    sp_index.drain_rotations()
                    size = sp_index.size
                    sp_index.close()
                    if dirty:
                        self._update_sp_metadata(sp_type, size)
                        logger.info(f"Saved UsearchSimprintIndex '{sp_type}' ({size} vectors)")
                except Exception:  # pragma: no cover
                    logger.exception(f"Failed to close UsearchSimprintIndex '{sp_type}'")

            self._simprint_indexes.clear()

            # Close LMDB
            try:
                self.env.close()
            except Exception:  # pragma: no cover
                logger.exception("Failed to close LMDB environment")

            self._closed = True
            logger.info(f"Closed UsearchIndex at {self.path}")

    def __len__(self):  # pragma: no cover
        # type: () -> int
        """Return number of assets in index."""
        try:
            with self._read_txn() as txn:
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

    @contextmanager
    def _read_txn(self):
        # type: () -> Iterator[lmdb.Transaction]
        """
        Open an LMDB read transaction, blocking during resize.

        Acquires shared access on ``_env_rwlock`` so that ``set_mapsize``
        (which requires exclusive access) waits for all active readers.

        :yield: Active LMDB read transaction
        """
        self._env_rwlock.acquire_read()
        try:
            with self.env.begin() as txn:
                yield txn
        finally:
            self._env_rwlock.release_read()

    def _safe_resize(self, new_size):
        # type: (int) -> None
        """
        Resize LMDB map with exclusive access (no active transactions).

        Acquires exclusive ``_env_rwlock``, draining all active read
        transactions before calling ``set_mapsize``.

        :param new_size: New map size in bytes
        """
        self._env_rwlock.acquire_write()
        try:
            self.env.set_mapsize(new_size)
        finally:
            self._env_rwlock.release_write()

    @property
    def tracked_unit_types(self):
        # type: () -> list[str]
        """Sorted list of NPHD unit_types tracked in LMDB metadata."""
        return sorted(self._get_all_tracked_unit_types())

    @property
    def tracked_simprint_types(self):
        # type: () -> list[str]
        """Sorted list of simprint types tracked in LMDB metadata."""
        return sorted(self._get_sp_types())

    def rebuild(self, unit_types, simprint_types):
        # type: (list[str], list[str]) -> dict[str, list[str]]
        """
        Rebuild specified derived indexes from LMDB source data.

        Each NPHD unit_type and simprint_type is rebuilt fresh: the existing shard
        directory is removed and a new index is built from LMDB entries. Use the
        ``tracked_unit_types`` and ``tracked_simprint_types`` properties to discover
        what is currently tracked.

        :param unit_types: NPHD unit_types to rebuild (e.g., ["CONTENT_TEXT_V0"])
        :param simprint_types: Simprint types to rebuild (e.g., ["CONTENT_TEXT_V0"])
        :return: Dict with lists of types that were actually rebuilt
        """
        rebuilt_unit_types = []  # type: list[str]
        for unit_type in unit_types:
            if self._rebuild_nphd_index(unit_type):
                rebuilt_unit_types.append(unit_type)

        rebuilt_simprint_types = []  # type: list[str]
        for sp_type in simprint_types:
            if self._rebuild_simprint_index(sp_type):
                rebuilt_simprint_types.append(sp_type)

        return {"unit_types": rebuilt_unit_types, "simprint_types": rebuilt_simprint_types}

    def _search_simprints(self, query, limit, exact=False):
        # type: (IsccQuery, int, bool) -> list[IsccChunkMatch]
        """
        Search simprints and convert results to IsccChunkMatch format.

        Routes to exact LMDB search or approximate ShardedIndex128 search.

        :param query: Query with simprints field
        :param limit: Maximum number of chunk matches to return
        :param exact: If True, use LMDB exact search; if False, use ShardedIndex128
        :return: List of IsccChunkMatch objects with metadata enrichment
        """
        if exact:
            return self._search_simprints_exact(query, limit)

        return self._search_simprints_approximate(query, limit)

    def _convert_simprint_match(self, raw_match, assets_db, txn):
        # type: (SimprintMatchMulti, lmdb._Database | None, lmdb.Transaction | None) -> IsccChunkMatch
        """
        Convert SimprintMatchMulti to IsccChunkMatch with metadata enrichment.

        Converts protocol-layer result (bytes) to schema-layer result (strings).
        Fetches metadata from LMDB assets database if available.

        :param raw_match: Protocol-layer match result
        :param assets_db: LMDB database handle (or None if no enrichment)
        :param txn: LMDB transaction (or None if no enrichment)
        :return: Schema-layer chunk match with metadata
        """
        from iscc_search.schema import IsccChunkMatch, IsccMatchedChunk, Types

        # Convert ISCC-ID from bytes to string
        iscc_id_str = "ISCC:" + ic.encode_base32(raw_match.iscc_id)

        # Fetch metadata from LMDB (if available)
        source = None
        metadata = None
        if assets_db is not None and txn is not None:
            # Extract 8-byte body from full 10-byte ISCC-ID (strip 2-byte header)
            iscc_id_body = raw_match.iscc_id[2:]
            key = int.from_bytes(iscc_id_body, "big", signed=False)
            key_bytes = struct.pack(">Q", key)

            asset_bytes = txn.get(key_bytes, db=assets_db)
            if asset_bytes is not None:
                asset = common.deserialize_asset(asset_bytes)
                if asset.metadata:
                    source = asset.metadata.get("source")
                    metadata = asset.metadata

        # Convert type results and chunks
        types_converted = {}  # type: dict[str, Types]
        for simprint_type, type_result in raw_match.types.items():
            # Convert chunks if detailed=True
            chunks_converted = None
            if type_result.chunks is not None:
                chunks_converted = [
                    IsccMatchedChunk(
                        query=ic.encode_base64(chunk.query),
                        match=ic.encode_base64(chunk.match),
                        score=chunk.score,
                        freq=chunk.freq,
                        offset=chunk.offset,
                        size=chunk.size,
                        content=None,  # Not populated in this integration
                    )
                    for chunk in type_result.chunks
                ]

            types_converted[simprint_type] = Types(
                score=type_result.score,
                matches=type_result.matches,
                queried=type_result.queried,
                chunks=chunks_converted,
            )

        return IsccChunkMatch(
            iscc_id=iscc_id_str, score=raw_match.score, types=types_converted, source=source, metadata=metadata
        )

    def _load_sp_databases(self):
        # type: () -> None
        """
        Load existing LMDB simprint database handles from metadata.

        Reads sp_types from __metadata__ and opens cached database handles
        for each known simprint type.
        """
        sp_types = self._get_sp_types()
        if not sp_types:
            return

        with self.env.begin(write=True) as txn:
            for sp_type in sp_types:
                data_db_name = f"__sp_{sp_type}__".encode()
                assets_db_name = f"__sp_assets_{sp_type}__".encode()

                data_db = self.env.open_db(
                    data_db_name,
                    txn=txn,
                    dupsort=True,
                    dupfixed=True,
                )
                assets_db = self.env.open_db(assets_db_name, txn=txn)

                self._sp_data_dbs[sp_type] = data_db
                self._sp_assets_dbs[sp_type] = assets_db

        logger.debug(f"Loaded LMDB simprint databases for types: {sp_types}")

    def _open_sp_databases_in_txn(self, txn, sp_type):
        # type: (lmdb.Transaction, str) -> tuple[lmdb._Database, lmdb._Database]
        """
        Open or get cached LMDB simprint database handles within a write transaction.

        If this is a new type, opens the databases and registers the type in metadata.

        :param txn: Active LMDB write transaction
        :param sp_type: Simprint type identifier (e.g., "CONTENT_TEXT_V0")
        :return: (data_db, sp_assets_db) database handles
        """
        if sp_type in self._sp_data_dbs:
            return self._sp_data_dbs[sp_type], self._sp_assets_dbs[sp_type]

        data_db_name = f"__sp_{sp_type}__".encode()
        assets_db_name = f"__sp_assets_{sp_type}__".encode()

        data_db = self.env.open_db(
            data_db_name,
            txn=txn,
            dupsort=True,
            dupfixed=True,
        )
        assets_db = self.env.open_db(assets_db_name, txn=txn)

        self._sp_data_dbs[sp_type] = data_db
        self._sp_assets_dbs[sp_type] = assets_db

        # Register new type in metadata
        metadata_db = self.env.open_db(b"__metadata__", txn=txn)
        sp_types = self._get_sp_types_from_txn(txn, metadata_db)
        if sp_type not in sp_types:
            sp_types.append(sp_type)
            txn.put(b"sp_types", json.dumps(sp_types).encode(), db=metadata_db)
            logger.debug(f"Registered new simprint type in LMDB: {sp_type}")

        return data_db, assets_db

    def _get_sp_types(self):
        # type: () -> list[str]
        """
        Read sp_types JSON list from __metadata__.

        :return: List of registered simprint type identifiers
        """
        try:
            with self._read_txn() as txn:
                metadata_db = self.env.open_db(b"__metadata__", txn=txn)
                return self._get_sp_types_from_txn(txn, metadata_db)
        except lmdb.ReadonlyError:  # pragma: no cover
            return []

    def _get_sp_types_from_txn(self, txn, metadata_db):
        # type: (lmdb.Transaction, lmdb._Database) -> list[str]
        """
        Read sp_types JSON list from __metadata__ within an existing transaction.

        :param txn: Active LMDB transaction
        :param metadata_db: Metadata database handle
        :return: List of registered simprint type identifiers
        """
        raw = txn.get(b"sp_types", db=metadata_db)
        if raw is None:
            return []
        return json.loads(raw.decode())

    def _search_simprints_exact(self, query, limit):
        # type: (IsccQuery, int) -> list[IsccChunkMatch]
        """
        Exact LMDB simprint search with per-type lookup and cross-type aggregation.

        For each simprint type in the query, performs hard-boundary search via
        lmdb_ops.search_simprints_exact. Groups results by asset, computes
        overall score as mean of per-type scores, and enriches with metadata.

        :param query: Query with simprints field
        :param limit: Maximum number of chunk matches to return
        :return: List of IsccChunkMatch objects with metadata enrichment
        """
        from iscc_search.indexes.simprint.models import SimprintMatchMulti, TypeMatchResult

        # Per-type search and aggregation
        # asset_type_results: iscc_id_body -> {sp_type: TypeMatchResult}
        asset_type_results = {}  # type: dict[bytes, dict[str, TypeMatchResult]]

        with self._read_txn() as txn:
            for sp_type, simprint_objs in query.simprints.items():
                if sp_type not in self._sp_data_dbs:
                    continue

                data_db = self._sp_data_dbs[sp_type]

                # Count total assets for this type
                if sp_type in self._sp_assets_dbs:
                    total_assets = txn.stat(db=self._sp_assets_dbs[sp_type])["entries"]
                else:
                    total_assets = 0

                # Convert query simprints from Simprint RootModel objects to bytes
                query_sp_bytes = [ic.decode_base64(s.root if hasattr(s, "root") else s) for s in simprint_objs]

                raw_matches = lmdb_ops.search_simprints_exact(
                    txn=txn,
                    db=data_db,
                    query_simprints=query_sp_bytes,
                    total_assets=total_assets,
                    limit=limit * 2,  # Over-fetch per type, trim after aggregation
                    threshold=self._opts.match_threshold_simprints,
                    detailed=True,
                )

                for raw_match in raw_matches:
                    body = raw_match.iscc_id_body
                    if body not in asset_type_results:
                        asset_type_results[body] = {}

                    # Convert SimprintMatchRaw chunks to MatchedChunkRaw (already in correct format)
                    asset_type_results[body][sp_type] = TypeMatchResult(
                        score=raw_match.score,
                        queried=raw_match.queried,
                        matches=raw_match.matches,
                        chunks=raw_match.chunks,
                    )

        if not asset_type_results:
            return []

        # Build SimprintMatchMulti objects (overall score = mean of type scores)
        multi_matches = []  # type: list[SimprintMatchMulti]
        for iscc_id_body, type_results in asset_type_results.items():
            asset_score = sum(tr.score for tr in type_results.values()) / len(type_results)

            # Reconstruct full ISCC-ID (2-byte header + 8-byte body)
            iscc_id_obj = IsccID.from_body(iscc_id_body, self._realm_id)
            iscc_id_bytes = ic.decode_base32(str(iscc_id_obj).split(":")[-1])

            multi_matches.append(
                SimprintMatchMulti(
                    iscc_id=iscc_id_bytes,
                    score=asset_score,
                    types=type_results,
                )
            )

        multi_matches.sort(key=lambda x: (-x.score, x.iscc_id))
        multi_matches = multi_matches[:limit]

        # Convert to IsccChunkMatch with metadata enrichment (reuse existing converter)
        chunk_matches = []  # type: list[IsccChunkMatch]
        try:
            with self._read_txn() as txn:
                assets_db = self.env.open_db(b"__assets__", txn=txn)
                for raw_match in multi_matches:
                    chunk_match = self._convert_simprint_match(raw_match, assets_db, txn)
                    chunk_matches.append(chunk_match)
        except lmdb.ReadonlyError:  # pragma: no cover
            for raw_match in multi_matches:
                chunk_match = self._convert_simprint_match(raw_match, None, None)
                chunk_matches.append(chunk_match)

        return chunk_matches

    def _search_simprints_approximate(self, query, limit):
        # type: (IsccQuery, int) -> list[IsccChunkMatch]
        """
        Approximate simprint search via derived ShardedIndex128 indexes.

        Uses IDF-weighted scoring with 20x oversampling for asset diversity.
        Doc frequencies are looked up via LMDB for each matched simprint.

        :param query: Query with simprints field
        :param limit: Maximum number of chunk matches to return
        :return: List of IsccChunkMatch objects with metadata enrichment
        """
        from iscc_search.indexes.simprint.models import SimprintMatchMulti, TypeMatchResult

        # Count total assets for IDF calculation
        total_assets = self._get_total_sp_assets()

        # Per-type search and aggregation
        # asset_type_results: iscc_id_body -> {sp_type: TypeMatchResult}
        asset_type_results = {}  # type: dict[bytes, dict[str, TypeMatchResult]]

        for sp_type, simprint_objs in query.simprints.items():
            if sp_type not in self._simprint_indexes:
                # Derived index missing for this type. Rebuilding from LMDB can take hours
                # at production scale, so we must NOT do it inside a search request. Log a
                # loud warning and skip the type; an explicit out-of-band rebuild is required.
                if sp_type in self._sp_data_dbs:
                    logger.warning(
                        f"UsearchSimprintIndex missing for type '{sp_type}' but LMDB has data - "
                        "skipping this type in search. Run an explicit rebuild to restore results."
                    )
                continue

            sp_index = self._simprint_indexes[sp_type]

            # Convert query simprints from Simprint RootModel objects to bytes
            query_sp_bytes = [ic.decode_base64(s.root if hasattr(s, "root") else s) for s in simprint_objs]

            # Create doc_freq_fn that looks up LMDB for true document frequency
            if sp_type in self._sp_data_dbs:
                data_db = self._sp_data_dbs[sp_type]

                def doc_freq_fn(sp_key, _db=data_db):
                    # type: (bytes, object) -> int
                    with self._read_txn() as txn:
                        return lmdb_ops.count_doc_freq(txn, _db, sp_key)

            else:
                doc_freq_fn = None

            raw_matches = sp_index.search_raw(
                simprints=query_sp_bytes,
                limit=limit * 2,  # Over-fetch per type, trim after aggregation
                threshold=self._opts.match_threshold_simprints,
                detailed=True,
                doc_freq_fn=doc_freq_fn,
                total_assets=total_assets,
            )

            for raw_match in raw_matches:
                body = raw_match.iscc_id_body
                if body not in asset_type_results:
                    asset_type_results[body] = {}

                asset_type_results[body][sp_type] = TypeMatchResult(
                    score=raw_match.score,
                    queried=raw_match.queried,
                    matches=raw_match.matches,
                    chunks=raw_match.chunks,
                )

        if not asset_type_results:
            return []

        # Build SimprintMatchMulti objects (overall score = mean of type scores)
        multi_matches = []  # type: list[SimprintMatchMulti]
        for iscc_id_body, type_results in asset_type_results.items():
            asset_score = sum(tr.score for tr in type_results.values()) / len(type_results)

            # Reconstruct full ISCC-ID (2-byte header + 8-byte body)
            iscc_id_obj = IsccID.from_body(iscc_id_body, self._realm_id)
            iscc_id_bytes = ic.decode_base32(str(iscc_id_obj).split(":")[-1])

            multi_matches.append(
                SimprintMatchMulti(
                    iscc_id=iscc_id_bytes,
                    score=asset_score,
                    types=type_results,
                )
            )

        multi_matches.sort(key=lambda x: (-x.score, x.iscc_id))
        multi_matches = multi_matches[:limit]

        # Convert to IsccChunkMatch with metadata enrichment
        chunk_matches = []  # type: list[IsccChunkMatch]
        try:
            with self._read_txn() as txn:
                assets_db = self.env.open_db(b"__assets__", txn=txn)
                for raw_match in multi_matches:
                    chunk_match = self._convert_simprint_match(raw_match, assets_db, txn)
                    chunk_matches.append(chunk_match)
        except lmdb.ReadonlyError:  # pragma: no cover
            for raw_match in multi_matches:
                chunk_match = self._convert_simprint_match(raw_match, None, None)
                chunk_matches.append(chunk_match)
        except Exception as e:
            logger.error(f"Failed to enrich simprint matches with metadata: {e}")
            for raw_match in multi_matches:
                chunk_match = self._convert_simprint_match(raw_match, None, None)
                chunk_matches.append(chunk_match)

        return chunk_matches

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
        Update ShardedNphdIndex metadata in LMDB.

        Tracks expected vector count for sync detection on next load.

        :param unit_type: Unit type identifier
        :param vector_count: Current number of vectors in ShardedNphdIndex
        """
        with self._write_lock, self.env.begin(write=True) as txn:
            metadata_db = self.env.open_db(b"__metadata__", txn=txn)
            key = f"nphd_count:{unit_type}".encode()
            txn.put(key, struct.pack(">Q", vector_count), db=metadata_db)

    def _get_nphd_metadata(self, unit_type):
        # type: (str) -> int | None
        """
        Get expected ShardedNphdIndex vector count from LMDB metadata.

        :param unit_type: Unit type identifier
        :return: Expected vector count, or None if not tracked
        """
        try:
            with self._read_txn() as txn:
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
            with self._read_txn() as txn:
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
        Load existing ShardedNphdIndex directories, accepting stale state on mismatch.

        Loads indexes for all unit_types tracked in LMDB metadata. ShardedNphdIndex
        auto-loads existing shards from its directory at construction time.
        Logs warning if vector count is out of sync but does NOT auto-rebuild
        (auto-rebuild can cause OOM on large indexes). Use CLI rebuild command instead.
        """
        tracked_unit_types = self._get_all_tracked_unit_types()

        for unit_type in tracked_unit_types:
            shard_dir = self.path / unit_type
            try:
                nphd_index = ShardedNphdIndex(
                    max_dim=self.max_dim,
                    path=shard_dir,
                    connectivity=self._opts.hnsw_connectivity_units,
                    expansion_add=self._opts.hnsw_expansion_add_units,
                    expansion_search=self._opts.hnsw_expansion_search_units,
                    shard_size=self._opts.shard_size_units * 1024 * 1024,
                    background_rotation=True,
                )

                # Check if index is in sync with LMDB
                expected_count = self._get_nphd_metadata(unit_type)
                actual_count = nphd_index.size

                shards = nphd_index.shard_count

                if expected_count is not None and expected_count != actual_count:
                    logger.warning(
                        f"ShardedNphdIndex '{unit_type}' out of sync: "
                        f"expected {expected_count} vectors, found {actual_count}. "
                        f"Skipping auto-rebuild. Run 'iscc-search index rebuild "
                        f"--unit-type {unit_type}' (or '--all') to repair."
                    )
                    # Accept stale index rather than risk OOM during rebuild
                    self._nphd_indexes[unit_type] = nphd_index
                else:
                    self._nphd_indexes[unit_type] = nphd_index

                logger.info(f"Loaded NPHD index '{unit_type}': {actual_count} vectors, {shards} shards")

            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to load ShardedNphdIndex '{unit_type}': {e}. Skipping.")

    def _rebuild_nphd_index(self, unit_type):
        # type: (str) -> bool
        """
        Rebuild ShardedNphdIndex from LMDB asset data.

        Full rebuild: iterates all assets, extracts vectors for unit_type,
        and creates fresh ShardedNphdIndex. Automatically saves and updates metadata.

        :param unit_type: Unit type identifier to rebuild
        :return: True if an index was rebuilt, False if no source vectors exist
        """
        import time as time_module

        start_time = time_module.time()
        logger.info(f"Rebuilding ShardedNphdIndex for unit_type '{unit_type}'...")

        # Collect all vectors for this unit_type from LMDB
        keys = []  # type: list[int]
        vectors = []  # type: list[bytes]

        with self._read_txn() as txn:
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
            return False

        # Remove stale/corrupt shard directory before creating fresh index
        shard_dir = self.path / unit_type
        old_index = self._nphd_indexes.pop(unit_type, None)
        if old_index is not None:
            old_index.reset()
        if shard_dir.exists():
            shutil.rmtree(shard_dir)

        # Create fresh ShardedNphdIndex and add all vectors
        nphd_index = ShardedNphdIndex(
            max_dim=self.max_dim,
            path=shard_dir,
            connectivity=self._opts.hnsw_connectivity_units,
            expansion_add=self._opts.hnsw_expansion_add_units,
            expansion_search=self._opts.hnsw_expansion_search_units,
            shard_size=self._opts.shard_size_units * 1024 * 1024,
            background_rotation=True,
        )
        nphd_index.add(keys, vectors)

        # Save to disk
        nphd_index.save()

        # Update metadata
        self._update_nphd_metadata(unit_type, nphd_index.size)

        # Store in memory
        self._nphd_indexes[unit_type] = nphd_index

        elapsed = time_module.time() - start_time
        logger.info(f"Rebuilt ShardedNphdIndex for unit_type '{unit_type}': {len(keys)} vectors in {elapsed:.2f}s")
        return True

    def _get_or_create_nphd_index(self, unit_type):
        # type: (str) -> ShardedNphdIndex
        """Get or create ShardedNphdIndex for unit_type."""
        if unit_type not in self._nphd_indexes:  # pragma: no branch
            nphd_index = ShardedNphdIndex(
                max_dim=self.max_dim,
                path=self.path / unit_type,
                connectivity=self._opts.hnsw_connectivity_units,
                expansion_add=self._opts.hnsw_expansion_add_units,
                expansion_search=self._opts.hnsw_expansion_search_units,
                shard_size=self._opts.shard_size_units * 1024 * 1024,
                background_rotation=True,
            )
            self._nphd_indexes[unit_type] = nphd_index
            logger.debug(f"Created new ShardedNphdIndex for unit_type '{unit_type}'")

        return self._nphd_indexes[unit_type]

    def _get_or_create_simprint_index(self, sp_type, ndim):
        # type: (str, int) -> UsearchSimprintIndex
        """Get or create derived UsearchSimprintIndex for a simprint type."""
        if sp_type not in self._simprint_indexes:
            sp_dir = self.path / f"SIMPRINT_{sp_type}"
            sp_index = UsearchSimprintIndex(
                path=sp_dir,
                ndim=ndim,
                connectivity=self._opts.hnsw_connectivity_simprints,
                expansion_add=self._opts.hnsw_expansion_add_simprints,
                expansion_search=self._opts.hnsw_expansion_search_simprints,
                shard_size=self._opts.shard_size_simprints * 1024 * 1024,
                oversampling_factor=self._opts.oversampling_factor,
                background_rotation=True,
            )
            self._simprint_indexes[sp_type] = sp_index
            logger.debug(f"Created new UsearchSimprintIndex for type '{sp_type}' (ndim={ndim})")
        return self._simprint_indexes[sp_type]

    def _load_simprint_indexes(self):
        # type: () -> None
        """
        Load existing derived simprint indexes (ShardedIndex128), accepting stale state.

        For each known simprint type from LMDB metadata, opens the ShardedIndex128
        directory and checks vector count against LMDB metadata. Logs warning on
        mismatch but does NOT auto-rebuild (auto-rebuild can cause OOM on large indexes).
        """
        sp_types = self._get_sp_types()
        if not sp_types:
            return

        for sp_type in sp_types:
            sp_dir = self.path / f"SIMPRINT_{sp_type}"
            if not sp_dir.exists():
                # Directory missing - searches of this type will return empty results until an
                # explicit rebuild is run. Auto-rebuild is intentionally disabled to avoid
                # hours-long work blocking a search request at production scale.
                logger.warning(
                    f"Simprint index directory missing for type '{sp_type}' - "
                    "searches of this type will be empty until an explicit rebuild is run."
                )
                continue

            try:
                # Detect ndim from LMDB data
                ndim = self._detect_sp_ndim(sp_type)
                if ndim is None:
                    continue

                sp_index = UsearchSimprintIndex(
                    path=sp_dir,
                    ndim=ndim,
                    connectivity=self._opts.hnsw_connectivity_simprints,
                    expansion_add=self._opts.hnsw_expansion_add_simprints,
                    expansion_search=self._opts.hnsw_expansion_search_simprints,
                    shard_size=self._opts.shard_size_simprints * 1024 * 1024,
                    oversampling_factor=self._opts.oversampling_factor,
                    background_rotation=True,
                )

                # Check sync with LMDB
                expected_count = self._get_sp_metadata(sp_type)
                actual_count = sp_index.size
                shards = sp_index.shard_count

                if expected_count is not None and expected_count != actual_count:
                    logger.warning(
                        f"UsearchSimprintIndex '{sp_type}' out of sync: "
                        f"expected {expected_count}, found {actual_count}. "
                        f"Skipping auto-rebuild. Run 'iscc-search index rebuild "
                        f"--simprint-type {sp_type}' (or '--all') to repair."
                    )
                    # Accept stale index rather than risk OOM during rebuild
                    self._simprint_indexes[sp_type] = sp_index
                else:
                    self._simprint_indexes[sp_type] = sp_index

                logger.info(f"Loaded simprint index '{sp_type}': {actual_count} vectors, {shards} shards")

            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to load UsearchSimprintIndex '{sp_type}': {e}. Skipping.")

    def _rebuild_simprint_index(self, sp_type):
        # type: (str) -> bool
        """
        Rebuild ShardedIndex128 for a simprint type from LMDB source of truth.

        :param sp_type: Simprint type identifier to rebuild
        :return: True if an index was rebuilt, False if no source vectors exist
        """
        start_time = time.time()
        logger.info(f"Rebuilding UsearchSimprintIndex for type '{sp_type}'...")

        if sp_type not in self._sp_data_dbs:
            logger.warning(f"No LMDB database for simprint type '{sp_type}' - skipping rebuild")
            return False

        data_db = self._sp_data_dbs[sp_type]

        # Remove stale in-memory reference and directory
        sp_dir = self.path / f"SIMPRINT_{sp_type}"
        old_index = self._simprint_indexes.pop(sp_type, None)
        if old_index is not None:
            old_index.reset()
        if sp_dir.exists():
            shutil.rmtree(sp_dir)

        # Detect ndim before iterating (avoids conditional inside batch loop)
        ndim = self._detect_sp_ndim(sp_type)
        if ndim is None:
            logger.info(f"No vectors found for simprint type '{sp_type}' - skipping rebuild")
            return False

        # Iterate LMDB in batches to avoid loading all vectors into RAM
        sp_index = UsearchSimprintIndex(
            path=sp_dir,
            ndim=ndim,
            connectivity=self._opts.hnsw_connectivity_simprints,
            expansion_add=self._opts.hnsw_expansion_add_simprints,
            expansion_search=self._opts.hnsw_expansion_search_simprints,
            shard_size=self._opts.shard_size_simprints * 1024 * 1024,
            oversampling_factor=self._opts.oversampling_factor,
            background_rotation=True,
        )
        total_vectors = 0
        with self._read_txn() as txn:
            for keys, vectors in lmdb_ops.iter_simprint_vectors(txn, data_db):
                sp_index.add_raw(keys, vectors)
                total_vectors += len(keys)

        sp_index.save()

        # Update metadata and store
        self._update_sp_metadata(sp_type, sp_index.size)
        self._simprint_indexes[sp_type] = sp_index

        elapsed = time.time() - start_time
        logger.info(f"Rebuilt UsearchSimprintIndex for type '{sp_type}': {total_vectors} vectors in {elapsed:.2f}s")
        return True

    def _update_sp_metadata(self, sp_type, vector_count):
        # type: (str, int) -> None
        """
        Update simprint index vector count in LMDB metadata.

        :param sp_type: Simprint type identifier
        :param vector_count: Current number of vectors in the derived index
        """
        with self._write_lock, self.env.begin(write=True) as txn:
            metadata_db = self.env.open_db(b"__metadata__", txn=txn)
            key = f"sp_count:{sp_type}".encode()
            txn.put(key, struct.pack(">Q", vector_count), db=metadata_db)

    def _get_sp_metadata(self, sp_type):
        # type: (str) -> int | None
        """
        Get expected simprint vector count from LMDB metadata.

        :param sp_type: Simprint type identifier
        :return: Expected vector count, or None if not tracked
        """
        try:
            with self._read_txn() as txn:
                metadata_db = self.env.open_db(b"__metadata__", txn=txn)
                key = f"sp_count:{sp_type}".encode()
                value = txn.get(key, db=metadata_db)
                if value is None:
                    return None
                return struct.unpack(">Q", value)[0]
        except lmdb.ReadonlyError:  # pragma: no cover
            return None

    def _get_total_sp_assets(self):
        # type: () -> int
        """
        Get total number of assets across all simprint types.

        Uses __assets__ entry count as the global total for IDF calculation.

        :return: Total asset count
        """
        try:
            with self._read_txn() as txn:
                assets_db = self.env.open_db(b"__assets__", txn=txn)
                return txn.stat(db=assets_db)["entries"]
        except lmdb.ReadonlyError:  # pragma: no cover
            return 0

    def _detect_sp_ndim(self, sp_type):
        # type: (str) -> int | None
        """
        Detect simprint dimensionality (in bits) from LMDB data.

        Reads first key from the simprint database and returns its length in bits.

        :param sp_type: Simprint type identifier
        :return: ndim in bits, or None if database is empty
        """
        if sp_type not in self._sp_data_dbs:
            return None
        data_db = self._sp_data_dbs[sp_type]
        try:
            with self._read_txn() as txn:
                cursor = txn.cursor(data_db)
                if cursor.first():
                    return len(cursor.key()) * 8
        except lmdb.ReadonlyError:  # pragma: no cover
            pass
        return None

    def _search_instance_unit(self, instance_code):
        # type: (bytes) -> dict[int, float]
        """
        Search INSTANCE unit in LMDB dupsort for exact matches.

        INSTANCE codes are identity codes (checksums/hashes), not similarity codes.
        Any prefix match indicates the same underlying data (with varying confidence
        based on bit length). Therefore, all matches score 1.0.

        **Binary scoring**: Any INSTANCE match (64/128/256-bit) scores 1.0, indicating
        the matched asset contains the same data. No match = not in results (implicitly 0.0).

        :param instance_code: Instance-Code digest bytes
        :return: Dict mapping integer keys to score (1.0 for all matches)
        """
        results = {}  # type: dict[int, float]

        with self._read_txn() as txn:
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
                    # Binary scoring: any match = 1.0 (identity match)
                    score = 1.0
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
                        score = 1.0  # Binary: match = 1.0
                        results[key] = max(results.get(key, 0.0), score)

            if query_len >= 16:  # 128-bit or 256-bit query
                # Check 64-bit prefix
                prefix_64 = instance_code[:8]
                if cursor.set_key(prefix_64):
                    for value_bytes in cursor.iternext_dup(keys=False, values=True):
                        key = struct.unpack(">Q", value_bytes)[0]
                        score = 1.0  # Binary: match = 1.0
                        results[key] = max(results.get(key, 0.0), score)

        return results

    def _search_similarity_unit(self, unit_type, vector, limit):
        # type: (str, bytes, int) -> dict[int, float]
        """
        Search similarity unit in ShardedNphdIndex.

        :param unit_type: Unit type identifier
        :param vector: Unit body bytes
        :param limit: Maximum number of results
        :return: Dict mapping integer keys to scores (1.0 - nphd_distance)
        """
        nphd_index = self._nphd_indexes[unit_type]

        query = np.frombuffer(vector, dtype=np.uint8)
        matches = nphd_index.search(query, count=limit)

        # Convert distances to scores: score = 1.0 - distance
        results = {}  # type: dict[int, float]
        for key, distance in zip(matches.keys, matches.distances):
            score = 1.0 - float(distance)
            results[int(key)] = max(0.0, score)  # Clamp to non-negative

        return results
