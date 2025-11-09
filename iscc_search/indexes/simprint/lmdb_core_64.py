"""
High-Performance LMDB-Based 64-bit Simprint Index

A hard-boundary simprint index optimized for 64-bit simprints with IDF-weighted scoring.
Uses LMDB's dupsort and integerkey features for maximum performance.

Key Features:
- 64-bit simprints stored as integer keys (integerkey optimization)
- 16-byte ChunkPointer values (ISCC-ID body + offset + size) with dupfixed
- Dedicated assets database for O(1) duplicate detection and asset counting
- IDF-weighted scoring based on document frequency
- Batch operations using putmulti/getmulti
- Automatic map_size expansion on MapFullError
- Thread-safe read operations
"""

import struct
from pathlib import Path
from typing import TYPE_CHECKING
from collections import defaultdict
import math

import lmdb
from loguru import logger

from iscc_search.indexes.simprint.models import MatchedChunkRaw, SimprintMatchRaw


if TYPE_CHECKING:
    from iscc_search.protocols.simprint_core import SimprintEntryRaw  # noqa: F401


class LmdbSimprintIndex64:
    """
    High-performance LMDB-based 64-bit simprint index with IDF-weighted scoring.

    This implementation uses hard boundaries (exact hash collisions) for clustering
    similar content, optimized for 64-bit simprints with chunk location tracking.

    Architecture:
    - Simprints DB: simprint (64-bit int) -> ChunkPointer (16 bytes) with dupsort+dupfixed
    - Assets DB: iscc_id_body (64-bit int) -> empty value, for O(1) duplicate detection
    - Global metadata (simprint_type, realm_id) managed externally by coordinators

    Scoring:
    - IDF (Inverse Document Frequency) weighting reduces impact of common simprints
    - Score = (matches/queried) * average(IDF of matched simprints)
    - IDF = log(total_assets / (1 + assets_with_simprint))
    """

    # Constants
    SIMPRINT_BYTES = 8  # 64-bit simprints
    CHUNK_POINTER_BYTES = 16  # 8 bytes ISCC-ID + 4 bytes offset + 4 bytes size
    MAX_OFFSET = 2**32 - 1  # 4 GB max offset
    MAX_SIZE = 2**32 - 1  # 4 GB max size
    MAX_RESIZE_RETRIES = 10
    MAX_MAP_SIZE = 1024 * 1024 * 1024 * 1024  # 1 TB
    _DB_SIMPRINTS = b"simprints"  # LMDB database name for simprint mappings
    _DB_ASSETS = b"assets"  # LMDB database name for asset tracking

    def __init__(self, uri, **kwargs):
        # type: (str, ...) -> None
        """
        Open or create a simprint index at the specified location.

        :param uri: Index location (path or file:// URI)
        :param kwargs: Backend-specific configuration options (currently unused)
        """
        # Parse URI
        if uri.startswith("file://"):
            self.path = Path(uri[7:])
        elif uri.startswith("lmdb://"):
            self.path = Path(uri[7:])
        else:
            self.path = Path(uri)

        self.path.mkdir(parents=True, exist_ok=True)

        # Open LMDB environment with 2 named databases
        # Use writemap=True and map_async=True for better Windows compatibility
        self.env = lmdb.open(
            str(self.path),
            max_dbs=2,  # simprints, assets
            writemap=True,  # Avoid Windows file reservation issues
            map_async=True,  # Better write performance
            sync=False,  # Bulk operations, sync manually
            metasync=False,
        )

        # Open simprints database with optimal flags for 64-bit keys and fixed values
        self.simprints_db = self.env.open_db(
            self._DB_SIMPRINTS,
            dupsort=True,  # Allow duplicate keys for multiple chunks per simprint
            dupfixed=True,  # Fixed 16-byte values (ChunkPointer)
        )

        # Open assets database for O(1) duplicate detection and counting
        # Key: iscc_id_body (8 bytes as 64-bit integer), Value: empty (existence matters)
        self.assets_db = self.env.open_db(self._DB_ASSETS)

    def add_raw(self, entries):
        # type: (list[SimprintEntryRaw]) -> None
        """
        Add entries to the index with add-once semantics.

        Uses batch operations for optimal performance.
        Silently ignores duplicate ISCC-ID bodies.

        :param entries: List of entries to add atomically
        """
        if not entries:
            return

        existing = self._check_existing_assets(entries)
        new_asset_ids, simprint_pairs = self._build_insert_pairs(entries, existing)

        if not new_asset_ids:
            return  # Nothing to add

        self._execute_batch_insert(new_asset_ids, simprint_pairs)

    def search_raw(self, simprints, limit=10, threshold=0.8, detailed=True):
        # type: (list[bytes], int, float, bool) -> list[SimprintMatchRaw]
        """
        Search for assets with similar simprints using IDF-weighted scoring.

        :param simprints: Binary simprints to search for
        :param limit: Maximum number of assets to return
        :param threshold: Minimum similarity score (0.0 to 1.0)
        :param detailed: If True, include individual chunk matches
        :return: List of matched assets ordered by similarity
        """
        if not simprints:
            return []

        query_simprints = [s[:8] for s in simprints]
        asset_matches, doc_frequencies = self._fetch_matches_and_frequencies(query_simprints)

        total_assets = max(len(self), 1)
        results = []

        for iscc_id_body, matches in asset_matches.items():
            score = self._calculate_idf_score(matches, doc_frequencies, total_assets, len(query_simprints))

            if score >= threshold:
                result = self._format_match_result(
                    iscc_id_body, matches, score, doc_frequencies, len(query_simprints), detailed
                )
                results.append(result)

        results.sort(key=lambda x: (-x.score, x.iscc_id_body))
        return results[:limit]

    def close(self):
        # type: () -> None
        """Close the index and release resources."""
        self.env.sync(True)
        self.env.close()

    def __contains__(self, iscc_id_body):
        # type: (bytes) -> bool
        """
        Check if an ISCC-ID body exists in the index.

        O(1) lookup using assets database.

        :param iscc_id_body: Binary ISCC-ID body (8 bytes)
        :return: True if the asset has been indexed
        """
        with self.env.begin() as txn:
            return txn.get(iscc_id_body, db=self.assets_db) is not None

    def __len__(self):
        # type: () -> int
        """
        Return the number of unique assets in the index.

        O(1) count using LMDB statistics.
        """
        with self.env.begin() as txn:
            stats = txn.stat(db=self.assets_db)
            return stats["entries"]

    def __enter__(self):
        # type: () -> LmdbSimprintIndex64
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # type: (type | None, Exception | None, object | None) -> None
        """Context manager cleanup."""
        self.close()

    def _pack_chunk_pointer(self, iscc_id_body, offset, size):
        # type: (bytes, int, int) -> bytes
        """
        Pack chunk pointer into 16-byte binary format.

        Layout: iscc_id_body(8) + offset(4) + size(4)
        Uses network byte order (big-endian) for cross-platform portability.
        """
        if len(iscc_id_body) != 8:
            raise ValueError(f"ISCC-ID body must be 8 bytes, got {len(iscc_id_body)}")
        if offset > self.MAX_OFFSET:
            raise ValueError(f"Offset {offset} exceeds max {self.MAX_OFFSET}")
        if size > self.MAX_SIZE:
            raise ValueError(f"Size {size} exceeds max {self.MAX_SIZE}")

        # Pack using network byte order (big-endian) for portability
        return iscc_id_body + struct.pack("!II", offset, size)

    def _unpack_chunk_pointer(self, data):
        # type: (bytes) -> tuple[bytes, int, int]
        """
        Unpack 16-byte chunk pointer.

        :return: (iscc_id_body, offset, size)
        """
        if len(data) != self.CHUNK_POINTER_BYTES:
            raise ValueError(f"Expected {self.CHUNK_POINTER_BYTES} bytes, got {len(data)}")

        iscc_id_body = data[:8]
        offset, size = struct.unpack("!II", data[8:16])
        return iscc_id_body, offset, size

    def _check_existing_assets(self, entries):
        # type: (list[SimprintEntryRaw]) -> set[bytes]
        """
        Check which ISCC-ID bodies already exist in the index.

        :param entries: Entries to check
        :return: Set of existing ISCC-ID bodies
        """
        unique_ids = list({entry.iscc_id_body for entry in entries})

        with self.env.begin() as txn:
            cursor = txn.cursor(db=self.assets_db)
            results = cursor.getmulti(unique_ids)
            return {key for key, _ in results}

    def _build_insert_pairs(self, entries, existing):
        # type: (list[SimprintEntryRaw], set[bytes]) -> tuple[list[bytes], list[tuple[bytes, bytes]]]
        """
        Build data structures for batch insert, filtering duplicates.

        :param entries: Entries to process
        :param existing: Set of existing ISCC-ID bodies to skip
        :return: (new_asset_ids, simprint_pairs)
        """
        simprint_pairs = []  # type: list[tuple[bytes, bytes]]
        new_asset_ids = []  # type: list[bytes]
        seen_in_batch = set()  # type: set[bytes]

        for entry in entries:
            if entry.iscc_id_body in existing or entry.iscc_id_body in seen_in_batch:
                continue

            seen_in_batch.add(entry.iscc_id_body)
            new_asset_ids.append(entry.iscc_id_body)
            for simprint in entry.simprints:
                # Ensure simprint is 64-bit
                simprint_key = simprint.simprint[:8]
                chunk_ptr = self._pack_chunk_pointer(entry.iscc_id_body, simprint.offset, simprint.size)
                simprint_pairs.append((simprint_key, chunk_ptr))

        return new_asset_ids, simprint_pairs

    def _execute_batch_insert(self, new_asset_ids, simprint_pairs):
        # type: (list[bytes], list[tuple[bytes, bytes]]) -> None
        """
        Execute atomic batch insert with automatic resize on MapFullError.

        :param new_asset_ids: List of new ISCC-ID bodies to register
        :param simprint_pairs: List of (simprint_key, chunk_pointer) pairs to insert
        """
        retry_count = 0
        while retry_count <= self.MAX_RESIZE_RETRIES:  # pragma: no branch
            try:
                with self.env.begin(write=True) as txn:
                    # Register new assets
                    assets_cursor = txn.cursor(db=self.assets_db)
                    assets_cursor.putmulti(
                        [(asset_id, b"") for asset_id in new_asset_ids],
                        dupdata=False,
                    )

                    # Insert simprint entries
                    simprints_cursor = txn.cursor(db=self.simprints_db)
                    simprints_cursor.putmulti(simprint_pairs, dupdata=False)
                break

            except lmdb.MapFullError:  # pragma: no cover
                retry_count += 1
                if retry_count > self.MAX_RESIZE_RETRIES:
                    raise RuntimeError(f"Failed to add after {self.MAX_RESIZE_RETRIES} resize attempts")

                old_size = self.env.info()["map_size"]
                new_size = min(old_size * 2, self.MAX_MAP_SIZE)
                if new_size == old_size:
                    raise RuntimeError(f"Cannot resize beyond {self.MAX_MAP_SIZE} bytes")

                logger.info(f"Resizing LMDB from {old_size:,} to {new_size:,} bytes")
                self.env.set_mapsize(new_size)

    def _fetch_matches_and_frequencies(self, query_simprints):
        # type: (list[bytes]) -> tuple[dict[bytes, list[tuple[bytes, bytes, int, int]]], dict[bytes, int]]
        """
        Fetch matches from database and calculate document frequencies.

        :param query_simprints: List of 64-bit simprint queries
        :return: (asset_matches, doc_frequencies)
        """
        asset_matches = defaultdict(list)  # type: dict[bytes, list[tuple[bytes, bytes, int, int]]]
        doc_frequencies = {}  # type: dict[bytes, int]

        with self.env.begin() as txn:
            cursor = txn.cursor(db=self.simprints_db)
            results = cursor.getmulti(query_simprints, dupdata=True, dupfixed_bytes=16)

            simprint_to_assets = defaultdict(set)  # type: dict[bytes, set[bytes]]
            for simprint_bytes, chunk_bytes in results:
                iscc_id_body, offset, size = self._unpack_chunk_pointer(chunk_bytes)
                asset_matches[iscc_id_body].append((simprint_bytes, simprint_bytes, offset, size))
                simprint_to_assets[simprint_bytes].add(iscc_id_body)

            for simprint_bytes, assets in simprint_to_assets.items():
                doc_frequencies[simprint_bytes] = len(assets)

        return asset_matches, doc_frequencies

    def _calculate_idf_score(self, matches, doc_frequencies, total_assets, num_queried):
        # type: (list[tuple[bytes, bytes, int, int]], dict[bytes, int], int, int) -> float
        """
        Calculate IDF-weighted similarity score.

        :param matches: List of (query_simprint, match_simprint, offset, size) tuples
        :param doc_frequencies: Document frequency for each simprint
        :param total_assets: Total number of assets in index
        :param num_queried: Number of simprints in query
        :return: Similarity score (0.0 to 1.0)
        """
        if not matches:
            return 0.0

        # Calculate average IDF
        idf_sum = 0.0
        for query_simprint, _, _, _ in matches:
            df = doc_frequencies.get(query_simprint, 1)
            idf = math.log(total_assets / (1 + df))
            idf_sum += idf

        avg_idf = idf_sum / len(matches)

        # Calculate match ratio
        match_ratio = len(matches) / num_queried

        # Normalize IDF against best achievable case (df=1 for all simprints)
        # This allows perfect matches with rare simprints to reach 1.0
        if total_assets > 2:
            max_idf = math.log(total_assets / 2)
            normalized_idf = min(avg_idf / max_idf, 1.0)
        else:
            # Edge case: 1-2 assets, treat as perfect match
            normalized_idf = 1.0

        # Blend match ratio with IDF weighting
        return match_ratio * (0.5 + 0.5 * normalized_idf)

    def _format_match_result(self, iscc_id_body, matches, score, doc_frequencies, num_queried, detailed):
        # type: (bytes, list[tuple[bytes, bytes, int, int]], float, dict[bytes, int], int, bool) -> SimprintMatchRaw
        """
        Format match data into SimprintMatchRaw result.

        :param iscc_id_body: ISCC-ID body of matched asset
        :param matches: List of (query_simprint, match_simprint, offset, size) tuples
        :param score: Calculated similarity score
        :param doc_frequencies: Document frequency for each simprint
        :param num_queried: Number of simprints in query
        :param detailed: Whether to include chunk details
        :return: Formatted match result
        """
        chunks = None
        if detailed:
            chunks = [
                MatchedChunkRaw(
                    query=query_simprint,
                    match=match_simprint,
                    score=1.0,  # Exact match within hard boundary
                    offset=offset,
                    size=size,
                    freq=doc_frequencies.get(match_simprint, 1),
                )
                for query_simprint, match_simprint, offset, size in matches
            ]

        return SimprintMatchRaw(
            iscc_id_body=iscc_id_body,
            score=score,
            queried=num_queried,
            matches=len(matches),
            chunks=chunks,
        )
