"""
Pure stateless functions for LMDB simprint operations.

Provides pack/unpack for 16-byte chunk pointers, IDF calculation,
document frequency counting, exact (hard-boundary) simprint search,
and rebuild iteration for derived usearch indexes.
All functions operate on raw LMDB transactions and database handles
without managing LMDB lifecycle.
"""

import math
import struct
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator  # noqa: F401

import numpy as np

from iscc_search.indexes.simprint.models import MatchedChunkRaw, SimprintMatchRaw


# Constants
CHUNK_POINTER_BYTES = 16  # 8 bytes ISCC-ID body + 4 bytes offset + 4 bytes size
MAX_OFFSET = 2**32 - 1  # 4 GB max offset
MAX_SIZE = 2**32 - 1  # 4 GB max size


def pack_chunk_pointer(iscc_id_body, offset, size):
    # type: (bytes, int, int) -> bytes
    """
    Pack chunk pointer into 16-byte binary format.

    Layout: iscc_id_body(8) + offset(4) + size(4)
    Uses network byte order (big-endian) for cross-platform portability.

    :param iscc_id_body: Binary ISCC-ID body (8 bytes)
    :param offset: Byte offset of chunk in asset
    :param size: Size of chunk in bytes
    :return: 16-byte packed chunk pointer
    """
    if len(iscc_id_body) != 8:
        raise ValueError(f"ISCC-ID body must be 8 bytes, got {len(iscc_id_body)}")
    if offset > MAX_OFFSET:
        raise ValueError(f"Offset {offset} exceeds max {MAX_OFFSET}")
    if size > MAX_SIZE:
        raise ValueError(f"Size {size} exceeds max {MAX_SIZE}")
    return iscc_id_body + struct.pack("!II", offset, size)


def unpack_chunk_pointer(data):
    # type: (bytes) -> tuple[bytes, int, int]
    """
    Unpack 16-byte chunk pointer.

    :param data: 16-byte packed chunk pointer
    :return: (iscc_id_body, offset, size)
    """
    if len(data) != CHUNK_POINTER_BYTES:
        raise ValueError(f"Expected {CHUNK_POINTER_BYTES} bytes, got {len(data)}")
    iscc_id_body = data[:8]
    offset, size = struct.unpack("!II", data[8:16])
    return iscc_id_body, offset, size


def calculate_idf(freq, total_assets):
    # type: (int, int) -> float
    """
    Calculate Inverse Document Frequency for a simprint.

    Uses smooth IDF formula: log(1 + total_assets / (1 + freq)).
    Always non-negative (minimum 0.0 when total_assets=0).

    :param freq: Document frequency (number of assets containing simprint)
    :param total_assets: Total number of assets in the index
    :return: IDF weight (always >= 0.0)
    """
    if total_assets <= 0:
        return 0.0
    return math.log(1 + total_assets / (1 + freq))


def delete_asset_simprints(txn, db, iscc_id_body):
    # type: (object, object, bytes) -> list[bytes]
    """
    Delete all simprint entries for a given asset from a dupsort data database.

    Iterates all key-value pairs and removes those where the value's first 8 bytes
    match the given ISCC-ID body. After cursor.delete(), the cursor advances to the
    next record automatically.

    :param txn: LMDB write transaction
    :param db: LMDB database handle (dupsort with 16-byte fixed values)
    :param iscc_id_body: 8-byte ISCC-ID body identifying the asset
    :return: List of deleted composite keys (16-byte chunk pointers)
    """
    cursor = txn.cursor(db)
    deleted_keys = []  # type: list[bytes]
    has_item = cursor.first()
    while has_item:
        value = cursor.value()
        if value[:8] == iscc_id_body:
            deleted_keys.append(bytes(value))
            has_item = cursor.delete()
        else:
            has_item = cursor.next()
    return deleted_keys


def count_doc_freq(txn, db, simprint_key, dup_limit=1000):
    # type: (object, object, bytes, int) -> int
    """
    Count distinct assets containing a simprint key.

    Iterates duplicate values for the key, extracting the 8-byte ISCC-ID body
    from each 16-byte chunk pointer to count unique assets.

    :param txn: LMDB read transaction
    :param db: LMDB database handle (dupsort with 16-byte fixed values)
    :param simprint_key: Simprint bytes to look up
    :param dup_limit: Maximum duplicates to scan (safety cap)
    :return: Number of distinct assets containing this simprint
    """
    cursor = txn.cursor(db)
    if not cursor.set_key(simprint_key):
        return 0

    seen_assets = set()  # type: set[bytes]
    count = 0
    for value in cursor.iternext_dup(keys=False, values=True):
        iscc_id_body = value[:8]
        seen_assets.add(iscc_id_body)
        count += 1
        if count >= dup_limit:
            break

    return len(seen_assets)


def search_simprints_exact(txn, db, query_simprints, total_assets, limit, threshold, detailed, dup_limit=1000):
    # type: (object, object, list[bytes], int, int, float, bool, int) -> list[SimprintMatchRaw]
    """
    Hard-boundary exact search with coverage x quality scoring.

    For each query simprint, finds all matching chunk pointers in the LMDB database.
    Groups matches by asset and scores using coverage (fraction matched) times
    quality (inverse-frequency weighting within the match set).

    :param txn: LMDB read transaction
    :param db: LMDB database handle (dupsort with 16-byte fixed values)
    :param query_simprints: List of binary simprints to search for
    :param total_assets: Total number of assets (for potential IDF use)
    :param limit: Maximum number of results to return
    :param threshold: Minimum score to include in results (0.0-1.0)
    :param detailed: If True, include individual chunk matches
    :param dup_limit: Maximum duplicates per simprint key (safety cap)
    :return: List of SimprintMatchRaw sorted by score descending
    """
    if not query_simprints:
        return []

    # Collect matches and doc frequencies
    # asset_matches: iscc_id_body -> [(query_sp, match_sp, offset, size), ...]
    asset_matches = defaultdict(list)  # type: dict[bytes, list[tuple[bytes, bytes, int, int]]]
    doc_frequencies = {}  # type: dict[bytes, int]
    simprint_to_assets = defaultdict(set)  # type: dict[bytes, set[bytes]]

    cursor = txn.cursor(db)
    for sp_bytes in query_simprints:
        if not cursor.set_key(sp_bytes):
            continue

        count = 0
        for value in cursor.iternext_dup(keys=False, values=True):
            iscc_id_body, offset, size = unpack_chunk_pointer(value)
            asset_matches[iscc_id_body].append((sp_bytes, sp_bytes, offset, size))
            simprint_to_assets[sp_bytes].add(iscc_id_body)
            count += 1
            if count >= dup_limit:
                break

    # Build doc frequencies from collected data
    for sp_bytes, assets in simprint_to_assets.items():
        doc_frequencies[sp_bytes] = len(assets)

    # Score and format results
    num_queried = len(query_simprints)
    results = []  # type: list[SimprintMatchRaw]

    for iscc_id_body, matches in asset_matches.items():
        score = _calculate_coverage_quality_score(matches, doc_frequencies, num_queried)

        if score < threshold:
            continue

        chunks = None
        if detailed:
            chunks = [
                MatchedChunkRaw(
                    query=query_sp,
                    match=match_sp,
                    score=1.0,  # Exact match within hard boundary
                    offset=offset,
                    size=size,
                    freq=doc_frequencies.get(match_sp, 1),
                )
                for query_sp, match_sp, offset, size in matches
            ]

        results.append(
            SimprintMatchRaw(
                iscc_id_body=iscc_id_body,
                score=score,
                queried=num_queried,
                matches=len(matches),
                chunks=chunks,
            )
        )

    results.sort(key=lambda x: (-x.score, x.iscc_id_body))
    return results[:limit]


def _calculate_coverage_quality_score(matches, doc_frequencies, num_queried):
    # type: (list[tuple[bytes, bytes, int, int]], dict[bytes, int], int) -> float
    """
    Calculate similarity score using coverage and relative rarity within match set.

    Score = Coverage x Quality
    - Coverage: fraction of unique query simprints matched (0.0 to 1.0)
    - Quality: min-max normalized inverse frequency (0.0 to 1.0)

    Uses coverage x quality scoring algorithm.

    :param matches: List of (query_simprint, match_simprint, offset, size) tuples
    :param doc_frequencies: Document frequency for each simprint
    :param num_queried: Number of simprints in query
    :return: Similarity score (0.0 to 1.0)
    """
    if not matches:
        return 0.0

    # Group by query simprint, keep best (lowest) frequency for each
    query_to_best_freq = {}  # type: dict[bytes, int]
    for query_simprint, match_simprint, _, _ in matches:
        freq = doc_frequencies.get(match_simprint, 1)
        if query_simprint not in query_to_best_freq:
            query_to_best_freq[query_simprint] = freq
        else:
            query_to_best_freq[query_simprint] = min(query_to_best_freq[query_simprint], freq)

    # Coverage: fraction of unique query simprints matched
    coverage = len(query_to_best_freq) / num_queried

    # Quality: average relative rarity within this match set
    freqs = list(query_to_best_freq.values())

    if len(freqs) == 1:
        quality = 1.0
    else:
        min_freq = min(freqs)
        max_freq = max(freqs)

        if min_freq == max_freq:
            quality = 1.0
        else:
            inverse_freqs = [1.0 / f for f in freqs]
            min_inv = 1.0 / max_freq
            max_inv = 1.0 / min_freq
            quality = sum((inv - min_inv) / (max_inv - min_inv) for inv in inverse_freqs) / len(inverse_freqs)

    return coverage * quality


REBUILD_BATCH_SIZE = 100_000  # Batch size for memory-efficient rebuild


def iter_simprint_vectors(txn, db, batch_size=REBUILD_BATCH_SIZE):
    # type: (object, object, int) -> Generator[tuple[list[bytes], list[np.ndarray]], None, None]
    """
    Yield (composite_keys, vectors) batches from LMDB for ShardedIndex128 rebuild.

    Iterates the simprint database and yields batches of composite keys (ChunkPointer values)
    paired with their simprint vectors for incremental insertion into ShardedIndex128.

    :param txn: LMDB read transaction
    :param db: LMDB simprint database handle (dupsort with 16-byte fixed values)
    :param batch_size: Maximum entries per yielded batch
    :return: Generator yielding (keys, vectors) tuples
    """
    keys = []  # type: list[bytes]
    vectors = []  # type: list[np.ndarray]
    cursor = txn.cursor(db)

    if not cursor.first():
        return

    while True:
        simprint_bytes = bytes(cursor.key())
        # Iterate all duplicate values for this key
        for value in cursor.iternext_dup(keys=False, values=True):
            chunk_pointer = bytes(value)
            keys.append(chunk_pointer)
            vectors.append(np.frombuffer(simprint_bytes, dtype=np.uint8))
            if len(keys) >= batch_size:
                yield keys, vectors
                keys = []
                vectors = []
        # Move to next distinct key
        if not cursor.next_nodup():
            break

    if keys:
        yield keys, vectors
