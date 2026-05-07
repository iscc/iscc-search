"""
Derived ShardedIndex128 for approximate simprint similarity search.

Persistence and asset dedup are handled by the main LMDB (unified architecture).
This class manages only the derived ShardedIndex128 for approximate search.
Scoring uses IDF-weighted averaging per the OpenAPI spec (IsccChunkMatch.yaml).

Storage:
    - ShardedIndex128 with composite 128-bit keys: iscc_id_body(8) + offset(4) + size(4)
    - Binary vectors (Hamming metric, B1 scalar type)
    - Sharded HNSW with bloom filter and tombstone-based deletion

Ranking:
    - Configurable oversampling to ensure enough distinct assets
    - IDF-weighted asset-level scoring: sum(idf_i * sim_i) / sum(all_idf_i)
    - Unmatched query simprints contribute idf * 0.0 (penalizes low coverage)
    - doc_freq_fn callback provides true document frequency via LMDB
"""

import struct
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from iscc_usearch import ShardedIndex128

from iscc_search.indexes.simprint import lmdb_ops

if TYPE_CHECKING:
    from collections.abc import Callable  # noqa: F401

    from iscc_search.indexes.simprint.models import SimprintMatchRaw, MatchedChunkRaw  # noqa: F401


class UsearchSimprintIndex:
    """
    Derived ShardedIndex128 for approximate simprint search.

    Persistence and asset dedup are handled by the main LMDB (unified architecture).
    This class manages only the derived ShardedIndex128 for approximate search.
    Scoring uses IDF-weighted averaging per the OpenAPI spec (IsccChunkMatch.yaml).
    """

    def __init__(
        self,
        path,
        ndim=128,
        connectivity=8,
        expansion_add=16,
        expansion_search=512,
        shard_size=1024 * 1024 * 1024,
        oversampling_factor=20,
        background_rotation=False,
    ):
        # type: (str | Path, int, int, int, int, int, int, bool) -> None
        """
        Create or open derived simprint index.

        :param path: Directory path for ShardedIndex128 shard storage
        :param ndim: Simprint dimensions in bits (e.g., 64, 128, 256)
        :param connectivity: HNSW graph connectivity parameter
        :param expansion_add: Build-time search depth
        :param expansion_search: Search depth for HNSW queries
        :param shard_size: Maximum shard file size in bytes
        :param oversampling_factor: Oversampling multiplier for asset diversity
        :param background_rotation: Serialize full shards in a background thread
        """
        self.path = Path(path)
        self.ndim = ndim
        self.oversampling_factor = oversampling_factor
        self._index = ShardedIndex128(
            ndim=ndim,
            metric="hamming",
            dtype="b1",
            path=self.path,
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
            shard_size=shard_size,
            background_rotation=background_rotation,
        )

    def add_raw(self, composite_keys, vectors):
        # type: (list[bytes], list[np.ndarray]) -> None
        """
        Add vectors with composite 128-bit keys to derived index.

        Deduplicates within batch (keeps first occurrence) but skips the
        expensive contains-check against the existing index. Asset-level
        dedup is handled by the caller (main LMDB registry).
        Does NOT call save() - caller is responsible for explicit flush.

        :param composite_keys: 16-byte composite keys (iscc_id_body + offset + size)
        :param vectors: Binary simprint vectors as numpy uint8 arrays
        """
        if not composite_keys:
            return
        keys_arr = self._index._normalize_batch_keys(composite_keys)
        vectors_array = np.stack(vectors)
        # Deduplicate within batch (keep first occurrence)
        _, first_indices = np.unique(keys_arr, return_index=True)
        if len(first_indices) < len(keys_arr):
            first_indices = np.sort(first_indices)
            keys_arr = keys_arr[first_indices]
            vectors_array = vectors_array[first_indices]
        self._index.add(keys_arr, vectors_array)

    def remove(self, composite_keys):
        # type: (list[bytes]) -> None
        """
        Remove vectors by composite keys from derived index.

        :param composite_keys: 16-byte composite keys to remove
        """
        if not composite_keys:
            return
        self._index.remove(composite_keys)

    def search_raw(self, simprints, limit=10, threshold=0.0, detailed=False, doc_freq_fn=None, total_assets=0):
        # type: (list[bytes], int, float, bool, Callable[[bytes], int] | None, int) -> list[SimprintMatchRaw]
        """
        Search with configurable oversampling and IDF-weighted asset-level scoring.

        For each query simprint, searches ShardedIndex128 with oversampling_factor * limit
        candidates, groups results by asset (key[:8]), computes best-per-query-per-asset,
        then calculates IDF-weighted score across all query simprints.

        :param simprints: Binary simprints to search for
        :param limit: Maximum results to return
        :param threshold: Minimum individual simprint match score (0.0-1.0)
        :param detailed: If True, include chunk matches with stored simprint bytes
        :param doc_freq_fn: Callable(simprint_bytes) -> int for true document frequency.
            Required for IDF scoring. If None, all frequencies default to 1.
        :param total_assets: Total assets in the index (for IDF calculation)
        :return: list[SimprintMatchRaw] ordered by IDF-weighted score
        """
        from iscc_search.indexes.simprint.models import SimprintMatchRaw, MatchedChunkRaw

        if not simprints or len(self._index) == 0:
            return []

        # Convert simprints to numpy array for batch search
        query_vectors = np.stack([np.frombuffer(s, dtype=np.uint8) for s in simprints])

        # Search with oversampling to get enough distinct assets
        count = max(1, limit * self.oversampling_factor)
        batch_results = self._index.search(query_vectors, count=count)

        # Handle single query case (returns Matches, not BatchMatches)
        if len(simprints) == 1:
            batch_results = [batch_results]

        # Group chunk results by asset, track best score per query per asset
        # asset_chunks: asset_id -> {query_idx: (offset, size, score, composite_key)}
        asset_best = defaultdict(dict)  # type: dict[bytes, dict[int, tuple[int, int, float, bytes]]]

        for query_idx in range(len(simprints)):
            matches = batch_results[query_idx]
            for i in range(len(matches.keys)):
                raw_key = bytes(matches.keys[i])
                distance = float(matches.distances[i])

                # Normalize Hamming distance to similarity score
                score = 1.0 - (distance / self.ndim)
                if score < threshold:
                    continue

                # Decompose composite key
                asset_id = raw_key[:8]
                offset, size = struct.unpack("!II", raw_key[8:16])

                # Keep best match per query per asset (store composite key for vector lookup)
                if query_idx not in asset_best[asset_id]:
                    asset_best[asset_id][query_idx] = (offset, size, score, raw_key)
                elif score > asset_best[asset_id][query_idx][2]:  # pragma: no cover
                    # Defensive: HNSW search returns sorted results, so this branch
                    # is only reachable if results arrive in non-sorted order
                    asset_best[asset_id][query_idx] = (offset, size, score, raw_key)

        if not asset_best:
            return []

        # Cache doc frequencies to avoid redundant LMDB lookups
        freq_cache = {}  # type: dict[bytes, int]

        def get_freq(sp_key):
            # type: (bytes) -> int
            if sp_key not in freq_cache:
                if doc_freq_fn is not None:
                    freq_cache[sp_key] = doc_freq_fn(sp_key)
                else:
                    freq_cache[sp_key] = 1
            return freq_cache[sp_key]

        # IDF-weighted scoring per asset
        scored_results = []  # type: list[SimprintMatchRaw]
        for asset_id, best_per_query in asset_best.items():
            # Calculate IDF-weighted score using matched (stored) simprint for freq lookup
            total_idf = 0.0
            weighted_sim = 0.0

            for query_idx, (offset, size, sim, composite_key) in best_per_query.items():
                stored_vector = self._index.get(composite_key)
                match_bytes = stored_vector.tobytes() if stored_vector is not None else simprints[query_idx]
                freq = get_freq(match_bytes)
                idf = lmdb_ops.calculate_idf(freq, total_assets)
                total_idf += idf
                weighted_sim += idf * sim

            # Unmatched query simprints: look up actual IDF for each
            matched_indices = set(best_per_query.keys())
            for qi in range(len(simprints)):
                if qi not in matched_indices:
                    freq = get_freq(simprints[qi])
                    idf = lmdb_ops.calculate_idf(freq, total_assets)
                    total_idf += idf

            asset_score = weighted_sim / total_idf if total_idf > 0 else 0.0

            # Build chunk details if requested
            chunks = None
            if detailed:
                chunks = []
                for query_idx, (offset, size, sim, composite_key) in best_per_query.items():
                    stored_vector = self._index.get(composite_key)
                    match_bytes = stored_vector.tobytes() if stored_vector is not None else simprints[query_idx]
                    freq = get_freq(match_bytes)
                    chunks.append(
                        MatchedChunkRaw(
                            query=simprints[query_idx],
                            match=match_bytes,
                            score=sim,
                            offset=offset,
                            size=size,
                            freq=freq,
                        )
                    )

            scored_results.append(
                SimprintMatchRaw(
                    iscc_id_body=asset_id,
                    score=asset_score,
                    queried=len(simprints),
                    matches=len(best_per_query),
                    chunks=chunks,
                )
            )

        # Sort by score descending, then by asset ID for stability
        scored_results.sort(key=lambda x: (-x.score, x.iscc_id_body))
        return scored_results[:limit]

    @property
    def dirty(self):
        # type: () -> int
        """Number of unsaved key mutations."""
        return self._index.dirty

    @property
    def size(self):
        # type: () -> int
        """Return number of vectors in the index."""
        return len(self._index)

    def save(self):
        # type: () -> None
        """Save derived index to disk."""
        self._index.save()

    def reset(self):
        # type: () -> None
        """Release all in-memory resources."""
        self._index.reset()

    def close(self):
        # type: () -> None
        """Save and release resources."""
        self._index.close()
