"""
Usearch-based soft-boundary simprint index with exponential confidence weighting.

Implements SimprintIndexRaw protocol for asset-level similarity search using pure Usearch
with multi-query aggregation and configurable confidence weighting.

Storage:
    - Pure Usearch with multi=True (multiple simprints per asset)
    - Keys: ISCC-ID body as uint64
    - Vectors: Binary simprints
    - No metadata store (asset-level only, no chunk offset/size tracking)

Ranking:
    - Threshold filters individual simprint matches (noise rejection)
    - Multi-query aggregation with exponential confidence weighting
    - Score: exponentially weighted average of matched chunks (no coverage penalty)
    - Returns top limit results (no final score threshold)
    - Efficient: fetches only limit candidates per query simprint
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from usearch.index import Index, MetricKind, ScalarKind

if TYPE_CHECKING:
    from iscc_search.indexes.simprint.models import SimprintEntryRaw, SimprintMatchRaw  # noqa: F401


class UsearchSimprintIndex:
    """
    Usearch-based soft-boundary simprint index with exponential confidence weighting.

    Implements SimprintIndexRaw protocol for asset-level similarity search.
    Uses multi-query aggregation with configurable confidence weighting.
    """

    # Global default constants (overridable per query)
    DEFAULT_CONFIDENCE_EXPONENT = 4  # Emphasize high-confidence matches
    DEFAULT_COVERAGE_WEIGHT = 0.2  # Coverage influence (0=ignore, higher=more separation)

    def __init__(self, uri, ndim=128, realm_id=None, connectivity=8, expansion_add=16, **kwargs):
        # type: (str, int, bytes | None, int, int, ...) -> None
        """
        Create or open Usearch simprint index.

        :param uri: Index location (file path or URI)
        :param ndim: Simprint dimensions in bits (e.g., 64, 128, 256)
        :param realm_id: ISCC-ID realm identifier (2 bytes, optional)
        :param connectivity: HNSW graph connectivity (default 8, higher=better recall, slower build)
        :param expansion_add: Build-time search depth (default 16, lower=faster bulk indexing)
        :param kwargs: match_threshold, confidence_exponent for global overrides
        """
        # Parse URI to file path
        if uri.startswith("file://"):
            self.path = Path(uri[7:])
        else:
            self.path = Path(uri)

        # Try loading metadata from disk (for existing indexes)
        metadata_path = self.path.with_suffix(self.path.suffix + ".metadata.json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                realm_id = bytes.fromhex(metadata["realm_id"]) if metadata.get("realm_id") else realm_id
                ndim = metadata.get("ndim", ndim)

        self.ndim = ndim
        self.realm_id = realm_id

        # Create Usearch Index with multi=True, metric=Hamming, dtype=B1
        self.index = Index(
            ndim=ndim,  # Bits (e.g., 128)
            metric=MetricKind.Hamming,
            dtype=ScalarKind.B1,
            multi=True,  # Multiple simprints per ISCC-ID
            connectivity=connectivity,
            expansion_add=expansion_add,
        )

        # Load from disk if exists
        if self.path.exists():
            self.index = Index.restore(str(self.path), view=False)

    def add_raw(self, entries):
        # type: (list[SimprintEntryRaw]) -> None
        """
        Add simprint entries with batch operations.

        Note: Unlike the protocol's add-once semantics, this implementation
        allows adding multiple simprints for the same ISCC-ID (multi=True).

        :param entries: list[SimprintEntryRaw]
        """
        if not entries:
            return

        keys = []
        vectors = []

        for entry in entries:
            # Convert ISCC-ID body to uint64 key
            key = int.from_bytes(entry.iscc_id_body, "big", signed=False)

            # Add all simprints for this asset
            for simprint in entry.simprints:
                keys.append(key)
                # Convert bytes to numpy array
                vectors.append(np.frombuffer(simprint.simprint, dtype=np.uint8))

        # Batch add to Usearch
        if keys:
            # Stack vectors into 2D array
            vectors_array = np.stack(vectors)
            self.index.add(np.array(keys, dtype=np.uint64), vectors_array)

    def search_raw(self, simprints, limit=10, threshold=0.0, detailed=False, **kwargs):
        # type: (list[bytes], int, float, bool, ...) -> list[SimprintMatchRaw]
        """
        Search with exponential confidence weighting.

        Fetches top limit candidates per query simprint, filters by threshold (noise rejection),
        aggregates by asset, returns top limit results sorted by combined score.

        Per-query configurable parameters via kwargs:
        - confidence_exponent: Override DEFAULT_CONFIDENCE_EXPONENT (e.g., 6 for more emphasis)
        - coverage_weight: Override DEFAULT_COVERAGE_WEIGHT (e.g., 0.0=ignore, 0.5=more influence)

        :param simprints: Binary simprints to search for
        :param limit: Maximum results to return (also candidate count per simprint, default 10)
        :param threshold: Minimum individual simprint match score to consider (0.0-1.0, default 0.0, typical 0.8)
        :param detailed: If True, include chunk matches (offset/size not tracked, will be 0)
        :return: list[SimprintMatchRaw] ordered by quality score (exponentially weighted average), limited to top limit results
        """
        from iscc_search.indexes.simprint.models import SimprintMatchRaw, MatchedChunkRaw

        if not simprints:
            return []

        # Use per-query overrides or fall back to class defaults
        confidence_exponent = kwargs.get("confidence_exponent", self.DEFAULT_CONFIDENCE_EXPONENT)
        coverage_weight = kwargs.get("coverage_weight", self.DEFAULT_COVERAGE_WEIGHT)

        # Convert simprints to numpy array
        query_vectors = np.stack([np.frombuffer(s, dtype=np.uint8) for s in simprints])

        # Step 1: Batch search - fetch top candidates per query simprint
        # Only fetch what we need - bad matches are filtered by threshold anyway
        batch_results = self.index.search(
            query_vectors,
            count=limit,
        )

        # Handle single query case (returns Matches) vs batch (returns BatchMatches)
        # When there's only 1 query, usearch returns Matches directly
        if len(simprints) == 1:
            batch_results = [batch_results]

        # Step 2: Aggregate by asset with individual match filtering
        # Track best score per query_idx per asset (naturally bounds coverage to [0,1])
        asset_scores = defaultdict(dict)
        for query_idx in range(len(simprints)):
            matches = batch_results[query_idx]
            # Access keys and distances arrays directly
            for i in range(len(matches.keys)):
                key = matches.keys[i]
                distance = matches.distances[i]
                # Normalize Hamming distance (raw bit count) to 0-1 score
                # distance is bit count, ndim is total bits
                score = 1.0 - (distance / self.ndim)
                # Filter weak matches (noise) by threshold
                if score >= threshold:
                    # Keep only best match per query per asset
                    if query_idx not in asset_scores[key]:
                        asset_scores[key][query_idx] = score
                    else:
                        asset_scores[key][query_idx] = max(asset_scores[key][query_idx], score)

        # Step 3: Calculate weighted scores
        scored_results = []
        for key_uint64, query_scores in asset_scores.items():
            scores = list(query_scores.values())  # One score per query index

            # Quality: exponentially weighted average of matched chunks
            # Each score is weighted by score^k to emphasize high-confidence matches
            weighted_sum = sum(s ** (confidence_exponent + 1) for s in scores)  # s^k * s = s^(k+1)
            weight_sum = sum(s**confidence_exponent for s in scores)  # sum of weights
            quality = weighted_sum / weight_sum if weight_sum > 0 else 0.0

            # Apply configurable coverage weighting to help separate true/false positives
            # Formula: coverage^w × quality where w controls influence (0=ignore)
            # Both terms in [0,1] so product naturally bounded to [0,1]
            if coverage_weight > 0:
                coverage = len(scores) / len(simprints)
                final_score = (coverage**coverage_weight) * quality
            else:
                # No coverage influence - quality only
                final_score = quality

            # Convert numpy.uint64 to Python int before to_bytes
            iscc_id_body = int(key_uint64).to_bytes(8, "big", signed=False)
            scored_results.append(
                SimprintMatchRaw(
                    iscc_id_body=iscc_id_body,
                    score=final_score,
                    queried=len(simprints),
                    matches=len(scores),
                    chunks=None,
                )
            )

        # Step 4: Build detailed chunk matches if requested
        if detailed:
            for result in scored_results:
                key_uint64 = int.from_bytes(result.iscc_id_body, "big", signed=False)

                # Retrieve all stored simprints for this asset (returns 2D array for multi=True)
                stored_vectors_2d = self.index.get(key_uint64)

                # Handle missing key (shouldn't happen for matched assets)
                if stored_vectors_2d is None:  # pragma: no cover
                    result.chunks = None
                    continue

                # Match each query simprint to best stored simprint using vectorized ops
                chunks = []
                for query_sp in simprints:
                    query_vec = np.frombuffer(query_sp, dtype=np.uint8)

                    # Vectorized XOR: query (1D) vs all stored vectors (2D)
                    # Broadcasting: (1, ndim_bytes) XOR (n_vectors, ndim_bytes)
                    xor_results = np.bitwise_xor(query_vec, stored_vectors_2d)

                    # Unpack bits and sum per row to get hamming distances
                    # Shape: (n_vectors, ndim_bytes) -> (n_vectors, ndim_bytes*8) -> (n_vectors,)
                    distances = np.unpackbits(xor_results, axis=1).sum(axis=1)

                    # Convert to scores (vectorized)
                    scores = 1.0 - (distances / self.ndim)

                    # Find best match above threshold
                    valid_mask = scores >= threshold
                    if valid_mask.any():
                        # Find best score among valid matches only
                        valid_scores = scores[valid_mask]
                        valid_indices = np.where(valid_mask)[0]
                        best_valid_idx = np.argmax(valid_scores)
                        best_idx = valid_indices[best_valid_idx]
                        best_score = scores[best_idx]

                        chunks.append(
                            MatchedChunkRaw(
                                query=query_sp,
                                match=stored_vectors_2d[best_idx].tobytes(),
                                score=float(best_score),
                                offset=0,  # Position tracking not supported
                                size=0,  # Position tracking not supported
                                freq=1,  # Frequency tracking not supported
                            )
                        )

                result.chunks = chunks if chunks else None

                # Recalculate score based on actual chunk matches
                # Note: chunks is guaranteed non-empty because asset is in scored_results
                # only if at least one query simprint matched >= threshold initially
                chunk_scores = [chunk.score for chunk in chunks]

                # Quality: exponentially weighted average of matched chunks
                weighted_sum = sum(s ** (confidence_exponent + 1) for s in chunk_scores)
                weight_sum = sum(s**confidence_exponent for s in chunk_scores)
                quality = weighted_sum / weight_sum if weight_sum > 0 else 0.0

                # Apply coverage weighting
                if coverage_weight > 0:
                    coverage = len(chunk_scores) / len(simprints)
                    final_score = (coverage**coverage_weight) * quality
                else:
                    final_score = quality

                result.score = final_score
                result.matches = len(chunks)

        # Step 5: Sort and limit
        scored_results.sort(key=lambda x: (-x.score, x.iscc_id_body))
        return scored_results[:limit]

    def __contains__(self, iscc_id_body):
        # type: (bytes) -> bool
        """Check if ISCC-ID exists using usearch's native contains."""
        key = int.from_bytes(iscc_id_body, "big", signed=False)
        return key in self.index

    def __len__(self):
        # type: () -> int
        """Return number of unique assets by counting unique keys in index."""
        return len(set(self.index.keys))

    def close(self):
        # type: () -> None
        """Save index and metadata to disk and release resources."""
        self.index.save(str(self.path))

        # Save metadata separately (usearch doesn't support custom metadata)
        metadata_path = self.path.with_suffix(self.path.suffix + ".metadata.json")
        metadata = {
            "realm_id": self.realm_id.hex() if self.realm_id else None,
            "ndim": self.ndim,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    def optimize(self):
        # type: () -> None
        """
        Compact index for better performance after bulk adds.

        Calls usearch's compact() method if available to optimize memory
        layout and query performance.
        """
        if hasattr(self.index, "compact"):  # pragma: no cover
            self.index.compact()  # pragma: no cover

    def __enter__(self):
        # type: () -> UsearchSimprintIndex
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # type: (type | None, Exception | None, object | None) -> None
        """Context manager exit - save and close."""
        self.close()
