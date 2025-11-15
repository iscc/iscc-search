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
    - Multi-query aggregation with exponential confidence weighting
    - Configurable match_threshold and confidence_exponent per query
    - Combined score: coverage * quality
"""

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
    DEFAULT_MATCH_THRESHOLD = 0.75  # Filter noise (~25% Hamming distance)
    DEFAULT_CONFIDENCE_EXPONENT = 4  # Emphasize high-confidence matches

    def __init__(self, uri, ndim=128, realm_id=None, **kwargs):
        # type: (str, int, bytes | None, ...) -> None
        """
        Create or open Usearch simprint index.

        :param uri: Index location (file path or URI)
        :param ndim: Simprint dimensions in bits (e.g., 64, 128, 256)
        :param realm_id: ISCC-ID realm identifier (2 bytes, optional)
        :param kwargs: match_threshold, confidence_exponent for global overrides
        """
        # Parse URI to file path
        if uri.startswith("file://"):
            self.path = Path(uri[7:])
        else:
            self.path = Path(uri)

        self.ndim = ndim
        self.realm_id = realm_id

        # Create Usearch Index with multi=True, metric=Hamming, dtype=B1
        self.index = Index(
            ndim=ndim,  # Bits (e.g., 128)
            metric=MetricKind.Hamming,
            dtype=ScalarKind.B1,
            multi=True,  # Multiple simprints per ISCC-ID
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

        Per-query configurable threshold and exponent via kwargs:
        - match_threshold: Override DEFAULT_MATCH_THRESHOLD (e.g., 0.8)
        - confidence_exponent: Override DEFAULT_CONFIDENCE_EXPONENT (e.g., 6)

        :param simprints: Binary simprints to search for
        :param limit: Maximum results
        :param threshold: Minimum score (0.0-1.0, default 0.0)
        :param detailed: Must be False (raises NotImplementedError if True, default False)
        :return: list[SimprintMatchRaw] ordered by score
        """
        from iscc_search.indexes.simprint.models import SimprintMatchRaw

        if detailed:
            raise NotImplementedError(
                "UsearchSimprintIndex doesn't support detailed=True (no chunk offset/size storage)"
            )

        if not simprints:
            return []

        # Use per-query overrides or fall back to class defaults
        match_threshold = kwargs.get("match_threshold", self.DEFAULT_MATCH_THRESHOLD)
        confidence_exponent = kwargs.get("confidence_exponent", self.DEFAULT_CONFIDENCE_EXPONENT)

        # Convert simprints to numpy array
        query_vectors = np.stack([np.frombuffer(s, dtype=np.uint8) for s in simprints])

        # Step 1: Batch search
        batch_results = self.index.search(
            query_vectors,
            count=1000,  # Candidate pool per query
        )

        # Handle single query case (returns Matches) vs batch (returns BatchMatches)
        # When there's only 1 query, usearch returns Matches directly
        if len(simprints) == 1:
            batch_results = [batch_results]

        # Step 2: Aggregate by asset with confidence filtering
        asset_scores = defaultdict(list)
        for query_idx in range(len(simprints)):
            matches = batch_results[query_idx]
            # Access keys and distances arrays directly
            for i in range(len(matches.keys)):
                key = matches.keys[i]
                distance = matches.distances[i]
                # Normalize Hamming distance (raw bit count) to 0-1 score
                # distance is bit count, ndim is total bits
                score = 1.0 - (distance / self.ndim)
                if score >= match_threshold:  # Filter noise
                    asset_scores[key].append(score)

        # Step 3: Calculate weighted scores
        scored_results = []
        for key_uint64, scores in asset_scores.items():
            # Coverage: fraction of query simprints matched
            coverage = len(scores) / len(simprints)

            # Quality: exponentially weighted average
            weighted_sum = sum(s**confidence_exponent for s in scores)
            weight_sum = sum(s for s in scores)
            quality = weighted_sum / weight_sum

            # Combined score
            final_score = coverage * quality

            if final_score >= threshold:
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

        # Step 4: Sort and limit
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
        """Save index to disk and release resources."""
        self.index.save(str(self.path))

    def __enter__(self):
        # type: () -> UsearchSimprintIndex
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # type: (type | None, Exception | None, object | None) -> None
        """Context manager exit - save and close."""
        self.close()
