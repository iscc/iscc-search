"""
Core Simprint Indexing Protocol

This module defines the minimal protocol for implementing backend-agnostic simprint indexes.
Simprints are granular content fingerprints that enable similarity matching at the chunk level
within digital assets identified by ISCC-IDs.

The protocol uses raw binary types (bytes) for maximum performance and leaves all ISCC-specific
serialization/deserialization to higher-level coordinators.
"""

from typing import Protocol


class SimprintRaw(Protocol):
    """
    Raw simprint representation for a content chunk.

    Represents a single simprint and its location within an asset.
    All simprints in a single entry must be of the same type.
    """

    simprint: bytes  # Binary simprint digest of the chunk
    offset: int  # Byte offset where chunk starts in the asset
    size: int  # Size in bytes of the chunk


class SimprintEntryRaw(Protocol):
    """
    Entry containing all simprints of a specific type for a single asset.

    Used for indexing operations. All simprints must be of the same type
    (e.g., all CONTENT_TEXT_V0 or all SEMANTIC_TEXT_V0).
    """

    iscc_id: bytes  # Binary ISCC-ID body (8 bytes, without header)
    simprints: list[SimprintRaw]  # List of simprints with location metadata


class MatchedChunkRaw(Protocol):
    """
    Individual chunk match result from a simprint search.

    Represents a single simprint from the query that matched a simprint
    in the index, including similarity score and location information.
    """

    query: bytes  # Binary simprint from the search query
    match: bytes  # Binary simprint from the index that matched
    score: float  # Similarity score (0.0 to 1.0) at chunk level
    offset: int  # Byte offset of matched chunk in the asset
    size: int  # Size in bytes of the matched chunk


class SimprintMatchRaw(Protocol):
    """
    Asset-level match result aggregating multiple simprint matches.

    Represents a single asset that matched the query based on overlapping simprints.
    The asset-level score aggregates individual chunk scores.

    Score Calculation:
        Implementations are free to choose their aggregation strategy (mean, max, weighted, etc.)
        The only requirement is that score is a float between 0.0 and 1.0.
    """

    iscc_id: bytes  # Binary ISCC-ID body of the matched asset (without header)
    score: float  # Aggregated similarity score (0.0 to 1.0) at asset level
    queried: int  # Number of simprints in the query
    matches: int  # Number of query simprints that found matches
    chunks: list[MatchedChunkRaw] | None  # Individual chunk matches (None when detailed=False)


class SimprintIndexRaw(Protocol):
    """
    Core protocol for backend-agnostic simprint indexes.

    Defines the minimal interface for indexes that match media assets by their
    granular chunk simprints. Every simprint index implementation must satisfy
    this protocol.

    Implementation Notes:
        - Indexes handle only raw binary data (bytes) for performance
        - ISCC-specific serialization is handled by higher-level coordinators
        - Global metadata (simprint_type, dimensions, realm_id) managed externally
        - All operations must be thread-safe for concurrent access
    """

    def __init__(self, path, **kwargs):
        # type: (str, ...) -> None
        """
        Open or create a simprint index at the specified path.

        Creates a new index if the path doesn't exist, otherwise opens the existing index.
        Implementation-specific parameters can be passed via kwargs.

        :param path: Path to the index directory (will be created if needed)
        :param kwargs: Backend-specific configuration options
        """

    def add_raw(self, entries):
        # type: (list[SimprintEntryRaw]) -> None
        """
        Add entries to the index with add-once semantics.

        Add-Once Semantics:
            Each ISCC-ID can only be added ONCE to the index.
            Duplicate ISCC-IDs are silently ignored (no-op).

        Rationale:
            Silent ignore allows efficient bulk loading without pre-checking.
            Callers needing feedback can check existence before/after.

        Transaction Semantics:
            MUST be atomic - either all new entries are added or none.
            Duplicates don't cause transaction failure.

        :param entries: List of entries to add atomically
        """

    def search_raw(self, simprints, limit=10, threshold=0.8, detailed=True):
        # type: (list[bytes], int, float, bool) -> list[SimprintMatchRaw]
        """
        Search for assets with similar simprints.

        Finds assets whose simprints match the query simprints above the threshold.
        Results are ordered by descending similarity score.

        :param simprints: Binary simprints to search for
        :param limit: Maximum number of assets to return
        :param threshold: Minimum similarity score (0.0 to 1.0) for matches
        :param detailed: If True, include individual chunk matches in results
        :return: List of matched assets ordered by similarity (best first)
        """

    def __contains__(self, iscc_id):
        # type: (bytes) -> bool
        """
        Check if an ISCC-ID exists in the index.

        Enables pythonic membership testing: `if iscc_id in index:`

        :param iscc_id: Binary ISCC-ID body (without header) to check
        :return: True if the asset has been indexed, False otherwise
        """

    def __len__(self):
        # type: () -> int
        """
        Return the number of unique assets in the index.

        Enables pythonic length checking: `len(index)`

        :return: Count of indexed assets (not total simprints)
        """

    def close(self):
        """
        Close the index and release resources.

        Flushes pending writes and releases file handles/connections.
        The index should not be used after closing.
        """


class SimprintIndexMutableRaw(SimprintIndexRaw):
    """
    Extended protocol for mutable simprint indexes that support deletion and retrieval.

    Not all backends need to support mutation - immutable/append-only indexes can
    implement only the base SimprintIndexRaw protocol.
    """

    def get_raw(self, iscc_ids):
        # type: (list[bytes]) -> list[SimprintEntryRaw]
        """
        Retrieve indexed entries by their ISCC-IDs.

        Returns the stored simprints for the specified assets. Useful for
        re-indexing, debugging, or implementing update operations.

        :param iscc_ids: Binary ISCC-ID bodies to retrieve
        :return: List of entries (empty entries for non-existent ISCC-IDs)
        """

    def delete_raw(self, iscc_ids):
        # type: (list[bytes]) -> None
        """
        Remove assets from the index by their ISCC-IDs.

        Completely removes all simprints associated with the specified assets.
        Non-existent ISCC-IDs are silently ignored.

        Transaction Semantics:
            MUST be atomic - either all entries are deleted or none are deleted.

        :param iscc_ids: Binary ISCC-ID bodies to delete
        """
