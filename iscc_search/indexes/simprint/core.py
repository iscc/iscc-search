"""Core Simprint Indexing Protocol"""

from typing import Protocol


class SimprintRaw(Protocol):
    """Minimal raw simprint representation"""

    simprint: bytes  # Simprint digest of the chunk
    offset: int  # Offset of the chunk in the asset
    size: int  # Extent of the chunk


class SimprintEntryRaw(Protocol):
    """Simprints of a given simprint-type for a given asset for indexing."""

    iscc_id: bytes  # ISCC-ID digest (without a header) of asset that includes the chunk
    simprints: list[SimprintRaw]  # Simprint from asset (all the same simprint_type)


class MatchedChunkRaw(Protocol):
    """A matched chunk for a given simprint search."""

    query: bytes  # The query simprint
    match: bytes  # The matched simprint
    score: float  # The score of the match (0.0 low to 1.0 high) at chunk level
    offset: int  # The offset of the matched chunk in the matched asset
    size: int  # The extent of the matched chunk


class SimprintMatchRaw(Protocol):
    """A single matched asset for a given query input (overlaping simprints).

    Score Calculation:
        Implementations are free to choose their aggregation strategy (mean, max, weighted, etc.)
        The only requirement is that score is a float between 0.0 and 1.0
    """

    iscc_id: bytes  # ISCC-ID of the matched asset (digest without header)
    score: float  # The score of the match (0.0 to 1.0) at asset level
    queried: int  # The number of simprints in the query
    matches: int  # The number of simprints that match the query simprints
    chunks: list[MatchedChunkRaw] | None  # None when detailed=False


class SimprintIndexRaw(Protocol):
    """
    Minimal core protocol for a backend agnostic index for matching media assets identified by
    ISCC-IDs based on granular chunk simprints. Every Simprint Index must implement this protocol.

    Global metadata such as simprint_type, dimension, realm_id, and data deserialization and
    serialization have to be managed outside the index.
    """

    def __init__(self, path, **kwargs):
        # type: (str|Path, Any) -> None
        """
        Open or create a new simprint index at the given path.

        :param path: Path to the index directory
        """

    def add_raw(self, entries):
        # type: (list[SimprintEntryRaw]) -> None
        """Add entries to the index.

        Add-Once Semantics:
          Each ISCC-ID can only be added ONCE to the index.
          Duplicate ISCC-IDs are silently ignored (no-op).

        Rationale:
          Silent ignore allows efficient bulk loading without pre-checking.
          Callers who need to know can check existence before/after.

        Transaction Semantics:
          MUST be atomic - either all new entries are added or none.
          Duplicates don't cause transaction failure.

        :param entries: List of entries to add atomically
        """

    def search_raw(self, simprints, limit=10, threshold=0.8, detailed=True):
        # type: (list[bytes], int, float, bool) -> list[SimprintMatchRaw]
        """
        Search for similar entries in the index based on simprints.

        :param simprints: List simprints to search for
        :param limit: The maximum number of matched documents to return
        :param threshold: Minimum similarity score required for a match
        :param detailed: Return detailed match information (matched chunks)
        """

    def __contains__(self, iscc_id):
        # type: (bytes) -> bool
        """
        Check if an ISCC-ID exists in the index.

        :param iscc_id: ISCC-ID digest (without header) to check
        :return: True if the ISCC-ID has been indexed, False otherwise
        """

    def __len__(self):
        # type: () -> int
        """
        Return the number of unique ISCC-IDs in the index.

        :return: Count of indexed assets (not simprints)
        """

    def close(self):
        """Close the index."""


class SimprintIndexMutableRaw(SimprintIndexRaw):
    """
    Extended protocol for mutable simprint indexes that support deletion and retrieval.

    Not all backends need to support mutation - immutable/append-only indexes can
    implement only the base SimprintIndexRaw protocol.
    """

    def get_raw(self, iscc_ids):
        # type: (list[bytes]) -> list[SimprintEntryRaw]
        """Retrieve entries by ISCC-IDs (for re-indexing, debugging)."""

    def delete_raw(self, iscc_ids):
        # type: (list[bytes]) -> None
        """
        Delete all entries for given ISCC-IDs.

        Transaction Semantics:
            MUST be atomic - either all entries are deleted or none are deleted.

        :param iscc_ids: List of ISCC-IDs to delete (digests without a header)
        """
