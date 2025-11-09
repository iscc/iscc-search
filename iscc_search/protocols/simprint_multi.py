"""
Multi Simprint-Type Indexing Protocol

This module defines the protocol for implementing backend-agnostic simprint with transparent
support for multiple simprint types.
"""

from typing import Protocol
from iscc_search.protocols.simprint_core import SimprintRaw, MatchedChunkRaw


class SimprintEntryMulti(Protocol):
    """
    Entry containing simprints of potentially multiple types for a single asset.

    This is the decoded form compatible with IsccQuery, where simprints are
    grouped by type. The type is identified by the dictionary key, avoiding
    redundant storage of type information per simprint.
    """

    iscc_id: bytes  # Full ISCC-ID digest (10 bytes: 2-byte header + 8-byte body)
    simprints: dict[str, list[SimprintRaw]]  # Simprints grouped by type (key = type identifier)


class TypeMatchResult(Protocol):
    """
    Search results for a specific simprint type within an asset.

    Aggregates all matches of a particular type (e.g., CONTENT_TEXT_V0) with
    type-specific scoring and match statistics.
    """

    score: float  # Aggregated similarity score (0.0 to 1.0) for this type
    queried: int  # Number of simprints queried for this type
    matches: int  # Number of query simprints that found matches for this type
    chunks: list[MatchedChunkRaw] | None  # Individual chunk matches (None when detailed=False)


class SimprintMatchMulti(Protocol):
    """
    Multi-type asset match result with type-grouped simprint matches.

    Represents a single asset that matched the query based on one or more simprint
    types. Results are organized hierarchically: asset → type → chunks.

    Score Calculation:
        - Asset-level score: Aggregated across all matching types
        - Type-level score: Aggregated across all chunks of that type
        - Chunk-level score: Individual simprint similarity
        Implementations choose aggregation strategy (mean, max, weighted, etc.)
    """

    iscc_id: bytes  # Full ISCC-ID digest (10 bytes: 2-byte header + 8-byte body)
    score: float  # Overall aggregated similarity score (0.0 to 1.0) across all types
    types: dict[str, TypeMatchResult]  # Results grouped by simprint type


class SimprintIndexMulti(Protocol):
    """
    Multi-type simprint index protocol with transparent type routing.

    This protocol extends the base SimprintIndexRaw to handle multiple simprint
    types transparently. It manages type-specific sub-indexes internally and
    routes operations based on simprint type identifiers.

    Implementation Notes:
        - Maintains separate indexes for each simprint type
        - Automatically creates new type-specific indexes as needed
        - Handles realm_id for multi-tenant scenarios (future enhancement)
        - Accepts entries in decoded IsccQuery-compatible format
    """

    def __init__(self, uri, **kwargs):
        # type: (str, ...) -> None
        """
        Open or create a multi-type simprint index at the specified location.

        Note: Multi-type indexes manage ndim per simprint type (auto-detected).
        realm_id is extracted from first entry and validated across all entries.

        :param uri: Index location as URI:
          - File path: '/path/to/index' or 'file:///path/to/index'
          - LMDB: 'lmdb:///path/to/index'
          - PostgreSQL: 'postgresql://user:pass@host:5432/dbname'
          - Future: 'redis://host:6379/0', 's3://bucket/prefix', etc.
        :param kwargs: Backend-specific configuration options
        """

    def add_raw_multi(self, entries):
        # type: (list[SimprintEntryMulti]) -> None
        """
        Add multi-type entries to the index with transparent type routing.

        Accepts entries in decoded IsccQuery-compatible format where simprints
        are grouped by type. Each entry can contain simprints of multiple types
        which are automatically routed to appropriate type-specific sub-indexes.

        Realm ID Management:
            - Extracts realm_id from first entry's ISCC-ID header
            - Persists realm_id and validates all subsequent entries
            - Sub-indexes receive only 8-byte bodies (header stripped)
            - Reconstructs full ISCC-IDs in results using stored realm_id

        Type Management:
            - Creates type-specific indexes on first use
            - Routes simprints to correct index based on dictionary key
            - Maintains type metadata (dimensions per type)

        Add-Once Semantics:
            - Each ISCC-ID can only be added ONCE per simprint type
            - Duplicate ISCC-IDs within a type are silently ignored
            - Same ISCC-ID can exist with different simprint types

        Transaction Semantics:
            - Atomic per type - all simprints of a type are added or none
            - Cross-type atomicity depends on backend capabilities

        Example Entry Structure:
            {
                'iscc_id': b'\\x00\\x10\\x12\\x34\\x56\\x78\\x9a\\xbc\\xde\\xf0',  # 10 bytes (header + body)
                'simprints': {
                    'CONTENT_TEXT_V0': [
                        {'simprint': b'...', 'offset': 0, 'size': 512},
                        {'simprint': b'...', 'offset': 512, 'size': 489}
                    ],
                    'SEMANTIC_TEXT_V0': [
                        {'simprint': b'...', 'offset': 0, 'size': 1024},
                        {'simprint': b'...', 'offset': 1024, 'size': 876}
                    ]
                }
            }

        :param entries: List of multi-type entries to add
        """

    def search_raw_multi(self, simprints, limit=10, threshold=0.8, detailed=True):
        # type: (dict[str, list[bytes]], int, float, bool) -> list[SimprintMatchMulti]
        """
        Search for assets with similar simprints across multiple types.

        Performs parallel searches across type-specific indexes and aggregates results
        into a unified ranking. Each simprint type is searched independently, then
        results are merged by asset with hierarchical scoring.

        Query Structure:
            Simprints grouped by type, matching IsccQuery format:
            {
                'CONTENT_TEXT_V0': [b'...', b'...'],
                'SEMANTIC_TEXT_V0': [b'...', b'...', b'...']
            }

        Result Structure:
            Hierarchical results with scores at multiple levels:
            - Asset level: Overall similarity across all types
            - Type level: Similarity for specific type
            - Chunk level: Individual simprint matches (if detailed=True)

        Scoring Strategy:
            - Implementations define aggregation (mean, max, weighted)
            - Type-level scores aggregate chunk scores
            - Asset-level scores aggregate type scores
            - Can weight types differently (e.g., semantic > content)

        Deduplication:
            - Same asset may match via multiple types
            - Results are deduplicated by iscc_id
            - All matching types included in single result

        Example Return Structure:
            [
                {
                    'iscc_id': b'\\x00\\x10\\x12\\x34\\x56\\x78\\x9a\\xbc\\xde\\xf0',  # 10 bytes
                    'score': 0.923,  # Overall score across all types
                    'types': {
                        'CONTENT_TEXT_V0': {
                            'score': 0.954,  # Type-specific score
                            'queried': 2,    # Query had 2 simprints of this type
                            'matches': 2,    # Both found matches
                            'chunks': [      # Individual matches (if detailed=True)
                                {
                                    'query': b'\\x01\\x7b\\xee...',  # Query simprint
                                    'match': b'\\x01\\x7b\\xee...',  # Matching simprint in index
                                    'score': 1.0,                    # Perfect match
                                    'offset': 0,
                                    'size': 512
                                },
                                {
                                    'query': b'\\x07\\x89\\x25...',
                                    'match': b'\\x07\\x89\\x24...',  # Near match
                                    'score': 0.908,
                                    'offset': 512,
                                    'size': 489
                                }
                            ]
                        },
                        'SEMANTIC_TEXT_V0': {
                            'score': 0.892,
                            'queried': 3,
                            'matches': 2,    # Only 2 of 3 found matches
                            'chunks': [...]  # Omitted for brevity
                        }
                    }
                },
                {
                    'iscc_id': b'\\x00\\x10\\xfe\\xdc\\xba\\x98\\x76\\x54\\x32\\x10',  # 10 bytes
                    'score': 0.871,
                    'types': {
                        'CONTENT_TEXT_V0': {
                            'score': 0.871,
                            'queried': 2,
                            'matches': 1,
                            'chunks': None  # When detailed=False
                        }
                    }
                }
            ]

        :param simprints: Binary simprints grouped by type identifier
        :param limit: Maximum number of unique assets to return
        :param threshold: Minimum similarity score (0.0 to 1.0) per type
        :param detailed: If True, include individual chunk matches
        :return: List of matched assets with type-grouped results
        """

    def get_indexed_types(self):
        # type: () -> list[str]
        """
        Get list of simprint types that have been indexed.

        Returns all simprint type identifiers (e.g., "CONTENT_TEXT_V0", "SEMANTIC_TEXT_V0")
        for which at least one simprint has been indexed. Useful for discovery and
        understanding what types of searches are available.

        :return: List of simprint type identifiers (sorted for consistency)
        """

    def __contains__(self, iscc_id):
        # type: (bytes) -> bool
        """
        Check if an ISCC-ID exists in any type-specific sub-index.

        Returns True if the asset has been indexed with at least one simprint type.
        An asset may exist in multiple type-specific indexes simultaneously.

        Enables pythonic membership testing: `if iscc_id in multi_index:`

        Implementation Note:
            This requires checking each type-specific sub-index until a match
            is found (short-circuit on first match for efficiency).

        :param iscc_id: Full ISCC-ID digest (10 bytes: 2-byte header + 8-byte body)
        :return: True if the asset exists in any type-specific index
        """

    def close(self):
        """
        Close the index and release resources.

        Closes all type-specific sub-indexes and releases their resources.
        Flushes pending writes and releases file handles/connections.
        The index should not be used after closing.
        """


class SimprintIndexMutableMulti(SimprintIndexMulti):
    """
    Extended protocol for mutable multi-type simprint indexes that support deletion and retrieval.

    Not all backends need to support mutation - immutable/append-only indexes can
    implement only the base SimprintIndexMulti protocol.
    """

    def get_raw_multi(self, iscc_ids):
        # type: (list[bytes]) -> list[SimprintEntryMulti]
        """
        Retrieve indexed entries by their ISCC-IDs across all types.

        Returns stored simprints for specified assets, grouped by type.
        Useful for "more-like-this" queries where the query asset is already
        indexed, for debugging, or for implementing update operations.

        Missing ISCC-IDs return empty entries (iscc_id set, empty simprints dict).

        :param iscc_ids: Full ISCC-ID digests (10 bytes each)
        :return: List of entries (empty simprints dict for non-existent ISCC-IDs)
        """

    def delete_raw_multi(self, iscc_ids):
        # type: (list[bytes]) -> None
        """
        Remove assets from all type-specific indexes by their ISCC-IDs.

        Completely removes all simprints (across all types) for the specified assets.
        Non-existent ISCC-IDs are silently ignored.

        Transaction Semantics:
            Should be atomic where possible - either all entries are deleted or none.

        :param iscc_ids: Full ISCC-ID digests (10 bytes each)
        """
