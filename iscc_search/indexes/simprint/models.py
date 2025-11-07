"""
Msgspec models for simprint search results.

These concrete implementations of the protocol types provide efficient serialization
and clean type checking for simprint index implementations.
"""

import msgspec


class MatchedChunkRaw(msgspec.Struct):
    """Individual chunk match result from a simprint search."""

    query: bytes  # Binary simprint from the search query
    match: bytes  # Binary simprint from the index that matched
    score: float  # Similarity score (0.0 to 1.0) at chunk level
    offset: int  # Byte offset of matched chunk in the asset
    size: int  # Size in bytes of the matched chunk
    freq: int  # Document frequency (number of assets containing this simprint)


class SimprintMatchRaw(msgspec.Struct):
    """Asset-level match result aggregating multiple simprint matches."""

    iscc_id_body: bytes  # Binary ISCC-ID body of the matched asset (without header)
    score: float  # Aggregated similarity score (0.0 to 1.0) at asset level
    queried: int  # Number of simprints in the query
    matches: int  # Number of query simprints that found matches
    chunks: list[MatchedChunkRaw] | None  # Individual chunk matches (None when detailed=False)
