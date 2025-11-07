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


class SimprintRaw(msgspec.Struct):
    """Raw simprint representation for a content chunk."""

    simprint: bytes  # Binary simprint digest of the chunk
    offset: int  # Byte offset where chunk starts in the asset
    size: int  # Size in bytes of the chunk


class SimprintEntryRaw(msgspec.Struct):
    """Entry containing all simprints of a specific type for a single asset."""

    iscc_id_body: bytes  # Binary ISCC-ID body (8 bytes, without header)
    simprints: list[SimprintRaw]  # List of simprints with location metadata


class SimprintEntryMulti(msgspec.Struct):
    """Entry containing simprints of multiple types for a single asset."""

    iscc_id: bytes  # Full ISCC-ID digest (10 bytes: 2-byte header + 8-byte body)
    simprints: dict[str, list[SimprintRaw]]  # Simprints grouped by type


class TypeMatchResult(msgspec.Struct):
    """Search results for a specific simprint type within an asset."""

    score: float  # Aggregated similarity score (0.0 to 1.0) for this type
    queried: int  # Number of simprints queried for this type
    matches: int  # Number of query simprints that found matches
    chunks: list[MatchedChunkRaw] | None  # Individual chunk matches (None when detailed=False)


class SimprintMatchMulti(msgspec.Struct):
    """Multi-type asset match result with type-grouped simprint matches."""

    iscc_id: bytes  # Full ISCC-ID digest (10 bytes: 2-byte header + 8-byte body)
    score: float  # Overall aggregated similarity score (0.0 to 1.0) across all types
    types: dict[str, TypeMatchResult]  # Results grouped by simprint type
