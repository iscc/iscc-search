"""
LMDB-Based Multi-Type Simprint Index Coordinator

Manages multiple LmdbSimprintIndex instances with transparent type routing.
Coordinates realm_id handling and aggregates results across simprint types.
"""

from pathlib import Path
from typing import TYPE_CHECKING
from collections import defaultdict

from loguru import logger

from iscc_search.indexes.simprint.lmdb_core import LmdbSimprintIndex
from iscc_search.indexes.simprint.models import (
    SimprintEntryRaw,
    SimprintMatchMulti,
    TypeMatchResult,
    MatchedChunkRaw,
)

if TYPE_CHECKING:
    from iscc_search.protocols.simprint_multi import SimprintIndexMulti  # noqa: F401


class LmdbSimprintIndexMulti:
    """
    Multi-type simprint index coordinator for variable-length simprints.

    Manages separate LmdbSimprintIndex instances for each simprint type and coordinates
    realm_id handling, type routing, and result aggregation.

    Architecture:
    - Root directory contains flat LMDB files: SIMPRINT_{type}.lmdb
    - No coordinator metadata persistence (stateless coordinator)
    - Each sub-index stores its own realm_id and ndim in LMDB metadata
    - Realm ID extracted from first entry's ISCC-ID header and propagated to sub-indexes
    - Sub-indexes work with 8-byte bodies only (headers stripped)
    - Results reconstruct full 10-byte ISCC-IDs using runtime realm_id
    """

    def __init__(self, uri, **kwargs):
        # type: (str, ...) -> None
        """
        Open or create a multi-type simprint index at the specified location.

        :param uri: Index location (path or file:// URI) - directory containing flat .lmdb files
        :param kwargs: Backend-specific configuration options (currently unused)
        """
        if uri.startswith("file://"):
            self.path = Path(uri[7:])
        elif uri.startswith("lmdb://"):
            self.path = Path(uri[7:])
        else:
            self.path = Path(uri)

        self.path.mkdir(parents=True, exist_ok=True)

        # Runtime-only state (no persistence)
        self.realm_id = None  # type: bytes | None
        self.indexes = {}  # type: dict[str, LmdbSimprintIndex]

        self._open_existing_indexes()

    def add_raw_multi(self, entries):
        # type: (list[SimprintEntryMulti]) -> None
        """
        Add multi-type entries with transparent type routing.

        Extracts realm_id from first entry, routes simprints to type-specific indexes,
        and enforces add-once semantics per type.

        Auto-detects ndim for each type from first simprints added.

        :param entries: List of multi-type entries to add
        """
        if not entries:
            return

        # Extract and validate realm_id from first entry if not set
        if self.realm_id is None:
            self._extract_realm_id(entries[0].iscc_id)

        # Batch validate all realm_ids upfront (fail fast before processing)
        for entry in entries:
            self._validate_realm_id(entry.iscc_id)

        # Group entries by type for batch processing
        type_entries = defaultdict(list)  # type: dict[str, list[SimprintEntryRaw]]

        for entry in entries:
            iscc_id_body = entry.iscc_id[2:]  # Strip 2-byte header

            for simprint_type, simprints in entry.simprints.items():
                type_entries[simprint_type].append(SimprintEntryRaw(iscc_id_body=iscc_id_body, simprints=simprints))

        # Add to type-specific indexes (passing realm_id when creating new ones)
        for simprint_type, type_entry_list in type_entries.items():
            index = self._get_or_create_index(simprint_type, type_entry_list, self.realm_id)
            index.add_raw(type_entry_list)

    def search_raw_multi(self, simprints, limit=10, threshold=0.8, detailed=True):
        # type: (dict[str, list[bytes]], int, float, bool) -> list[SimprintMatchMulti]
        """
        Search for assets with similar simprints across multiple types.

        Performs parallel searches across type-specific indexes and aggregates results
        by asset with hierarchical scoring.

        :param simprints: Binary simprints grouped by type identifier
        :param limit: Maximum number of unique assets to return
        :param threshold: Minimum similarity score (0.0 to 1.0) per type
        :param detailed: If True, include individual chunk matches
        :return: List of matched assets with type-grouped results
        """
        if not simprints:
            return []

        # Search each type and collect results
        asset_type_matches = defaultdict(dict)  # type: dict[bytes, dict[str, TypeMatchResult]]

        for simprint_type, type_simprints in simprints.items():
            if simprint_type not in self.indexes:
                continue

            index = self.indexes[simprint_type]
            matches = index.search_raw(type_simprints, limit=limit * 2, threshold=threshold, detailed=detailed)

            for match in matches:
                # Convert single-type match to TypeMatchResult
                chunks = None
                if detailed and match.chunks:
                    chunks = [
                        MatchedChunkRaw(
                            query=chunk.query,
                            match=chunk.match,
                            score=chunk.score,
                            offset=chunk.offset,
                            size=chunk.size,
                            freq=chunk.freq,
                        )
                        for chunk in match.chunks
                    ]

                type_result = TypeMatchResult(
                    score=match.score,
                    queried=match.queried,
                    matches=match.matches,
                    chunks=chunks,
                )

                asset_type_matches[match.iscc_id_body][simprint_type] = type_result

        # Aggregate results per asset
        results = []
        for iscc_id_body, type_results in asset_type_matches.items():
            # Calculate asset-level score as mean of type scores
            asset_score = sum(tr.score for tr in type_results.values()) / len(type_results)

            # Reconstruct full ISCC-ID
            iscc_id = self.realm_id + iscc_id_body if self.realm_id else iscc_id_body

            results.append(
                SimprintMatchMulti(
                    iscc_id=iscc_id,
                    score=asset_score,
                    types=type_results,
                )
            )

        # Sort by score and apply limit
        results.sort(key=lambda x: (-x.score, x.iscc_id))
        return results[:limit]

    def get_indexed_types(self):
        # type: () -> list[str]
        """
        Get list of simprint types that have been indexed.

        :return: List of simprint type identifiers (sorted for consistency)
        """
        return sorted(self.indexes.keys())

    def __contains__(self, iscc_id):
        # type: (bytes) -> bool
        """
        Check if an ISCC-ID exists in any type-specific sub-index.

        :param iscc_id: Full ISCC-ID digest (10 bytes: 2-byte header + 8-byte body)
        :return: True if the asset exists in any type-specific index
        """
        if len(iscc_id) != 10:
            return False

        iscc_id_body = iscc_id[2:]
        return any(iscc_id_body in index for index in self.indexes.values())

    def close(self):
        # type: () -> None
        """
        Close all type-specific indexes and release resources.
        """
        for index in self.indexes.values():
            index.close()
        self.indexes.clear()

    def __enter__(self):
        # type: () -> LmdbSimprintIndexMulti
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # type: (type | None, Exception | None, object | None) -> None
        """Context manager cleanup."""
        self.close()

    def _extract_realm_id(self, iscc_id):
        # type: (bytes) -> None
        """
        Extract realm_id from first ISCC-ID (runtime only, no persistence).

        :param iscc_id: Full 10-byte ISCC-ID
        """
        if len(iscc_id) != 10:
            raise ValueError(f"ISCC-ID must be 10 bytes, got {len(iscc_id)}")

        self.realm_id = iscc_id[:2]
        logger.debug(f"Extracted realm_id: {self.realm_id.hex()}")

    def _validate_realm_id(self, iscc_id):
        # type: (bytes) -> None
        """
        Validate that ISCC-ID matches stored realm_id.

        :param iscc_id: Full 10-byte ISCC-ID
        """
        if self.realm_id is None:
            return

        if len(iscc_id) != 10:
            raise ValueError(f"ISCC-ID must be 10 bytes, got {len(iscc_id)}")

        if iscc_id[:2] != self.realm_id:
            raise ValueError(f"ISCC-ID realm mismatch: expected {self.realm_id.hex()}, got {iscc_id[:2].hex()}")

    def _open_existing_indexes(self):
        # type: () -> None
        """
        Open all existing type-specific indexes on startup via flat file discovery.

        Scans directory for files matching SIMPRINT_*.lmdb pattern.
        """
        for entry in self.path.iterdir():
            if entry.is_file() and entry.name.startswith("SIMPRINT_") and entry.name.endswith(".lmdb"):
                simprint_type = entry.name[9:-5]  # Extract type from SIMPRINT_{type}.lmdb
                try:
                    # Open with realm_id=None - sub-index loads its own metadata
                    index = LmdbSimprintIndex(str(entry), realm_id=None)
                    self.indexes[simprint_type] = index

                    # Extract realm_id from first sub-index if not yet set
                    if self.realm_id is None and index.realm_id is not None:
                        self.realm_id = index.realm_id
                        logger.debug(f"Loaded realm_id={self.realm_id.hex()} from sub-index {simprint_type}")

                    # Validate realm_id consistency across sub-indexes
                    if index.realm_id is not None and self.realm_id != index.realm_id:  # pragma: no cover
                        raise ValueError(  # pragma: no cover
                            f"Realm ID mismatch: coordinator has {self.realm_id.hex()}, "
                            f"but sub-index {simprint_type} has {index.realm_id.hex()}"
                        )

                    logger.debug(f"Opened existing index: {simprint_type}")
                except Exception as e:  # pragma: no cover
                    logger.warning(f"Failed to open index {simprint_type}: {e}")  # pragma: no cover

    def _get_or_create_index(self, simprint_type, entries=None, realm_id=None):
        # type: (str, list[SimprintEntryRaw] | None, bytes | None) -> LmdbSimprintIndex
        """
        Get existing index or create new one for simprint type.

        :param simprint_type: Type identifier (e.g., "CONTENT_TEXT_V0")
        :param entries: Optional entries to detect ndim from (for auto-detection)
        :param realm_id: Optional realm_id to pass to new sub-index
        :return: Type-specific index instance
        """
        if simprint_type in self.indexes:
            return self.indexes[simprint_type]

        # Determine ndim for new index by auto-detection from entries
        ndim = None
        if entries:
            for entry in entries:
                if entry.simprints:
                    ndim = len(entry.simprints[0].simprint) * 8
                    logger.debug(f"Auto-detected ndim={ndim} for type {simprint_type}")
                    break

        # Create new index with flat file naming: SIMPRINT_{type}.lmdb
        type_path = self.path / f"SIMPRINT_{simprint_type}.lmdb"
        index = LmdbSimprintIndex(str(type_path), ndim=ndim, realm_id=realm_id)
        self.indexes[simprint_type] = index

        logger.debug(
            f"Created new index: {simprint_type} (ndim={ndim}, realm_id={realm_id.hex() if realm_id else None})"
        )

        return index
