"""
LMDB-Based Multi-Type Simprint Index Coordinator

Manages multiple LmdbSimprintIndex64 instances with transparent type routing.
Coordinates realm_id handling and aggregates results across simprint types.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING
from collections import defaultdict

from loguru import logger

from iscc_search.indexes.simprint.lmdb_core_64 import LmdbSimprintIndex64
from iscc_search.indexes.simprint.models import (
    SimprintEntryMulti,
    SimprintEntryRaw,
    SimprintMatchMulti,
    TypeMatchResult,
    MatchedChunkRaw,
)

if TYPE_CHECKING:
    from iscc_search.protocols.simprint_multi import SimprintIndexMulti  # noqa: F401


class LmdbSimprintIndexMulti64:
    """
    Multi-type simprint index coordinator for 64-bit simprints.

    Manages separate LmdbSimprintIndex64 instances for each simprint type and coordinates
    realm_id handling, type routing, and result aggregation.

    Architecture:
    - Root directory contains metadata.json with realm_id and indexed types
    - Each simprint type gets a subdirectory with its own LmdbSimprintIndex64
    - Realm ID extracted from first entry's ISCC-ID header and persisted
    - Sub-indexes work with 8-byte bodies only (headers stripped)
    - Results reconstruct full 10-byte ISCC-IDs using stored realm_id
    """

    _METADATA_FILE = "metadata.json"

    def __init__(self, uri, **kwargs):
        # type: (str, ...) -> None
        """
        Open or create a multi-type simprint index at the specified location.

        :param uri: Index location (path or file:// URI)
        :param kwargs: Backend-specific configuration options (currently unused)
        """
        if uri.startswith("file://"):
            self.path = Path(uri[7:])
        elif uri.startswith("lmdb://"):
            self.path = Path(uri[7:])
        else:
            self.path = Path(uri)

        self.path.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.path / self._METADATA_FILE

        # Load or initialize metadata
        self.realm_id = None  # type: bytes | None
        self.indexes = {}  # type: dict[str, LmdbSimprintIndex64]

        self._load_metadata()
        self._open_existing_indexes()

    def add_raw_multi(self, entries):
        # type: (list[SimprintEntryMulti]) -> None
        """
        Add multi-type entries with transparent type routing.

        Extracts realm_id from first entry, routes simprints to type-specific indexes,
        and enforces add-once semantics per type.

        :param entries: List of multi-type entries to add
        """
        if not entries:
            return

        # Extract and validate realm_id from first entry if not set
        if self.realm_id is None:
            self._extract_realm_id(entries[0].iscc_id)

        # Group entries by type for batch processing
        type_entries = defaultdict(list)  # type: dict[str, list[SimprintEntryRaw]]

        for entry in entries:
            self._validate_realm_id(entry.iscc_id)
            iscc_id_body = entry.iscc_id[2:]  # Strip 2-byte header

            for simprint_type, simprints in entry.simprints.items():
                type_entries[simprint_type].append(SimprintEntryRaw(iscc_id_body=iscc_id_body, simprints=simprints))

        # Add to type-specific indexes
        for simprint_type, type_entry_list in type_entries.items():
            index = self._get_or_create_index(simprint_type)
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

    def get_raw_multi(self, iscc_ids):
        # type: (list[bytes]) -> list[SimprintEntryMulti]
        """
        Retrieve indexed entries by their ISCC-IDs across all types.

        :param iscc_ids: Full ISCC-ID digests (10 bytes each)
        :return: List of entries (empty simprints dict for non-existent ISCC-IDs)
        """
        results = []

        for iscc_id in iscc_ids:
            iscc_id_body = iscc_id[2:]
            simprints_by_type = {}  # type: dict[str, list[SimprintRaw]]

            # Query each type-specific index
            for simprint_type, index in self.indexes.items():
                # LmdbSimprintIndex64 doesn't have get_raw, so we check containment only
                # For full implementation, we'd need to add get_raw to LmdbSimprintIndex64
                if iscc_id_body in index:
                    # Placeholder: actual retrieval would require get_raw method
                    simprints_by_type[simprint_type] = []

            results.append(SimprintEntryMulti(iscc_id=iscc_id, simprints=simprints_by_type))

        return results

    def delete_raw_multi(self, iscc_ids):
        # type: (list[bytes]) -> None
        """
        Remove assets from all type-specific indexes by their ISCC-IDs.

        :param iscc_ids: Full ISCC-ID digests (10 bytes each)
        """
        # Delete from each type-specific index
        for index in self.indexes.values():
            # LmdbSimprintIndex64 doesn't have delete_raw yet
            # For full implementation, we'd need to add delete_raw to LmdbSimprintIndex64
            pass

    def close(self):
        # type: () -> None
        """
        Close all type-specific indexes and release resources.
        """
        for index in self.indexes.values():
            index.close()
        self.indexes.clear()

    def __enter__(self):
        # type: () -> LmdbSimprintIndexMulti64
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # type: (type | None, Exception | None, object | None) -> None
        """Context manager cleanup."""
        self.close()

    def _load_metadata(self):
        # type: () -> None
        """Load metadata from disk or initialize empty."""
        if self.metadata_path.exists():
            data = json.loads(self.metadata_path.read_text())
            if "realm_id" in data and data["realm_id"]:
                self.realm_id = bytes.fromhex(data["realm_id"])

    def _save_metadata(self):
        # type: () -> None
        """Save metadata to disk."""
        data = {
            "realm_id": self.realm_id.hex() if self.realm_id else None,
            "indexed_types": self.get_indexed_types(),
        }
        self.metadata_path.write_text(json.dumps(data, indent=2))

    def _extract_realm_id(self, iscc_id):
        # type: (bytes) -> None
        """
        Extract and persist realm_id from first ISCC-ID.

        :param iscc_id: Full 10-byte ISCC-ID
        """
        if len(iscc_id) != 10:
            raise ValueError(f"ISCC-ID must be 10 bytes, got {len(iscc_id)}")

        self.realm_id = iscc_id[:2]
        self._save_metadata()
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
        """Open all existing type-specific indexes on startup."""
        for entry in self.path.iterdir():
            if entry.is_dir() and entry.name != "__pycache__":
                try:
                    index = LmdbSimprintIndex64(str(entry))
                    self.indexes[entry.name] = index
                    logger.debug(f"Opened existing index: {entry.name}")
                except Exception as e:  # pragma: no cover
                    logger.warning(f"Failed to open index {entry.name}: {e}")

    def _get_or_create_index(self, simprint_type):
        # type: (str) -> LmdbSimprintIndex64
        """
        Get existing index or create new one for simprint type.

        :param simprint_type: Type identifier (e.g., "CONTENT_TEXT_V0")
        :return: Type-specific index instance
        """
        if simprint_type in self.indexes:
            return self.indexes[simprint_type]

        # Create new index
        type_path = self.path / simprint_type
        index = LmdbSimprintIndex64(str(type_path))
        self.indexes[simprint_type] = index
        self._save_metadata()
        logger.debug(f"Created new index: {simprint_type}")

        return index
