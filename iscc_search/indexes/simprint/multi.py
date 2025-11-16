"""
Backend-Agnostic Multi-Type Simprint Index Coordinator

Manages multiple backend instances (LMDB or LanceDB) with transparent type routing.
Coordinates realm_id handling and aggregates results across simprint types.
"""

from pathlib import Path
from typing import TYPE_CHECKING
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from iscc_search.indexes.simprint.models import (
    SimprintEntryRaw,
    SimprintMatchMulti,
    TypeMatchResult,
    MatchedChunkRaw,
)

if TYPE_CHECKING:
    from iscc_search.protocols.simprint_core import SimprintIndexRaw  # noqa: F401
    from iscc_search.protocols.simprint_multi import SimprintIndexMulti  # noqa: F401


class SimprintMultiIndex:
    """
    Backend-agnostic multi-type simprint index coordinator.

    Manages separate backend instances for each simprint type and coordinates
    realm_id handling, type routing, and result aggregation.

    Supported Backends:
    - "lmdb": LmdbSimprintIndex (hard-boundary, IDF scoring, single-file storage)
    - "lancedb": LancedbSimprintIndex (soft-boundary, exponential weighting, directory storage)
    - "usearch": UsearchSimprintIndex (soft-boundary, exponential weighting, single-file storage)

    Architecture:
    - Root directory contains backend-specific files: SIMPRINT_{type}{ext}
    - No coordinator metadata persistence (stateless coordinator)
    - Each sub-index stores its own realm_id and ndim in backend-specific metadata
    - Realm ID extracted from first entry's ISCC-ID header and propagated to sub-indexes
    - Sub-indexes work with 8-byte bodies only (headers stripped)
    - Results reconstruct full 10-byte ISCC-IDs using runtime realm_id
    """

    # Backend registry: (module, class, file_extension)
    BACKEND_MAP = {
        "lmdb": ("iscc_search.indexes.simprint.lmdb_core", "LmdbSimprintIndex", ".lmdb"),
        "lancedb": ("iscc_search.indexes.simprint.lancedb_core", "LancedbSimprintIndex", ""),
        "usearch": ("iscc_search.indexes.simprint.usearch_core", "UsearchSimprintIndex", ".usearch"),
    }

    def __init__(self, uri, backend="lmdb", **kwargs):
        # type: (str, str, ...) -> None
        """
        Open or create a multi-type simprint index at the specified location.

        :param uri: Index location (path or file:// URI) - directory containing backend files
        :param backend: Backend implementation ("lmdb", "lancedb", or "usearch", default "lmdb")
        :param kwargs: Backend-specific configuration options
        """
        if backend not in self.BACKEND_MAP:
            raise ValueError(f"Unknown backend: {backend}. Supported backends: {list(self.BACKEND_MAP.keys())}")

        self.backend = backend
        self.backend_kwargs = kwargs

        # Load backend class dynamically
        module_path, class_name, file_ext = self.BACKEND_MAP[backend]
        module = __import__(module_path, fromlist=[class_name])
        self.backend_class = getattr(module, class_name)  # type: type[SimprintIndexRaw]
        self.file_extension = file_ext

        # Parse URI to path
        if uri.startswith("file://"):
            self.path = Path(uri[7:])
        elif uri.startswith(f"{backend}://"):
            self.path = Path(uri[len(backend) + 3 :])
        else:
            self.path = Path(uri)

        self.path.mkdir(parents=True, exist_ok=True)

        # Runtime-only state (no persistence)
        self.realm_id = None  # type: bytes | None
        self.indexes = {}  # type: dict[str, SimprintIndexRaw]

        self._open_existing_indexes()

    def add_raw_multi(self, entries):
        # type: (list[SimprintEntryMulti]) -> None
        """
        Add multi-type entries with transparent type routing.

        Extracts realm_id from first entry, routes simprints to type-specific indexes,
        and delegates add semantics to backend implementation.

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

    def search_raw_multi(self, simprints, limit=10, threshold=0.0, detailed=True):
        # type: (dict[str, list[bytes]], int, float, bool) -> list[SimprintMatchMulti]
        """
        Search for assets with similar simprints across multiple types.

        Performs parallel searches across type-specific indexes and aggregates results
        by asset with hierarchical scoring.

        :param simprints: Binary simprints grouped by type identifier
        :param limit: Maximum number of unique assets to return
        :param threshold: Minimum similarity score (0.0 to 1.0) per type (default 0.0 returns all)
        :param detailed: If True, include individual chunk matches
        :return: List of matched assets with type-grouped results (limited by limit parameter)
        """
        if not simprints:
            return []

        # Search each type in parallel and collect results
        asset_type_matches = defaultdict(dict)  # type: dict[bytes, dict[str, TypeMatchResult]]

        # Helper function to search a single type
        def search_type(simprint_type, type_simprints):
            # type: (str, list[bytes]) -> tuple[str, list]
            """Search a single simprint type and return results."""
            index = self.indexes[simprint_type]
            matches = index.search_raw(type_simprints, limit=limit * 2, threshold=threshold, detailed=detailed)
            return simprint_type, matches

        # Execute searches in parallel using ThreadPoolExecutor
        # Filter to only include types that have indexes
        search_tasks = [
            (simprint_type, type_simprints)
            for simprint_type, type_simprints in simprints.items()
            if simprint_type in self.indexes
        ]

        # Early return if no types to search
        if not search_tasks:
            return []

        # Run searches in parallel (one thread per simprint type)
        with ThreadPoolExecutor(max_workers=len(search_tasks)) as executor:
            # Submit all search tasks
            futures = {
                executor.submit(search_type, simprint_type, type_simprints): simprint_type
                for simprint_type, type_simprints in search_tasks
            }

            # Collect results as they complete
            for future in as_completed(futures):
                simprint_type, matches = future.result()

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

    def optimize(self):
        # type: () -> None
        """
        Optimize all type-specific indexes for better performance.

        For LanceDB: Builds vector indexes for fast ANN search.
        For Usearch: Compacts index for better memory layout and performance.
        For LMDB: Currently a no-op (LMDB doesn't need optimization).

        Call after adding data to ensure best query performance.
        """
        for simprint_type, index in self.indexes.items():
            if hasattr(index, "optimize"):
                logger.debug(f"Optimizing {simprint_type} index (backend={self.backend})")
                index.optimize()

    def __enter__(self):
        # type: () -> SimprintMultiIndex
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
        # Decode realm for logging
        import iscc_core as ic

        _mt, realm, _vs, _len, _body = ic.decode_header(iscc_id)
        logger.debug(f"Extracted realm_id: {realm}")

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
        Open all existing type-specific indexes on startup via file/directory discovery.

        Scans directory for entries matching SIMPRINT_* pattern (files or directories).
        """
        if not self.path.exists():  # pragma: no cover
            return  # pragma: no cover

        for entry in self.path.iterdir():
            # Check if entry matches backend pattern
            if not entry.name.startswith("SIMPRINT_"):  # pragma: no cover
                continue  # pragma: no cover

            # For LMDB: match .lmdb files
            # For LanceDB: match directories (no extension)
            is_valid = False
            if self.file_extension:
                # File-based backend (e.g., .lmdb)
                is_valid = entry.is_file() and entry.name.endswith(self.file_extension)
                simprint_type = entry.name[9 : -len(self.file_extension)]  # Extract type from SIMPRINT_{type}{ext}
            else:
                # Directory-based backend (e.g., lancedb)
                is_valid = entry.is_dir()
                simprint_type = entry.name[9:]  # Extract type from SIMPRINT_{type}

            if not is_valid:
                continue

            try:
                # Open with realm_id=None - sub-index loads its own metadata
                index = self.backend_class(str(entry), realm_id=None, **self.backend_kwargs)
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

                logger.debug(f"Opened existing index: {simprint_type} (backend={self.backend})")
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to open index {simprint_type}: {e}")  # pragma: no cover

    def _get_or_create_index(self, simprint_type, entries=None, realm_id=None):
        # type: (str, list[SimprintEntryRaw] | None, bytes | None) -> SimprintIndexRaw
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

        # Create new index with backend-specific naming
        type_path = self.path / f"SIMPRINT_{simprint_type}{self.file_extension}"
        index = self.backend_class(str(type_path), ndim=ndim, realm_id=realm_id, **self.backend_kwargs)
        self.indexes[simprint_type] = index

        logger.debug(
            f"Created new index: {simprint_type} (backend={self.backend}, ndim={ndim}, "
            f"realm_id={realm_id.hex() if realm_id else None})"
        )

        return index

    @property
    def realm_id_int(self):
        # type: () -> int | None
        """
        Extract realm ID as integer from stored 2-byte header.

        Properly decodes the ISCC-ID header to extract realm field.

        :return: Realm ID (0 or 1) or None if not set
        """
        if self.realm_id is None:
            return None
        import iscc_core as ic

        # Pad to 10 bytes for decode_header (2-byte header + 8-byte body)
        _mt, realm, _vs, _len, _body = ic.decode_header(self.realm_id + b"\x00" * 8)
        return realm
