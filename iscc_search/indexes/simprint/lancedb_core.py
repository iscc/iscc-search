"""
High-Performance LanceDB-Based Variable-Length Simprint Index

A disk-based simprint index optimized for variable-length simprints with exponential
confidence weighting and full chunk metadata support.

Key Features:
- Variable-length simprints (configurable via ndim parameter in bits)
- Full chunk metadata storage (offset, size per simprint)
- Out-of-core operation for datasets larger than RAM
- Exponential confidence weighting for soft-boundary matching
- Incremental updates via optimize() method
- Append semantics (no duplicate checking for performance)
- Native Hamming distance search via LanceDB
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING
from collections import defaultdict

import lancedb
import pyarrow as pa
import numpy as np
from loguru import logger

from iscc_search.indexes.simprint.models import MatchedChunkRaw, SimprintMatchRaw

if TYPE_CHECKING:
    from iscc_search.protocols.simprint_core import SimprintEntryRaw  # noqa: F401


class LancedbSimprintIndex:
    """
    High-performance LanceDB-based variable-length simprint index.

    This implementation uses LanceDB's disk-based storage for out-of-core operation,
    supporting datasets larger than available RAM. Uses exponential confidence weighting
    for soft-boundary matching (approximate nearest neighbors via HNSW).

    Architecture:
    - LanceDB Table: Columnar storage with PyArrow schema
    - Schema: iscc_id_body (8B) + simprint (bytes) + offset (4B) + size (4B) + vector (list[uint8])
    - Metadata: ndim and realm_id stored in table metadata
    - File Structure: LanceDB database directory with Arrow files

    Add Semantics: "append"
    - Simple append without duplicate checking
    - Users should handle deduplication externally if needed
    - Maximum performance for batch ingestion

    Scoring:
    - Exponential confidence weighting (configurable match_threshold and confidence_exponent)
    - Coverage × Quality approach
    - Score = (matches/queried) × weighted_quality
    """

    # Constants
    MAX_OFFSET = 2**32 - 1  # 4 GB max offset
    MAX_SIZE = 2**32 - 1  # 4 GB max size
    DEFAULT_MATCH_THRESHOLD = 0.0  # Accept all matches (filter at asset level instead)
    DEFAULT_CONFIDENCE_EXPONENT = 4  # Emphasize high-confidence matches
    _TABLE_NAME = "simprints"  # Fixed table name in database
    _METADATA_KEY = "index_metadata"  # Key for metadata storage

    def __init__(self, uri, ndim=None, realm_id=None, **kwargs):
        # type: (str, int | None, bytes | None, ...) -> None
        """
        Open or create a simprint index at the specified location.

        :param uri: Index location (path or file:// URI) - directory for LanceDB database
        :param ndim: Simprint dimensions in bits (e.g., 64, 128, 256).
                     If None, auto-detect from first simprint added.
        :param realm_id: ISCC-ID realm identifier (2 bytes). If None, loaded from metadata.
        :param kwargs: Backend-specific configuration options (currently unused)
        """
        # Parse URI to database path
        if uri.startswith("file://"):
            self.path = Path(uri[7:])
        elif uri.startswith("lancedb://"):
            self.path = Path(uri[10:])
        else:
            self.path = Path(uri)

        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize state
        self.ndim = ndim  # type: int | None
        self.simprint_bytes = None  # type: int | None
        self.realm_id = realm_id  # type: bytes | None
        self._row_count = None  # type: int | None
        self._num_partitions = None  # type: int | None

        # Connect to LanceDB
        self.db = lancedb.connect(str(self.path))

        # Open or create table
        self._init_table()

        # Load or initialize ndim and realm_id from metadata
        self._load_metadata()

        # If ndim provided in constructor, validate and override metadata
        if ndim is not None:
            if self.ndim is not None and self.ndim != ndim:
                raise ValueError(f"Index has ndim={self.ndim} but constructor specified ndim={ndim}")
            self.ndim = ndim
            self.simprint_bytes = ndim // 8
            self._save_metadata()

        # If realm_id provided in constructor, validate and store
        if realm_id is not None:
            if len(realm_id) != 2:
                raise ValueError(f"realm_id must be 2 bytes, got {len(realm_id)}")
            if self.realm_id is not None and self.realm_id != realm_id:
                raise ValueError(
                    f"Index has realm_id={self.realm_id.hex()} but constructor specified realm_id={realm_id.hex()}"
                )
            self.realm_id = realm_id
            self._save_metadata()

    @property
    def add_semantics(self):
        # type: () -> str
        """
        Return identifier for this index's add behavior.

        :return: "append" - Simple append without duplicate checking
        """
        return "append"

    def add_raw(self, entries):
        # type: (list[SimprintEntryRaw]) -> None
        """
        Add entries to the index with append semantics (no duplicate checking).

        Add Semantics: "append"
        - Simply appends all entries to the table
        - No duplicate checking for maximum performance
        - Users should deduplicate externally if needed

        Auto-detects ndim from first simprint if not yet configured.

        :param entries: List of entries to add
        """
        if not entries:
            return

        # Auto-detect ndim from first simprint if not configured
        if self.ndim is None:
            self._auto_detect_ndim(entries)

        # Build Arrow records from entries
        records = self._build_records(entries)

        if not records:
            return  # Nothing to add (all entries had empty simprints)

        # Create PyArrow Table
        pa_table = pa.Table.from_pylist(records, schema=self._get_schema())

        # Create table if it doesn't exist yet (lazy creation)
        if self.table is None:
            self.table = self.db.create_table(self._TABLE_NAME, pa_table)
            logger.debug(f"Created new LanceDB table: {self._TABLE_NAME}")
        else:
            # Add to existing table
            self.table.add(pa_table)

        # Invalidate cached row statistics (new data added)
        self._row_count = None
        self._num_partitions = None

    def search_raw(self, simprints, limit=10, threshold=0.0, detailed=True, **kwargs):
        # type: (list[bytes], int, float, bool, ...) -> list[SimprintMatchRaw]
        """
        Search for assets with similar simprints using exponential confidence weighting.

        Per-query configurable threshold and exponent via kwargs:
        - match_threshold: Override DEFAULT_MATCH_THRESHOLD (e.g., 0.8)
        - confidence_exponent: Override DEFAULT_CONFIDENCE_EXPONENT (e.g., 6)

        :param simprints: Binary simprints to search for
        :param limit: Maximum number of assets to return
        :param threshold: Minimum similarity score (0.0 to 1.0, default 0.0 returns all)
        :param detailed: If True, include individual chunk matches
        :return: List of matched assets ordered by similarity (limited by limit parameter)
        """
        if not simprints:
            return []

        if self.table is None or self.ndim is None:
            return []  # No data indexed yet

        # Use per-query overrides or fall back to class defaults
        match_threshold = kwargs.get("match_threshold", self.DEFAULT_MATCH_THRESHOLD)
        confidence_exponent = kwargs.get("confidence_exponent", self.DEFAULT_CONFIDENCE_EXPONENT)

        # Convert query simprints to vectors (zero-copy view over bytes)
        query_vectors = [np.frombuffer(s, dtype=np.uint8) for s in simprints]

        # Search all queries in a single batch and collect results
        asset_scores = defaultdict(list)  # type: dict[bytes, list[tuple[float, bytes, bytes, int, int]]]

        # Calculate optimal nprobes for IVF_FLAT index (search a subset of partitions).
        # Must match index creation formula: num_partitions = max(1, num_rows // 6000)
        if self._row_count is None:
            self._row_count = self.table.count_rows()
        num_rows = self._row_count
        if num_rows == 0:  # pragma: no cover
            return []  # pragma: no cover

        if self._num_partitions is None:
            self._num_partitions = max(1, num_rows // 6000)
        num_partitions = self._num_partitions

        # For small indexes, a single probe is usually sufficient and much faster.
        # For larger indexes, search a modest fraction of partitions.
        if num_partitions <= 8:
            nprobes = 1
        else:  # pragma: no cover
            # Requires 48k+ rows (num_partitions > 8) - impractical for tests
            nprobes = min(num_partitions, max(8, num_partitions // 5))  # pragma: no cover

        # Cap backend-level candidate count to avoid effectively scanning the
        # whole table for each query simprint. Over-fetch modestly relative to
        # the logical limit to preserve recall at the asset level.
        if limit <= 0:
            backend_limit = 64
        else:
            backend_limit = min(4 * limit, 256)

        # Batch all query vectors into single search call for performance
        # Stack into 2D array: shape (num_queries, ndim)
        batch_query = np.vstack(query_vectors)

        # LanceDB batch vector search with IVF_FLAT tuning
        # - distance_type: Must match index metric (use distance_type, not deprecated metric)
        # - nprobes: Number of IVF partitions to search (higher = better recall, slower)
        # - refine_factor: Over-fetch and re-rank for accuracy
        # Results include 'query_index' field mapping each result to its query vector
        try:
            results = (
                self.table.search(batch_query)
                .distance_type("hamming")
                .nprobes(nprobes)
                .refine_factor(2)
                .limit(backend_limit)
                .to_list()
            )
        except Exception:  # pragma: no cover
            # Table might be empty or index not built yet
            return []

        # Process batched results
        for result in results:
            # Map result back to original query simprint using query_index
            # Note: LanceDB only includes query_index for multi-query batches
            if len(simprints) == 1:
                query_idx = 0  # Single query - all results map to first query
            else:
                query_idx = result["query_index"]  # Multi-query - use field
            query_simprint = simprints[query_idx]

            # Extract distance and normalize to score
            distance = result.get("_distance", 0)
            score = 1.0 - (distance / self.ndim)

            # Filter by match threshold
            if score >= match_threshold:
                iscc_id_body = result["iscc_id_body"]
                match_simprint = result["simprint"]
                offset = result["offset"]
                size = result["size"]

                # Store (score, query_simprint, match_simprint, offset, size)
                asset_scores[iscc_id_body].append((score, query_simprint, match_simprint, offset, size))

        # Aggregate results per asset with exponential weighting
        results = []

        for iscc_id_body, matches in asset_scores.items():
            # Extract scores
            scores = [score for score, _, _, _, _ in matches]

            # Coverage: fraction of query simprints matched
            coverage = len(set(m[1] for m in matches)) / len(simprints)

            # Quality: exponentially weighted average
            if len(scores) == 1:
                quality = scores[0]
            else:
                weighted_sum = sum(s**confidence_exponent for s in scores)
                weight_sum = sum(scores)
                quality = weighted_sum / weight_sum if weight_sum > 0 else 0.0

            # Combined score
            final_score = coverage * quality

            if final_score >= threshold:
                # Build chunk details if requested
                chunks = None
                if detailed:
                    chunks = [
                        MatchedChunkRaw(
                            query=query_simprint,
                            match=match_simprint,
                            score=score,
                            offset=offset,
                            size=size,
                            freq=1,  # Default freq (exponential weighting doesn't need global freq)
                        )
                        for score, query_simprint, match_simprint, offset, size in matches
                    ]

                results.append(
                    SimprintMatchRaw(
                        iscc_id_body=iscc_id_body,
                        score=final_score,
                        queried=len(simprints),
                        matches=len(matches),
                        chunks=chunks,
                    )
                )

        # Sort by score descending, then by iscc_id_body for stability
        results.sort(key=lambda x: (-x.score, x.iscc_id_body))
        return results[:limit]

    def __contains__(self, iscc_id_body):
        # type: (bytes) -> bool
        """
        Check if an ISCC-ID body exists in the index.

        Note: With append semantics, may return True even if duplicates exist.

        :param iscc_id_body: Binary ISCC-ID body (8 bytes)
        :return: True if the asset has been indexed
        """
        if self.table is None:
            return False

        try:
            # Scan through table data to find matching iscc_id_body
            # Using to_arrow() to avoid pandas dependency
            arrow_table = self.table.to_arrow()
            if arrow_table.num_rows == 0:  # pragma: no cover
                return False  # pragma: no cover

            # Check if any row has this iscc_id_body
            iscc_id_column = arrow_table["iscc_id_body"]
            for i in range(arrow_table.num_rows):
                if iscc_id_column[i].as_py() == iscc_id_body:  # pragma: no branch
                    return True
            return False  # pragma: no cover
        except Exception as e:  # pragma: no cover
            logger.warning(f"Error checking contains: {e}")  # pragma: no cover
            return False  # pragma: no cover

    def __len__(self):
        # type: () -> int
        """
        Return the number of unique assets in the index.

        Counts distinct ISCC-ID bodies (not total rows, due to append semantics).

        :return: Count of unique indexed assets
        """
        if self.table is None:
            return 0

        try:
            # Use count_rows for total
            total_rows = self.table.count_rows()
            if total_rows == 0:  # pragma: no cover
                return 0  # pragma: no cover

            # Get all iscc_id_body values and count unique using set
            # Use to_arrow() instead of to_pandas() to avoid pandas dependency
            arrow_table = self.table.to_arrow()
            if arrow_table.num_rows == 0:  # pragma: no cover
                return 0  # pragma: no cover

            # Extract iscc_id_body column and count unique values
            iscc_id_column = arrow_table["iscc_id_body"]
            unique_ids = set()
            for i in range(arrow_table.num_rows):
                id_body = iscc_id_column[i].as_py()  # Get Python bytes object
                unique_ids.add(id_body)
            return len(unique_ids)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Error counting unique assets: {e}")  # pragma: no cover
            return 0  # pragma: no cover

    def close(self):
        # type: () -> None
        """Close the index and release resources."""
        # LanceDB connections don't need explicit closing
        pass

    def optimize(self):
        # type: () -> None
        """
        Optimize the index for better performance.

        Triggers LanceDB's optimization operations:
        - Compact and prune old versions
        - Build/update vector indexes for fast ANN search

        CRITICAL: Call after adding data to build vector index.
        Without this, queries use exhaustive linear search (extremely slow).
        """
        if self.table is None:  # pragma: no cover
            logger.debug("No table to optimize")  # pragma: no cover
            return  # pragma: no cover

        try:
            # Use modern optimize() API (replaces deprecated compact_files())
            self.table.optimize()
            logger.debug("Optimized LanceDB table (compacted and pruned)")

            # Create/update vector index if we have data and ndim is known
            if self.ndim is not None:  # pragma: no branch
                try:
                    num_rows = self.table.count_rows()
                    if num_rows == 0:  # pragma: no cover
                        logger.info("Skipping index creation for empty LanceDB table")  # pragma: no cover
                        return  # pragma: no cover

                    # Calculate optimal num_partitions (target 4K-8K rows per partition)
                    target_rows_per_partition = 6000  # midpoint of 4K-8K range
                    num_partitions = max(1, num_rows // target_rows_per_partition)

                    # Use IVF-FLAT for Hamming distance (IVF-PQ doesn't support Hamming)
                    # LanceDB synchronous API signature: create_index(metric, num_partitions, ...)
                    self.table.create_index(
                        metric="hamming",
                        num_partitions=num_partitions,
                        index_type="IVF_FLAT",
                        replace=True,
                    )
                    logger.info(f"Created IVF-FLAT index with {num_partitions} partitions for {num_rows:,} rows")
                except Exception as e:  # pragma: no cover
                    logger.warning(f"Vector index creation failed: {e}")  # pragma: no cover

        except Exception as e:  # pragma: no cover
            logger.warning(f"Optimization failed: {e}")  # pragma: no cover

    def __enter__(self):
        # type: () -> LancedbSimprintIndex
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # type: (type | None, Exception | None, object | None) -> None
        """Context manager cleanup."""
        self.close()

    def _init_table(self):
        # type: () -> None
        """Initialize or open the LanceDB table."""
        table_names = self.db.table_names()

        if self._TABLE_NAME in table_names:
            # Open existing table
            self.table = self.db.open_table(self._TABLE_NAME)
            logger.debug(f"Opened existing LanceDB table: {self._TABLE_NAME}")
        else:
            # Don't create table yet - will be created lazily on first add_raw
            self.table = None
            logger.debug("Table will be created on first add_raw")

    def _get_schema(self):
        # type: () -> pa.Schema
        """
        Get PyArrow schema for the table based on current ndim.

        :return: PyArrow schema
        """
        if self.simprint_bytes is None:  # pragma: no cover
            # Fallback to variable-length schema  # pragma: no cover
            return pa.schema([  # pragma: no cover
                pa.field("iscc_id_body", pa.binary(8)),  # pragma: no cover
                pa.field("simprint", pa.binary()),  # pragma: no cover
                pa.field("offset", pa.uint32()),  # pragma: no cover
                pa.field("size", pa.uint32()),  # pragma: no cover
                pa.field("vector", pa.list_(pa.uint8())),  # pragma: no cover
            ])  # pragma: no cover

        # Fixed-length schema based on ndim
        return pa.schema([
            pa.field("iscc_id_body", pa.binary(8)),
            pa.field("simprint", pa.binary(self.simprint_bytes)),
            pa.field("offset", pa.uint32()),
            pa.field("size", pa.uint32()),
            pa.field("vector", pa.list_(pa.uint8(), list_size=self.simprint_bytes)),
        ])

    def _load_metadata(self):
        # type: () -> None
        """Load metadata from metadata file or initialize empty."""
        metadata_path = self.path / f"{self._TABLE_NAME}_metadata.json"

        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    data = json.load(f)

                if "ndim" in data and data["ndim"]:  # pragma: no branch
                    self.ndim = data["ndim"]
                    self.simprint_bytes = self.ndim // 8
                    logger.debug(f"Loaded ndim={self.ndim} from metadata file")

                if "realm_id" in data and data["realm_id"]:  # pragma: no branch
                    self.realm_id = bytes.fromhex(data["realm_id"])
                    logger.debug(f"Loaded realm_id={self.realm_id.hex()} from metadata file")

            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to load metadata: {e}")  # pragma: no cover

    def _save_metadata(self):
        # type: () -> None
        """Save metadata to metadata file."""
        metadata_path = self.path / f"{self._TABLE_NAME}_metadata.json"

        data = {
            "ndim": self.ndim,
            "realm_id": self.realm_id.hex() if self.realm_id else None,
        }

        try:
            with open(metadata_path, "w") as f:
                json.dump(data, f)
            logger.debug(f"Saved metadata: ndim={self.ndim}, realm_id={self.realm_id.hex() if self.realm_id else None}")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to save metadata: {e}")  # pragma: no cover

    def _auto_detect_ndim(self, entries):
        # type: (list[SimprintEntryRaw]) -> None
        """
        Auto-detect ndim from first simprint in entries.

        :param entries: List of entries with simprints
        """
        for entry in entries:
            if entry.simprints:
                first_simprint = entry.simprints[0].simprint
                self.ndim = len(first_simprint) * 8
                self.simprint_bytes = len(first_simprint)
                self._save_metadata()
                logger.debug(f"Auto-detected ndim={self.ndim} from first simprint")
                return

        raise ValueError("Cannot auto-detect ndim: no simprints in entries")

    def _build_records(self, entries):
        # type: (list[SimprintEntryRaw]) -> list[dict]
        """
        Build Arrow records from SimprintEntryRaw entries.

        :param entries: List of entries to convert
        :return: List of dict records for PyArrow
        """
        records = []  # type: list[dict]

        for entry in entries:
            for simprint in entry.simprints:
                # Validate simprint length if ndim is configured
                if self.ndim is not None and len(simprint.simprint) != self.simprint_bytes:
                    raise ValueError(
                        f"Simprint length mismatch: expected {self.simprint_bytes} bytes "
                        f"(ndim={self.ndim}), got {len(simprint.simprint)} bytes"
                    )

                # Validate offset and size
                if simprint.offset > self.MAX_OFFSET:
                    raise ValueError(f"Offset {simprint.offset} exceeds max {self.MAX_OFFSET}")
                if simprint.size > self.MAX_SIZE:
                    raise ValueError(f"Size {simprint.size} exceeds max {self.MAX_SIZE}")

                # Convert binary simprint to vector (list of uint8)
                vector = list(simprint.simprint)

                # Create record
                records.append({
                    "iscc_id_body": entry.iscc_id_body,
                    "simprint": simprint.simprint,
                    "offset": simprint.offset,
                    "size": simprint.size,
                    "vector": vector,
                })

        return records
