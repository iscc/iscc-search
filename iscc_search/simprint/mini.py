"""Minimal LMDB-Based High Performance Simprint Inverted Index"""

from pathlib import Path
from collections import Counter
import lmdb
from loguru import logger


class SimprintMiniIndexRaw:
    """
    Minimal space-efficient LMDB Simprint Index without simprint positional metadata.

    Simprints stored as 64-bit integer keys mapping to ISCC-IDs stored as 64-bit integer values.
    Uses LMDB dupsort for efficient inverted index: one simprint key -> multiple ISCC-ID values.

    Features:
    - Single simprint type per index instance
    - Fixed 64-bit simprints and ISCC-IDs for optimal LMDB performance
    - Append-only (no deletion support)
    - Ranked search results by match count
    - Prevents duplicate (simprint, iscc_id) pairs for accurate match counts
    - Automatic map_size expansion on MapFullError
    """

    # MapFullError retry limits
    MAX_RESIZE_RETRIES = 10
    MAX_MAP_SIZE = 1024 * 1024 * 1024 * 1024  # 1 TB

    def __init__(self, path, simprint_type):
        # type: (Path | str, str) -> None
        """
        Open or create the simprint index.

        Uses LMDB default map_size (10MB) with automatic expansion on demand.

        :param path: Directory path for LMDB database
        :param simprint_type: Simprint type identifier (e.g., 'SEMANTIC_TEXT_V0')
        """
        self.path = Path(path)
        self.simprint_type = simprint_type
        self.path.mkdir(parents=True, exist_ok=True)

        # Open LMDB environment with default map_size (10MB)
        # writemap=True and map_async=True avoid Windows file reservation issue
        self.env = lmdb.open(
            str(self.path),
            max_dbs=1,
            writemap=True,  # Use writable mmap, avoids Windows file reservation
            map_async=True,  # Asynchronous flushing for better write performance
        )

        # Open database with optimal flags for fixed-size integer keys and values
        self.db = self.env.open_db(
            self.simprint_type.encode("utf-8"),
            integerkey=True,  # 64-bit integer keys (simprints)
            integerdup=True,  # 64-bit integer values (ISCC-IDs) - implies dupsort and dupfixed
        )

    def add(self, pairs):
        # type: (list[tuple[bytes, bytes]]) -> None
        """
        Add simprint-to-ISCC-ID pairs to the index.

        Duplicate pairs are automatically skipped to ensure accurate match counts.
        Automatically resizes map_size if MapFullError occurs.

        :param pairs: List of (simprint, iscc_id) tuples in native byte order (64-bit each)
        """
        retry_count = 0

        while retry_count <= self.MAX_RESIZE_RETRIES:  # pragma: no branch
            try:
                # Batch insert using putmulti (moves loop to C for better performance)
                with self.env.begin(write=True, db=self.db) as txn:
                    cursor = txn.cursor()
                    cursor.putmulti(pairs, dupdata=False)
                break  # Success

            except lmdb.MapFullError:  # pragma: no cover
                retry_count += 1

                # Check if we've exceeded retry limit
                if retry_count > self.MAX_RESIZE_RETRIES:
                    raise RuntimeError(
                        f"Failed to add pairs after {self.MAX_RESIZE_RETRIES} resize attempts. "
                        f"Current map_size: {self.map_size:,} bytes. "
                        f"This may indicate disk space issues, permissions problems, or filesystem limits."
                    )

                old_size = self.map_size
                new_size = old_size * 2

                # Check if new size would exceed maximum
                if new_size > self.MAX_MAP_SIZE:
                    raise RuntimeError(
                        f"Cannot resize LMDB map beyond MAX_MAP_SIZE ({self.MAX_MAP_SIZE:,} bytes). "
                        f"Current size: {old_size:,}, attempted size: {new_size:,}. "
                        f"Consider splitting data across multiple indexes or increasing MAX_MAP_SIZE."
                    )

                logger.info(
                    f"SimprintMiniIndexRaw map_size increased from {old_size:,} to {new_size:,} bytes "
                    f"(retry {retry_count}/{self.MAX_RESIZE_RETRIES})"
                )
                self.env.set_mapsize(new_size)

    def search(self, simprints):
        # type: (list[bytes]) -> list[tuple[bytes, int]]
        """
        Search the index for matching items.

        Returns ISCC-IDs ranked by number of matching simprints (collision count).
        ISCC-IDs with more matching simprints are ranked higher.

        :param simprints: List of 64-bit simprint digests to search for (native byte order)
        :return: List of (iscc_id, match_count) tuples sorted by match_count descending
        """
        # Counter to aggregate match counts per ISCC-ID
        iscc_id_counter = Counter()  # type: Counter[bytes]

        with self.env.begin(db=self.db) as txn:
            cursor = txn.cursor()

            # Retrieve all ISCC-IDs for all simprints in one optimized call
            # dupdata=True: get all duplicate values for each key
            # dupfixed_bytes=8: values are 64-bit integers (8 bytes)
            results = cursor.getmulti(simprints, dupdata=True, dupfixed_bytes=8)

            # Count ISCC-ID occurrences
            for _, iscc_id in results:
                iscc_id_counter[iscc_id] += 1

        # Sort by match count descending, then by ISCC-ID for stable ordering
        return sorted(
            iscc_id_counter.items(),
            key=lambda x: (-x[1], x[0]),  # Descending count, ascending ID
        )

    def close(self):
        # type: () -> None
        """Close the index and flush to disk."""
        self.env.sync()  # Ensure all data is flushed
        self.env.close()

    @property
    def map_size(self):
        # type: () -> int
        """Get current LMDB map_size in bytes."""
        return self.env.info()["map_size"]

    def __enter__(self):
        # type: () -> SimprintMiniIndexRaw
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # type: (type | None, Exception | None, object | None) -> None
        """Context manager exit."""
        self.close()
