"""Minimal LMDB-Based High Performance Simprint Inverted Index"""

from pathlib import Path
from collections import Counter
import lmdb


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
    """

    def __init__(self, path, simprint_type, map_size=1024**3):
        # type: (Path | str, str, int) -> None
        """
        Open or create the simprint index.

        :param path: Directory path for LMDB database
        :param simprint_type: Simprint type identifier (e.g., 'SEMANTIC_TEXT_V0')
        :param map_size: Maximum size of database in bytes (default 1GB)
        """
        self.path = Path(path)
        self.simprint_type = simprint_type
        self.path.mkdir(parents=True, exist_ok=True)

        # Open LMDB environment
        self.env = lmdb.open(
            str(self.path),
            map_size=map_size,
            max_dbs=1,
            writemap=True,  # Use writable mmap for better performance
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

        :param pairs: List of (simprint, iscc_id) tuples in native byte order (64-bit each)
        """
        # Batch insert using putmulti (moves loop to C for better performance)
        with self.env.begin(write=True, db=self.db) as txn:
            cursor = txn.cursor()
            cursor.putmulti(pairs, dupdata=False)

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

    def __enter__(self):
        # type: () -> SimprintMiniIndexRaw
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # type: (type | None, Exception | None, object | None) -> None
        """Context manager exit."""
        self.close()
