"""Minimal LMDB-Based High Performance Simprint Inverted Index"""

import struct
from pathlib import Path
from typing import Protocol
from collections import Counter
import lmdb


class SimprintMiniItem(Protocol):
    """Protocol for items to be indexed."""

    iscc_id: bytes  # 64-bit ISCC-ID body (realm needs to be tracked at index level)
    simprints: list[bytes]  # List of 64-bit simprint digests


class SimprintMiniResult:
    """Search result with ranked ISCC-IDs by match count."""

    def __init__(self, matches):
        # type: (list[tuple[bytes, int]]) -> None
        """
        Initialize search result.

        :param matches: List of (iscc_id, match_count) tuples sorted by match_count descending
        """
        self.matches = matches

    def __repr__(self):
        # type: () -> str
        return f"SimprintMiniResult(matches={len(self.matches)})"


class SimprintMiniIndex:
    """
    Minimal space-efficient LMDB Simprint Index without simprint positional metadata.

    Simprints stored as 64-bit integer keys mapping to ISCC-IDs stored as 64-bit integer values.
    Uses LMDB dupsort for efficient inverted index: one simprint key -> multiple ISCC-ID values.

    Features:
    - Single simprint type per index instance
    - Fixed 64-bit simprints and ISCC-IDs for optimal LMDB performance
    - Append-only (no deletion support)
    - Ranked search results by match count
    - Allows duplicate (simprint, iscc_id) pairs for faster writes
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

    def add(self, items):
        # type: (list[SimprintMiniItem]) -> None
        """
        Add items to the index.

        Each simprint from each item is added as a key with the item's ISCC-ID as value.
        Duplicates are allowed for faster writes (same simprint-iscc_id pair can exist multiple times).

        :param items: List of items with iscc_id and simprints to index
        """
        with self.env.begin(write=True, db=self.db) as txn:
            for item in items:
                # Convert ISCC-ID once per item
                iscc_id_int = struct.unpack("<Q", item.iscc_id)[0]
                iscc_id_bytes = struct.pack("=Q", iscc_id_int)  # Native byte order for LMDB

                # Add each simprint with this ISCC-ID
                for simprint in item.simprints:
                    simprint_int = struct.unpack("<Q", simprint)[0]
                    simprint_bytes = struct.pack("=Q", simprint_int)  # Native byte order for LMDB

                    # dupdata=True allows duplicates (default behavior)
                    txn.put(simprint_bytes, iscc_id_bytes, dupdata=True)

    def search(self, simprints):
        # type: (list[bytes]) -> SimprintMiniResult
        """
        Search the index for matching items.

        Returns ISCC-IDs ranked by number of matching simprints (collision count).
        ISCC-IDs with more matching simprints are ranked higher.

        :param simprints: List of 64-bit simprint digests to search for
        :return: Search result with ranked matches
        """
        # Counter to aggregate match counts per ISCC-ID
        iscc_id_counter = Counter()  # type: Counter[bytes]

        with self.env.begin(db=self.db) as txn:
            cursor = txn.cursor()

            # Look up each simprint and collect all matching ISCC-IDs
            for simprint in simprints:
                simprint_int = struct.unpack("<Q", simprint)[0]
                simprint_bytes = struct.pack("=Q", simprint_int)

                # Position cursor at this simprint key
                if cursor.set_key(simprint_bytes):
                    # Iterate through all ISCC-IDs for this simprint
                    for iscc_id_bytes in cursor.iternext_dup(keys=False, values=True):
                        # Convert back to little-endian for consistency
                        iscc_id_int = struct.unpack("=Q", iscc_id_bytes)[0]
                        iscc_id = struct.pack("<Q", iscc_id_int)
                        iscc_id_counter[iscc_id] += 1

        # Sort by match count descending, then by ISCC-ID for stable ordering
        ranked_matches = sorted(
            iscc_id_counter.items(),
            key=lambda x: (-x[1], x[0]),  # Descending count, ascending ID
        )

        return SimprintMiniResult(matches=ranked_matches)

    def close(self):
        # type: () -> None
        """Close the index and flush to disk."""
        self.env.sync()  # Ensure all data is flushed
        self.env.close()

    def __enter__(self):
        # type: () -> SimprintMiniIndex
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # type: (type | None, Exception | None, object | None) -> None
        """Context manager exit."""
        self.close()
