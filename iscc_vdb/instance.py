"""
An index for variable length ISCC Instance-Codes.

Uses LMDB with dupsort/dupfixed/integerdup for efficient storage of multiple 8-byte ISCC-IDs per Instance-Code.

Storage format:
- Key: instance_code digest (variable-length bytes)
- Values: 8-byte iscc_id digests (stored as sorted duplicates)

LMDB configuration:
- dupsort=True: Allows multiple values per key with O(log n) duplicate checking
- dupfixed=True: All values are fixed 8-byte size (optimized storage without per-value headers)
- integerdup=True: Values treated as 8-byte integers (optimized sorting/comparison)
"""

import os
from typing import Protocol
import lmdb
import iscc_core as ic


IsccIds = str | list[str] | bytes | list[bytes]
InstanceCodes = str | list[str] | bytes | list[bytes]

# ISCC-ID header constants for realm_id 0 and 1
ISCC_ID_HEADER_REALM_0 = ic.encode_header(ic.MT.ID, 0, ic.VS.V1, 0)
ISCC_ID_HEADER_REALM_1 = ic.encode_header(ic.MT.ID, 1, ic.VS.V1, 0)


class PInstanceIndex(Protocol):
    """
    Maps variable length Instance-Codes to ISCC-IDs.
    """

    path: os.PathLike
    realm_id: int

    def add(self, iscc_ids, instance_codes):
        # type: (IsccIds, InstanceCodes) -> int
        """
        Add ISCC-ID under key Instance-Code.

        :param iscc_ids: ISCC-ID string or digest or list of ISCC-ID strings or digests
        :param instance_codes: Instance-Code string or digest or list of Instance-Code strings or digests
        :return: Number of new keys added
        """

    def get(self, instance_codes):
        # type: (InstanceCodes) -> list[str]
        """
        Return all ISCC-IDS for a given instance_code (exact match).

        :param instance_codes: Instance-Code string or digest or list of Instance-Code strings or digests
        :return: List of matching ISCC-IDs or empty list if not found.
        """

    def search(self, instance_codes):
        # type: (InstanceCodes) -> dict[str, list[str]]
        """
        Return ISCC-IDs mapped to their matching Instance-Codes.

        :param instance_codes: Instance-Code string or digest or list of Instance-Code strings or digests
        :return: Dict with ISCC-IDs as keys and lists of matched Instance-Codes as values.
        """


class InstanceIndex:
    """LMDB-backed index for Instance-Code prefix search with dupsort/dupfixed optimization."""

    def __init__(self, path, realm_id=0, durable=False, readahead=False):
        # type: (os.PathLike, int, bool, bool) -> None
        """Create or open LMDB instance index.

        :param path: Directory path for LMDB environment
        :param realm_id: ISCC realm ID for ISCC-ID reconstruction (default 0)
        :param durable: If True, flush to disk on commit (slower, ACID compliant).
                        If False, defer flushes for performance (default, maintains ACI).
        :param readahead: If True, enable OS readahead (better for sequential access).
                          If False, disable readahead (better for random access, default).
        """
        self.path = os.fspath(path)
        self.realm_id = realm_id
        os.makedirs(self.path, exist_ok=True)

        self.env = lmdb.open(
            self.path,
            max_dbs=1,
            writemap=True,
            metasync=False,
            sync=durable,
            readahead=readahead,
            max_spare_txns=16,
        )

        # Open named database with dupsort/dupfixed for 8-byte ISCC-IDs
        with self.env.begin(write=True) as txn:
            self.db = self.env.open_db(
                b"instance",
                txn=txn,
                dupsort=True,
                dupfixed=True,
                integerdup=True,
            )

    def add(self, iscc_ids, instance_codes):
        # type: (IsccIds, InstanceCodes) -> int
        """Add ISCC-ID(s) under Instance-Code key(s).

        :param iscc_ids: ISCC-ID string/bytes or list
        :param instance_codes: Instance-Code string/bytes or list
        :return: Number of new mappings added

        Note: Automatically doubles map_size if full and retries operation.
        """
        id_list = self._normalize_to_bytes_list(iscc_ids)
        ic_list = self._normalize_to_bytes_list(instance_codes)

        if len(id_list) != len(ic_list):
            raise ValueError("Number of ISCC-IDs must match Instance-Codes")

        try:
            with self.env.begin(write=True) as txn:
                cursor = txn.cursor(self.db)
                # Build (key, value) tuples: (instance_code, iscc_id)
                items = list(zip(ic_list, id_list))
                # putmulti returns (consumed, added) - dupdata=False prevents duplicates
                _, added = cursor.putmulti(items, dupdata=False)
        except lmdb.MapFullError:
            new_size = self.map_size * 2
            self.env.set_mapsize(new_size)
            with self.env.begin(write=True) as txn:
                cursor = txn.cursor(self.db)
                items = list(zip(ic_list, id_list))
                _, added = cursor.putmulti(items, dupdata=False)

        return added

    def get(self, instance_codes):
        # type: (InstanceCodes) -> list[str]
        """Return all ISCC-IDs for exact instance_code match(es).

        :param instance_codes: Instance-Code string/bytes or list
        :return: List of ISCC-ID strings
        """
        ic_list = self._normalize_to_bytes_list(instance_codes)

        result_ids = set()
        with self.env.begin() as txn:
            cursor = txn.cursor(self.db)
            # Use getmulti for optimized batch retrieval of fixed-size duplicates
            for _, value in cursor.getmulti(ic_list, dupdata=True, dupfixed_bytes=8):
                result_ids.add(value)

        return [self._bytes_to_iscc_id(iid) for iid in sorted(result_ids)]

    def search(self, instance_codes, bidirectional=True):
        # type: (InstanceCodes, bool) -> dict[str, list[str]]
        """Return ISCC-IDs mapped to their matching Instance-Codes.

        ISCC-IDs with longer matches appear first in the dict.
        For each ISCC-ID, Instance-Codes are sorted by length (longest first).

        :param instance_codes: Instance-Code prefix(es) to search
        :param bidirectional: If True, also match shorter stored codes (default: True)
        :return: Dict mapping ISCC-ID to list of matching Instance-Codes
        """
        ic_list = self._normalize_to_bytes_list(instance_codes)
        temp_results = {}  # type: dict[str, set[str]]  # IC -> ISCC-IDs

        with self.env.begin() as txn:
            cursor = txn.cursor(self.db)

            for search_code in ic_list:
                search_len = len(search_code)

                # Search with full length (forward search + exact match)
                self._search_prefix(cursor, search_code, temp_results)

                if bidirectional:
                    # Then check shorter prefixes (128-bit, then 64-bit)
                    if search_len == 32:  # 256-bit search
                        self._search_prefix(cursor, search_code[:16], temp_results)

                    if search_len >= 16:  # 128-bit or 256-bit search
                        self._search_prefix(cursor, search_code[:8], temp_results)

        # Invert: ISCC-ID -> [(length, Instance-Code), ...]
        inverted = {}  # type: dict[str, list[tuple[int, str]]]
        for ic_str, iscc_ids in temp_results.items():
            ic_bytes_len = len(self._to_bytes(ic_str))
            for iscc_id in iscc_ids:
                if iscc_id not in inverted:
                    inverted[iscc_id] = []
                inverted[iscc_id].append((ic_bytes_len, ic_str))

        # Sort each ISCC-ID's matches by length (longest first)
        for iscc_id in inverted:
            inverted[iscc_id].sort(reverse=True)

        # Build final dict ordered by longest match length
        results = {}  # type: dict[str, list[str]]
        for iscc_id in sorted(inverted.keys(), key=lambda x: inverted[x][0][0], reverse=True):
            results[iscc_id] = [ic for _, ic in inverted[iscc_id]]

        return results

    def remove_by_iscc_id(self, iscc_ids):
        # type: (IsccIds) -> int
        """Remove mappings by ISCC-ID(s).

        :param iscc_ids: ISCC-ID(s) to remove
        :return: Number of mappings removed
        """
        id_list = self._normalize_to_bytes_list(iscc_ids)
        id_set = set(id_list)

        count = 0
        with self.env.begin(write=True) as txn:
            cursor = txn.cursor(self.db)

            # Collect entries to delete
            to_delete = []
            for key, value in cursor:
                if value in id_set:
                    to_delete.append((key, value))

            # Delete collected entries
            for key, value in to_delete:
                cursor.set_key_dup(key, value)
                cursor.delete()
                count += 1

        return count

    def remove_by_instance_code(self, instance_codes):
        # type: (InstanceCodes) -> int
        """Remove mappings by Instance-Code(s).

        :param instance_codes: Instance-Code(s) to remove
        :return: Number of ISCC-IDs removed
        """
        ic_list = self._normalize_to_bytes_list(instance_codes)

        count = 0
        with self.env.begin(write=True) as txn:
            cursor = txn.cursor(self.db)
            for ic in ic_list:
                if cursor.set_key(ic):
                    # Count all duplicates for this key
                    for _ in cursor.iternext_dup():
                        count += 1

                    # Delete all duplicates
                    cursor.set_key(ic)
                    cursor.delete(dupdata=True)

        return count

    def __len__(self):
        # type: () -> int
        """Return total number of mappings."""
        count = 0
        with self.env.begin() as txn:
            cursor = txn.cursor(self.db)
            # With dupsort, each duplicate is counted separately
            for _ in cursor:
                count += 1
        return count

    def close(self):
        # type: () -> None
        """Close LMDB environment."""
        self.env.close()

    @property
    def map_size(self):
        # type: () -> int
        """Get current map_size from LMDB environment."""
        return self.env.info()["map_size"]

    def set_mapsize(self, new_size):
        # type: (int) -> None
        """Increase the maximum size the database may grow to.

        :param new_size: New maximum size in bytes (must be larger than current size)
        :raises lmdb.Error: If active transactions exist in current process
        :raises ValueError: If new_size would shrink the database

        Note: Must be called when no transactions are active. Only increases persist.
        """
        self.env.set_mapsize(new_size)

    def __del__(self):
        # type: () -> None
        """Cleanup on deletion."""
        if hasattr(self, "env"):
            self.env.close()

    # Helper methods

    @staticmethod
    def _to_bytes(code):
        # type: (str | bytes) -> bytes
        """Convert ISCC code to bytes digest."""
        if isinstance(code, bytes):
            return code
        decoded = ic.decode_base32(code.removeprefix("ISCC:"))
        return ic.decode_header(decoded)[4]

    @staticmethod
    def _normalize_to_bytes_list(codes):
        # type: (str | bytes | list[str] | list[bytes]) -> list[bytes]
        """Normalize input to list of bytes digests."""
        if isinstance(codes, (str, bytes)):
            return [InstanceIndex._to_bytes(codes)]
        return [InstanceIndex._to_bytes(c) for c in codes]

    def _bytes_to_iscc_id(self, digest):
        # type: (bytes) -> str
        """Convert ISCC-ID digest back to string."""
        if self.realm_id == 0:
            header = ISCC_ID_HEADER_REALM_0
        elif self.realm_id == 1:
            header = ISCC_ID_HEADER_REALM_1
        else:
            raise ValueError(f"Invalid realm_id {self.realm_id}, must be 0 or 1")
        return "ISCC:" + ic.encode_base32(header + digest)

    def _bytes_to_instance_code(self, digest):
        # type: (bytes) -> str
        """Convert Instance-Code digest back to string."""
        bit_length = len(digest) * 8
        header = ic.encode_header(ic.MT.INSTANCE, ic.ST.NONE, ic.VS.V0, bit_length)
        return "ISCC:" + ic.encode_base32(header + digest)

    def _search_prefix(self, cursor, prefix, results):
        # type: (lmdb.Cursor, bytes, dict[str, set[str]]) -> None
        """Search for all entries with matching prefix and accumulate results.

        :param cursor: LMDB cursor for iteration
        :param prefix: Prefix bytes to search for
        :param results: Dict to accumulate Instance-Code -> set of ISCC-IDs
        """
        if not cursor.set_range(prefix):
            return

        # With dupsort, iteration yields each duplicate separately
        for key, value in cursor:
            if not key.startswith(prefix):
                break

            ic_str = self._bytes_to_instance_code(key)
            if ic_str not in results:
                results[ic_str] = set()
            results[ic_str].add(self._bytes_to_iscc_id(value))
