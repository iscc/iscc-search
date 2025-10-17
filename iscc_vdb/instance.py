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
        Return all ISCC-IDS for all instance_codes with a matching prefix.

        :param instance_codes: Instance-Code string or digest or list of Instance-Code strings or digests
        :return: Dict with ISCC-IDs as keys and sets of matched ISCC-CODEs.
        """


class InstanceIndex:
    """LMDB-backed index for Instance-Code prefix search with dupsort/dupfixed optimization."""

    def __init__(self, path, realm_id=0, map_size=10 * 1024 * 1024 * 1024):
        # type: (os.PathLike, int, int) -> None
        """Create or open LMDB instance index.

        :param path: Directory path for LMDB environment
        :param realm_id: ISCC realm ID for ISCC-ID reconstruction (default 0)
        :param map_size: Maximum size in bytes (default 10GB)
        """
        self.path = os.fspath(path)
        self.realm_id = realm_id
        os.makedirs(self.path, exist_ok=True)

        self.env = lmdb.open(
            self.path,
            map_size=map_size,
            max_dbs=1,
            writemap=True,
            metasync=False,
            sync=True,
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
        """
        id_list = self._normalize_to_bytes_list(iscc_ids)
        ic_list = self._normalize_to_bytes_list(instance_codes)

        if len(id_list) != len(ic_list):
            raise ValueError("Number of ISCC-IDs must match Instance-Codes")

        count = 0
        with self.env.begin(write=True) as txn:
            cursor = txn.cursor(self.db)
            for iscc_id, instance_code in zip(id_list, ic_list):
                # dupdata=False prevents duplicates, returns False if exists
                if cursor.put(instance_code, iscc_id, dupdata=False):
                    count += 1

        return count

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

    def search(self, instance_codes):
        # type: (InstanceCodes) -> dict[str, list[str]]
        """Return all ISCC-IDs for instance codes with matching prefix(es).

        :param instance_codes: Instance-Code prefix(es) to search
        :return: Dict mapping Instance-Code to list of ISCC-IDs
        """
        ic_list = self._normalize_to_bytes_list(instance_codes)

        results = {}  # type: dict[str, list[str]]

        with self.env.begin() as txn:
            cursor = txn.cursor(self.db)

            for prefix in ic_list:
                if not cursor.set_range(prefix):
                    continue

                # With dupsort, iteration yields each duplicate separately
                for key, value in cursor:
                    if not key.startswith(prefix):
                        break

                    ic_str = self._bytes_to_instance_code(key)
                    if ic_str not in results:
                        results[ic_str] = []
                    results[ic_str].append(self._bytes_to_iscc_id(value))

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
