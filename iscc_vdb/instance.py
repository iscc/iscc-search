"""
An index for variable length ISCC Instance-Codes.

Basic Datastructure

{
<instance_code>: set(<iscc_id>, <iscc_id>, ...),
...
}

Where:
- <instance_code> is a variable length digest (Body of ISCC Instance-Code).
- <iscc_id> is an 8-byte digest of the ISCC-ID body.

InstanceIndex methods accept multiple representations of ISCC-IDs and Instance-Codes:

ISCC-ID as string - decode: iscc_core.decode_base32(iscc_id.removeprefix("ISCC:"))[2:]
ISCC-ID as bytes - decode:  iscc_id

Instance-Code as string - decode: iscc_core.decode_base32(iscc_code.removeprefix("ISCC:"))[2:]
Instance-Code as bytes - decode: iscc_code

For return values reconstruct ISCC-ID based on Realm-ID

Lists of ISCC-IDs and Instance-Codes are also supported for batch operations.
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
    """LMDB-backed index for Instance-Code prefix search.

    Storage format:
    - Key: instance_code digest (bytes)
    - Value: concatenated 8-byte iscc_id digests (id1 + id2 + id3...)

    Keys are sorted lexicographically for efficient prefix iteration.
    ISCC-IDs are fixed 8-byte digests, stored sequentially without separators.
    """

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

        # Open LMDB environment
        self.env = lmdb.open(
            self.path,
            map_size=map_size,
            max_dbs=0,
            writemap=True,  # Use writable mmap (faster on Windows)
            metasync=False,  # Sync metadata less frequently
            sync=True,  # But do sync data
        )

    def add(self, iscc_ids, instance_codes):
        # type: (IsccIds, InstanceCodes) -> int
        """Add ISCC-ID(s) under Instance-Code key(s).

        :param iscc_ids: ISCC-ID string/bytes or list
        :param instance_codes: Instance-Code string/bytes or list
        :return: Number of new mappings added
        """
        # Normalize to lists of bytes
        id_list = self._normalize_to_bytes_list(iscc_ids, is_iscc_id=True)
        ic_list = self._normalize_to_bytes_list(instance_codes, is_iscc_id=False)

        if len(id_list) != len(ic_list):
            raise ValueError("Number of ISCC-IDs must match Instance-Codes")

        count = 0
        with self.env.begin(write=True) as txn:
            for iscc_id, instance_code in zip(id_list, ic_list):
                # Get existing IDs for this instance code
                existing = txn.get(instance_code)

                if existing:
                    # Parse existing IDs (8 bytes each)
                    existing_ids = [existing[i : i + 8] for i in range(0, len(existing), 8)]
                    if iscc_id not in existing_ids:
                        # Append new ID
                        txn.put(instance_code, existing + iscc_id)
                        count += 1
                else:
                    # New instance code
                    txn.put(instance_code, iscc_id)
                    count += 1

        return count

    def get(self, instance_codes):
        # type: (InstanceCodes) -> list[str]
        """Return all ISCC-IDs for exact instance_code match(es).

        :param instance_codes: Instance-Code string/bytes or list
        :return: List of ISCC-ID strings
        """
        # Normalize to list
        ic_list = self._normalize_to_bytes_list(instance_codes, is_iscc_id=False)

        result_ids = set()
        with self.env.begin() as txn:
            for ic_bytes in ic_list:
                value = txn.get(ic_bytes)
                if value:
                    # Parse stored ISCC-IDs (8 bytes each)
                    id_bytes_list = [value[i : i + 8] for i in range(0, len(value), 8)]
                    result_ids.update(id_bytes_list)

        return [self._bytes_to_iscc_id(iid) for iid in sorted(result_ids)]

    def search(self, instance_codes):
        # type: (InstanceCodes) -> dict[str, list[str]]
        """Return all ISCC-IDs for instance codes with matching prefix(es).

        :param instance_codes: Instance-Code prefix(es) to search
        :return: Dict mapping Instance-Code to list of ISCC-IDs
        """
        # Normalize to list
        ic_list = self._normalize_to_bytes_list(instance_codes, is_iscc_id=False)

        results = {}  # type: dict[str, list[str]]

        with self.env.begin() as txn:
            cursor = txn.cursor()

            for prefix in ic_list:
                # Position cursor at prefix start
                if not cursor.set_range(prefix):
                    continue

                # Iterate while keys match prefix
                for key, value in cursor:
                    if not key.startswith(prefix):
                        break

                    # Convert instance code back to string
                    ic_str = self._bytes_to_instance_code(key)

                    # Parse ISCC-IDs from value (8 bytes each)
                    id_bytes_list = [value[i : i + 8] for i in range(0, len(value), 8)]
                    iscc_ids = [self._bytes_to_iscc_id(iid) for iid in id_bytes_list]

                    results[ic_str] = iscc_ids

        return results

    def remove_by_iscc_id(self, iscc_ids):
        # type: (IsccIds) -> int
        """Remove mappings by ISCC-ID(s).

        :param iscc_ids: ISCC-ID(s) to remove
        :return: Number of mappings removed
        """
        id_list = self._normalize_to_bytes_list(iscc_ids, is_iscc_id=True)
        id_set = set(id_list)

        count = 0
        with self.env.begin(write=True) as txn:
            cursor = txn.cursor()
            keys_to_delete = []

            # Scan all entries
            for key, value in cursor:
                # Parse ISCC-IDs (8 bytes each)
                id_bytes_list = [value[i : i + 8] for i in range(0, len(value), 8)]
                # Filter out matching IDs
                remaining = [iid for iid in id_bytes_list if iid not in id_set]

                if len(remaining) < len(id_bytes_list):
                    count += len(id_bytes_list) - len(remaining)
                    if remaining:
                        # Update with remaining IDs (concatenate)
                        txn.put(key, b"".join(remaining))
                    else:
                        # No IDs left, mark for deletion
                        keys_to_delete.append(key)

            # Delete empty entries
            for key in keys_to_delete:
                txn.delete(key)

        return count

    def remove_by_instance_code(self, instance_codes):
        # type: (InstanceCodes) -> int
        """Remove mappings by Instance-Code(s).

        :param instance_codes: Instance-Code(s) to remove
        :return: Number of ISCC-IDs removed
        """
        ic_list = self._normalize_to_bytes_list(instance_codes, is_iscc_id=False)

        count = 0
        with self.env.begin(write=True) as txn:
            for ic in ic_list:
                value = txn.get(ic)
                if value:
                    # Count IDs before deleting (8 bytes each)
                    count += len(value) // 8
                    txn.delete(ic)

        return count

    def __len__(self):
        # type: () -> int
        """Return total number of mappings."""
        count = 0
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                # Each ISCC-ID is 8 bytes
                count += len(value) // 8
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
    def _to_bytes(code, is_iscc_id):
        # type: (str | bytes, bool) -> bytes
        """Convert ISCC code to bytes digest."""
        if isinstance(code, bytes):
            return code

        # Decode ISCC string and extract digest (body)
        decoded = ic.decode_base32(code.removeprefix("ISCC:"))
        # Use iscc_core to properly decode header (variable length 2-8 bytes)
        _mt, _st, _vs, _ln, body = ic.decode_header(decoded)
        return body

    @staticmethod
    def _normalize_to_bytes_list(codes, is_iscc_id):
        # type: (str | bytes | list[str] | list[bytes], bool) -> list[bytes]
        """Normalize input to list of bytes digests."""
        if isinstance(codes, (str, bytes)):
            return [InstanceIndex._to_bytes(codes, is_iscc_id)]
        return [InstanceIndex._to_bytes(c, is_iscc_id) for c in codes]

    def _bytes_to_iscc_id(self, digest):
        # type: (bytes) -> str
        """Convert ISCC-ID digest back to string."""
        # Use pre-computed header constants
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
        # Calculate bit length from digest size
        bit_length = len(digest) * 8
        # Reconstruct with header
        header = ic.encode_header(ic.MT.INSTANCE, ic.ST.NONE, ic.VS.V0, bit_length)
        return "ISCC:" + ic.encode_base32(header + digest)
