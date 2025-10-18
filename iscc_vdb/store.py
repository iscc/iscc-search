"""
LMDB-backed primary storage for ISCC entries and metadata.

IsccStore serves as the source of truth with two named databases:
- entries: 64-bit integer keys (ISCC-IDs), JSON values
- metadata: string keys, JSON values
"""

import os
import struct
from typing import Any, Iterator

import lmdb
import simdjson


class IsccStore:
    """Durable LMDB-backed storage for ISCC entries and metadata.

    Provides atomic writes, efficient lookups by ISCC-ID, and persistent metadata storage.
    Time ordering is preserved as LMDB sorts integer keys by value.
    """

    def __init__(self, path, realm_id=0, durable=True):
        # type: (str | os.PathLike, int, bool) -> None
        """Initialize IsccStore with LMDB environment and named databases.

        :param path: Directory path for LMDB storage
        :param realm_id: ISCC realm ID (0-1) for ISCC-ID reconstruction (default: 0)
        :param durable: If True, full persistence; if False, reduced durability for testing
        """
        self.env = lmdb.open(str(path), max_dbs=2, sync=durable, metasync=durable, lock=durable)

        self.entries_db = self.env.open_db(b"entries", integerkey=True)
        self.metadata_db = self.env.open_db(b"metadata")

        # Initialize or restore realm_id from metadata
        stored_realm_id = self.get_metadata("__realm_id__")
        if stored_realm_id is None:
            self.realm_id = realm_id
            self.put_metadata("__realm_id__", realm_id)
        else:
            self.realm_id = stored_realm_id

    def put(self, iscc_id, entry):
        # type: (int, dict) -> None
        """Store entry with ISCC-ID as key.

        :param iscc_id: 64-bit integer ISCC-ID
        :param entry: Entry dict with iscc_id, iscc_code, units keys

        Note: Automatically doubles map_size if full and retries operation.
        """
        key = struct.pack("Q", iscc_id)
        value = simdjson.dumps(entry).encode("utf-8")
        try:
            with self.env.begin(write=True, db=self.entries_db) as txn:
                txn.put(key, value)
        except lmdb.MapFullError:
            new_size = self.map_size * 2
            self.env.set_mapsize(new_size)
            with self.env.begin(write=True, db=self.entries_db) as txn:
                txn.put(key, value)

    def get(self, iscc_id):
        # type: (int) -> dict | None
        """Retrieve entry by ISCC-ID.

        :param iscc_id: 64-bit integer ISCC-ID
        :return: Entry dict or None if not found
        """
        key = struct.pack("Q", iscc_id)
        with self.env.begin(db=self.entries_db) as txn:
            value = txn.get(key)
        if value is None:
            return None
        return simdjson.loads(value)

    def delete(self, iscc_id):
        # type: (int) -> bool
        """Delete entry by ISCC-ID.

        :param iscc_id: 64-bit integer ISCC-ID
        :return: True if deleted, False if not found
        """
        key = struct.pack("Q", iscc_id)
        with self.env.begin(write=True, db=self.entries_db) as txn:
            return txn.delete(key)

    def iter_entries(self):
        # type: () -> Iterator[tuple[int, dict]]
        """Iterate all entries in ascending ISCC-ID order.

        Yields tuples of (iscc_id_int, entry_dict).
        """
        with self.env.begin(db=self.entries_db) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                iscc_id = struct.unpack("Q", key)[0]
                entry = simdjson.loads(value)
                yield iscc_id, entry

    def get_metadata(self, key):
        # type: (str) -> Any
        """Retrieve metadata value by key.

        :param key: Metadata key string
        :return: Metadata value or None if not found
        """
        key_bytes = key.encode("utf-8")
        with self.env.begin(db=self.metadata_db) as txn:
            value = txn.get(key_bytes)
        if value is None:
            return None
        return simdjson.loads(value)

    def put_metadata(self, key, value):
        # type: (str, Any) -> None
        """Store metadata key-value pair.

        :param key: Metadata key string
        :param value: Metadata value (must be JSON-serializable)

        Note: Automatically doubles map_size if full and retries operation.
        """
        key_bytes = key.encode("utf-8")
        value_bytes = simdjson.dumps(value).encode("utf-8")
        try:
            with self.env.begin(write=True, db=self.metadata_db) as txn:
                txn.put(key_bytes, value_bytes)
        except lmdb.MapFullError:
            new_size = self.map_size * 2
            self.env.set_mapsize(new_size)
            with self.env.begin(write=True, db=self.metadata_db) as txn:
                txn.put(key_bytes, value_bytes)

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

    def close(self):
        # type: () -> None
        """Close LMDB environment and release resources."""
        self.env.close()
