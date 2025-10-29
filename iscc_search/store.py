"""
LMDB-backed primary storage for ISCC entries and metadata.

IsccStore serves as the source of truth with two named databases:
- entries: 64-bit big-endian keys (ISCC-IDs), JSON values
- metadata: string keys, JSON values
"""

import iscc_core
import lmdb
import simdjson
from loguru import logger


class IsccStore:
    """Durable LMDB-backed storage for ISCC entries and metadata.

    Provides atomic writes, efficient lookups by ISCC-ID, and persistent metadata storage.
    Time ordering is preserved as LMDB lexicographically sorts big-endian keys.

    CONCURRENCY: LMDB supports multi-reader/single-writer with built-in locking (lock=True).
    However, when used with NphdIndex via UsearchIndex, the combined system is single-process
    only due to .usearch file limitations. See NphdIndex and UsearchIndexManager for details.
    """

    DEFAULT_LMDB_OPTIONS = {
        "readonly": False,
        "metasync": True,  # Full durability with metadata flush
        "sync": True,  # Full ACID compliance
        "mode": 0o644,  # Standard file permissions
        "create": True,  # Create directory if missing
        "readahead": False,  # Better for random access pattern
        "writemap": False,  # Safer, prevents corruption from bad writes
        "meminit": True,  # Security: zero-initialize buffers
        "map_async": False,  # Not applicable without writemap
        "max_readers": 126,  # LMDB default for concurrent reads
        "max_spare_txns": 1,  # Simple operations, minimal caching
        "lock": True,  # Enable locking for concurrent access
    }

    def __init__(self, path, realm_id=0, lmdb_options=None):
        # type: (str | os.PathLike, int, dict[str, Any] | None) -> None
        """Initialize IsccStore with LMDB environment and named databases.

        :param path: Directory path for LMDB storage
        :param realm_id: ISCC realm ID (0-1) for ISCC-ID reconstruction (default: 0)
        :param lmdb_options: Optional LMDB configuration dict (merged with defaults)
        :raises ValueError: If realm_id is not 0 or 1
        """
        # Validate realm_id
        if realm_id not in (0, 1):
            raise ValueError(f"realm_id must be 0 or 1, got {realm_id}")

        # Merge user options with defaults
        options = self.DEFAULT_LMDB_OPTIONS.copy()
        if lmdb_options:
            options.update(lmdb_options)

        # Force internal parameters that users cannot override
        options["max_dbs"] = 2
        options["subdir"] = False

        self.env = lmdb.open(str(path), **options)

        self.entries_db = self.env.open_db(b"entries", integerkey=False)
        self.metadata_db = self.env.open_db(b"metadata")

        # Initialize or restore realm_id from metadata
        stored_realm_id = self.get_metadata("__realm_id__")
        if stored_realm_id is None:
            self.realm_id = realm_id
            self.put_metadata("__realm_id__", realm_id)
        else:
            self.realm_id = stored_realm_id

    def _iscc_id_to_key(self, iscc_id):
        # type: (int | str) -> bytes
        """Convert ISCC-ID (string or int) to big-endian LMDB key bytes.

        :param iscc_id: 64-bit integer ISCC-ID or ISCC-ID string (e.g., "ISCC:...")
        :return: 8-byte key in big-endian byte order
        """
        if isinstance(iscc_id, int):
            return iscc_id.to_bytes(8, "big")
        # Decode full ISCC-ID, extract body (skip 2-byte header) - already big-endian
        return iscc_core.decode_base32(iscc_id.removeprefix("ISCC:"))[2:]

    def add(self, iscc_ids, entries):
        # type: (int | str | list[int | str], dict | list[dict]) -> int
        """Store entries with ISCC-IDs as keys.

        :param iscc_ids: ISCC-ID (int/string) or list of ISCC-IDs
        :param entries: Entry dict or list of entry dicts
        :return: Number of entries added

        Note: Automatically doubles map_size if full and retries operation.
        """
        id_list = [iscc_ids] if isinstance(iscc_ids, (int, str)) else list(iscc_ids)
        entry_list = [entries] if isinstance(entries, dict) else list(entries)

        if len(id_list) != len(entry_list):
            raise ValueError("Number of ISCC-IDs must match entries")

        # Convert all ISCC-IDs to LMDB key bytes
        items = [
            (self._iscc_id_to_key(iid), simdjson.dumps(entry).encode("utf-8"))
            for iid, entry in zip(id_list, entry_list)
        ]

        try:
            with self.env.begin(write=True, db=self.entries_db) as txn:
                cursor = txn.cursor(self.entries_db)
                _, added = cursor.putmulti(items, dupdata=False)
        except lmdb.MapFullError:
            old_size = self.map_size
            new_size = old_size * 2
            logger.info(f"IsccStore map_size increased from {old_size:,} to {new_size:,} bytes")
            self.env.set_mapsize(new_size)
            with self.env.begin(write=True, db=self.entries_db) as txn:
                cursor = txn.cursor(self.entries_db)
                _, added = cursor.putmulti(items, dupdata=False)

        return added

    def get(self, iscc_id):
        # type: (int | str) -> dict | None
        """Retrieve entry by ISCC-ID.

        :param iscc_id: ISCC-ID as integer or string (e.g., "ISCC:...")
        :return: Entry dict or None if not found
        """
        key = self._iscc_id_to_key(iscc_id)
        with self.env.begin(db=self.entries_db) as txn:
            value = txn.get(key)
        if value is None:
            return None
        return simdjson.loads(value)

    def delete(self, iscc_id):
        # type: (int | str) -> bool
        """Delete entry by ISCC-ID.

        :param iscc_id: ISCC-ID as integer or string (e.g., "ISCC:...")
        :return: True if deleted, False if not found
        """
        key = self._iscc_id_to_key(iscc_id)
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
                iscc_id = int.from_bytes(key, "big")
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
            old_size = self.map_size
            new_size = old_size * 2
            logger.info(f"IsccStore metadata map_size increased from {old_size:,} to {new_size:,} bytes")
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
        current_size = self.map_size
        if new_size < current_size:
            raise ValueError(f"Cannot shrink database: new_size ({new_size:,}) < current map_size ({current_size:,})")
        self.env.set_mapsize(new_size)

    def close(self):
        # type: () -> None
        """Close LMDB environment and release resources."""
        self.env.close()

    def __del__(self):
        # type: () -> None
        """Ensure environment is closed on object deletion."""
        if hasattr(self, "env"):
            self.env.close()
