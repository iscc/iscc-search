"""
Extended LMDB-Based Simprint Index with Chunk Metadata

Stores simprint-to-ISCC-ID mappings with chunk location metadata (offset, size) for
collision analysis and text retrieval. Uses fixed 16-byte ChunkPointer values for
efficient LMDB dupfixed storage.
"""

from pathlib import Path
from collections import Counter
from typing import TYPE_CHECKING
import lmdb
from loguru import logger
import iscc_core as ic

from iscc_search.models import IsccID

if TYPE_CHECKING:
    pass  # noqa: F401


class ChunkPointer:
    """
    Fixed 16-byte chunk pointer for LMDB dupfixed storage.

    Binary layout: iscc_id_body (8) + offset (4) + size (4) = 16 bytes

    Limits:
    - Max offset: 4,294,967,295 bytes (4 GB)
    - Max size: 4,294,967,295 bytes (4 GB)
    """

    MAX_OFFSET = 4_294_967_295  # 2^32 - 1
    MAX_SIZE = 4_294_967_295  # 2^32 - 1

    def __init__(self, iscc_id_body, offset, size):
        # type: (bytes, int, int) -> None
        """
        Create a chunk pointer.

        :param iscc_id_body: 8-byte ISCC-ID body (without header)
        :param offset: Byte offset in source file (max 4 GB)
        :param size: Chunk size in bytes (max 4 GB)
        """
        self.iscc_id_body = iscc_id_body
        self.offset = offset
        self.size = size

    def to_bytes(self):
        # type: () -> bytes
        """Serialize to 16-byte binary format using native byte order."""
        import sys

        if self.offset > self.MAX_OFFSET:
            raise ValueError(f"Offset {self.offset} exceeds max {self.MAX_OFFSET}")
        if self.size > self.MAX_SIZE:
            raise ValueError(f"Size {self.size} exceeds max {self.MAX_SIZE}")

        # Use native byte order to match LMDB's integer handling
        return self.iscc_id_body + self.offset.to_bytes(4, sys.byteorder) + self.size.to_bytes(4, sys.byteorder)

    @classmethod
    def from_bytes(cls, data):
        # type: (bytes) -> ChunkPointer
        """Deserialize from 16-byte binary format using native byte order."""
        import sys

        if len(data) != 16:
            raise ValueError(f"ChunkPointer requires exactly 16 bytes, got {len(data)}")

        iscc_id_body = data[:8]
        # Use native byte order to match LMDB's integer handling
        offset = int.from_bytes(data[8:12], sys.byteorder)
        size = int.from_bytes(data[12:16], sys.byteorder)

        # Sanity check: detect corrupted data
        if offset > cls.MAX_OFFSET or size > cls.MAX_SIZE:
            from loguru import logger

            logger.error(
                f"Corrupted ChunkPointer detected! offset={offset:,} (max={cls.MAX_OFFSET:,}), "
                f"size={size:,} (max={cls.MAX_SIZE:,}). "
                f"Raw bytes: {data.hex()}"
            )
            # This might indicate database was created with wrong byte order

        return cls(iscc_id_body=iscc_id_body, offset=offset, size=size)


class SimprintIndexRaw:
    """
    LMDB Simprint Index with chunk metadata storage.

    Simprints stored as binary keys (64, 128, 192, or 256 bits) mapping to ChunkPointer values (16 bytes).
    Uses LMDB dupfixed for efficient inverted index: one simprint key -> multiple ChunkPointer values.

    Features:
    - Single simprint type per index instance
    - Configurable simprint key size (64, 128, 192, or 256 bits) and fixed 16-byte ChunkPointer values
    - Append-only (no deletion support)
    - Ranked search results by match count
    - Prevents duplicate (simprint, chunk_pointer) pairs for accurate match counts
    - Automatic map_size expansion on MapFullError
    - Stores chunk location metadata for collision analysis
    """

    # MapFullError retry limits
    MAX_RESIZE_RETRIES = 10
    MAX_MAP_SIZE = 1024 * 1024 * 1024 * 1024  # 1 TB

    def __init__(self, path, simprint_type, bits=64):
        # type: (Path | str, str, int) -> None
        """
        Open or create the simprint index.

        Uses LMDB default map_size (10MB) with automatic expansion on demand.

        :param path: Directory path for LMDB database
        :param simprint_type: Simprint type identifier (e.g., 'SEMANTIC_TEXT_V0')
        :param bits: Simprint key size in bits (64, 128, 192, or 256)
        :raises ValueError: If bits is not one of the valid values
        """
        # Validate bits parameter
        if bits not in (64, 128, 192, 256):
            raise ValueError(f"bits must be 64, 128, 192, or 256, got {bits}")

        self.bits = bits
        self.simprint_bytes = bits // 8
        self.path = Path(path)
        self.simprint_type = simprint_type
        self.path.mkdir(parents=True, exist_ok=True)

        # Open LMDB environment with default map_size (10MB)
        # writemap=True and map_async=True avoid Windows file reservation issue
        self.env = lmdb.open(
            str(self.path),
            max_dbs=3,  # simprint_type db, index_metadata db, asset_metadata db
            writemap=True,
            map_async=True,
            sync=False,
            metasync=False,
        )

        # Open database with optimal flags for fixed-size keys and values
        # integerkey only works for 64-bit keys
        db_flags = {"dupfixed": True}  # Fixed 16-byte ChunkPointer values (implies dupsort)
        if bits == 64:
            db_flags["integerkey"] = True  # 64-bit integer keys (simprints)

        self.db = self.env.open_db(self.simprint_type.encode("utf-8"), **db_flags)

    def add(self, pairs):
        # type: (list[tuple[bytes, bytes]]) -> None
        """
        Add simprint-to-ChunkPointer pairs to the index.

        Duplicate pairs are automatically skipped to ensure accurate match counts.
        Automatically resizes map_size if MapFullError occurs.

        :param pairs: List of (simprint, chunk_pointer_bytes) tuples (8 bytes, 16 bytes each)
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
                    f"SimprintIndexRaw map_size increased from {old_size:,} to {new_size:,} bytes "
                    f"(retry {retry_count}/{self.MAX_RESIZE_RETRIES})"
                )
                self.env.set_mapsize(new_size)

    def search(self, simprints):
        # type: (list[bytes]) -> list[tuple[bytes, int]]
        """
        Search the index for matching items.

        Returns ISCC-IDs ranked by number of matching simprints (collision count).
        ISCC-IDs with more matching simprints are ranked higher.

        :param simprints: List of simprint digests (size determined by bits parameter)
        :return: List of (iscc_id_body, match_count) tuples sorted by match_count descending
        """
        # Counter to aggregate match counts per ISCC-ID
        iscc_id_counter = Counter()  # type: Counter[bytes]

        with self.env.begin(db=self.db) as txn:
            cursor = txn.cursor()

            # Retrieve all ChunkPointers for all simprints in one optimized call
            # dupdata=True: get all duplicate values for each key
            # dupfixed_bytes=16: values are 16-byte ChunkPointers
            results = cursor.getmulti(simprints, dupdata=True, dupfixed_bytes=16)

            # Count ISCC-ID occurrences (extract from ChunkPointer)
            for _, chunk_bytes in results:
                ptr = ChunkPointer.from_bytes(chunk_bytes)
                iscc_id_counter[ptr.iscc_id_body] += 1

        # Sort by match count descending, then by ISCC-ID for stable ordering
        return sorted(
            iscc_id_counter.items(),
            key=lambda x: (-x[1], x[0]),  # Descending count, ascending ID
        )

    def get_chunk_pointers(self, simprints):
        # type: (list[bytes]) -> dict[bytes, list[ChunkPointer]]
        """
        Get chunk pointers for simprints.

        :param simprints: List of simprint digests (size determined by bits parameter)
        :return: Dict mapping simprint -> list[ChunkPointer]
        """
        result = {}  # type: dict[bytes, list[ChunkPointer]]

        with self.env.begin(db=self.db) as txn:
            cursor = txn.cursor()

            # Use getmulti for efficient dupfixed retrieval
            # Returns list of (key, value) tuples for all duplicates
            getmulti_results = cursor.getmulti(simprints, dupdata=True, dupfixed_bytes=16)

            # Group results by simprint (key)
            for simprint_bytes, chunk_bytes in getmulti_results:
                if simprint_bytes not in result:
                    result[simprint_bytes] = []
                result[simprint_bytes].append(ChunkPointer.from_bytes(chunk_bytes))

        return result

    def close(self):
        # type: () -> None
        """Close the index and flush to disk."""
        self.env.sync(True)  # Ensure all data is flushed
        self.env.close()

    @property
    def map_size(self):
        # type: () -> int
        """Get current LMDB map_size in bytes."""
        return self.env.info()["map_size"]

    def __enter__(self):
        # type: () -> SimprintIndexRaw
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # type: (type | None, Exception | None, object | None) -> None
        """Context manager exit."""
        self.close()


class SimprintIndex(SimprintIndexRaw):
    """
    Developer-friendly simprint index with automatic data transformation and chunk metadata.

    Accepts high-level features objects and ISCC-ID strings, handles all encoding/decoding
    internally. Returns IsccChunk-compliant search results with ISCC-ID strings and chunk metadata.

    Automatically tracks REALM-ID from the first indexed ISCC-ID for correct reconstruction.
    Stores asset-level metadata (source, modality, track) per ISCC-ID for text retrieval.
    """

    def __init__(self, path, simprint_type, text_base_dir=None, bits=64):
        # type: (Path | str, str, Path | str | None, int) -> None
        """
        Open or create the simprint index with metadata tracking.

        :param path: Directory path for LMDB database
        :param simprint_type: Simprint type identifier (e.g., 'SEMANTIC_TEXT_V0')
        :param text_base_dir: Optional directory containing .iscc.utf8 files for text retrieval
        :param bits: Simprint key size in bits (64, 128, 192, or 256)
        :raises ValueError: If bits parameter conflicts with stored bits in existing index
        """
        # Check stored bits if the database already exists
        stored_bits = None
        index_path = Path(path)

        # Check if this is an existing LMDB database (not just an empty directory)
        # An LMDB database will have data.mdb file
        if index_path.exists() and (index_path / "data.mdb").exists():
            # Open metadata database first to check stored bits
            # We need to do this before calling super().__init__ which needs bits parameter
            temp_env = lmdb.open(
                str(index_path),
                max_dbs=3,
                readonly=True,
                lock=False,
            )
            temp_index_metadata_db = temp_env.open_db(b"index_metadata")

            try:
                with temp_env.begin(db=temp_index_metadata_db) as txn:
                    bits_bytes = txn.get(b"bits")
                    if bits_bytes:
                        stored_bits = int.from_bytes(bits_bytes, "big")
            finally:
                temp_env.close()

        # Use stored bits for existing indexes, ignore bits parameter
        if stored_bits is not None:
            if bits != 64 and bits != stored_bits:
                # Only warn if user explicitly passed a non-default value that conflicts
                logger.warning(f"Ignoring bits parameter ({bits}) - using stored bits from index ({stored_bits})")
            logger.debug(f"Opened existing index with bits={stored_bits}")

        # Initialize parent with determined bits value
        super().__init__(path, simprint_type, bits=stored_bits if stored_bits else bits)

        # Open index metadata database for storing realm_id and bits
        self.index_metadata_db = self.env.open_db(b"index_metadata")

        # Open asset metadata database for ISCC-ID -> asset metadata mapping
        # Stores JSON with: source (URL/path), modality (text/image/audio/video/mixed), track (optional)
        self.asset_metadata_db = self.env.open_db(b"asset_metadata")

        # Store bits in index metadata if not already stored
        if stored_bits is None:
            with self.env.begin(write=True, db=self.index_metadata_db) as txn:
                txn.put(b"bits", self.bits.to_bytes(2, "big"))
            logger.info(f"SimprintIndex bits set to {self.bits}")

        # Load realm_id from index metadata database if it exists
        with self.env.begin(db=self.index_metadata_db) as txn:
            realm_id_bytes = txn.get(b"realm_id")
            self.realm_id = int.from_bytes(realm_id_bytes, "big") if realm_id_bytes else None

        self.text_base_dir = Path(text_base_dir) if text_base_dir else None

    def add(self, iscc_id, features, source=None, modality=None, track=None):
        # type: (str, dict, str | None, str | None, int | None) -> None
        """
        Add simprints from features object to the index with chunk metadata.

        Automatically decodes ISCC-ID and simprints, creates ChunkPointer pairs, and indexes them.
        Sets REALM-ID from the first ISCC-ID added to the index.
        Stores asset-level metadata (source, modality, track) per ISCC-ID.

        :param iscc_id: ISCC-ID in canonical string format (with or without "ISCC:" prefix)
        :param features: Features dict with 'simprints', 'offsets', and 'sizes' keys
        :param source: Optional resolvable URL/path to source content
        :param modality: Optional content modality (text/image/audio/video/mixed)
        :param track: Optional track/layer/page identifier
        """
        import json

        # Decode ISCC-ID and extract 8-byte body (strip 2-byte header)
        iscc_id_obj = IsccID(iscc_id)
        iscc_id_body = iscc_id_obj.body

        # Store realm_id from first ISCC-ID if not already set
        if self.realm_id is None:
            self.realm_id = iscc_id_obj.realm_id
            with self.env.begin(write=True, db=self.index_metadata_db) as txn:
                txn.put(b"realm_id", self.realm_id.to_bytes(1, "big"))
            logger.info(f"SimprintIndex realm_id set to {self.realm_id}")

        # Get simprints, offsets, sizes from features
        simprints = features.get("simprints", [])
        offsets = features.get("offsets", [])
        sizes = features.get("sizes", [])

        if not (simprints and offsets and sizes):
            logger.warning(f"Missing simprints, offsets, or sizes in features for {iscc_id}")
            return

        if len(simprints) != len(offsets) or len(simprints) != len(sizes):
            logger.warning(f"Length mismatch in simprints/offsets/sizes for {iscc_id}")
            return

        # Store asset metadata if any field is provided
        if source or modality or track is not None:
            metadata = {}  # type: dict[str, str | int]
            if source:
                metadata["source"] = source
            if modality:
                metadata["modality"] = modality
            if track is not None:
                metadata["track"] = track

            with self.env.begin(write=True, db=self.asset_metadata_db) as txn:
                txn.put(iscc_id_body, json.dumps(metadata).encode("utf-8"))

        # Build pairs: (simprint, ChunkPointer.to_bytes())
        pairs = []
        for simprint_b64, offset, size in zip(simprints, offsets, sizes):
            simprint_bytes = ic.decode_base64(simprint_b64)[: self.simprint_bytes]
            chunk_ptr = ChunkPointer(iscc_id_body, offset, size)
            pairs.append((simprint_bytes, chunk_ptr.to_bytes()))

        # Call parent add method
        super().add(pairs)

    def search(self, simprints, limit=None):
        # type: (list[str], int | None) -> list[dict]
        """
        Search for matching ISCC-IDs by simprints.

        :param simprints: List of base64-encoded simprint strings
        :param limit: Optional maximum number of results to return
        :return: List of dicts with 'iscc_id' (str) and 'match_count' (int), sorted by match_count
        :raises ValueError: If realm_id is not set (index is empty)
        """
        # Verify realm_id is set (index must have at least one entry)
        if self.realm_id is None:
            raise ValueError("Cannot search empty index - realm_id not set")

        # Base64 decode simprints and truncate to configured bits
        simprint_bytes = [ic.decode_base64(s)[: self.simprint_bytes] for s in simprints]

        # Call parent search method
        raw_results = super().search(simprint_bytes)

        # Convert to developer-friendly format
        results = []
        for iscc_id_body, match_count in raw_results:
            # Reconstruct ISCC-ID with header using tracked realm_id
            iscc_id_str = str(IsccID.from_body(iscc_id_body, self.realm_id))

            results.append({"iscc_id": iscc_id_str, "match_count": match_count})

            if limit and len(results) >= limit:
                break

        return results

    def get_chunk_pointers(self, simprints):
        # type: (list[str]) -> dict[str, list[ChunkPointer]]
        """
        Get chunk pointers for simprints.

        :param simprints: List of base64-encoded simprint strings
        :return: Dict mapping simprint -> list[ChunkPointer]
        """
        # Base64 decode simprints and truncate to configured bits
        simprint_bytes_list = [ic.decode_base64(s)[: self.simprint_bytes] for s in simprints]

        # Call parent method
        raw_results = super().get_chunk_pointers(simprint_bytes_list)

        # Convert keys back to base64 strings
        return {ic.encode_base64(k): v for k, v in raw_results.items()}

    def get_chunks(self, simprints, limit_per_simprint=None):
        # type: (list[str], int | None) -> dict[str, list[dict]]
        """
        Get actual content chunks for simprints in IsccChunk format.

        :param simprints: List of base64-encoded simprint strings
        :param limit_per_simprint: Max chunks per simprint (None = all)
        :return: Dict mapping simprint -> list of IsccChunk dicts
        :raises ValueError: If text_base_dir not configured or realm_id not set
        """
        import json
        import base64

        if not self.text_base_dir:
            raise ValueError("text_base_dir not configured")
        if self.realm_id is None:
            raise ValueError("Cannot retrieve chunks from empty index - realm_id not set")

        pointer_map = self.get_chunk_pointers(simprints)
        result = {}  # type: dict[str, list[dict]]

        # Get asset metadata for all ISCC-IDs
        iscc_id_to_metadata = {}  # type: dict[bytes, dict]
        with self.env.begin(db=self.asset_metadata_db) as txn:
            for pointers in pointer_map.values():
                for ptr in pointers:
                    if ptr.iscc_id_body not in iscc_id_to_metadata:
                        metadata_bytes = txn.get(ptr.iscc_id_body)
                        if metadata_bytes:
                            iscc_id_to_metadata[ptr.iscc_id_body] = json.loads(metadata_bytes.decode("utf-8"))
                        else:
                            logger.debug(f"No asset metadata in DB for ISCC-ID body {ptr.iscc_id_body.hex()}")
                            iscc_id_to_metadata[ptr.iscc_id_body] = {}

        # Extract chunks
        for simprint_b64, pointers in pointer_map.items():
            chunks = []

            for ptr in pointers[:limit_per_simprint] if limit_per_simprint else pointers:
                metadata = iscc_id_to_metadata.get(ptr.iscc_id_body, {})
                source = metadata.get("source")
                modality = metadata.get("modality")
                track = metadata.get("track")

                if not source:
                    logger.debug(f"No source URL found for ISCC-ID body {ptr.iscc_id_body.hex()}")
                    continue

                # Parse source URL to get file path
                # Support file:// URLs and plain paths
                if source.startswith("file://"):
                    file_path = Path(source[7:])  # Strip "file://"
                else:
                    # Assume relative path, resolve against text_base_dir
                    file_path = self.text_base_dir / source

                # Try multiple path variations
                text_path = None
                tried_paths = []
                for suffix in ["", ".iscc.utf8"]:
                    candidate = file_path.parent / (file_path.name + suffix) if suffix else file_path
                    tried_paths.append(str(candidate))
                    if candidate.exists():
                        text_path = candidate
                        break

                if not text_path:
                    logger.debug(f"Text file not found. Tried: {tried_paths}")
                    continue

                # Read and extract chunk (slice by bytes, not characters)
                try:
                    # Read file in binary mode to preserve byte offsets
                    with open(text_path, "rb") as f:
                        file_bytes = f.read()

                    # Slice by byte offsets (as stored in ChunkPointer)
                    chunk_bytes = file_bytes[ptr.offset : ptr.offset + ptr.size]

                    # Create IsccChunk dict conforming to schema
                    chunk = {
                        "iscc_id": str(IsccID.from_body(ptr.iscc_id_body, self.realm_id)),
                        "offset": ptr.offset,
                        "size": ptr.size,
                    }

                    # Add optional fields
                    if source:
                        chunk["source"] = source
                    if modality:
                        chunk["modality"] = modality
                    if track is not None:
                        chunk["track"] = track

                    # Add content as data-url (RFC 2397)
                    content_b64 = base64.b64encode(chunk_bytes).decode("ascii")
                    chunk["content"] = f"data:text/plain;base64,{content_b64}"

                    chunks.append(chunk)
                except Exception as e:
                    logger.warning(f"Error reading {text_path}: {e}")
                    continue

            if chunks:
                result[simprint_b64] = chunks

        # Log summary for debugging
        if not result:
            total_pointers = sum(len(ptrs) for ptrs in pointer_map.values())
            logger.warning(
                f"get_chunks returned no results! "
                f"Simprints queried: {len(simprints)}, "
                f"Chunk pointers found: {total_pointers}, "
                f"Asset metadata in DB: {len(iscc_id_to_metadata)}, "
                f"Text base dir: {self.text_base_dir}"
            )

        return result
