"""Multi-Index ISCC Vector Database.

This module provides a multi-index class for ISCC vector database that manages
multiple NphdIndex instances for different ISCC component types, provides a
usearch-compatible API with rich result structures, and handles multi-component
ISCCs by decomposing and routing to appropriate indices.

The IsccIndex class stores:
- Multiple NphdIndex instances for different component types (meta, semantic, content, data)
- SQLite database for Instance-Code storage with prefix matching
- Metadata about the index configuration

Directory structure:
    index_path/
    ├── index.json     # Index metadata (max_bits, version)
    ├── instances.db   # SQLite database for Instance-Codes
    └── *.usearch      # Component-specific indices (e.g., meta-semantic.usearch)

Example:
    >>> from iscc_vdb.iscc_index import IsccIndex
    >>>
    >>> # Create new index
    >>> index = IsccIndex("/path/to/index")
    >>>
    >>> # Add ISCCs
    >>> index.add("ISCC:MAIGHFECJMOPMIAB", "ISCC:KACT4EBWK27737D2AYCJRAL5Z36G76RFRMO4554RU26HZ4ORJGIVHDI")
    >>>
    >>> # Search for similar ISCCs
    >>> results = index.search("ISCC:KACT4EBWK27737D2AYCJRAL5Z36G76RFRMO4554RU26HZ4ORJGIVHDI", count=10)
    >>> print(results.keys)  # ['ISCC:MAIGHFECJMOPMIAB']
"""

import json
import typing
from datetime import datetime, timezone
from pathlib import Path

import iscc_core as ic
import platformdirs

from iscc_vdb.nphd_index import NphdIndex


class IsccIndex:
    """
    Multi-index ISCC vector database managing multiple NphdIndex instances
    for different ISCC component types.
    """

    def __init__(self, path=None, max_bits=256):
        # type: (str | Path | None, int) -> None
        """
        Initialize IsccIndex with path and maximum bits configuration.

        :param path: Directory path where index files will be stored.
                     If None, uses default directory in user data folder.
        :param max_bits: Maximum supported vector size in bits (default: 256)
        """
        if path is None:
            # Use platformdirs for cross-platform user data directory
            user_data_dir = platformdirs.user_data_dir("iscc-vdb", "iscc")
            self.path = Path(user_data_dir) / "default"
        else:
            self.path = Path(path)

        self.max_bits = max_bits
        self.indices = {}  # type: dict[str, NphdIndex]

        # Create directory if it doesn't exist
        try:
            self.path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            msg = f"Failed to create index directory {self.path}: {e}"
            raise RuntimeError(msg) from e

        # Load or create index metadata
        self._load_or_create_metadata()

    def _load_or_create_metadata(self):
        # type: () -> None
        """Load existing metadata or create new metadata file."""
        metadata_path = self.path / "index.json"

        if metadata_path.exists():
            try:
                with metadata_path.open("r") as f:
                    metadata = json.load(f)

                # Validate metadata
                if "max_bits" not in metadata:
                    msg = f"Invalid metadata in {metadata_path}: missing max_bits"
                    raise ValueError(msg)

                # Update max_bits from metadata if different
                if metadata["max_bits"] != self.max_bits:
                    self.max_bits = metadata["max_bits"]

            except (json.JSONDecodeError, OSError) as e:
                msg = f"Failed to load metadata from {metadata_path}: {e}"
                raise RuntimeError(msg) from e
        else:
            # Create new metadata file
            metadata = {
                "max_bits": self.max_bits,
                "version": "0.0.1",
                "created": datetime.now(timezone.utc).isoformat(),
            }

            try:
                with metadata_path.open("w") as f:
                    json.dump(metadata, f, indent=2)
            except OSError as e:
                msg = f"Failed to create metadata file {metadata_path}: {e}"
                raise RuntimeError(msg) from e

    def _validate_iscc(self, iscc):
        # type: (str) -> bool
        """
        Validate that the input is a valid ISCC string.

        :param iscc: ISCC string to validate
        :return: True if valid, False otherwise
        """
        try:
            # Use iscc_core.codec.iscc_validate for validation
            from iscc_core.codec import iscc_validate

            iscc_validate(iscc, strict=True)
        except (ValueError, TypeError):
            return False
        else:
            return True

    def _decompose_iscc(self, iscc):
        # type: (str) -> list[tuple[str, str]]
        """
        Decompose multi-component ISCCs into individual components.

        :param iscc: Full ISCC string
        :return: List of (component_code, type_id) tuples
        """
        from iscc_core.codec import iscc_decompose, iscc_type_id

        # Use iscc_core to decompose the ISCC
        components = iscc_decompose(iscc)
        result = []

        for component in components:
            # Extract type_id using iscc_type_id function
            type_id = iscc_type_id(component)
            result.append((component, type_id))

        return result

    def _iscc_id_to_uint64(self, iscc_id):
        # type: (str) -> int
        """
        Convert ISCC-ID (base32 string) to uint64.

        :param iscc_id: ISCC-ID as base32 string (with or without ISCC: prefix)
        :return: uint64 integer representation
        """
        from iscc_core.codec import decode_base32, iscc_clean

        # Clean the ISCC-ID (remove ISCC: prefix if present)
        clean_id = iscc_clean(iscc_id)

        # Decode base32 to bytes
        # ISCC-IDs have 2-byte header + 8 bytes payload = 10 bytes total
        id_bytes = decode_base32(clean_id)
        if len(id_bytes) < 10:
            msg = f"Invalid ISCC-ID: expected at least 10 bytes, got {len(id_bytes)}"
            raise ValueError(msg)

        # Skip the 2-byte header and extract the 8-byte ID
        id_payload = id_bytes[2:10]

        # Convert bytes to uint64 (big-endian)
        return int.from_bytes(id_payload, byteorder="big")

    def _uint64_to_iscc_id(self, uint64_key):
        # type: (int) -> str
        """
        Convert uint64 back to ISCC-ID string.

        :param uint64_key: uint64 integer
        :return: ISCC-ID as base32 string
        """
        from iscc_core.codec import encode_component
        from iscc_core.constants import MT, ST_ID

        # Convert uint64 to 8 bytes (big-endian)
        id_bytes = uint64_key.to_bytes(8, byteorder="big")

        # Create ISCC-ID using encode_component
        # Default to v0 ISCC-IDs: MainType=ID, SubType=PRIVATE, Version=0
        # This creates deterministic ISCC-IDs for internal use
        iscc_id = encode_component(mtype=MT.ID, stype=ST_ID.PRIVATE, version=0, bit_length=64, digest=id_bytes)

        return iscc_id
