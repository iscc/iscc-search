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
from pathlib import Path

from iscc_vdb.nphd_index import NphdIndex


class IsccIndex:
    """
    Multi-index ISCC vector database managing multiple NphdIndex instances
    for different ISCC component types.
    """

    def __init__(self, path, max_bits=256):
        # type: (str | Path, int) -> None
        """
        Initialize IsccIndex with path and maximum bits configuration.

        :param path: Directory path where index files will be stored
        :param max_bits: Maximum supported vector size in bits (default: 256)
        """
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
                "created": "2024-01-01T00:00:00Z",  # Will be updated with actual timestamp
            }

            try:
                with metadata_path.open("w") as f:
                    json.dump(metadata, f, indent=2)
            except OSError as e:
                msg = f"Failed to create metadata file {metadata_path}: {e}"
                raise RuntimeError(msg) from e
