"""ISCC-aware vector index with automatic type conversion.

This module provides the IsccUnitIndex class that extends NphdIndex to support
ISCC-specific data types (ISCC-UNIT and ISCC-ID) with transparent conversion.
"""

import json
import typing
from collections.abc import Sequence
from pathlib import Path

import iscc_core as ic  # type: ignore[import-untyped]
import numpy as np

from iscc_vdb.metrics import pack_binary_vector, unpack_binary_vector
from iscc_vdb.nphd_index import NphdIndex


class IsccUnitIndex(NphdIndex):
    """ISCC-aware vector index with automatic type conversion."""

    def __init__(self, sample_unit=None, realm_id=0, version=1, **kwargs):
        # type: (typing.Optional[str], int, int, typing.Any) -> None
        """Initialize IsccUnitIndex with optional sample unit for type locking.

        :param sample_unit: Optional ISCC-UNIT string to lock the index to a specific type
        :param realm_id: Realm ID for ISCC-ID generation (default: 0)
        :param version: Version for ISCC-ID generation (default: 1)
        :param **kwargs: Additional parameters passed to NphdIndex
        """
        # Store ISCC-ID parameters
        self.realm_id = realm_id
        self.version = version

        # Store decoded header components (not string)
        self.unit_header = None  # type: typing.Optional[typing.Tuple[int, int, int]]

        # Process sample unit if provided
        if sample_unit:
            self._set_unit_type_from_sample(sample_unit)

        super().__init__(**kwargs)

    def _set_unit_type_from_sample(self, sample_unit):
        # type: (str) -> None
        """Store the actual header components from sample.

        :param sample_unit: ISCC-UNIT string to extract type information from
        :raises ValueError: If the unit is invalid or has unsupported body length
        """
        decoded = ic.iscc_decode(sample_unit)
        maintype, subtype, version, length, body = decoded

        # Validate body length
        if len(body) not in {8, 16, 24, 32}:
            msg = f"Invalid ISCC body length: {len(body)} bytes"
            raise ValueError(msg)

        # Validate that it's not a composite code
        if maintype == ic.MT.ISCC:
            msg = "Composite codes (MT.ISCC) are not supported"
            raise ValueError(msg)

        # Store header components directly
        self.unit_header = (maintype, subtype, version)

    def _is_batch(self, obj):
        # type: (typing.Any) -> bool
        """Detect if input is a batch (excluding strings/bytes).

        :param obj: Object to check
        :return: True if the object is a batch of items
        """
        return isinstance(obj, Sequence) and not isinstance(obj, str | bytes | bytearray)

    def _normalize_key(self, key):
        # type: (typing.Union[int, str]) -> int
        """Normalize a key to integer format.

        :param key: Integer key or ISCC-ID string
        :return: Integer key
        :raises ValueError: If ISCC-ID doesn't decode to exactly 8 bytes
        """
        if isinstance(key, str):
            # Decode ISCC-ID to integer
            decoded = ic.iscc_decode(key)
            maintype, subtype, version, length, body = decoded

            if maintype != ic.MT.ID:
                msg = f"Expected ISCC-ID (MainType.ID), got MainType={maintype}"
                raise ValueError(msg)

            if len(body) != 8:
                msg = f"ISCC-ID must decode to 8 bytes, got {len(body)} bytes"
                raise ValueError(msg)

            # Convert bytes to unsigned integer
            return int.from_bytes(body, "big", signed=False)

        # Validate integer key
        if key < 0 or key >= 2**64:
            msg = f"Integer key must be in range [0, 2^64), got {key}"
            raise ValueError(msg)

        return key

    def _normalize_vector(self, vector):
        # type: (typing.Union[str, bytes, np.ndarray]) -> bytes
        """Normalize a vector to bytes format.

        :param vector: ISCC-UNIT string, bytes, or numpy array
        :return: Binary vector bytes
        :raises ValueError: If type doesn't match locked type or body length is invalid
        """
        if isinstance(vector, str):
            # Decode ISCC-UNIT
            decoded = ic.iscc_decode(vector)
            maintype, subtype, version, length, body = decoded

            # Validate body length
            if len(body) not in {8, 16, 24, 32}:
                msg = f"Invalid ISCC body length: {len(body)} bytes"
                raise ValueError(msg)

            # Check type consistency if type is locked
            if self.unit_header is not None:
                expected_mt, expected_st, expected_vs = self.unit_header
                if (maintype, subtype, version) != (expected_mt, expected_st, expected_vs):
                    msg = (
                        f"Type mismatch: expected MainType={expected_mt}, SubType={expected_st}, Version={expected_vs}, "
                        f"got MainType={maintype}, SubType={subtype}, Version={version}"
                    )
                    raise ValueError(msg)
            else:
                # First vector sets the type
                self.unit_header = (maintype, subtype, version)

            return body  # type: ignore[no-any-return]

        # Convert numpy array to bytes if needed
        if isinstance(vector, np.ndarray):
            vector = bytes(vector)

        # Validate byte length
        if len(vector) not in {8, 16, 24, 32}:
            msg = f"Vector must be 8, 16, 24, or 32 bytes, got {len(vector)} bytes"
            raise ValueError(msg)

        return vector

    def _reconstruct_iscc_unit(self, packed_vector):
        # type: (np.ndarray) -> typing.Union[str, np.ndarray]
        """Reconstruct ISCC-UNIT from packed vector.

        :param packed_vector: Packed vector with length signal
        :return: ISCC-UNIT string or raw bytes if no type info
        """
        if self.unit_header is None:
            return packed_vector  # Can't reconstruct without type info

        # Unpack to get original bytes
        body_bytes = unpack_binary_vector(packed_vector)

        # Use stored header components
        maintype, subtype, version = self.unit_header
        bit_length = len(body_bytes) * 8

        # Reconstruct using iscc-core
        unit_str = ic.encode_component(maintype, subtype, version, bit_length, body_bytes)

        return "ISCC:" + unit_str  # type: ignore[no-any-return]

    def _int_to_iscc_id(self, key_int):
        # type: (int) -> str
        """Convert integer key to ISCC-ID string.

        :param key_int: Integer key
        :return: ISCC-ID string
        """
        # Convert to 8 bytes (unsigned)
        body_bytes = key_int.to_bytes(8, "big", signed=False)

        # Create ID using same component encoder
        # For ISCC-ID, version is always 1 per spec, length is 64 bits
        id_str = ic.encode_component(
            ic.MT.ID,
            self.realm_id,  # SubType is realm for IDs
            1,  # Version is always 1 for ISCC-IDs
            64,  # 64-bit ID
            body_bytes,
        )

        return "ISCC:" + id_str  # type: ignore[no-any-return]

    def add(self, keys, vectors, **kwargs):
        # type: (typing.Any, typing.Any, typing.Any) -> typing.Any
        """Add vectors with automatic ISCC conversion.

        :param keys: Integer keys or ISCC-ID strings
        :param vectors: ISCC-UNIT strings, bytes, or numpy arrays
        :param **kwargs: Additional parameters passed to parent
        :return: Result from parent add method
        """
        # Handle batch operations
        if self._is_batch(keys):
            # Process batch
            normalized_keys = []
            normalized_vectors = []

            for key, vector in zip(keys, vectors, strict=False):
                normalized_keys.append(self._normalize_key(key))
                normalized_vectors.append(self._normalize_vector(vector))

            # Use parent's add with normalized data
            return super().add(normalized_keys, normalized_vectors, **kwargs)
        else:
            # Single item
            normalized_key = self._normalize_key(keys)
            normalized_vector = self._normalize_vector(vectors)
            return super().add(normalized_key, normalized_vector, **kwargs)

    def _get_single(self, key, dtype=None):
        # type: (typing.Any, typing.Any) -> typing.Any
        """Get a single vector by key.

        :param key: Integer key or ISCC-ID string
        :param dtype: Data type for result
        :return: ISCC-UNIT string or None
        """
        try:
            normalized_key = self._normalize_key(key)
        except ValueError:
            return None

        # Check if key exists
        if normalized_key not in self:
            return None

        packed_vector = super().get(normalized_key, dtype)

        if packed_vector is None:
            return None

        return self._reconstruct_iscc_unit(packed_vector)

    def get(self, keys, dtype=None):
        # type: (typing.Any, typing.Any) -> typing.Any
        """Retrieve vectors by keys with ISCC reconstruction.

        :param keys: Integer keys or ISCC-ID strings
        :param dtype: Data type for results (unused, for compatibility)
        :return: ISCC-UNIT strings or packed vectors if no type info
        """
        # Handle batch operations
        if self._is_batch(keys):
            normalized_keys = []
            for k in keys:
                try:
                    normalized_keys.append(self._normalize_key(k))
                except ValueError:
                    # Invalid key
                    normalized_keys.append(-1)

            # Check if keys exist first
            valid_keys = [k for k in normalized_keys if k >= 0 and k in self]
            if not valid_keys:
                return [None] * len(keys)

            packed_vectors = super().get(normalized_keys, dtype)

            if packed_vectors is None:
                return None

            # Reconstruct ISCC units for batch
            results = []
            for key, packed in zip(normalized_keys, packed_vectors, strict=False):
                if key in self and packed is not None:
                    results.append(self._reconstruct_iscc_unit(packed))
                else:
                    results.append(None)  # type: ignore[arg-type]
            return results
        else:
            return self._get_single(keys, dtype)

    def search(self, vectors, count=10, radius=float("inf"), *, threads=0, exact=False, log=False, progress=None):
        # type: (typing.Any, int, float, int, bool, typing.Any, typing.Any) -> typing.Any
        """Search with ISCC-ID results.

        :param vectors: Query vectors (ISCC-UNIT strings, bytes, or numpy arrays)
        :param count: Number of results to return
        :param radius: Maximum distance for results
        :param threads: Number of threads to use
        :param exact: Whether to use exact search
        :param log: Enable logging
        :param progress: Progress callback
        :return: Search results with ISCC-ID keys
        """
        # Normalize query vectors
        if self._is_batch(vectors):
            normalized_vectors = [self._normalize_vector(v) for v in vectors]
        else:
            normalized_vectors = self._normalize_vector(vectors)  # type: ignore[assignment]

        # Perform search with parent class
        results = super().search(
            normalized_vectors, count, radius, threads=threads, exact=exact, log=log, progress=progress
        )

        # Convert integer keys to ISCC-IDs in results
        if hasattr(results, "keys"):
            # Check if batch or single result
            if results.keys.ndim == 2:
                # Batch results
                batch_iscc_keys = []
                for i in range(results.keys.shape[0]):
                    row_keys = []
                    for j in range(results.keys.shape[1]):
                        key = int(results.keys[i, j])
                        if key >= 0:  # Valid key (usearch uses unsigned integers)
                            row_keys.append(self._int_to_iscc_id(key))
                        else:
                            row_keys.append(None)  # type: ignore[arg-type]
                    batch_iscc_keys.append(row_keys)
                results.keys = batch_iscc_keys
            else:
                # Single result
                single_iscc_keys = []  # type: list[typing.Optional[str]]
                for key in results.keys:
                    key_int = int(key)
                    # Check if it's a valid key (in usearch, keys are unsigned)
                    # For empty results, usearch might return max uint64 value
                    if key_int >= 0 and key_int < 2**64 - 1:  # Valid key
                        single_iscc_keys.append(self._int_to_iscc_id(key_int))
                    else:
                        single_iscc_keys.append(None)
                results.keys = single_iscc_keys

        return results

    def save(self, path, progress=None):  # type: ignore[override]
        # type: (typing.Union[str, Path], typing.Any) -> None
        """Save index with metadata sidecar.

        :param path: Path to save the index
        :param progress: Progress callback
        """
        # Save index data
        super().save(path, progress)

        # Create metadata
        metadata = {
            "format_version": 1,
            "realm_id": self.realm_id,
            "version": self.version,
        }

        # Add unit header if set
        if self.unit_header:
            maintype, subtype, version = self.unit_header
            # Store as integers for simplicity
            metadata["unit_header"] = {"maintype": maintype, "subtype": subtype, "version": version}  # type: ignore[assignment]

        # Save metadata to sidecar
        meta_path = Path(str(path) + ".meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def restore(cls, path, view=True, **kwargs):  # type: ignore[override]
        # type: (typing.Union[str, Path], bool, typing.Any) -> "IsccUnitIndex"
        """Restore index with metadata.

        :param path: Path to saved index file
        :param view: If True, use memory mapping
        :param **kwargs: Additional parameters
        :return: Restored IsccUnitIndex instance
        """
        # Load metadata
        meta_path = Path(str(path) + ".meta.json")

        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

            # Create instance with metadata
            instance = cls(realm_id=metadata.get("realm_id", 0), version=metadata.get("version", 1))

            # Restore unit header if present
            if "unit_header" in metadata:
                h = metadata["unit_header"]
                instance.unit_header = (h["maintype"], h["subtype"], h["version"])
        else:
            # No metadata - create basic instance
            instance = cls()

        # Load index data using parent's restore
        NphdIndex.restore(path, view, **kwargs)

        # Copy internal state from parent
        # Since we're using subclassing, we need to copy the loaded state
        if view:
            instance.view(path)
        else:
            instance.load(path)

        return instance
