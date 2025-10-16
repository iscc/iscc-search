"""
Scalable ANNS search for variable-length ISCC-UNITs.
"""

from collections.abc import Sequence
from functools import cached_property
from typing import Any

import iscc_core as ic
import numpy as np
from numpy.typing import NDArray

from iscc_vdb.nphd import NphdIndex


class UnitIndex(NphdIndex):
    """Fast approximate nearest neighbor search for variable-length ISCC-UNITs of the same type.

    Instead of integer keys and uint8-vectors, we accept ISCC-IDs as keys and ISCC-UNITs as vectors.
    """

    def __init__(self, unit_type=None, max_dim=256, realm_id=None, **kwargs):
        # type: (str | None, int, int | None, Any) -> None
        """Create a new ISCC-UNIT index.

        :param unit_type: ISCC type string (e.g. 'META-NONE-V0') or None for auto-detection
        :param max_dim: Maximum vector dimension in bits (default 256)
        :param realm_id: ISCC realm ID (0-15) or None for auto-detection
        :param kwargs: Additional arguments passed to NphdIndex
        """
        super().__init__(max_dim=max_dim, **kwargs)
        self.unit_type = unit_type
        self.realm_id = realm_id

    def add(self, keys, vectors, **kwargs):
        # type: (Any, str | Sequence[str], Any) -> list[str]
        """
        Add ISCC-UNITs to the index with ISCC-ID keys.

        :param keys: ISCC-ID string(s) or None for auto-generation
        :param vectors: ISCC-UNIT string(s) to add
        :param kwargs: Additional arguments passed to parent Index.add()
        :return: List of ISCC-ID strings for added vectors
        """
        # Normalize vectors to list
        iscc_units = [vectors] if isinstance(vectors, str) else list(vectors)

        if keys is None:
            iscc_ids = None
            # Set realm_id to 0 if not set
            if self.realm_id is None:
                self.realm_id = 0
        else:
            iscc_ids = [keys] if isinstance(keys, str) else list(keys)
            # Set realm_id from first ISCC-ID if not set
            if self.realm_id is None:
                self.realm_id = self._extract_realm_id(iscc_ids[0])

        # Set unit_type from first ISCC-UNIT if not set
        if self.unit_type is None:
            self.unit_type = self._extract_unit_type(iscc_units[0])

        # Convert ISCC-IDs to integer keys (or None for auto-generation)
        int_keys = None if iscc_ids is None else self._to_keys(iscc_ids)

        # Convert ISCC-UNITs to binary vectors
        bin_vectors = self._to_vectors(iscc_units)

        # Call parent add
        result_keys = super().add(int_keys, bin_vectors, **kwargs)

        # Convert result keys back to ISCC-IDs
        return self._to_iscc_ids(result_keys)

    @cached_property
    def _iscc_id_header(self):
        # type: () -> bytes
        """Return header bytes for ISCC-ID reconstruction."""
        if self.realm_id is None:
            raise ValueError("realm_id must be set before converting keys to ISCC-IDs")
        return ic.encode_header(ic.MT.ID, self.realm_id, ic.VS.V1, 0)

    def _extract_realm_id(self, iscc_id):
        # type: (str) -> int
        """Extract realm_id from ISCC-ID."""
        _mt, st, _vs, _ln, _body = ic.iscc_decode(iscc_id)
        return st

    def _extract_unit_type(self, iscc_unit):
        # type: (str) -> str
        """Extract unit_type from ISCC-UNIT."""
        type_id = ic.iscc_type_id(iscc_unit)
        # Remove the length part (last segment)
        return "-".join(type_id.split("-")[:-1])

    def _to_keys(self, iscc_ids):
        # type: (list[str]) -> list[int]
        """Convert ISCC-ID strings to integer keys."""
        keys = []
        for iid in iscc_ids:
            body = ic.decode_base32(iid.removeprefix("ISCC:"))[2:]
            keys.append(int.from_bytes(body, "big", signed=False))
        return keys

    def _to_vectors(self, iscc_units):
        # type: (list[str]) -> list[NDArray[np.uint8]]
        """Convert ISCC-UNIT strings to binary vectors."""
        vectors = []
        for iunit in iscc_units:
            vector = np.frombuffer(ic.decode_base32(iunit.removeprefix("ISCC:"))[2:], dtype=np.uint8)
            vectors.append(vector)
        return vectors

    def _to_iscc_ids(self, keys):
        # type: (NDArray[np.uint64]) -> list[str]
        """Convert integer keys to ISCC-ID strings."""
        arr_big = keys.astype(">u8")
        byte_array = arr_big.view(np.uint8).reshape(-1, 8)
        digests = [row.tobytes() for row in byte_array]
        iscc_ids = ["ISCC:" + ic.encode_base32(self._iscc_id_header + digest) for digest in digests]
        return iscc_ids
