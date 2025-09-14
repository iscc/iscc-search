"""ISCC-aware vector index with automatic type conversion.

This module provides the IsccUnitIndex class that extends NphdIndex to support
ISCC-specific data types (ISCC-UNIT and ISCC-ID) with transparent conversion.
"""

from typing import Sequence

import iscc_core as ic
from iscc_vdb import NphdIndex


class IsccUnitIndex(NphdIndex):
    ISCC_ID_REALM = 0
    UNIT_HEADER = None

    def add(self, keys, vectors, **kwargs):
        iscc_ids = [keys] if isinstance(keys, str) else keys
        iscc_units = [vectors] if isinstance(vectors, str) else vectors

        if not self.UNIT_HEADER:
            self.UNIT_HEADER = ic.iscc_decode(iscc_units[0])[3]

        keys = self._iscc_ids_to_keys(iscc_ids)
        vectors = self._iscc_units_to_vectors(iscc_units)
        return super().add(keys, vectors, **kwargs)

    def _iscc_ids_to_keys(self, iscc_ids):
        # type: (Sequence[str]) -> Sequence[int]
        """Extract 64-bit integer keys from ISCC-ID strings."""
        int_keys = []
        for iscc_id in iscc_ids:
            body = ic.decode_base32(iscc_id.removeprefix("ISCC:"))[2:]
            int_keys.append(int.from_bytes(body, "big", signed=False))
        return int_keys

    def _iscc_units_to_vectors(self, vectors):
        # type: (Sequence[str]) -> Sequence[bytes]
        """Extract binary bit-vectors from ISCC-UNIT strings."""
        int_vectors = []
        for vector in vectors:
            int_vectors.append(ic.decode_base32(vector.removeprefix("ISCC:"))[2:])
        return int_vectors

    def _keys_to_iscc_ids(self, keys):
        # type: (Sequence[int]) -> Sequence[str]
        """Reconstruct ISCC-ID strings from integer keys."""
        iscc_ids = []
        for key in keys:
            iscc_ids.append(
                "ISCC:" + ic.encode_component(
                    ic.MT.ID, self.ISCC_ID_REALM, 1, 64, key.to_bytes(8, "big", signed=False)
                )
            )
        return iscc_ids

    def _vectors_to_iscc_units(self, vectors):
        # type: (Sequence[bytes]) -> Sequence[str]
        """Reconstruct ISCC-UNIT strings from binary bit-vectors."""
        iscc_units = []
        for vector in vectors:
            iscc_tuple = (*self.UNIT_HEADER, len(vector) * 8, vector)
            iscc_units.append("ISCC:" + ic.encode_component(*iscc_tuple))
        return iscc_units
