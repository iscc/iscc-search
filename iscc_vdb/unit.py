"""
Scalable ANNS search for variable-length ISCC-UNITs.
"""

from collections.abc import Sequence
from functools import cached_property
from typing import Any

import iscc_core as ic
import numpy as np
from numpy.typing import NDArray
from usearch.index import BatchMatches, Matches

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

    def get(self, keys, dtype=None):
        # type: (str | Sequence[str], Any) -> str | list[str | None] | None
        """
        Retrieve ISCC-UNITs by ISCC-ID key(s).

        :param keys: ISCC-ID string(s) to lookup
        :param dtype: Optional data type (defaults to index dtype)
        :return: ISCC-UNIT string(s) or None for missing keys
        """
        # Track if single key was passed and normalize to list
        single = isinstance(keys, str)
        iscc_ids = [keys] if single else list(keys)

        # Convert ISCC-IDs to integer keys
        int_keys = self._to_keys(iscc_ids)

        # Call parent get with list of keys
        results = super().get(int_keys, dtype=dtype)

        # Convert binary vectors to ISCC-UNITs
        iscc_units = [None if r is None else self._vector_to_iscc_unit(r) for r in results]

        # Unwrap single result
        if single:
            return iscc_units[0]

        return iscc_units

    def search(self, vectors, count=10, **kwargs):
        # type: (str | Sequence[str], int, Any) -> Matches | BatchMatches
        """
        Search for nearest neighbors of ISCC-UNIT query vector(s).

        :param vectors: ISCC-UNIT string(s) to query
        :param count: Maximum number of nearest neighbors to return per query
        :param kwargs: Additional arguments passed to parent Index.search()
        :return: Matches for single query or BatchMatches for batch queries with ISCC-ID keys
        """
        # Normalize input and convert ISCC-UNITs to binary vectors
        iscc_units = [vectors] if isinstance(vectors, str) else list(vectors)
        bin_vectors = self._to_vectors(iscc_units)

        # Call parent search
        result = super().search(bin_vectors, count=count, **kwargs)

        # Convert integer keys to ISCC-IDs
        if result.keys.ndim == 1:
            # Matches object (single query)
            iscc_ids = np.array(self._to_iscc_ids(result.keys))
            return Matches(
                keys=iscc_ids,
                distances=result.distances,
                visited_members=result.visited_members,
                computed_distances=result.computed_distances,
            )

        # BatchMatches object (multiple queries)
        iscc_ids_list = [self._to_iscc_ids(row) for row in result.keys]
        iscc_ids_array = np.array(iscc_ids_list)
        return BatchMatches(
            keys=iscc_ids_array,
            distances=result.distances,
            counts=result.counts,
            visited_members=result.visited_members,
            computed_distances=result.computed_distances,
        )

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

    def _vector_to_iscc_unit(self, vector):
        # type: (NDArray[np.uint8]) -> str
        """Convert binary vector to ISCC-UNIT string."""
        # Parse unit_type to extract MT, ST, VS
        parts = self.unit_type.split("-")
        mt = getattr(ic.MT, parts[0])

        # Handle subtype (either ST.NONE or ST_CC.*)
        if parts[1] == "NONE":
            st = ic.ST.NONE
        else:
            st = getattr(ic.ST_CC, parts[1])

        vs = getattr(ic.VS, parts[2])

        # Calculate bit length from vector size
        bit_length = len(vector) * 8

        # Reconstruct ISCC-UNIT using encode_component
        return "ISCC:" + ic.encode_component(mt, st, vs, bit_length, vector.tobytes())
