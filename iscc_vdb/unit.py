"""
Scalable ANNS search for variable-length ISCC-UNITs.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import iscc_core as ic
import numpy as np
from numpy.typing import NDArray
from usearch.index import BatchMatches, Matches

from iscc_vdb.nphd import NphdIndex


@dataclass
class UnitMatch:
    """Single search result with ISCC-ID key and distance."""

    key: str
    distance: float

    def to_tuple(self):
        # type: () -> tuple
        """Convert to (key, distance) tuple."""
        return self.key, self.distance


class UnitMatches:
    """Search results for a single ISCC-UNIT query with ISCC-ID string keys.

    Wraps usearch Matches object to provide ISCC-ID strings as keys instead of integers.
    """

    def __init__(self, keys, distances, visited_members, computed_distances):
        # type: (NDArray, NDArray, int, int) -> None
        """Create UnitMatches from search results.

        :param keys: NumPy array of ISCC-ID strings
        :param distances: NumPy array of distances
        :param visited_members: Number of graph nodes visited
        :param computed_distances: Number of distance computations
        """
        self.keys = keys
        self.distances = distances
        self.visited_members = visited_members
        self.computed_distances = computed_distances

    def __len__(self):
        # type: () -> int
        """Return number of matches."""
        return len(self.keys)

    def __getitem__(self, index):
        # type: (int) -> UnitMatch
        """Get single match by index."""
        if isinstance(index, int) and index < len(self):
            return UnitMatch(
                key=self.keys[index],
                distance=float(self.distances[index]),
            )
        else:
            raise IndexError(f"`index` must be an integer under {len(self)}")

    def to_list(self):
        # type: () -> list[tuple]
        """Convert to list of (key, distance) tuples."""
        return [(key, float(distance)) for key, distance in zip(self.keys, self.distances)]

    def __repr__(self):
        # type: () -> str
        """Return string representation."""
        return f"UnitMatches({len(self)})"


class UnitBatchMatches(Sequence):
    """Search results for multiple ISCC-UNIT queries with ISCC-ID string keys.

    Wraps usearch BatchMatches object to provide ISCC-ID strings as keys instead of integers.
    """

    def __init__(self, keys, distances, counts, visited_members, computed_distances):
        # type: (NDArray, NDArray, NDArray, int, int) -> None
        """Create UnitBatchMatches from search results.

        :param keys: 2D NumPy array of ISCC-ID strings (n_queries, k)
        :param distances: 2D NumPy array of distances (n_queries, k)
        :param counts: 1D NumPy array with actual matches per query
        :param visited_members: Total graph nodes visited
        :param computed_distances: Total distance computations
        """
        self.keys = keys
        self.distances = distances
        self.counts = counts
        self.visited_members = visited_members
        self.computed_distances = computed_distances

    def __len__(self):
        # type: () -> int
        """Return number of queries."""
        return len(self.counts)

    def __getitem__(self, index):
        # type: (int) -> UnitMatches
        """Get UnitMatches for a single query by index."""
        if isinstance(index, int) and index < len(self):
            return UnitMatches(
                keys=self.keys[index, : self.counts[index]],
                distances=self.distances[index, : self.counts[index]],
                visited_members=self.visited_members // len(self),
                computed_distances=self.computed_distances // len(self),
            )
        else:
            raise IndexError(f"`index` must be an integer under {len(self)}")

    def to_list(self):
        # type: () -> list[list[tuple]]
        """Convert to list of lists of (key, distance) tuples."""
        list_of_matches = [self.__getitem__(row) for row in range(self.__len__())]
        return [match.to_tuple() for matches in list_of_matches for match in matches]

    def mean_recall(self, expected, count=None):
        # type: (NDArray, int | None) -> float
        """Measure recall [0, 1] of matches containing expected ISCC-IDs."""
        return self.count_matches(expected, count=count) / len(expected)

    def count_matches(self, expected, count=None):
        # type: (NDArray, int | None) -> int
        """Count how many queries found their expected ISCC-ID."""
        assert len(expected) == len(self)
        recall = 0
        if count is None:
            count = self.keys.shape[1]

        if count == 1:
            recall = np.sum(self.keys[:, 0] == expected)
        else:
            for i in range(len(self)):
                recall += expected[i] in self.keys[i, :count]
        return recall

    def __repr__(self):
        # type: () -> str
        """Return string representation."""
        return f"UnitBatchMatches({np.sum(self.counts)} across {len(self)} queries)"


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
        # type: (str | Sequence[str], int, Any) -> UnitMatches | UnitBatchMatches
        """
        Search for nearest neighbors of ISCC-UNIT query vector(s).

        :param vectors: ISCC-UNIT string(s) to query
        :param count: Maximum number of nearest neighbors to return per query
        :param kwargs: Additional arguments passed to parent Index.search()
        :return: UnitMatches for single query or UnitBatchMatches for batch queries with ISCC-ID keys
        """
        # Normalize input and convert ISCC-UNITs to binary vectors
        iscc_units = [vectors] if isinstance(vectors, str) else list(vectors)
        bin_vectors = self._to_vectors(iscc_units)

        # Call parent search
        result = super().search(bin_vectors, count=count, **kwargs)

        # Convert integer keys to ISCC-IDs
        if result.keys.ndim == 1:
            # Single query - return UnitMatches
            iscc_ids = np.array(self._to_iscc_ids(result.keys))
            return UnitMatches(
                keys=iscc_ids,
                distances=result.distances,
                visited_members=result.visited_members,
                computed_distances=result.computed_distances,
            )

        # Batch queries - return UnitBatchMatches
        iscc_ids_list = [self._to_iscc_ids(row) for row in result.keys]
        iscc_ids_array = np.array(iscc_ids_list)
        return UnitBatchMatches(
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
