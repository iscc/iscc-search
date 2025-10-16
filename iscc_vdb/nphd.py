"""
Scalable ANNS search for variable-length binary bit-vectors with NPHD metric.
"""

from collections.abc import Sequence
from typing import Any, Union

import numpy as np
from numba import njit
from numpy.typing import NDArray
from usearch.index import BatchMatches, Index, Matches, ScalarKind

from iscc_vdb.metrics import create_nphd_metric

# Type aliases for cleaner type comments
Key = int | None
Keys = Sequence[int] | None
Vector = NDArray[np.uint8]
Vectors = Sequence[NDArray[np.uint8]] | NDArray[np.uint8]


@njit(cache=True)
def pad_vectors(vectors, nbytes):  # pragma: no cover
    # type: (Vectors, int) -> NDArray[np.uint8]
    """
    Add length prefix and padding to a batch of variable-length bit-vectors.

    Prepends each vector with a length byte and pads to uniform size for fixed-size index storage.
    First byte stores original length, followed by vector bytes and zero padding up to nbytes.

    :param vectors: Sequence of variable-length uint8 arrays or 2D uint8 array with uniform-length vectors.
    :param nbytes: Maximum number of bytes per vector (excluding length prefix byte).
    :return: 2D array of shape (batch_size, nbytes + 1) with length-prefixed padded vectors.
    """
    batch_size = len(vectors)
    padded = np.zeros((batch_size, nbytes + 1), dtype=np.uint8)
    for i in range(batch_size):
        vec = vectors[i]
        length = len(vec)
        padded[i, 0] = length
        for j in range(min(length, nbytes)):
            padded[i, j + 1] = vec[j]
    return padded


@njit(cache=True)
def unpad_vectors(padded):  # pragma: no cover
    # type: (NDArray[np.uint8]) -> list[Vector]
    """
    Extract variable-length bit-vectors from length-prefixed padded matrix.

    Reverses the pad_vectors operation by reading length prefix from first byte
    and extracting only the valid data bytes for each vector.

    :param padded: 2D array of shape (batch_size, nbytes + 1) with length-prefixed padded vectors.
    :return: List of variable-length uint8 arrays without padding.
    """
    batch_size = len(padded)
    result = []
    for i in range(batch_size):
        length = int(padded[i, 0])
        vector = padded[i, 1 : length + 1].copy()
        result.append(vector)
    return result


class NphdIndex(Index):
    """Fast approximate nearest neighbor search for variable-length binary bit-vectors.

    Supports Normalized Prefix Hamming Distance (NPHD) metric and packed binary vectors
    as np.uint8 arrays of variable length. Vector keys must be integers.
    """

    def __init__(self, max_dim=256, **kwargs):
        # type: (int, Any) -> None
        """Create a new NPHD index."""
        self.max_dim = max_dim
        self.max_bytes = max_dim // 8

        assert "ndim" not in kwargs, "`ndim` is calculated from `max_dim`"
        assert "metric" not in kwargs, "`metric` is set automatically (NPHD)"
        assert "dtype" not in kwargs, "`dtype` is set automatically (ScalarKind.B1)"

        metric = create_nphd_metric()

        super().__init__(
            ndim=max_dim + 8,  # + 8 bits for length signal byte
            metric=metric,
            dtype=ScalarKind.B1,
            **kwargs,
        )

    def add(self, keys, vectors, **kwargs):
        # type: (Key | Sequence[int], Vectors, Any) -> NDArray[np.uint64]
        """
        Add variable-length binary vectors to the index.

        :param keys: Integer key(s) or None for auto-generation
        :param vectors: Single vector, 2D array of uniform vectors, or list of variable-length vectors
        :param kwargs: Additional arguments passed to parent Index.add()
        :return: Array of keys for added vectors
        """
        # Handle single vector - wrap in list for padding
        if hasattr(vectors, "ndim") and vectors.ndim == 1:
            vectors = [vectors]

        # Pad vectors to uniform size
        padded = pad_vectors(vectors, self.max_bytes)

        # Call parent add with padded vectors
        return super().add(keys, padded, **kwargs)

    def get(self, keys, dtype=None):
        # type: (int | Sequence[int], Any) -> Vector | list | None
        """
        Retrieve unpadded variable-length vectors by key(s).

        :param keys: Integer key(s) to lookup
        :param dtype: Optional data type (defaults to index dtype)
        :return: Unpadded vector(s) or None for missing keys
        """
        results = super().get(keys, dtype=dtype)

        if results is None:
            return None

        if isinstance(results, np.ndarray):
            if results.ndim == 1:
                return unpad_vectors(results.reshape(1, -1))[0]
            return unpad_vectors(results)

        return [
            None if r is None else unpad_vectors(r.reshape(1, -1))[0] if r.ndim == 1 else unpad_vectors(r)
            for r in results
        ]

    def search(self, vectors, count=10, **kwargs):
        # type: (Vectors, int, Any) -> Union[Matches, BatchMatches]
        """
        Search for nearest neighbors of query vector(s).

        :param vectors: Single vector or batch of variable-length vectors to query
        :param count: Maximum number of nearest neighbors to return per query
        :param kwargs: Additional arguments passed to parent Index.search()
        :return: Matches for single query or BatchMatches for batch queries
        """
        # Handle single vector - wrap in list for padding
        if hasattr(vectors, "ndim") and vectors.ndim == 1:
            vectors = [vectors]

        # Pad vectors to uniform size
        padded = pad_vectors(vectors, self.max_bytes)

        # Call parent search with padded vectors
        return super().search(padded, count=count, **kwargs)
