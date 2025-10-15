"""
Scalable ANNS search for variable-length binary bit-vectors with NPHD metric.
"""

from typing import Any, Sequence

import numpy as np
from numba import njit
from numpy.typing import NDArray
from usearch.index import Index, KeyOrKeysLike, ScalarKind, VectorOrVectorsLike

from iscc_vdb.metrics import create_nphd_metric


class NphdIndex(Index):
    """Fast approximate nearest neighbor search for variable-length binary bit-vectors.

    Supports Normalized Prefix Hamming Distance (NPHD) metric and packed binary vectors
    as np.uint8 arrays of variable length. Vector keys must be integers.
    Vector keys must be integers.
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

    def add_one(self, key, vector, **kwargs):
        # type: (int|None, NDArray[np.uint8], Any) -> int
        """Add a single vector to the index."""
        length = len(vector)
        padded = np.zeros(self.max_bytes + 1, dtype=np.uint8)
        padded[0] = length
        padded[1 : length + 1] = vector
        return super().add(key, padded, **kwargs)

    def add_many(self, keys, vectors, **kwargs):
        """Add multiple vectors to the index."""
        return super().add(keys, pad_vectors(vectors), **kwargs)

    def get_one(self, key):
        # type: (int) -> NDArray[np.uint8] | None
        padded = super().get(key)
        if padded is not None and padded.ndim == 1:
            length = padded[0]
            return padded[1 : length + 1]
        return None

    def add(self, keys, vectors, **kwargs):
        # type: (KeyOrKeysLike, VectorOrVectorsLike, Any) -> int | NDArray[np.uint64]
        """Inserts one or more variable length binary bit-vectors into the index."""
        pass


@njit(cache=True)
def pad_vectors(vectors, nbytes):
    # type: (Sequence[NDArray[np.uint8]] | NDArray[np.uint8], int) -> NDArray[np.uint8]
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
def unpad_vectors(padded):
    # type: (NDArray[np.uint8]) -> list[NDArray[np.uint8]]
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
