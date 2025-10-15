"""
Scalable ANNS search for variable-length binary bit-vectors with NPHD metric.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from usearch.index import Index, KeyOrKeysLike, ScalarKind, VectorOrVectorsLike

from iscc_vdb.metrics import create_nphd_metric


class NphdIndex(Index):
    """Fast approximate nearest neighbor search for variable-length binary bit-vectors.

    Supports Normalized Prefix Hamming Distance (NPHD) metric and packed binary vectors
    as np.uint8 arrays of variable length. Vector keys must be integers.
    Vector keys must be integers.

    Example:

        >>> index = NphdIndex(max_dim=32)
        >>> vector = np.array([32, 64, 128, 255], dtype=np.uint8)])
        >>> index.add(1, vector))
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
        # type: (KeyOrKeysLike, VectorOrVectorsLike, Any) -> int | NDArray[np.uint64]
        """Insert a single vector into the index."""

        # Convert 1D vector to 2D matrix
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, len(vectors))

        return super().add(keys, pad_vector(vectors, self.max_bytes), **kwargs)


def pad_vector(vector, nbytes):
    # type: (NDArray[np.uint8], int) -> NDArray[np.uint8]
    """Add length prefix and padding to a packed binary bit-vector."""
    length = len(vector)
    padded = np.zeros(nbytes + 1, dtype=np.uint8)
    padded[0] = length
    padded[1 : length + 1] = vector
    return padded


def unpad_vector(vector):
    # type: (NDArray[np.uint8]) -> NDArray[np.uint8]
    """Remove length prefix and padding from the packed binary bit-vector."""
    original_length = vector[0]
    return vector[1 : original_length + 1]
