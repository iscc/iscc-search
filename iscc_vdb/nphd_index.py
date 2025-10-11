"""Minimal NphdIndex implementation using subclassing.

This module provides a simple subclass of usearch.Index that transparently handles
variable-length binary vectors (8, 16, 24, or 32 bytes) using NPHD metric.

Example:
    >>> from iscc_vdb.nphd_index import NphdIndex
    >>> import numpy as np
    >>>
    >>> # Create index
    >>> index = NphdIndex()
    >>>
    >>> # Add variable-length vectors - automatically padded internally
    >>> index.add([1, 2, 3], [
    ...     b'\\xFF' * 8,   # 8-byte vector
    ...     b'\\xAA' * 16,  # 16-byte vector
    ...     b'\\x55' * 32,  # 32-byte vector
    ... ])
    >>>
    >>> # Search works transparently
    >>> matches = index.search(b'\\xFF' * 8, count=2)
    >>> print(matches.keys, matches.distances)
    >>>
    >>> # Save and restore
    >>> index.save("my_index.usearch")
    >>> restored = NphdIndex.restore("my_index.usearch")
"""

import typing

import numpy as np
from usearch.index import Index, ScalarKind

from iscc_vdb.metrics import create_nphd_metric, pack_binary_vector


class NphdIndex(Index):
    """
    Subclass of usearch.Index that transparently handles variable-length binary vectors
    with NPHD (Normalized Prefix Hamming Distance) metric.
    """

    def __init__(self, max_bits=256, **kwargs):
        # type: (int, typing.Any) -> None
        """
        Initialize NphdIndex with NPHD metric.

        :param max_bits: Maximum supported vector size in bits (default: 256)
        :param **kwargs: Additional parameters passed to usearch.Index
        """
        self.max_bits = max_bits
        self.max_bytes = max_bits // 8

        # Create NPHD metric
        metric = create_nphd_metric()

        # Initialize parent with fixed 264-bit vectors (for length signal + max ISCC)
        super().__init__(
            ndim=max_bits + 8,  # +8 bits for length signal byte
            metric=metric,
            dtype=ScalarKind.B1,
            **kwargs,
        )

    def add(self, keys, vectors, **kwargs):
        # type: (typing.Any, typing.Any, typing.Any) -> typing.Any
        """Add vectors with automatic length encoding."""
        # Handle single vector case
        if isinstance(vectors, bytes | bytearray | np.ndarray) and (
            not isinstance(vectors, np.ndarray) or vectors.ndim == 1
        ):
            # Single vector
            packed = pack_binary_vector(vectors, max_bytes=self.max_bytes)
            return super().add(keys, packed, **kwargs)

        # Handle multiple vectors
        prepared = []
        if isinstance(vectors, list):
            for v in vectors:
                prepared.append(pack_binary_vector(v, max_bytes=self.max_bytes))
        elif isinstance(vectors, np.ndarray) and vectors.ndim == 2:
            for i in range(vectors.shape[0]):
                prepared.append(pack_binary_vector(vectors[i], max_bytes=self.max_bytes))
        else:
            # Single vector in unexpected format
            packed = pack_binary_vector(vectors, max_bytes=self.max_bytes)
            return super().add(keys, packed, **kwargs)

        # Stack prepared vectors and call parent
        prepared_array = np.vstack(prepared) if len(prepared) > 1 else prepared[0]
        return super().add(keys, prepared_array, **kwargs)

    def search(self, vectors, count=10, radius=float("inf"), *, threads=0, exact=False, log=False, progress=None):
        # type: (typing.Any, int, float, int, bool, typing.Any, typing.Any) -> typing.Any
        """Search with automatic vector preparation."""
        # Handle single vector case
        if isinstance(vectors, bytes | bytearray | np.ndarray) and (
            not isinstance(vectors, np.ndarray) or vectors.ndim == 1
        ):
            # Single vector
            packed = pack_binary_vector(vectors, max_bytes=self.max_bytes)
            return super().search(packed, count, radius, threads=threads, exact=exact, log=log, progress=progress)

        # Handle multiple vectors
        prepared = []
        if isinstance(vectors, list):
            for v in vectors:
                prepared.append(pack_binary_vector(v, max_bytes=self.max_bytes))
        elif isinstance(vectors, np.ndarray) and vectors.ndim == 2:
            for i in range(vectors.shape[0]):
                prepared.append(pack_binary_vector(vectors[i], max_bytes=self.max_bytes))
        else:
            # Single vector in unexpected format
            packed = pack_binary_vector(vectors, max_bytes=self.max_bytes)
            return super().search(packed, count, radius, threads=threads, exact=exact, log=log, progress=progress)

        # Stack prepared vectors and call parent
        prepared_array = np.vstack(prepared) if len(prepared) > 1 else prepared[0]
        return super().search(prepared_array, count, radius, threads=threads, exact=exact, log=log, progress=progress)

    def get(self, keys, dtype=None):
        # type: (typing.Any, typing.Any) -> typing.Any
        """
        Retrieve vectors by keys.

        Note: Returns packed vectors with length signal byte.
        To get original vectors, extract bytes 1:length+1 where length = vector[0].
        """
        # Just return the packed vectors - user can unpack if needed
        # WARNING: usearch 2.21.0 returns uninitialized memory for non-existent keys with multi=False
        # The issue is reported and should be fixed in the next release.
        return super().get(keys, dtype)

    @staticmethod
    def restore(path_or_buffer, view=True, **kwargs):
        # type: (typing.Any, bool, typing.Any) -> typing.Any
        """
        Restore an NphdIndex from disk.

        :param path_or_buffer: Path to saved index file
        :param view: If True, use memory mapping
        :return: Restored NphdIndex instance
        """
        # First get metadata to determine dimensions
        metadata = Index.metadata(path_or_buffer)
        if metadata is None:
            msg = f"Failed to read metadata from {path_or_buffer}"
            raise RuntimeError(msg)

        # Calculate max_bits from stored ndim
        max_bits = metadata["dimensions"] - 8  # -8 for length signal

        # Create new NphdIndex instance (this will set up NPHD metric)
        instance = NphdIndex(max_bits=max_bits)

        # Now load the actual data
        if view:
            instance.view(path_or_buffer)
        else:
            instance.load(path_or_buffer)

        return instance
