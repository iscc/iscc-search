"""
Custom metrics for ISCC vector database.

This module provides custom distance metrics for binary vectors, specifically
the Normalized Prefix Hamming Distance (NPHD) for variable-length ISCC codes.
"""

import numpy as np
from numba import carray, cfunc, types  # type: ignore[import-untyped]
from usearch.index import CompiledMetric, MetricKind, MetricSignature

# Maximum supported vector size: 264 bits (33 bytes including length signal)
MAX_BYTES = 33

# Define the Numba signature for binary vectors
# We use uint8 (unsigned char) for binary data
NPHD_SIGNATURE = types.float32(
    types.CPointer(types.uint8),
    types.CPointer(types.uint8),
)


@cfunc(NPHD_SIGNATURE)
def nphd_distance(a, b):  # type: ignore[no-any-unimported]  # pragma: no cover
    # type: (types.CPointer[types.uint8], types.CPointer[types.uint8]) -> types.float32
    """
    Calculate NPHD between two variable-length binary vectors.

    Uses MAX_BYTES constant for buffer size.
    """
    a_array = carray(a, MAX_BYTES)
    b_array = carray(b, MAX_BYTES)

    # Extract length from first byte (1-255 bytes)
    a_bytes = types.int32(a_array[0])
    b_bytes = types.int32(b_array[0])

    # Use the shorter length for comparison (prefix compatibility)
    min_bytes = min(a_bytes, b_bytes)

    # Calculate Hamming distance over the common prefix
    # Start from byte 1 (skip length signal byte)
    hamming_distance = types.int32(0)
    for byte_idx in range(1, min_bytes + 1):
        # XOR the bytes and count differing bits
        xor_result = types.uint8(a_array[byte_idx] ^ b_array[byte_idx])
        # Count set bits using Brian Kernighan's algorithm
        while xor_result > 0:
            hamming_distance += 1
            xor_result = types.uint8(xor_result & (xor_result - 1))

    # Convert to bits for normalization
    min_bits = min_bytes * 8

    # Normalize by the length of the shorter vector
    if min_bits == 0:
        return types.float32(0.0)

    normalized_distance = types.float32(hamming_distance) / types.float32(min_bits)
    return normalized_distance


def create_nphd_metric():  # type: ignore[no-any-unimported]
    # type: () -> CompiledMetric
    """
    Create a Normalized Prefix Hamming Distance (NPHD) metric for variable-length binary vectors.

    NPHD is designed for prefix-compatible codes where shorter codes are prefixes of longer codes.
    It normalizes the Hamming distance by the length of the common prefix (shorter vector).

    The metric expects binary vectors with a length signal in the first byte:
    - First byte: length signal (1-255) indicating actual vector length in bytes
    - Remaining bytes: actual binary vector data

    Uses a fixed maximum of 264 bits (33 bytes) for vector storage.

    :return: CompiledMetric instance for use with usearch Index
    """
    # Create and return the compiled metric
    return CompiledMetric(
        pointer=nphd_distance.address,
        kind=MetricKind.Hamming,  # Use Hamming as base kind
        signature=MetricSignature.ArrayArray,
    )


def pack_binary_vector(vector_bytes, max_bytes=32):
    # type: (bytes | np.ndarray, int) -> np.ndarray
    """
    Pack a binary vector with a length signal.

    :param vector_bytes: Binary data (1-255 bytes)
    :param max_bytes: Maximum bytes for padding (default: 32)
    :return: Packed vector with length signal as first byte
    """
    if isinstance(vector_bytes, bytes):
        vector_bytes = np.frombuffer(vector_bytes, dtype=np.uint8)

    length = len(vector_bytes)
    if length < 1 or length > 255:
        msg = f"Vector must be 1-255 bytes, got {length}"
        raise ValueError(msg)

    if length > max_bytes:
        msg = f"Vector length {length} exceeds max_bytes {max_bytes}"
        raise ValueError(msg)

    # Create packed vector with length signal + padded data
    packed = np.zeros(max_bytes + 1, dtype=np.uint8)
    packed[0] = length  # Store actual length in first byte
    packed[1 : length + 1] = vector_bytes

    return packed


def unpack_binary_vector(packed_vector):
    # type: (np.ndarray) -> bytes
    """
    Unpack a binary vector to extract the original binary data.

    :param packed_vector: Packed vector with length signal
    :return: Original binary data as bytes
    """
    actual_bytes = int(packed_vector[0])  # Convert to Python int to avoid numpy overflow
    return bytes(packed_vector[1 : actual_bytes + 1])
