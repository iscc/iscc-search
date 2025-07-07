"""
Custom metrics for ISCC vector database.

This module provides custom distance metrics for binary vectors, specifically
the Normalized Prefix Hamming Distance (NPHD) for variable-length ISCC codes.
"""

import numpy as np
from numba import carray, cfunc, types  # type: ignore[import-untyped]
from usearch.index import CompiledMetric, MetricKind, MetricSignature


def create_nphd_metric(max_bits=264):  # type: ignore[no-any-unimported]
    # type: (int) -> CompiledMetric
    """
    Create a Normalized Prefix Hamming Distance (NPHD) metric for variable-length binary vectors.

    NPHD is designed for prefix-compatible codes where shorter codes are prefixes of longer codes.
    It normalizes the Hamming distance by the length of the common prefix (shorter vector).

    The metric expects binary vectors with a length signal in the first byte:
    - First byte: length signal (0-3) indicating vector length in bytes
    - Remaining bytes: actual binary vector data

    :param max_bits: Maximum number of bits supported (default: 264 for 33 bytes)
    :return: CompiledMetric instance for use with usearch Index
    """
    # Calculate maximum bytes needed (including length signal byte)
    max_bytes = (max_bits + 7) // 8

    # Define the Numba signature for binary vectors
    # We use uint8 (unsigned char) for binary data
    signature = types.float32(
        types.CPointer(types.uint8),
        types.CPointer(types.uint8),
    )

    @cfunc(signature)
    def nphd_distance(a, b):  # type: ignore[no-any-unimported]
        # type: (types.CPointer[types.uint8], types.CPointer[types.uint8]) -> types.float32
        """Calculate NPHD between two variable-length binary vectors."""
        a_array = carray(a, max_bytes)
        b_array = carray(b, max_bytes)

        # Extract length signals from first byte
        a_length_signal = types.int32(a_array[0])
        b_length_signal = types.int32(b_array[0])

        # Convert length signals to actual byte lengths
        # 0 -> 8 bytes, 1 -> 16 bytes, 2 -> 24 bytes, 3 -> 32 bytes
        a_bytes = (a_length_signal + 1) * 8
        b_bytes = (b_length_signal + 1) * 8

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

    # Create and return the compiled metric
    return CompiledMetric(
        pointer=nphd_distance.address,
        kind=MetricKind.Hamming,  # Use Hamming as base kind
        signature=MetricSignature.ArrayArray,
    )


def pack_iscc_vector(iscc_bytes, max_bytes=32):
    # type: (bytes | np.ndarray, int) -> np.ndarray
    """
    Pack an ISCC binary vector with a length signal.

    :param iscc_bytes: ISCC binary data (8, 16, 24, or 32 bytes)
    :param max_bytes: Maximum bytes for padding (default: 32)
    :return: Packed vector with length signal as first byte
    """
    if isinstance(iscc_bytes, bytes):
        iscc_bytes = np.frombuffer(iscc_bytes, dtype=np.uint8)

    length = len(iscc_bytes)
    if length not in [8, 16, 24, 32]:
        msg = f"ISCC must be 8, 16, 24, or 32 bytes, got {length}"
        raise ValueError(msg)

    # Calculate length signal (0-3)
    length_signal = (length // 8) - 1

    # Create packed vector with length signal + padded data
    packed = np.zeros(max_bytes + 1, dtype=np.uint8)
    packed[0] = length_signal
    packed[1 : length + 1] = iscc_bytes

    return packed


def unpack_iscc_vector(packed_vector):
    # type: (np.ndarray) -> bytes
    """
    Unpack an ISCC vector to extract the original binary data.

    :param packed_vector: Packed vector with length signal
    :return: Original ISCC binary data as bytes
    """
    length_signal = packed_vector[0]
    actual_bytes = (length_signal + 1) * 8
    return bytes(packed_vector[1 : actual_bytes + 1])
