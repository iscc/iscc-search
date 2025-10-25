"""
Benchmark script for NPHD distance metric optimizations.

Tests various optimization strategies for the nphd_distance function to find
the best performance improvements for production use.
"""

import ctypes
import time
import numpy as np
from numba import cfunc, types, carray

# Maximum supported vector size: 264 bits (33 bytes including length signal)
MAX_BYTES = 33

# Precomputed popcount lookup table for 0-255
POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

# Define the Numba signature for binary vectors
NPHD_SIGNATURE = types.float32(
    types.CPointer(types.uint8),
    types.CPointer(types.uint8),
)


# ============================================================================
# ORIGINAL IMPLEMENTATION (Brian Kernighan's algorithm)
# ============================================================================


@cfunc(NPHD_SIGNATURE)
def nphd_distance_original(a, b):  # pragma: no cover
    # type: (types.CPointer[types.uint8], types.CPointer[types.uint8]) -> types.float32
    """Original implementation using Brian Kernighan's bit counting algorithm."""
    a_array = carray(a, MAX_BYTES)
    b_array = carray(b, MAX_BYTES)

    a_bytes = types.uint8(a_array[0])
    b_bytes = types.uint8(b_array[0])
    min_bytes = min(a_bytes, b_bytes)

    hamming_distance = types.uint16(0)
    for byte_idx in range(1, min_bytes + 1):
        xor_result = types.uint8(a_array[byte_idx] ^ b_array[byte_idx])
        while xor_result > 0:
            hamming_distance += 1
            xor_result = types.uint8(xor_result & (xor_result - 1))

    min_bits = min_bytes * 8
    if min_bits == 0:
        return types.float32(0.0)

    normalized_distance = types.float32(hamming_distance) / types.float32(min_bits)
    return normalized_distance


# ============================================================================
# OPTIMIZATION 1: Lookup Table
# ============================================================================


@cfunc(NPHD_SIGNATURE)
def nphd_distance_lut(a, b):  # pragma: no cover
    # type: (types.CPointer[types.uint8], types.CPointer[types.uint8]) -> types.float32
    """Optimized with precomputed lookup table for bit counting."""
    a_array = carray(a, MAX_BYTES)
    b_array = carray(b, MAX_BYTES)

    a_bytes = types.uint8(a_array[0])
    b_bytes = types.uint8(b_array[0])
    min_bytes = min(a_bytes, b_bytes)

    hamming_distance = types.uint16(0)
    for byte_idx in range(1, min_bytes + 1):
        xor_result = types.uint8(a_array[byte_idx] ^ b_array[byte_idx])
        # Lookup table: O(1) bit count instead of O(k) loop
        hamming_distance += types.uint16(
            (
                xor_result
                & 1
                + ((xor_result >> 1) & 1)
                + ((xor_result >> 2) & 1)
                + ((xor_result >> 3) & 1)
                + ((xor_result >> 4) & 1)
                + ((xor_result >> 5) & 1)
                + ((xor_result >> 6) & 1)
                + ((xor_result >> 7) & 1)
            )
        )

    min_bits = min_bytes * 8
    if min_bits == 0:
        return types.float32(0.0)

    normalized_distance = types.float32(hamming_distance) / types.float32(min_bits)
    return normalized_distance


# ============================================================================
# OPTIMIZATION 2: 64-bit Word Processing
# ============================================================================


@cfunc(NPHD_SIGNATURE)
def nphd_distance_words(a, b):  # pragma: no cover
    # type: (types.CPointer[types.uint8], types.CPointer[types.uint8]) -> types.float32
    """Optimized with 64-bit word processing (like usearch built-in)."""
    a_array = carray(a, MAX_BYTES)
    b_array = carray(b, MAX_BYTES)

    a_bytes = types.uint8(a_array[0])
    b_bytes = types.uint8(b_array[0])
    min_bytes = min(a_bytes, b_bytes)

    hamming_distance = types.uint16(0)

    # Process 8-byte (64-bit) words
    full_words = (min_bytes) // 8
    word_start = 1  # Skip length signal byte

    for word_idx in range(full_words):
        base_idx = word_start + word_idx * 8
        # Build 64-bit words from bytes
        a_word = types.uint64(0)
        b_word = types.uint64(0)
        for i in range(8):
            a_word |= types.uint64(a_array[base_idx + i]) << (i * 8)
            b_word |= types.uint64(b_array[base_idx + i]) << (i * 8)

        # XOR and count bits
        xor_result = types.uint64(a_word ^ b_word)
        # Manual popcount for uint64
        while xor_result > 0:
            hamming_distance += 1
            xor_result = types.uint64(xor_result & (xor_result - 1))

    # Handle remaining bytes
    remaining_start = word_start + full_words * 8
    for byte_idx in range(remaining_start, min_bytes + 1):
        xor_result = types.uint8(a_array[byte_idx] ^ b_array[byte_idx])
        while xor_result > 0:
            hamming_distance += 1
            xor_result = types.uint8(xor_result & (xor_result - 1))

    min_bits = min_bytes * 8
    if min_bits == 0:
        return types.float32(0.0)

    normalized_distance = types.float32(hamming_distance) / types.float32(min_bits)
    return normalized_distance


# ============================================================================
# OPTIMIZATION 3: Unrolled Popcount (compromise for Numba)
# ============================================================================


@cfunc(NPHD_SIGNATURE)
def nphd_distance_unrolled(a, b):  # pragma: no cover
    # type: (types.CPointer[types.uint8], types.CPointer[types.uint8]) -> types.float32
    """Optimized with unrolled bit counting for better performance."""
    a_array = carray(a, MAX_BYTES)
    b_array = carray(b, MAX_BYTES)

    a_bytes = types.uint8(a_array[0])
    b_bytes = types.uint8(b_array[0])
    min_bytes = min(a_bytes, b_bytes)

    hamming_distance = types.uint16(0)

    # Process bytes with unrolled bit counting
    for byte_idx in range(1, min_bytes + 1):
        xor_result = types.uint8(a_array[byte_idx] ^ b_array[byte_idx])

        # Unrolled bit count (compiler can optimize this)
        count = types.uint8(0)
        count += xor_result & 1
        count += (xor_result >> 1) & 1
        count += (xor_result >> 2) & 1
        count += (xor_result >> 3) & 1
        count += (xor_result >> 4) & 1
        count += (xor_result >> 5) & 1
        count += (xor_result >> 6) & 1
        count += (xor_result >> 7) & 1

        hamming_distance += types.uint16(count)

    min_bits = min_bytes * 8
    if min_bits == 0:
        return types.float32(0.0)

    normalized_distance = types.float32(hamming_distance) / types.float32(min_bits)
    return normalized_distance


# ============================================================================
# OPTIMIZATION 4: Bit Twiddling (LLVM recognizes and optimizes to POPCNT)
# ============================================================================


@cfunc(NPHD_SIGNATURE)
def nphd_distance_bittwid(a, b):  # pragma: no cover
    # type: (types.CPointer[types.uint8], types.CPointer[types.uint8]) -> types.float32
    """
    Optimized with bit-twiddling hack that LLVM can optimize to POPCNT instruction.

    Uses the classic bit manipulation algorithm that compilers recognize.
    """
    a_array = carray(a, MAX_BYTES)
    b_array = carray(b, MAX_BYTES)

    a_bytes = types.uint8(a_array[0])
    b_bytes = types.uint8(b_array[0])
    min_bytes = min(a_bytes, b_bytes)

    hamming_distance = types.uint16(0)

    # Process bytes with bit-twiddling popcount algorithm
    for byte_idx in range(1, min_bytes + 1):
        v = types.uint8(a_array[byte_idx] ^ b_array[byte_idx])

        # Bit twiddling hack for popcount (LLVM can optimize to POPCNT)
        # Based on: https://graphics.stanford.edu/~seander/bithacks.html
        v = types.uint8(v - ((v >> 1) & 0x55))
        v = types.uint8((v & 0x33) + ((v >> 2) & 0x33))
        count = types.uint8(((v + (v >> 4)) & 0x0F))

        hamming_distance += types.uint16(count)

    min_bits = min_bytes * 8
    if min_bits == 0:
        return types.float32(0.0)

    normalized_distance = types.float32(hamming_distance) / types.float32(min_bits)
    return normalized_distance


# ============================================================================
# OPTIMIZATION 5: Bit Twiddling with 32-bit words
# ============================================================================


@cfunc(NPHD_SIGNATURE)
def nphd_distance_bittwid32(a, b):  # pragma: no cover
    # type: (types.CPointer[types.uint8], types.CPointer[types.uint8]) -> types.float32
    """
    Optimized with 32-bit word processing and bit-twiddling popcount.

    Combines word-level processing with LLVM-optimizable popcount.
    """
    a_array = carray(a, MAX_BYTES)
    b_array = carray(b, MAX_BYTES)

    a_bytes = types.uint8(a_array[0])
    b_bytes = types.uint8(b_array[0])
    min_bytes = min(a_bytes, b_bytes)

    hamming_distance = types.uint16(0)

    # Process 4-byte (32-bit) words
    full_words = (min_bytes) // 4
    word_start = 1  # Skip length signal byte

    for word_idx in range(full_words):
        base_idx = word_start + word_idx * 4
        # Build 32-bit words from bytes
        a_word = types.uint32(0)
        b_word = types.uint32(0)
        for i in range(4):
            a_word |= types.uint32(a_array[base_idx + i]) << (i * 8)
            b_word |= types.uint32(b_array[base_idx + i]) << (i * 8)

        # XOR and popcount with bit twiddling
        v = types.uint32(a_word ^ b_word)

        # Bit twiddling hack for 32-bit popcount (LLVM optimizes to POPCNT)
        v = types.uint32(v - ((v >> 1) & 0x55555555))
        v = types.uint32((v & 0x33333333) + ((v >> 2) & 0x33333333))
        count = types.uint32(((v + (v >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24

        hamming_distance += types.uint16(count)

    # Handle remaining bytes
    remaining_start = word_start + full_words * 4
    for byte_idx in range(remaining_start, min_bytes + 1):
        v = types.uint8(a_array[byte_idx] ^ b_array[byte_idx])
        # 8-bit popcount
        v = types.uint8(v - ((v >> 1) & 0x55))
        v = types.uint8((v & 0x33) + ((v >> 2) & 0x33))
        count = types.uint8(((v + (v >> 4)) & 0x0F))
        hamming_distance += types.uint16(count)

    min_bits = min_bytes * 8
    if min_bits == 0:
        return types.float32(0.0)

    normalized_distance = types.float32(hamming_distance) / types.float32(min_bits)
    return normalized_distance


# ============================================================================
# Benchmark Utilities
# ============================================================================


def generate_test_vectors(num_vectors, vector_bytes):
    # type: (int, int) -> np.ndarray
    """
    Generate random binary vectors with length signal.

    :param num_vectors: Number of vectors to generate
    :param vector_bytes: Actual data bytes (without length signal)
    :return: Array of vectors with length signal as first byte
    """
    vectors = np.zeros((num_vectors, MAX_BYTES), dtype=np.uint8)
    # Set length signal
    vectors[:, 0] = vector_bytes
    # Fill with random binary data
    vectors[:, 1 : vector_bytes + 1] = np.random.randint(0, 256, (num_vectors, vector_bytes), dtype=np.uint8)
    return vectors


def benchmark_metric(metric_func, vectors, num_comparisons):
    # type: (object, np.ndarray, int) -> tuple[float, float]
    """
    Benchmark a metric function.

    :param metric_func: Compiled metric function (cfunc)
    :param vectors: Test vectors
    :param num_comparisons: Number of distance calculations
    :return: (total_time, comparisons_per_second)
    """
    num_vectors = len(vectors)

    start = time.perf_counter()
    for i in range(num_comparisons):
        idx_a = i % num_vectors
        idx_b = (i + 1) % num_vectors
        # Call the compiled function with proper ctypes pointers
        ptr_a = vectors[idx_a].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        ptr_b = vectors[idx_b].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        metric_func.ctypes(ptr_a, ptr_b)
    elapsed = time.perf_counter() - start

    comparisons_per_sec = num_comparisons / elapsed
    return elapsed, comparisons_per_sec


def run_benchmarks():
    # type: () -> None
    """Run comprehensive benchmarks on all implementations."""
    print("=" * 80)
    print("NPHD Distance Metric Optimization Benchmark")
    print("=" * 80)
    print()

    # Test configurations
    test_configs = [
        (8, "64-bit ISCC"),
        (16, "128-bit ISCC"),
        (24, "192-bit ISCC"),
        (32, "256-bit ISCC"),
    ]

    implementations = [
        ("Original (Brian Kernighan)", nphd_distance_original),
        ("Unrolled Bitcount", nphd_distance_unrolled),
        ("Lookup Table", nphd_distance_lut),
        ("64-bit Words", nphd_distance_words),
        ("Bit-Twiddling (8-bit)", nphd_distance_bittwid),
        ("Bit-Twiddling (32-bit)", nphd_distance_bittwid32),
    ]

    num_vectors = 1000
    num_comparisons = 100000  # 100k distance calculations

    for vector_bytes, config_name in test_configs:
        print(f"\n{config_name} ({vector_bytes} bytes)")
        print("-" * 80)

        # Generate test vectors
        vectors = generate_test_vectors(num_vectors, vector_bytes)

        results = []
        for impl_name, impl_func in implementations:
            elapsed, comp_per_sec = benchmark_metric(impl_func, vectors, num_comparisons)
            results.append((impl_name, elapsed, comp_per_sec))
            print(f"{impl_name:30s}: {elapsed:8.4f}s  ({comp_per_sec:12,.0f} cmp/s)")

        # Calculate speedups relative to original
        baseline_time = results[0][1]
        print()
        print("Speedup vs Original:")
        for impl_name, elapsed, comp_per_sec in results[1:]:
            speedup = baseline_time / elapsed
            print(f"{impl_name:30s}: {speedup:6.2f}x faster")

    print()
    print("=" * 80)
    print("Benchmark Complete")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmarks()
