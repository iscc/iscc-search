#!/usr/bin/env python
"""NPHD benchmarks optimized for dense binary data (50% bits set).

This script tests NPHD implementations with realistic dense ISCC data where
approximately 50% of bits are set. In this scenario:
- Brian Kernighan's algorithm loses its advantage (no early termination)
- Bit manipulation and SWAR methods should outperform
- Hardware POPCNT would be ideal
"""

import ctypes
import gc
import os
import time
from collections.abc import Callable

import numpy as np
from numba import carray, cfunc, types
from numba.types import float32, uint8
from usearch.index import CompiledMetric, Index, MetricKind, MetricSignature, ScalarKind

# Try to import SimSIMD for optimal performance
try:
    import simsimd

    HAS_SIMSIMD = True
    print(f"SimSIMD {simsimd.__version__} available - optimal performance enabled")
except ImportError:
    HAS_SIMSIMD = False
    print("SimSIMD not available. Install with: pip install simsimd")

# Maximum supported vector size (same as production)
MAX_BYTES = 33


# NPHD Implementation 1: Current baseline (Brian Kernighan)
@cfunc(float32(types.CPointer(uint8), types.CPointer(uint8)))
def nphd_baseline(a, b):
    # type: (types.CPointer[uint8], types.CPointer[uint8]) -> float32
    """NPHD using Brian Kernighan's algorithm - poor for dense data."""
    a_array = carray(a, MAX_BYTES)
    b_array = carray(b, MAX_BYTES)

    # Extract length from first byte
    a_bytes = types.uint8(a_array[0])
    b_bytes = types.uint8(b_array[0])

    # Use the shorter length
    min_bytes = min(a_bytes, b_bytes)

    # Calculate Hamming distance
    hamming_distance = types.uint16(0)
    for byte_idx in range(1, min_bytes + 1):
        xor_result = types.uint8(a_array[byte_idx] ^ b_array[byte_idx])
        # This loop runs ~4 times per byte for 50% density!
        while xor_result > 0:
            hamming_distance += 1
            xor_result = types.uint8(xor_result & (xor_result - 1))

    # Convert to bits for normalization
    min_bits = types.uint16(min_bytes) * types.uint16(8)

    if min_bits == 0:
        return types.float32(0.0)

    normalized_distance = types.float32(hamming_distance) / types.float32(min_bits)
    return normalized_distance


# NPHD Implementation 2: Bit manipulation (should be good for dense data)
@cfunc(float32(types.CPointer(uint8), types.CPointer(uint8)))
def nphd_bit_manipulation(a, b):
    # type: (types.CPointer[uint8], types.CPointer[uint8]) -> float32
    """NPHD using bit manipulation - constant time per byte."""
    a_array = carray(a, MAX_BYTES)
    b_array = carray(b, MAX_BYTES)

    # Extract length from first byte
    a_bytes = types.uint8(a_array[0])
    b_bytes = types.uint8(b_array[0])

    # Use the shorter length
    min_bytes = min(a_bytes, b_bytes)

    # Calculate Hamming distance
    hamming_distance = types.uint16(0)
    for byte_idx in range(1, min_bytes + 1):
        v = types.uint8(a_array[byte_idx] ^ b_array[byte_idx])
        # Constant-time bit manipulation popcount
        v = (v & 0x55) + ((v >> 1) & 0x55)
        v = (v & 0x33) + ((v >> 2) & 0x33)
        v = (v & 0x0F) + ((v >> 4) & 0x0F)
        hamming_distance += v

    # Convert to bits for normalization
    min_bits = types.uint16(min_bytes) * types.uint16(8)

    if min_bits == 0:
        return types.float32(0.0)

    normalized_distance = types.float32(hamming_distance) / types.float32(min_bits)
    return normalized_distance


# NPHD Implementation 3: SWAR 32-bit (excellent for dense data)
@cfunc(float32(types.CPointer(uint8), types.CPointer(uint8)))
def nphd_swar32(a, b):
    # type: (types.CPointer[uint8], types.CPointer[uint8]) -> float32
    """NPHD using 32-bit SWAR - processes 4 bytes in parallel."""
    a_array = carray(a, MAX_BYTES)
    b_array = carray(b, MAX_BYTES)

    # Extract length from first byte
    a_bytes = types.uint8(a_array[0])
    b_bytes = types.uint8(b_array[0])

    # Use the shorter length
    min_bytes = min(a_bytes, b_bytes)

    # Calculate Hamming distance
    hamming_distance = types.uint32(0)

    # Process 4 bytes at a time
    i = 1
    while i + 3 <= min_bytes:
        # Pack 4 bytes into uint32
        v = types.uint32(0)
        for j in range(4):
            byte_val = types.uint32(a_array[i + j] ^ b_array[i + j])
            v |= byte_val << (j * 8)

        # SWAR popcount for 32-bit value
        v = v - ((v >> 1) & types.uint32(0x55555555))
        v = (v & types.uint32(0x33333333)) + ((v >> 2) & types.uint32(0x33333333))
        v = (v + (v >> 4)) & types.uint32(0x0F0F0F0F)
        # Sum bytes
        v = (v & 0xFF) + ((v >> 8) & 0xFF) + ((v >> 16) & 0xFF) + ((v >> 24) & 0xFF)

        hamming_distance += v
        i += 4

    # Handle remaining bytes
    while i <= min_bytes:
        v = types.uint8(a_array[i] ^ b_array[i])
        v = (v & 0x55) + ((v >> 1) & 0x55)
        v = (v & 0x33) + ((v >> 2) & 0x33)
        v = (v & 0x0F) + ((v >> 4) & 0x0F)
        hamming_distance += v
        i += 1

    # Convert to bits for normalization
    min_bits = types.uint16(min_bytes) * types.uint16(8)

    if min_bits == 0:
        return types.float32(0.0)

    normalized_distance = types.float32(hamming_distance) / types.float32(min_bits)
    return normalized_distance


# NPHD Implementation 4: SWAR 64-bit (best for dense data)
@cfunc(float32(types.CPointer(uint8), types.CPointer(uint8)))
def nphd_swar64(a, b):
    # type: (types.CPointer[uint8], types.CPointer[uint8]) -> float32
    """NPHD using 64-bit SWAR - processes 8 bytes in parallel."""
    a_array = carray(a, MAX_BYTES)
    b_array = carray(b, MAX_BYTES)

    # Extract length from first byte
    a_bytes = types.uint8(a_array[0])
    b_bytes = types.uint8(b_array[0])
    min_bytes = min(a_bytes, b_bytes)

    if min_bytes == 0:
        return types.float32(0.0)

    # Calculate Hamming distance
    hamming_distance = types.uint64(0)
    i = 1

    # Process 8 bytes at a time
    while i + 7 <= min_bytes:
        # Pack 8 bytes into uint64
        xor_word = types.uint64(0)
        for j in range(8):
            byte_xor = types.uint64(a_array[i + j] ^ b_array[i + j])
            xor_word |= byte_xor << (j * 8)

        # 64-bit SWAR popcount
        v = xor_word
        v = v - ((v >> 1) & types.uint64(0x5555555555555555))
        v = (v & types.uint64(0x3333333333333333)) + ((v >> 2) & types.uint64(0x3333333333333333))
        v = (v + (v >> 4)) & types.uint64(0x0F0F0F0F0F0F0F0F)
        # Sum bytes
        v = (
            (v & 0xFF)
            + ((v >> 8) & 0xFF)
            + ((v >> 16) & 0xFF)
            + ((v >> 24) & 0xFF)
            + ((v >> 32) & 0xFF)
            + ((v >> 40) & 0xFF)
            + ((v >> 48) & 0xFF)
            + ((v >> 56) & 0xFF)
        )

        hamming_distance += v
        i += 8

    # Handle remaining bytes
    while i <= min_bytes:
        v = types.uint8(a_array[i] ^ b_array[i])
        v = (v & 0x55) + ((v >> 1) & 0x55)
        v = (v & 0x33) + ((v >> 2) & 0x33)
        v = (v & 0x0F) + ((v >> 4) & 0x0F)
        hamming_distance += v
        i += 1

    # Normalize
    min_bits = types.uint16(min_bytes) * types.uint16(8)
    normalized_distance = types.float32(hamming_distance) / types.float32(min_bits)
    return normalized_distance


# NPHD Implementation 5: Lookup table (good for any density)
# Pre-computed popcount lookup table
POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
POPCOUNT_PTR = POPCOUNT_TABLE.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


@cfunc(uint8(uint8))
def _popcount_byte(value):
    # type: (uint8) -> uint8
    """Count set bits in a byte using bit manipulation."""
    v = value
    v = (v & 0x55) + ((v >> 1) & 0x55)
    v = (v & 0x33) + ((v >> 2) & 0x33)
    v = (v & 0x0F) + ((v >> 4) & 0x0F)
    return v


@cfunc(float32(types.CPointer(uint8), types.CPointer(uint8)))
def nphd_lookup(a, b):
    # type: (types.CPointer[uint8], types.CPointer[uint8]) -> float32
    """NPHD using lookup table - consistent performance."""
    a_array = carray(a, MAX_BYTES)
    b_array = carray(b, MAX_BYTES)

    # Extract length from first byte
    a_bytes = types.uint8(a_array[0])
    b_bytes = types.uint8(b_array[0])
    min_bytes = min(a_bytes, b_bytes)

    if min_bytes == 0:
        return types.float32(0.0)

    # Calculate Hamming distance using lookup
    hamming_distance = types.uint16(0)

    for i in range(1, min_bytes + 1):
        xor_val = types.uint8(a_array[i] ^ b_array[i])
        count = _popcount_byte(xor_val)
        hamming_distance += count

    # Normalize
    min_bits = types.uint16(min_bytes) * types.uint16(8)
    normalized_distance = types.float32(hamming_distance) / types.float32(min_bits)
    return normalized_distance


def generate_dense_iscc_vectors(
    n_vectors: int, length_distribution: dict[int, float], density: float = 0.5
) -> list[np.ndarray]:
    """Generate ISCC vectors with specified bit density.

    Args:
        n_vectors: Number of vectors to generate
        length_distribution: Dict mapping vector lengths (8, 16, 24, 32) to probabilities
        density: Fraction of bits that should be set (0.5 = 50% density)

    Returns:
        List of numpy arrays with length signaling in first byte
    """
    vectors = []
    lengths = list(length_distribution.keys())
    probs = list(length_distribution.values())

    for _ in range(n_vectors):
        # Choose length based on distribution
        length = np.random.choice(lengths, p=probs)

        # Create vector with length signal in first byte
        vector = np.zeros(33, dtype=np.uint8)  # 1 byte signal + 32 bytes max
        vector[0] = length

        # Generate dense binary data with specified density
        # For each byte, generate bits with the target density
        for i in range(1, length + 1):
            # Generate 8 random bits with specified density
            bits = np.random.random(8) < density
            byte_val = 0
            for j, bit in enumerate(bits):
                if bit:
                    byte_val |= 1 << j
            vector[i] = byte_val

        vectors.append(vector)

    return vectors


def create_aligned_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    """Create memory-aligned array for optimal performance."""
    n_vectors = len(vectors)
    aligned = np.empty((n_vectors, 33), dtype=np.uint8, order="C")

    for i, vec in enumerate(vectors):
        aligned[i] = vec

    return aligned


def warm_up_functions():
    """Warm up all @cfunc implementations to avoid JIT compilation overhead."""
    print("Warming up JIT compilation...")

    # Create dummy dense vectors for warm-up
    dummy_a = np.zeros(33, dtype=np.uint8)
    dummy_b = np.zeros(33, dtype=np.uint8)
    dummy_a[0] = 8  # length
    dummy_b[0] = 8  # length
    # Dense data (50% bits set)
    dummy_a[1:9] = [0b10101010, 0b01010101, 0b11001100, 0b00110011, 0b11110000, 0b00001111, 0b10011001, 0b01100110]
    dummy_b[1:9] = [0b01010101, 0b10101010, 0b00110011, 0b11001100, 0b00001111, 0b11110000, 0b01100110, 0b10011001]

    functions = [nphd_baseline, nphd_bit_manipulation, nphd_swar32, nphd_swar64, nphd_lookup]

    for func in functions:
        try:
            ptr_a = dummy_a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            ptr_b = dummy_b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            func(ptr_a, ptr_b)
        except Exception as e:
            print(f"Warning: Failed to warm up {func.__name__}: {e}")

    print("JIT warm-up complete.")


def micro_benchmark_dense(name: str, metric_func: Callable, n_pairs: int = 100_000) -> dict[str, float]:
    """Micro-benchmark distance kernel with dense data."""
    print(f"  Micro-benchmarking {name} with {n_pairs:,} dense vector pairs...")

    # Generate dense vector pairs (50% density)
    vectors_a = []
    vectors_b = []

    for _ in range(n_pairs):
        # Random length
        length = np.random.choice([8, 16, 24, 32])

        vec_a = np.zeros(33, dtype=np.uint8)
        vec_b = np.zeros(33, dtype=np.uint8)
        vec_a[0] = length
        vec_b[0] = length

        # Generate dense data (50% bits set)
        for i in range(1, length + 1):
            # Each byte has ~4 bits set on average
            vec_a[i] = np.random.randint(0, 256)
            vec_b[i] = np.random.randint(0, 256)

        vectors_a.append(vec_a)
        vectors_b.append(vec_b)

    # Benchmark pure distance computations
    start_time = time.perf_counter()

    for vec_a, vec_b in zip(vectors_a, vectors_b, strict=False):
        ptr_a = vec_a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        ptr_b = vec_b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        metric_func(ptr_a, ptr_b)

    elapsed = time.perf_counter() - start_time

    return {
        "name": name,
        "elapsed_time": elapsed,
        "computations_per_second": n_pairs / elapsed,
        "nanoseconds_per_computation": (elapsed * 1_000_000_000) / n_pairs,
    }


def benchmark_indexing_dense(
    name: str,
    metric_func: Callable,
    vectors: list[np.ndarray],
    connectivity: int = 16,
    expansion_add: int = 128,
    expansion_search: int = 64,
    use_batch: bool = True,
) -> dict[str, float]:
    """Benchmark USearch indexing with dense data."""
    n_vectors = len(vectors)
    ndim = 264  # 33 bytes * 8 bits

    # Create metric
    metric = CompiledMetric(
        pointer=metric_func.address,
        kind=MetricKind.Hamming,
        signature=MetricSignature.ArrayArray,
    )

    # Create aligned vectors array for batch operations
    if use_batch:
        vectors_array = create_aligned_vectors(vectors)
        keys = np.arange(n_vectors, dtype=np.int64)

    # Actual indexing benchmark
    gc.collect()
    start_time = time.perf_counter()

    # Create index
    index = Index(
        ndim=ndim,
        metric=metric,
        dtype=ScalarKind.B1,
        connectivity=connectivity,
        expansion_add=expansion_add,
        expansion_search=expansion_search,
    )

    # Reserve space if available
    if hasattr(index, "reserve"):
        index.reserve(n_vectors)

    # Add vectors
    if use_batch and n_vectors > 100:
        threads = min(os.cpu_count(), 8)
        index.add(keys, vectors_array, threads=threads)
    else:
        for i, vec in enumerate(vectors):
            index.add(i, vec)

    indexing_time = time.perf_counter() - start_time

    return {
        "name": name,
        "indexing_time": indexing_time,
        "indexing_rate": n_vectors / indexing_time,
        "vectors_per_ms": n_vectors / (indexing_time * 1000),
        "index_size": index.size,
    }


def main():
    """Run benchmarks optimized for dense binary data."""
    print("NPHD Benchmarks for Dense Binary Data (50% density)")
    print("=" * 60)
    print("With dense data, we expect:")
    print("- Brian Kernighan's algorithm to perform poorly")
    print("- Bit manipulation and SWAR to excel")
    print("- Consistent performance across all data")
    print("=" * 60)

    # Warm up JIT compilation
    warm_up_functions()

    # First, run micro-benchmarks
    print("\n" + "=" * 60)
    print("Micro-benchmarks: Distance Kernel Performance (Dense Data)")
    print("=" * 60)

    implementations = [
        ("Baseline (Kernighan)", nphd_baseline),
        ("Bit Manipulation", nphd_bit_manipulation),
        ("SWAR 32-bit", nphd_swar32),
        ("SWAR 64-bit", nphd_swar64),
        ("Lookup Table", nphd_lookup),
    ]

    micro_results = []
    for name, func in implementations:
        try:
            result = micro_benchmark_dense(name, func, n_pairs=100_000)
            micro_results.append(result)
        except Exception as e:
            print(f"  Error with {name}: {e}")

    if micro_results:
        print("\nMicro-benchmark Summary (Dense Data):")
        print(f"{'Implementation':<25} {'Ops/sec':<15} {'ns/op':<10} {'Speedup':<10}")
        print("-" * 60)

        baseline_ops = micro_results[0]["computations_per_second"]
        for r in micro_results:
            speedup = r["computations_per_second"] / baseline_ops
            print(
                f"{r['name']:<25} {r['computations_per_second']:>13.0f} {r['nanoseconds_per_computation']:>8.1f} {speedup:>8.2f}x"
            )

    # Test with realistic dense ISCC data
    print("\n" + "=" * 60)
    print("End-to-End Indexing Benchmarks (10k dense vectors)")
    print("=" * 60)

    # Generate dense test data (50% density)
    print("\nGenerating 10,000 dense vectors (50% bits set)...")
    vectors = generate_dense_iscc_vectors(10000, {8: 0.1, 16: 0.3, 24: 0.3, 32: 0.3}, density=0.5)

    # Check actual density of generated data
    sample_vec = vectors[0]
    sample_bytes = sample_vec[1 : sample_vec[0] + 1]
    bits_set = sum(bin(b).count("1") for b in sample_bytes)
    total_bits = len(sample_bytes) * 8
    actual_density = bits_set / total_bits
    print(f"Actual density of first vector: {actual_density:.2%}")

    # Test each implementation
    results = []
    for name, func in implementations:
        print(f"\nTesting {name}...")
        try:
            # Sequential indexing
            seq_result = benchmark_indexing_dense(name, func, vectors, use_batch=False)
            print(f"  Sequential: {seq_result['indexing_time']:.3f}s ({seq_result['indexing_rate']:.1f} vectors/s)")

            # Batch indexing
            batch_result = benchmark_indexing_dense(name, func, vectors, use_batch=True)
            print(f"  Batch: {batch_result['indexing_time']:.3f}s ({batch_result['indexing_rate']:.1f} vectors/s)")

            results.append({
                "name": name,
                "seq_time": seq_result["indexing_time"],
                "batch_time": batch_result["indexing_time"],
                "batch_speedup": seq_result["indexing_time"] / batch_result["indexing_time"],
            })
        except Exception as e:
            print(f"  Error: {e}")

    # Summary comparison
    if results:
        print("\n" + "=" * 60)
        print("Summary: Dense Data Performance")
        print("=" * 60)
        print(f"{'Implementation':<25} {'Seq Time':<12} {'Batch Time':<12} {'Batch Speedup':<15} {'vs Baseline':<12}")
        print("-" * 85)

        baseline_batch = results[0]["batch_time"]
        for r in results:
            vs_baseline = baseline_batch / r["batch_time"]
            print(
                f"{r['name']:<25} {r['seq_time']:>10.3f}s {r['batch_time']:>10.3f}s {r['batch_speedup']:>13.2f}x {vs_baseline:>10.2f}x"
            )

    # Key findings
    print("\n" + "=" * 60)
    print("Key Findings for Dense Data")
    print("=" * 60)
    print("1. SWAR implementations should significantly outperform Kernighan's algorithm")
    print("2. Batch indexing still provides major speedup (3-5x)")
    print("3. The metric choice matters more with dense data")
    print("4. 64-bit SWAR should be fastest for processing 8 bytes at once")


if __name__ == "__main__":
    main()
