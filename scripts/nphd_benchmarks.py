#!/usr/bin/env python
"""End-to-end USearch indexing benchmarks comparing NPHD implementations.

This script benchmarks the most promising NPHD implementations that can be
integrated with USearch using the @cfunc decorator. It measures indexing
performance with realistic ISCC data patterns.

Key metrics:
- Indexing time: Total time to build the index
- Indexing rate: Vectors indexed per second
- Distance computations: Estimated based on HNSW algorithm complexity
"""

import gc
import time
from typing import Callable

import numpy as np
from numba import carray, cfunc, types
from numba.types import float32, uint8
from usearch.index import CompiledMetric, Index, MetricKind, MetricSignature, ScalarKind

# Maximum supported vector size (same as production)
MAX_BYTES = 33


# NPHD Implementation 1: Current baseline (Brian Kernighan's algorithm)
@cfunc(float32(types.CPointer(uint8), types.CPointer(uint8)))
def nphd_baseline(a, b):
    # type: (types.CPointer[uint8], types.CPointer[uint8]) -> float32
    """NPHD using Brian Kernighan's algorithm (current implementation)."""
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
        while xor_result > 0:
            hamming_distance += 1
            xor_result = types.uint8(xor_result & (xor_result - 1))

    # Convert to bits for normalization
    min_bits = min_bytes * 8

    if min_bits == 0:
        return types.float32(0.0)

    normalized_distance = types.float32(hamming_distance) / types.float32(min_bits)
    return normalized_distance


# NPHD Implementation 2: Bit manipulation (recommended)
@cfunc(float32(types.CPointer(uint8), types.CPointer(uint8)))
def nphd_optimized(a, b):
    # type: (types.CPointer[uint8], types.CPointer[uint8]) -> float32
    """NPHD using bit manipulation for popcount."""
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
        # Bit manipulation popcount
        v = (v & 0x55) + ((v >> 1) & 0x55)
        v = (v & 0x33) + ((v >> 2) & 0x33)
        v = (v & 0x0F) + ((v >> 4) & 0x0F)
        hamming_distance += v

    # Convert to bits for normalization
    min_bits = min_bytes * 8

    if min_bits == 0:
        return types.float32(0.0)

    normalized_distance = types.float32(hamming_distance) / types.float32(min_bits)
    return normalized_distance


# NPHD Implementation 3: SWAR (SIMD Within A Register)
@cfunc(float32(types.CPointer(uint8), types.CPointer(uint8)))
def nphd_swar(a, b):
    # type: (types.CPointer[uint8], types.CPointer[uint8]) -> float32
    """NPHD using SWAR technique for parallel bit counting."""
    a_array = carray(a, MAX_BYTES)
    b_array = carray(b, MAX_BYTES)

    # Extract length from first byte
    a_bytes = types.uint8(a_array[0])
    b_bytes = types.uint8(b_array[0])

    # Use the shorter length
    min_bytes = min(a_bytes, b_bytes)

    # Calculate Hamming distance
    hamming_distance = types.uint32(0)

    # Process 4 bytes at a time when possible
    i = 1
    while i + 3 <= min_bytes:
        # Pack 4 bytes into uint32
        v = types.uint32(0)
        for j in range(4):
            byte_val = types.uint32(a_array[i + j] ^ b_array[i + j])
            v |= byte_val << (j * 8)

        # SWAR popcount for 32-bit value
        v = v - ((v >> 1) & 0x55555555)
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
        v = (v + (v >> 4)) & 0x0F0F0F0F
        v = (v * 0x01010101) >> 24

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
    min_bits = min_bytes * 8

    if min_bits == 0:
        return types.float32(0.0)

    normalized_distance = types.float32(hamming_distance) / types.float32(min_bits)
    return normalized_distance


# NPHD Implementation 4: Unrolled loops
@cfunc(float32(types.CPointer(uint8), types.CPointer(uint8)))
def nphd_unrolled(a, b):
    # type: (types.CPointer[uint8], types.CPointer[uint8]) -> float32
    """NPHD with manually unrolled loops for better performance."""
    a_array = carray(a, MAX_BYTES)
    b_array = carray(b, MAX_BYTES)

    # Extract length from first byte
    a_bytes = types.uint8(a_array[0])
    b_bytes = types.uint8(b_array[0])

    # Use the shorter length
    min_bytes = min(a_bytes, b_bytes)

    # Calculate Hamming distance
    hamming_distance = types.uint16(0)
    i = 1

    # Process 8 bytes at a time
    while i + 7 <= min_bytes:
        # Unroll 8 iterations
        for j in range(8):
            v = types.uint8(a_array[i + j] ^ b_array[i + j])
            v = (v & 0x55) + ((v >> 1) & 0x55)
            v = (v & 0x33) + ((v >> 2) & 0x33)
            v = (v & 0x0F) + ((v >> 4) & 0x0F)
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

    # Convert to bits for normalization
    min_bits = min_bytes * 8

    if min_bits == 0:
        return types.float32(0.0)

    normalized_distance = types.float32(hamming_distance) / types.float32(min_bits)
    return normalized_distance


def generate_iscc_vectors(n_vectors: int, length_distribution: dict[int, float]) -> list[np.ndarray]:
    """Generate realistic ISCC vectors with specified length distribution.

    Args:
        n_vectors: Number of vectors to generate
        length_distribution: Dict mapping vector lengths (8, 16, 24, 32) to probabilities

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

        # Fill with random binary data
        vector[1 : length + 1] = np.random.randint(0, 256, size=length, dtype=np.uint8)

        vectors.append(vector)

    return vectors


def benchmark_indexing(name: str, metric_func: Callable, vectors: list[np.ndarray]) -> dict[str, float]:
    """Benchmark USearch indexing with a specific NPHD implementation.

    Args:
        name: Name of the implementation
        metric_func: Compiled metric function
        vectors: List of vectors to index

    Returns:
        Dict with timing results
    """
    n_vectors = len(vectors)
    ndim = 264  # 33 bytes * 8 bits

    # Create metric
    metric = CompiledMetric(
        pointer=metric_func.address,
        kind=MetricKind.Hamming,  # Use Hamming as base kind
        signature=MetricSignature.ArrayArray,
    )

    # Warm up
    warmup_index = Index(
        ndim=ndim, metric=metric, dtype=ScalarKind.B1, connectivity=16, expansion_add=128, expansion_search=64
    )
    for i in range(min(100, n_vectors)):
        warmup_index.add(i, vectors[i])
    del warmup_index

    # Actual indexing benchmark
    gc.collect()
    start_time = time.time()

    index = Index(
        ndim=ndim, metric=metric, dtype=ScalarKind.B1, connectivity=16, expansion_add=128, expansion_search=64
    )

    # Add progress reporting for large datasets
    report_interval = max(1000, n_vectors // 10)
    for i, vec in enumerate(vectors):
        index.add(i, vec)

        if n_vectors > 10000 and (i + 1) % report_interval == 0:
            print(f"      {name}: {i + 1:,}/{n_vectors:,} vectors ({(i + 1) / n_vectors * 100:.0f}%)")

    indexing_time = time.time() - start_time

    return {
        "name": name,
        "indexing_time": indexing_time,
        "indexing_rate": n_vectors / indexing_time,
        "vectors_per_ms": n_vectors / (indexing_time * 1000),
        "index_size": index.size,
    }


def print_results_table(results):
    # type: (list[dict[str, float]]) -> None
    """Print formatted results table."""
    if not results:
        return

    print(f"\n{'Implementation':<30} {'Index Time':<12} {'Index Rate':<15} {'Vectors/ms':<15} {'Speedup':<10}")
    print("-" * 85)

    baseline_index_time = results[0]["indexing_time"]

    for r in results:
        index_speedup = baseline_index_time / r["indexing_time"] if r["indexing_time"] > 0 else 0

        print(
            f"{r['name']:<30} "
            f"{r['indexing_time']:>10.3f}s "
            f"{r['indexing_rate']:>13.1f}/s "
            f"{r['vectors_per_ms']:>13.2f} "
            f"{index_speedup:>8.2f}x"
        )


def run_benchmark_suite(implementations, vectors, test_name=""):
    # type: (list[tuple[str, Callable]], list[np.ndarray], str) -> list[dict[str, float]]
    """Run benchmarks for all implementations."""
    results = []
    for name, func in implementations:
        try:
            if test_name:
                print(f"\nTesting {name}...")
            result = benchmark_indexing(name, func, vectors)
            results.append(result)
            if test_name:
                print(f"  Indexing: {result['indexing_time']:.2f}s ({result['indexing_rate']:.1f} vectors/s)")
                print(f"  Vectors per ms: {result['vectors_per_ms']:.2f}")
                print(f"  Index size: {result['index_size']:,} vectors")
        except Exception as e:
            print(f"Error with {name}: {e}")
    return results


def main():
    """Run end-to-end benchmarks comparing NPHD implementations."""
    print("NPHD End-to-End USearch Indexing Benchmarks")
    print("=" * 60)

    # Implementation configurations
    implementations = [
        ("Baseline (Kernighan)", nphd_baseline),
        ("Optimized (Bit Manipulation)", nphd_optimized),
        ("SWAR (Parallel)", nphd_swar),
        ("Unrolled Loops", nphd_unrolled),
    ]

    # Test configurations
    test_sizes = [1000, 5000, 10000]  # Reduced max size for faster runs
    length_distributions = {
        "Mixed": {8: 0.1, 16: 0.3, 24: 0.3, 32: 0.3},
    }

    # Run benchmarks
    for size in test_sizes:
        print(f"\nDataset size: {size:,} vectors")
        print("-" * 60)

        for dist_name, dist in length_distributions.items():
            print(f"\nLength distribution: {dist_name}")
            print(f"Distribution: {dist}")

            # Generate test data
            vectors = generate_iscc_vectors(size, dist)

            # Benchmark each implementation
            results = run_benchmark_suite(implementations, vectors)

            # Display results
            print_results_table(results)

    # Additional performance test with larger dataset
    print("\n" + "=" * 60)
    print("Large-scale performance test (25k vectors)")
    print("=" * 60)

    large_vectors = generate_iscc_vectors(25000, {8: 0.1, 16: 0.3, 24: 0.3, 32: 0.3})

    results = run_benchmark_suite(implementations, large_vectors, "large-scale")

    # Summary comparison
    if results:
        print("\n" + "=" * 60)
        print("Summary: Indexing Performance Comparison")
        print("=" * 60)

        baseline = results[0]["indexing_time"]
        for r in results:
            speedup = baseline / r["indexing_time"]
            improvement = (speedup - 1) * 100
            print(f"{r['name']:<30}: {speedup:>6.2f}x speedup ({improvement:>+6.1f}% improvement)")

        # Estimated distance computations analysis
        print("\n" + "=" * 60)
        print("Estimated Distance Computations Per Second")
        print("=" * 60)
        print("\nNote: HNSW indexing performs approximately M * log2(N) distance computations")
        print("where M is the connectivity parameter (16) and N is the number of vectors.")

        n = 25000
        m = 16
        estimated_computations = n * m * np.log2(n)

        print(f"\nFor {n:,} vectors with M={m}:")
        print(f"Estimated total distance computations: {estimated_computations:,.0f}")

        for r in results:
            comp_per_sec = estimated_computations / r["indexing_time"]
            print(f"{r['name']:<30}: {comp_per_sec:>15,.0f} computations/s")


if __name__ == "__main__":
    main()
