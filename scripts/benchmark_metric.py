"""
Benchmark comparing usearch internal hamming metric against our custom NPHD metric for index
add/search methods.

Compares "Writes per second" and "Searches per second" with 1000, 100.000, entries in indexes
with 64-bit and 256-bit vectors. For all index sizes we add entries with a single batch call, we search
entries in batches of 100.
"""

import time
import numpy as np
from usearch.index import Index, MetricKind, ScalarKind

from iscc_vdb.nphd import NphdIndex


def generate_test_vectors(count, nbits):
    # type: (int, int) -> tuple[np.ndarray, np.ndarray]
    """
    Generate random binary vectors and integer keys for testing.

    :param count: Number of vectors to generate
    :param nbits: Number of bits per vector (64 or 256)
    :return: (keys, vectors) tuple where vectors are uint8 arrays
    """
    nbytes = nbits // 8
    keys = np.arange(count, dtype=np.uint64)
    vectors = np.random.randint(0, 256, (count, nbytes), dtype=np.uint8)
    return keys, vectors


def benchmark_hamming_index(count, nbits):
    # type: (int, int) -> tuple[float, float]
    """
    Benchmark usearch Index with built-in hamming metric.

    :param count: Number of vectors to add
    :param nbits: Number of bits per vector
    :return: (writes_per_second, searches_per_second)
    """
    # Create index with built-in hamming metric
    index = Index(
        ndim=nbits,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
    )

    # Generate test data
    keys, vectors = generate_test_vectors(count, nbits)

    # Benchmark: Add all entries in single batch
    start = time.perf_counter()
    index.add(keys, vectors)
    add_elapsed = time.perf_counter() - start
    writes_per_second = count / add_elapsed

    # Benchmark: Search in batches of 100
    batch_size = 100
    num_batches = count // batch_size
    search_vectors = vectors[: num_batches * batch_size]

    start = time.perf_counter()
    for i in range(0, num_batches * batch_size, batch_size):
        batch = search_vectors[i : i + batch_size]
        index.search(batch, count=10)
    search_elapsed = time.perf_counter() - start
    searches_per_second = (num_batches * batch_size) / search_elapsed

    return writes_per_second, searches_per_second


def benchmark_nphd_index(count, nbits):
    # type: (int, int) -> tuple[float, float]
    """
    Benchmark NphdIndex with custom NPHD metric.

    :param count: Number of vectors to add
    :param nbits: Number of bits per vector
    :return: (writes_per_second, searches_per_second)
    """
    # Create index with custom NPHD metric
    index = NphdIndex(max_dim=256)

    # Generate test data
    keys, vectors = generate_test_vectors(count, nbits)

    # Benchmark: Add all entries in single batch
    start = time.perf_counter()
    index.add(keys, vectors)
    add_elapsed = time.perf_counter() - start
    writes_per_second = count / add_elapsed

    # Benchmark: Search in batches of 100
    batch_size = 100
    num_batches = count // batch_size
    search_vectors = vectors[: num_batches * batch_size]

    start = time.perf_counter()
    for i in range(0, num_batches * batch_size, batch_size):
        batch = search_vectors[i : i + batch_size]
        index.search(batch, count=10)
    search_elapsed = time.perf_counter() - start
    searches_per_second = (num_batches * batch_size) / search_elapsed

    return writes_per_second, searches_per_second


def print_results(metric_name, nbits, count, writes_per_sec, searches_per_sec):
    # type: (str, int, int, float, float) -> None
    """Print formatted benchmark results."""
    print(f"  {metric_name:15s} | {nbits:3d}-bit | {count:8,} | {writes_per_sec:15,.0f} | {searches_per_sec:17,.0f}")


def main():
    # type: () -> None
    """Run comprehensive benchmarks comparing hamming vs NPHD metrics."""
    print("=" * 90)
    print("Metric Comparison Benchmark: Hamming vs NPHD")
    print("=" * 90)
    print()
    print("Comparing usearch built-in Hamming metric against custom NPHD metric.")
    print("Tests measure add (writes) and search performance.")
    print()

    # Test configurations
    test_configs = [
        (1_000, 64, "1K entries, 64-bit"),
        (1_000, 256, "1K entries, 256-bit"),
        (100_000, 64, "100K entries, 64-bit"),
        (100_000, 256, "100K entries, 256-bit"),
    ]

    print(f"{'Metric':<15s} | {'Bits':>7s} | {'Count':>8s} | {'Writes/sec':>15s} | {'Searches/sec':>17s}")
    print("-" * 90)

    results = []  # type: list[tuple[str, int, int, float, float, float, float]]

    for count, nbits, config_name in test_configs:
        print(f"\n{config_name}")
        print("-" * 90)

        # Benchmark Hamming
        hamming_writes, hamming_searches = benchmark_hamming_index(count, nbits)
        print_results("Hamming", nbits, count, hamming_writes, hamming_searches)

        # Benchmark NPHD
        nphd_writes, nphd_searches = benchmark_nphd_index(count, nbits)
        print_results("NPHD", nbits, count, nphd_writes, nphd_searches)

        results.append((config_name, nbits, count, hamming_writes, hamming_searches, nphd_writes, nphd_searches))

    # Summary comparison
    print()
    print("=" * 90)
    print("Performance Comparison Summary")
    print("=" * 90)
    print()
    print(f"{'Configuration':<25s} | {'Operation':>10s} | {'Hamming':>15s} | {'NPHD':>15s} | {'Ratio':>10s}")
    print("-" * 90)

    for config_name, nbits, count, h_writes, h_searches, n_writes, n_searches in results:
        write_ratio = h_writes / n_writes
        search_ratio = h_searches / n_searches

        print(f"{config_name:<25s} | {'Writes':>10s} | {h_writes:15,.0f} | {n_writes:15,.0f} | {write_ratio:9.2f}x")
        print(f"{'':<25s} | {'Searches':>10s} | {h_searches:15,.0f} | {n_searches:15,.0f} | {search_ratio:9.2f}x")
        print()

    print("=" * 90)
    print("Benchmark Complete")
    print()
    print("Note: Ratio > 1.0 means Hamming is faster, < 1.0 means NPHD is faster")
    print("=" * 90)


if __name__ == "__main__":
    main()
