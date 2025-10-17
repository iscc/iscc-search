"""
Benchmark script for InstanceIndex performance testing.

Measures:
1. Adding 1 million entries to the index
2. Exact match query speed (get)
3. Prefix search query speed (search)
4. Performance across different Instance-Code bit lengths
"""

import random
import time
import tempfile
from pathlib import Path
from collections import defaultdict
import iscc_core as ic
from iscc_vdb.instance import InstanceIndex


def generate_test_data(count, bit_lengths=(64, 128, 192, 256)):
    # type: (int, tuple[int, ...]) -> tuple[list[str], list[str]]
    """Generate test ISCC-IDs and Instance-Codes with varying bit lengths."""
    print(f"Generating {count:,} test entries...")

    iscc_ids = []
    instance_codes = []
    length_dist = defaultdict(int)  # type: dict[int, int]

    start = time.perf_counter()
    for i in range(count):
        # Generate ISCC-ID
        iscc_id = ic.gen_iscc_id(
            timestamp=1000000 + i,
            hub_id=i % 1000,  # Cycle through hub_ids for diversity
            realm_id=0,
        )["iscc"]
        iscc_ids.append(iscc_id)

        # Generate Instance-Code with varying lengths
        bit_length = bit_lengths[i % len(bit_lengths)]
        instance_code_obj = ic.Code.rnd(ic.MT.INSTANCE, bits=bit_length)
        instance_code = f"ISCC:{instance_code_obj}"
        instance_codes.append(instance_code)
        length_dist[bit_length] += 1

    elapsed = time.perf_counter() - start

    print(f"Generated {len(iscc_ids):,} entries in {elapsed:.2f}s")
    print("\nInstance-Code length distribution:")
    for bits in sorted(length_dist.keys()):
        count = length_dist[bits]
        pct = (count / len(instance_codes)) * 100
        print(f"  {bits:3d} bits: {count:8,} ({pct:5.1f}%)")

    return iscc_ids, instance_codes


def benchmark_add(index, iscc_ids, instance_codes):
    # type: (InstanceIndex, list[str], list[str]) -> float
    """Benchmark adding entries to the index."""
    print(f"\n{'=' * 70}")
    print("Benchmark: add()")
    print(f"{'=' * 70}")
    print(f"Adding {len(iscc_ids):,} entries...")

    start = time.perf_counter()
    count = index.add(iscc_ids, instance_codes)
    elapsed = time.perf_counter() - start

    throughput = count / elapsed
    print("\nResults:")
    print(f"  Entries added: {count:,}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:,.0f} entries/sec")

    return elapsed


def benchmark_get(index, instance_codes, sample_size=1000):
    # type: (InstanceIndex, list[str], int) -> tuple[float, dict[int, float]]
    """Benchmark exact match queries."""
    print(f"\n{'=' * 70}")
    print("Benchmark: get() - Exact Match Queries")
    print(f"{'=' * 70}")
    print(f"Testing {sample_size:,} random queries...")

    # Sample random instance codes for testing
    sample = random.sample(instance_codes, min(sample_size, len(instance_codes)))

    # Overall benchmark
    start = time.perf_counter()
    for ic_code in sample:
        index.get(ic_code)
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / sample_size) * 1000
    throughput = sample_size / elapsed

    print("\nResults:")
    print(f"  Queries: {sample_size:,}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Avg query time: {avg_time_ms:.3f}ms")
    print(f"  Throughput: {throughput:,.0f} queries/sec")

    # Per-length breakdown
    print("\nPer-length breakdown:")
    length_times = {}  # type: dict[int, float]
    for bit_length in [64, 128, 192, 256]:
        # Filter samples by length
        length_samples = [
            ic_code
            for ic_code in sample
            if len(ic.decode_header(ic.decode_base32(ic_code.removeprefix("ISCC:")))[4]) * 8 == bit_length
        ]

        if length_samples:
            start = time.perf_counter()
            for ic_code in length_samples:
                index.get(ic_code)
            elapsed_len = time.perf_counter() - start
            avg_ms = (elapsed_len / len(length_samples)) * 1000
            length_times[bit_length] = avg_ms
            print(f"  {bit_length:3d} bits: {avg_ms:.3f}ms avg ({len(length_samples):,} samples)")

    return elapsed, length_times


def benchmark_search(index, instance_codes, sample_size=1000):
    # type: (InstanceIndex, list[str], int) -> tuple[float, dict[int, float]]
    """Benchmark prefix search queries."""
    print(f"\n{'=' * 70}")
    print("Benchmark: search() - Prefix Search with Bidirectional")
    print(f"{'=' * 70}")
    print(f"Testing {sample_size:,} random searches...")

    # Sample random instance codes for testing
    sample = random.sample(instance_codes, min(sample_size, len(instance_codes)))

    # Overall benchmark
    start = time.perf_counter()
    total_matches = 0
    for ic_code in sample:
        results = index.search(ic_code, bidirectional=True)
        total_matches += len(results)
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / sample_size) * 1000
    avg_matches = total_matches / sample_size
    throughput = sample_size / elapsed

    print("\nResults:")
    print(f"  Searches: {sample_size:,}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Avg search time: {avg_time_ms:.3f}ms")
    print(f"  Avg matches per search: {avg_matches:.1f}")
    print(f"  Throughput: {throughput:,.0f} searches/sec")

    # Per-length breakdown
    print("\nPer-length breakdown:")
    length_times = {}  # type: dict[int, float]
    for bit_length in [64, 128, 192, 256]:
        # Filter samples by length
        length_samples = [
            ic_code
            for ic_code in sample
            if len(ic.decode_header(ic.decode_base32(ic_code.removeprefix("ISCC:")))[4]) * 8 == bit_length
        ]

        if length_samples:
            start = time.perf_counter()
            for ic_code in length_samples:
                results = index.search(ic_code, bidirectional=True)
            elapsed_len = time.perf_counter() - start
            avg_ms = (elapsed_len / len(length_samples)) * 1000
            length_times[bit_length] = avg_ms
            print(f"  {bit_length:3d} bits: {avg_ms:.3f}ms avg ({len(length_samples):,} samples)")

    return elapsed, length_times


def benchmark_search_unidirectional(index, instance_codes, sample_size=1000):
    # type: (InstanceIndex, list[str], int) -> float
    """Benchmark prefix search queries (unidirectional only)."""
    print(f"\n{'=' * 70}")
    print("Benchmark: search() - Prefix Search (Unidirectional)")
    print(f"{'=' * 70}")
    print(f"Testing {sample_size:,} random searches...")

    # Sample random instance codes for testing
    sample = random.sample(instance_codes, min(sample_size, len(instance_codes)))

    start = time.perf_counter()
    total_matches = 0
    for ic_code in sample:
        results = index.search(ic_code, bidirectional=False)
        total_matches += len(results)
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / sample_size) * 1000
    avg_matches = total_matches / sample_size
    throughput = sample_size / elapsed

    print("\nResults:")
    print(f"  Searches: {sample_size:,}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Avg search time: {avg_time_ms:.3f}ms")
    print(f"  Avg matches per search: {avg_matches:.1f}")
    print(f"  Throughput: {throughput:,.0f} searches/sec")

    return elapsed


def main():
    # type: () -> None
    """Run all benchmarks."""
    print("=" * 70)
    print("InstanceIndex Performance Benchmark")
    print("=" * 70)

    # Configuration
    num_entries = 1_000_000
    query_sample_size = 1_000

    # Create temporary directory for LMDB
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "benchmark_db"

        # Initialize index
        print(f"\nInitializing InstanceIndex at {db_path}")
        print("Configuration: durable=False (optimized for speed)")
        index = InstanceIndex(
            db_path,
            realm_id=0,
            map_size=20 * 1024 * 1024 * 1024,  # 20GB
            durable=False,  # Optimize for speed (benchmark mode)
            readahead=False,  # Better for random access
        )

        # Generate test data
        iscc_ids, instance_codes = generate_test_data(num_entries)

        # Run benchmarks
        add_time = benchmark_add(index, iscc_ids, instance_codes)
        get_time, get_length_times = benchmark_get(index, instance_codes, query_sample_size)
        search_time, search_length_times = benchmark_search(index, instance_codes, query_sample_size)
        search_uni_time = benchmark_search_unidirectional(index, instance_codes, query_sample_size)

        # Final summary
        print(f"\n{'=' * 70}")
        print("Summary")
        print(f"{'=' * 70}")
        print(f"Database size: {len(index):,} entries")
        print("\nOperation Performance:")
        print(
            f"  Add:             {(add_time / num_entries) * 1000:.4f}ms avg  ({num_entries / add_time:,.0f} ops/sec)"
        )
        print(
            f"  Get (exact):     {(get_time / query_sample_size) * 1000:.3f}ms avg  ({query_sample_size / get_time:,.0f} ops/sec)"
        )
        print(
            f"  Search (bi):     {(search_time / query_sample_size) * 1000:.3f}ms avg  ({query_sample_size / search_time:,.0f} ops/sec)"
        )
        print(
            f"  Search (uni):    {(search_uni_time / query_sample_size) * 1000:.3f}ms avg  ({query_sample_size / search_uni_time:,.0f} ops/sec)"
        )
        print("=" * 70)

        # Cleanup
        index.close()
        print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
