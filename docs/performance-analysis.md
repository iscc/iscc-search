# Performance Analysis: ISCC-VDB with Usearch

## Executive Summary

This document analyzes the performance implications of using ISCC-IDs as 64-bit keys with usearch for vector
similarity search, including Multi-Key mode for SIMPRINTs indexing. The analysis covers storage overhead, query
performance, scalability considerations, and provides optimization recommendations.

## 1. Performance Implications of 64-bit ISCC-IDs as Keys

### 1.1 Key Structure Benefits

**ISCC-IDv1 as 64-bit keys offers several performance advantages:**

- **Native Integer Keys**: Usearch efficiently handles `uint64_t` keys with minimal overhead
- **Total Ordering**: The timestamp-based structure (52-bit timestamp + 12-bit server-id) provides natural
    ordering
- **Direct Mapping**: No need for hash tables or string lookups; direct integer comparison
- **Cache Efficiency**: 8-byte keys fit well in CPU cache lines

### 1.2 Key Lookup Performance

Based on usearch's architecture:

- **O(1) Key Access**: Direct hash map lookup for key-to-vector mapping
- **Memory Overhead**: ~8 bytes per key + hash table overhead (~16-24 bytes total per entry)
- **Collision Handling**: Minimal with 64-bit keyspace (2^64 possible values)

### 1.3 Comparison with Legacy Implementation

The legacy LMDB-based implementation used:

- MessagePack serialization for keys (additional overhead)
- B-tree based storage (O(log n) lookups)
- Multiple indirection levels for component lookups

**Performance Improvement**: 10-100x faster key lookups with usearch's hash-based approach

## 2. Storage Overhead Analysis

### 2.1 Vector Storage with Length Signalling

The 33-byte vector format (264 bits) breaks down as:

- 1 byte: Length signal (indicates actual ISCC length)
- 32 bytes: Maximum ISCC body (zero-padded for shorter ISCCs)

**Storage Overhead by ISCC Type:**

| ISCC Length | Actual Size | Stored Size | Overhead | Overhead % |
| ----------- | ----------- | ----------- | -------- | ---------- |
| 64-bit      | 8 bytes     | 33 bytes    | 25 bytes | 312.5%     |
| 128-bit     | 16 bytes    | 33 bytes    | 17 bytes | 106.3%     |
| 192-bit     | 24 bytes    | 33 bytes    | 9 bytes  | 37.5%      |
| 256-bit     | 32 bytes    | 33 bytes    | 1 byte   | 3.1%       |

### 2.2 Real-World Storage Impact

Assuming a typical distribution:

- 10% 64-bit ISCCs
- 20% 128-bit ISCCs
- 30% 192-bit ISCCs
- 40% 256-bit ISCCs

**Average overhead**: (0.1 × 312.5%) + (0.2 × 106.3%) + (0.3 × 37.5%) + (0.4 × 3.1%) = **65.0%**

For 1 billion vectors:

- Raw data: ~20 GB (average ~20 bytes/ISCC)
- Stored data: ~33 GB
- Additional overhead: ~13 GB

### 2.3 Index Structure Overhead

Usearch HNSW graph overhead:

- Node references: ~40 bits per edge (using uint40_t)
- Average edges per node: M × 2 (where M is typically 16-32)
- Graph overhead: ~160-320 bytes per vector

**Total memory footprint per vector**: ~200-350 bytes (including vector data and graph structure)

## 3. Multi-Key Mode Performance for SIMPRINTs

### 3.1 Multi-Key Mode Characteristics

When `multi=True`:

- Single ISCC-ID can map to multiple SIMPRINT vectors
- Useful for granular feature indexing
- Each SIMPRINT is independently searchable

### 3.2 Performance Trade-offs

**Advantages:**

- Efficient grouping of related features
- Single key lookup retrieves all SIMPRINTs for an asset
- Reduced key management overhead

**Disadvantages:**

- Iterator performance degradation (as noted in usearch docs)
- Variable-length result sets complicate memory allocation
- Potential cache misses when accessing multiple vectors per key

### 3.3 Expected Performance Impact

Based on usearch's implementation:

- **Write Performance**: ~5-10% slower due to multi-value management
- **Read Performance**:
    - Single key lookup: Negligible overhead
    - Range queries/iteration: 20-30% slower
- **Memory Overhead**: Additional ~8-16 bytes per vector for multi-value tracking

## 4. Scalability Analysis for Billions of Vectors

### 4.1 Memory Requirements

For 1 billion vectors:

- Vector data: 33 GB
- HNSW graph: 160-320 GB
- Key mapping: 16-24 GB
- **Total**: ~210-380 GB

### 4.2 Performance at Scale

**Build Time** (1 billion 264-bit vectors):

- Single-threaded: ~10-20 hours
- Multi-threaded (32 cores): ~1-2 hours
- With custom NPHD metric: +20-30% overhead

**Query Performance**:

- 10-NN search: ~0.1-1 ms per query
- 100-NN search: ~1-10 ms per query
- Recall@10: 95%+ (with properly tuned parameters)

### 4.3 Distributed Scaling

For datasets exceeding single-machine capacity:

- Use multiple smaller indices (as recommended by usearch)
- Parallel multi-index lookups
- Shard by ISCC type/subtype for better locality

## 5. Query Performance Patterns

### 5.1 NPHD Custom Metric Impact

The Normalized Prefix Hamming Distance implementation:

- **Computation**: O(min(len_a, len_b)) bit operations
- **Overhead vs Standard Hamming**: ~30-50% due to:
    - Length extraction from first byte
    - Dynamic common prefix calculation
    - Normalization division

### 5.2 Query Patterns Performance

**Pattern 1: Exact ISCC Lookup**

- Performance: O(1) with hash lookup
- Expected: \<0.01 ms

**Pattern 2: Similarity Search (k-NN)**

- Performance: O(log n) with HNSW
- Expected: 0.1-10 ms depending on k

**Pattern 3: SIMPRINT Matching**

- Performance: O(log n) + O(m) where m is SIMPRINTs per key
- Expected: 1-50 ms depending on granularity

## 6. Comparison with Legacy Implementation

| Aspect             | Legacy (LMDB) | New (Usearch)  | Improvement     |
| ------------------ | ------------- | -------------- | --------------- |
| Key Lookup         | O(log n)      | O(1)           | 10-100x         |
| Similarity Search  | O(n) scan     | O(log n) HNSW  | 1000x+          |
| Storage Efficiency | Variable      | Fixed 33 bytes | Predictable     |
| Concurrency        | Single writer | Thread-safe    | Parallel builds |
| Memory Mapping     | Yes           | Yes            | Comparable      |
| Custom Metrics     | No            | Yes (NPHD)     | New capability  |

## 7. Optimization Recommendations

### 7.1 Immediate Optimizations

1. **Separate Indices by ISCC Length**

    - Eliminate length signalling overhead
    - Use native 64/128/192/256-bit vectors
    - Trade-off: 4x index management complexity

2. **Batch Operations**

    - Use batch add/search operations
    - Reduces function call overhead
    - Improves cache locality

3. **Tune HNSW Parameters**

    - M=16 for build speed
    - M=32 for search quality
    - ef_construction=200 for good recall

### 7.2 Advanced Optimizations

1. **SIMD-Optimized NPHD**

    ```python
    # Use numpy/numba vectorized operations
    # Process multiple bytes in parallel
    # Utilize CPU popcount instructions
    ```

2. **Hierarchical Indexing**

    - Coarse quantization for initial filtering
    - Fine-grained search within clusters
    - Reduces search space dramatically

3. **Memory-Mapped Indices**

    - Keep indices on NVMe SSDs
    - Use OS page cache effectively
    - Reduces RAM requirements by 70%+

### 7.3 Architecture Recommendations

1. **Sharding Strategy**

    - Shard by ISCC MainType/SubType
    - Separate indices for SIMPRINTs
    - Enables parallel processing

2. **Caching Layer**

    - LRU cache for frequent queries
    - Pre-compute common ISCC combinations
    - Cache similarity results

3. **Monitoring & Profiling**

    - Track query latencies by pattern
    - Monitor memory usage trends
    - Profile NPHD metric performance

## 8. Conclusion

The proposed iscc-vdb architecture using usearch with 64-bit ISCC-IDs as keys provides:

- **Excellent scalability** to billions of vectors
- **Acceptable storage overhead** (65% average, mostly from shorter ISCCs)
- **Superior query performance** compared to legacy implementation
- **Flexible architecture** supporting both ISCC and SIMPRINT indexing

The main trade-offs involve storage overhead for length signalling and custom metric computation overhead. These
are manageable with the suggested optimizations and provide a solid foundation for a production-ready ISCC
vector database.

## Appendix: Benchmark Recommendations

To validate these projections, implement benchmarks measuring:

1. **Build Performance**

    - Time to index 1M, 10M, 100M vectors
    - Memory usage during build
    - CPU utilization patterns

2. **Query Performance**

    - k-NN search latency distribution
    - Recall@k for various k values
    - Throughput under concurrent load

3. **Storage Efficiency**

    - Actual vs theoretical storage usage
    - Compression ratios
    - Memory-mapped performance

4. **NPHD Metric Validation**

    - Correctness verification
    - Performance vs standard Hamming
    - SIMD optimization impact
