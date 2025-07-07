# USearch Examples and Best Practices for ISCC-VDB

This document compiles examples and insights from the usearch library relevant to implementing ISCC-VDB with
binary vectors and custom metrics.

## Binary Vector Implementation with ScalarKind.B1

### Key Concepts

- **ScalarKind.B1**: Represents 1-bit binary values, packed 8 per byte
- **ndim parameter**: Specifies number of BITS (not bytes) when using B1
- **Bit packing**: 8 bits are packed into each byte for memory efficiency

### Python Example - Binary Vector Index

```python
import numpy as np
from usearch.index import Index, MetricKind, ScalarKind

# Create index for 256-bit binary vectors
index = Index(
    ndim=256,  # 256 bits = 32 bytes
    metric=MetricKind.Hamming,
    dtype=ScalarKind.B1
)

# Create binary vectors using numpy
# Method 1: From boolean array
bool_vector = np.random.randint(0, 2, 256, dtype=np.uint8)
bit_vector = np.packbits(bool_vector)  # Results in 32 bytes

# Method 2: Direct byte array
byte_vector = np.random.bytes(32)  # 32 bytes = 256 bits

# Add vectors to index
index.add(42, bit_vector)
index.add(43, byte_vector)

# Search
query = np.random.bytes(32)
matches = index.search(query, 10)
```

## Custom Metrics Implementation

### Python Custom Metric with Numba

```python
from numba import cfunc, types, carray
from usearch.index import Index, MetricKind, MetricSignature, CompiledMetric

# Define custom metric for binary vectors
@cfunc(types.float32(types.CPointer(types.uint8), types.CPointer(types.uint8)))
def custom_binary_metric(a, b):
    # Access arrays - note: length must be known at compile time
    # For 256-bit vectors = 32 bytes
    a_array = carray(a, 32)
    b_array = carray(b, 32)

    distance = 0.0
    for i in range(32):
        # XOR to find differing bits, then count them
        diff = a_array[i] ^ b_array[i]
        # Count set bits (simple method)
        for j in range(8):
            if diff & (1 << j):
                distance += 1.0

    # Normalize by total bits
    return distance / 256.0

# Create compiled metric
metric = CompiledMetric(
    pointer=custom_binary_metric.address,
    kind=MetricKind.IP,  # Use IP as placeholder
    signature=MetricSignature.ArrayArray
)

# Use with index
index = Index(ndim=256, metric=metric, dtype=ScalarKind.B1)
```

### NPHD-Style Metric Considerations

For implementing NPHD (Normalized Prefix Hamming Distance) with variable-length vectors:

```python
# Note: USearch expects fixed dimensions, so we need workarounds
# Option 1: Use maximum dimension with length encoding

@cfunc(types.float32(types.CPointer(types.uint8), types.CPointer(types.uint8)))
def nphd_metric(a, b):
    # Assume first byte encodes length, next 32 bytes are max ISCC
    a_array = carray(a, 33)
    b_array = carray(b, 33)

    # Extract lengths
    len_a = a_array[0]
    len_b = b_array[0]
    common_len = min(len_a, len_b)

    # Calculate hamming distance on common prefix
    distance = 0.0
    common_bytes = common_len // 8

    for i in range(1, common_bytes + 1):
        diff = a_array[i] ^ b_array[i]
        for j in range(8):
            if diff & (1 << j):
                distance += 1.0

    # Handle remaining bits if not byte-aligned
    remaining_bits = common_len % 8
    if remaining_bits > 0:
        byte_idx = common_bytes + 1
        diff = a_array[byte_idx] ^ b_array[byte_idx]
        for j in range(remaining_bits):
            if diff & (1 << j):
                distance += 1.0

    # Normalize by common length
    return distance / common_len if common_len > 0 else 0.0
```

## Performance Optimization Techniques

### 1. Memory-Efficient Storage

```python
# Add vectors without copying (view mode)
index.add(key, vector, copy=False)

# Use batch operations
keys = np.arange(1000)
vectors = np.random.bytes(1000 * 32).reshape(1000, 32)
index.add(keys, vectors, threads=8)
```

### 2. Multi-threading

```python
# Enable multi-threading for operations
index.add(keys, vectors, threads=8)
results = index.search(queries, k=10, threads=8)
```

### 3. Expansion Factors

```python
# Adjust expansion factors for accuracy vs speed tradeoff
index = Index(
    ndim=256,
    metric=MetricKind.Hamming,
    dtype=ScalarKind.B1,
    expansion_add=40,     # Default: 40
    expansion_search=16   # Default: 16
)

# Can be changed after creation
index.change_expansion_add(64)
index.change_expansion_search(32)
```

### 4. Hardware Acceleration

```python
# Check SIMD capabilities
print(f"Hardware acceleration: {index.hardware_acceleration}")
```

## Best Practices for Binary Vector Search

1. **Bit Packing**: Always use `np.packbits()` or equivalent to pack binary data
2. **Batch Operations**: Process multiple vectors at once for better performance
3. **Memory Management**: Use `copy=False` when possible to avoid duplicating data
4. **Thread Tuning**: Experiment with thread counts based on your CPU cores
5. **Expansion Tuning**: Higher expansion = better recall but slower performance

## Testing Binary Vector Operations

```python
import numpy as np
from usearch.index import Index, MetricKind, ScalarKind

def test_binary_index():
    # Create index
    index = Index(ndim=256, metric=MetricKind.Hamming, dtype=ScalarKind.B1)

    # Generate test data
    n_vectors = 1000
    vectors = np.random.randint(0, 2, (n_vectors, 256), dtype=np.uint8)
    packed_vectors = np.packbits(vectors, axis=1)

    # Add vectors
    keys = np.arange(n_vectors)
    index.add(keys, packed_vectors)

    # Test retrieval
    retrieved = index.get(keys[0])
    assert np.array_equal(retrieved, packed_vectors[0])

    # Test search
    query = packed_vectors[0]
    matches = index.search(query, 10)
    assert matches.keys[0] == keys[0]
    assert matches.distances[0] == 0.0

    print(f"Index stats: {index.stats}")
```

## Limitations and Workarounds

### Variable-Length Vectors

USearch expects fixed dimensions, but ISCC vectors can be 64, 128, 192, or 256 bits. Workarounds:

1. **Multiple Indices**: Create separate indices for each vector length
2. **Length Encoding**: Use max dimension (264 bits) with first byte as length indicator
3. **Custom Wrapper**: Implement a wrapper class that manages multiple indices
4. **Using Indexes Class**: Leverage USearch's built-in `Indexes` class for multi-index search

### Option 1: Custom Wrapper Class

```python
class ISCCIndex:
    def __init__(self):
        self.indices = {
            64: Index(ndim=64, metric=MetricKind.Hamming, dtype=ScalarKind.B1),
            128: Index(ndim=128, metric=MetricKind.Hamming, dtype=ScalarKind.B1),
            192: Index(ndim=192, metric=MetricKind.Hamming, dtype=ScalarKind.B1),
            256: Index(ndim=256, metric=MetricKind.Hamming, dtype=ScalarKind.B1),
        }

    def add(self, key, vector):
        bit_length = len(vector) * 8
        if bit_length in self.indices:
            self.indices[bit_length].add(key, vector)
        else:
            raise ValueError(f"Unsupported vector length: {bit_length} bits")

    def search(self, query, k=10):
        bit_length = len(query) * 8
        if bit_length in self.indices:
            return self.indices[bit_length].search(query, k)
        else:
            # Search all indices and merge results
            all_results = []
            for idx_bits, index in self.indices.items():
                # Truncate or pad query as needed
                adjusted_query = self._adjust_query(query, idx_bits)
                results = index.search(adjusted_query, k)
                # Adjust distances based on length difference
                # ... implement NPHD logic here ...
                all_results.extend(results)
            # Sort and return top k
            return sorted(all_results, key=lambda x: x.distance)[:k]
```

### Option 2: Using USearch's Indexes Class

USearch provides an `Indexes` class that can wrap multiple `Index` instances and search across them:

```python
from usearch.index import Index, Indexes, MetricKind, ScalarKind

# Create separate indices for each ISCC length
index_64 = Index(ndim=64, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
index_128 = Index(ndim=128, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
index_192 = Index(ndim=192, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
index_256 = Index(ndim=256, metric=MetricKind.Hamming, dtype=ScalarKind.B1)

# Add vectors to appropriate indices
index_64.add(1, vector_64_bits)
index_128.add(2, vector_128_bits)
index_192.add(3, vector_192_bits)
index_256.add(4, vector_256_bits)

# Create a multi-index wrapper
multi_index = Indexes([index_64, index_128, index_192, index_256])

# Search across all indices
# Note: This requires handling different vector lengths appropriately
results = multi_index.search(query_vector, k=10)
```

### Option 3: Serialization and Dynamic Loading

For large-scale deployments, you can save indices to disk and load them on demand:

```python
# Save indices
index_64.save("iscc_64.usearch")
index_128.save("iscc_128.usearch")
index_192.save("iscc_192.usearch")
index_256.save("iscc_256.usearch")

# Load specific index based on query length
def load_index_for_query(query_vector):
    bit_length = len(query_vector) * 8
    index_path = f"iscc_{bit_length}.usearch"
    return Index.restore(index_path)

# Or use Indexes with paths
multi_index = Indexes([
    "iscc_64.usearch",
    "iscc_128.usearch",
    "iscc_192.usearch",
    "iscc_256.usearch"
])
```

## Advanced Storage Patterns

### Memory-Mapped Indices

For very large indices, use memory mapping to serve from disk without loading into RAM:

```python
# Save index to disk
index.save("large_index.usearch")

# Load as memory-mapped (read-only)
index_view = Index.restore("large_index.usearch", view=True)

# Search works normally but uses disk-backed memory
results = index_view.search(query, k=10)
```

### SQLite Integration

Store and search binary vectors directly in SQLite:

```python
import sqlite3
import usearch

# Load USearch SQLite extension
conn = sqlite3.connect(":memory:")
conn.enable_load_extension(True)
conn.load_extension(usearch.sqlite_path())

# Create table for binary vectors
conn.execute("""
    CREATE TABLE iscc_vectors (
        id INTEGER PRIMARY KEY,
        iscc_id TEXT,
        vector BLOB
    )
""")

# Insert binary vectors
vector = np.random.bytes(32)  # 256-bit vector
conn.execute("INSERT INTO iscc_vectors (iscc_id, vector) VALUES (?, ?)",
             ("ISCC:123456", vector))

# Search using Hamming distance
query = np.random.bytes(32)
results = conn.execute("""
    SELECT id, iscc_id, distance_hamming_binary(vector, ?) as distance
    FROM iscc_vectors
    ORDER BY distance
    LIMIT 10
""", (query,)).fetchall()
```

### Filtering with Predicates

Use predicates to filter search results based on metadata:

```python
# Define a filter function
def filter_by_date(key):
    # Assuming keys encode timestamp
    return key > 1000000  # Only recent entries

# Search with filter
results = index.search(query, k=10, predicate=filter_by_date)
```

## Performance Benchmarking

### Simple Benchmark Script

```python
import time
import numpy as np
from usearch.index import Index, MetricKind, ScalarKind

def benchmark_binary_index(n_vectors=100000, n_queries=1000, bits=256):
    # Create index
    index = Index(
        ndim=bits,
        metric=MetricKind.Hamming,
        dtype=ScalarKind.B1,
        expansion_add=40,
        expansion_search=16
    )

    # Generate random binary vectors
    bytes_per_vector = bits // 8
    vectors = np.random.bytes(n_vectors * bytes_per_vector).reshape(n_vectors, bytes_per_vector)
    keys = np.arange(n_vectors)

    # Benchmark insertion
    start = time.time()
    index.add(keys, vectors, threads=8)
    insert_time = time.time() - start
    print(f"Insertion: {n_vectors / insert_time:.0f} vectors/sec")

    # Generate queries
    queries = np.random.bytes(n_queries * bytes_per_vector).reshape(n_queries, bytes_per_vector)

    # Benchmark search
    start = time.time()
    for query in queries:
        results = index.search(query, k=10)
    search_time = time.time() - start
    print(f"Search: {n_queries / search_time:.0f} queries/sec")

    # Memory usage
    print(f"Memory usage: {index.memory_usage / 1024 / 1024:.1f} MB")
    print(f"Hardware acceleration: {index.hardware_acceleration}")

    return index

# Run benchmark
index = benchmark_binary_index()
```

## Key Takeaways for ISCC-VDB

1. **Binary Vector Support**: USearch has excellent support for binary vectors with `ScalarKind.B1`
2. **Performance**: Hardware acceleration (SIMD) is automatically used for binary operations
3. **Flexibility**: Multiple indices can be managed with the `Indexes` class
4. **Storage Options**: Memory mapping and SQLite integration provide scalability options
5. **Custom Metrics**: CompiledMetric allows implementing NPHD, though with some constraints
6. **Variable Length Challenge**: Requires workarounds since USearch expects fixed dimensions

## References

- [USearch Python API](https://github.com/unum-cloud/usearch/blob/main/python/README.md)
- [USearch Rust API](https://github.com/unum-cloud/usearch/blob/main/rust/README.md)
- [Benchmarks Documentation](https://github.com/unum-cloud/usearch/blob/main/BENCHMARKS.md)
- [Core Architecture Wiki](https://github.com/unum-cloud/usearch/wiki/Core-Architecture)
- [SQLite Extension](https://github.com/unum-cloud/usearch/blob/main/sqlite/README.md)
