# Deployment Notes

> **Note:** This document captures deployment considerations as they arise during development. Full deployment
> documentation will be written when the project approaches production readiness.

## Concurrency Model by Index Type

ISCC-SEARCH provides multiple index implementations with different concurrency characteristics. Choose the right
index for your deployment based on these constraints.

### Memory Index (`memory://`)

**Concurrency:** Single-process only (in-memory Python objects)

The memory index exists only in the process's memory space and cannot be shared between processes.

**Deployment:** Development and testing only. Not suitable for production.

### LMDB Index (`lmdb:///path`)

**Concurrency:** LMDB supports multi-reader/single-writer, but manager cache is process-local

**Constraints:**

1. **LMDB database files** support multi-reader/single-writer with built-in locking (lock=True)
2. **Manager instance cache** (`LmdbIndexManager._index_cache`) does not synchronize between processes
3. Multiple reader processes work correctly, but cache warming is per-process

**Impact:** Multiple processes can read safely, but the cache doesn't benefit subsequent processes.

**Deployment:** Single-process recommended for simplicity. Multi-process reads are safe but may have redundant
cache warming overhead.

### Usearch Index (`usearch:///path`)

**Concurrency:** Single-process only

**Constraints:**

1. **Usearch files** (`.usearch`) have no file locking or multi-process coordination
2. **Manager instance cache** (`UsearchIndexManager._index_cache`) does not synchronize between processes
3. While **LMDB component** supports multi-reader/single-writer, the combined system is limited by usearch's
    single-process constraint

**Impact:** Running multiple processes against the same indexes **will corrupt data**.

**Deployment:** Single-process only. See recommended patterns below.

### Postgres Index (`postgresql://...`)

**Concurrency:** Multi-process (planned)

The planned Postgres index will naturally support multi-process deployments as PostgreSQL handles concurrency
and locking internally.

**Deployment:** Multi-process and distributed deployments will be supported.

______________________________________________________________________

## Impact Summary

| Index Type | Multi-Process Reads | Multi-Process Writes | Recommended Pattern        |
| ---------- | ------------------- | -------------------- | -------------------------- |
| Memory     | ❌ No               | ❌ No                | Single process             |
| LMDB       | ✅ Yes (safe)       | ⚠️ Single writer     | Single process (preferred) |
| Usearch    | ❌ No               | ❌ No                | Single process             |
| Postgres   | ✅ Yes (planned)    | ✅ Yes (planned)     | Multi-process              |

______________________________________________________________________

## Recommended Deployment Patterns

### For Usearch Index (High-Performance Similarity Search)

**Use single-process async servers:**

- **FastAPI + Uvicorn** (single process, async workers)
- **Starlette** or other async frameworks
- Single process can handle thousands of concurrent connections via async/await

**Example:**

```bash
# Good: Single process with async concurrency
uvicorn myapp:app --host 0.0.0.0 --port 8000

# Bad: Multiple worker processes (WILL CORRUPT USEARCH FILES)
uvicorn myapp:app --workers 4  # DON'T DO THIS with usearch://
gunicorn myapp:app --workers 4  # DON'T DO THIS with usearch://
```

**Why this works:**

- Async/await handles I/O concurrency efficiently
- Usearch search operations release the GIL during computation
- Single process eliminates coordination overhead
- No risk of .usearch file corruption

### For LMDB Index (Prefix Matching Only)

**Single-process recommended, multi-process reads acceptable:**

```bash
# Preferred: Single process
uvicorn myapp:app --host 0.0.0.0 --port 8000

# Acceptable: Multi-process reads (if needed)
# Note: Each process warms its own cache
uvicorn myapp:app --workers 4  # Safe but cache-inefficient with lmdb://
```

**Multi-process considerations:**

- Safe for concurrent reads (LMDB handles locking)
- Single writer process required for writes
- Each process maintains separate instance cache (redundant memory usage)
- May be useful for CPU-bound workloads that can't use async

### For Memory Index (Testing Only)

Development/testing only - concurrency not applicable.

### For Postgres Index (Future)

Multi-process deployments will be fully supported when implemented.

______________________________________________________________________

## When to Consider Multi-Process Enhancements

For **Usearch Index**, only consider multi-process support if you encounter:

1. Real use case requiring process parallelism (not just web serving)
2. Benchmarks showing single-process async is insufficient
3. CPU-bound workload that can't leverage async

See `docs/roadmap.md` for read-only mode enhancement (10% solution) that would enable limited multi-process
deployments for Usearch Index.
