# Roadmap

This document tracks potential future enhancements and features for iscc-search. Items are unscheduled unless
otherwise noted.

## Possible Future Enhancements

### Multi-Process Support for Usearch Index (Read-Only Mode)

**Applies to:** Usearch Index (`usearch://`) only **Status:** Unscheduled **Effort:** ~10% of full multi-process
support **Complexity:** Low-Medium

A lightweight enhancement to enable multi-process deployments of **Usearch Index** with reduced corruption risk,
without implementing full multi-process write coordination.

**Note:** LMDB Index already supports multi-reader/single-writer. This enhancement is specifically for Usearch
Index which is currently single-process only.

#### Concept

Add read-only mode support to allow multiple reader processes while restricting writes to a single dedicated
writer process.

#### Components

1. **Read-only flag for Usearch Index components**

    - Add `readonly=True` parameter to `NphdIndex` (`.usearch` files)
    - Add `readonly=True` parameter to `InstanceIndex` and `IsccStore` (LMDB already supports this)
    - Prevent write operations in read-only mode
    - LMDB already supports read-only mode natively (propagate flag)

2. **Basic file locking on .usearch files**

    - Use `fcntl.flock()` (Unix) / `msvcrt.locking()` (Windows) for cross-platform locking
    - Lock `.usearch` files on open to detect concurrent write access
    - Fail fast with clear error message if write lock cannot be acquired
    - Allow multiple read locks simultaneously

3. **Manager-level readonly support**

    - `UsearchIndexManager(readonly=True)`
    - Propagate readonly flag to all cached index instances
    - Prevent creation of new indexes in readonly mode

#### Benefits

- Enables multi-process deployments (1 writer + N readers)
- Prevents obvious corruption from accidental concurrent writes
- ~10% implementation effort vs. full multi-process coordination
- Clean error messages guide users to correct deployment patterns

#### Limitations

- Does not enable multi-writer scenarios
- Does not provide cache synchronization between processes
- Readers may see stale data until process restart or manual reload
- Still requires deployment discipline (designate writer vs. readers)

#### Use Cases

**Usearch Index deployments requiring multi-process:**

- Web server: Read-only API processes + separate background indexer process
- Batch processing: Multiple read-only similarity search jobs + single indexing process
- Development: Prevent accidental .usearch corruption during experimentation

**Note:** LMDB Index users can already use multi-reader mode natively without this enhancement.

#### Implementation Notes

When implementing for **Usearch Index**:

1. Start with read-only mode in LMDB-backed classes (`InstanceIndex`, `IsccStore` - already supported)
2. Add file locking wrapper for `.usearch` files in `NphdIndex`
3. Propagate readonly flag through `UsearchIndexManager`
4. Add comprehensive tests for lock acquisition/release (cross-platform)
5. Update deployment patterns in `docs/deployment.md`

#### Alternatives Considered

**Option 1: Full multi-process support for Usearch Index**

With:

- File locking for all .usearch operations (read and write)
- LMDB write transaction coordination
- Cache invalidation across processes
- Atomic updates with rollback support

**Rejected because:** 10x implementation complexity with unclear real-world benefit. Current async
single-process pattern handles typical workloads efficiently.

**Option 2: Just use LMDB Index**

LMDB Index already supports multi-reader/single-writer natively. Why not just use that?

**Rejected because:** Usearch Index provides HNSW-based similarity search with sub-millisecond query latency for
large datasets. LMDB Index uses prefix-based matching which doesn't scale as well for high-dimensional
similarity search. Different use cases warrant different solutions.

______________________________________________________________________

## Scheduled Features

_No scheduled features at this time._

______________________________________________________________________

## Completed Features

_No completed roadmap items yet (project in early development)._
