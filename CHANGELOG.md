# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-05-07

### Added

- Remote client retries batch requests on transient network errors (connection refused, timeout,
    read/write failures) with exponential backoff (3 attempts, 1s/2s/4s delays). Previously, a
    server restart during ingestion silently dropped every in-flight batch with no recovery.
- NPHD and simprint index loaders log vector count and shard count at info level on startup,
    giving operators visibility into shard proliferation and expected startup I/O.
- `UsearchSimprintIndex.shard_count` property exposes the underlying `ShardedIndex128` shard count.
- Deployment docs: "Memory budget under container limits" section documenting the three-way
    competition between Python heap, LMDB mmap, and shard mmap pages under cgroup memory limits,
    with concrete sizing guidance from stress testing at 2 GB / 3000 IOPS.

### Changed

- **Dependency**: `iscc-usearch` minimum bumped from `0.6.1` to `0.8.0` (includes `usearch-iscc` 2.24.2 → 2.24.5).
    This upgrade brings critical upstream bug fixes (heap buffer overflow in HNSW search, double-free in
    add/remove cycles, stale vector references during refine operations) and two architectural improvements
    that directly address operational issues documented in the v0.1.1 post-incident analysis:
    - **Durable writes**: `save()` now serializes to an in-memory buffer and writes via a single `os.write()` +
        `fdatasync` + atomic rename. Write syscalls drop from ~30,000 to 3 per save, collapsing the IOPS-bound
        save duration on gp3 default storage from ~80 s to sub-second. This directly mitigates the lock-convoy
        freeze under sustained ingest (previously requiring a background flush worker or IOPS provisioning).
    - **Crash-safe persistence ordering**: `save()` and shard rotation now persist bloom → shard → tombstones,
        preventing deleted keys from reappearing after a crash between shard and tombstone writes.
    - **Bloom filter auto-recovery**: Missing or corrupt `bloom.isbf` files are automatically rebuilt from
        shard keys on load, reducing the stale-bloom false-negative risk during incremental repair.
    - **Stale shard cleanup**: Empty shard files after removals are cleaned up on `save()`.
    - **Duplicate key handling**: `add()` with an already-present key now silently skips instead of raising
        `RuntimeError` with partial commit. The remove-before-add pattern in `add_assets` is retained for
        update semantics.
    - New `stats()`, `close()`, and `drain_rotations()` APIs available for future use.
- Shutdown sequence uses upstream `close()` (save + resource release) instead of manual `save()` + `reset()`.
- Enable background shard rotation for all NPHD and simprint indexes. When an active shard reaches
    `shard_size`, serialization runs in a background thread so `add()` returns immediately instead of
    blocking for the full write. Backpressure limits pending rotations to 2.

### Fixed

- Drain pending background rotations before reading `.size` for metadata counts. Without draining,
    `add_assets()` and `close()` could persist stale vector counts (e.g. 0 instead of 50), causing
    false "out of sync" warnings on next index load.
- `/healthz` and `/readyz` probes converted from sync to async handlers. Sync probes shared the
    anyio threadpool with `add_assets` writers and became unreachable during threadpool saturation.

## [0.1.1] - 2026-05-04

### Added

- New CLI command `iscc-search index rebuild [--unit-type X] [--simprint-type Y] [--all] [--index NAME]`
    that rebuilds derived NPHD and simprint indexes for a local usearch-backed index from LMDB source data.
    The "out-of-sync" startup warnings now point at this command instead of an imaginary one. Operators no
    longer need a Python REPL to recover after a crash that left LMDB ahead of the on-disk shards.

### Changed

- **Breaking default**: `ISCC_SEARCH_FLUSH_INTERVAL` default raised from `0` (disabled) to `100000`. Derived
    HNSW indexes (NPHD, simprint) now auto-flush every 100,000 dirty mutations. Previously, indexes were
    only saved on graceful `close()` — small indexes that never reached the shard-size rotation threshold
    could lose every vector added since process start, leaving the server with 0% of recent data searchable
    after a crash until a manual rebuild completed. With the new default, the post-crash gap is bounded to
    ~100K vectors per sub-index, so the server remains in degraded-but-mostly-functional state and can
    serve queries while a rebuild is scheduled. Note: this **does not reduce rebuild cost** — current
    `_rebuild_nphd_index` / `_rebuild_simprint_index` are destructive (rmtree the shard dir, re-add
    everything from LMDB) and rebuild the full per-type vector set regardless of how much was already
    persisted. An incremental repair path that exploits the persisted state via per-shard bloom filters is
    planned. Trade-off: ~25× write amplification on heavy SIMPRINT ingestion. Set
    `ISCC_SEARCH_FLUSH_INTERVAL=0` to restore the old behavior if you need maximum ingestion throughput
    and accept unbounded loss on crash.
- **Breaking default**: `ISCC_SEARCH_SHARD_SIZE_UNITS` and `ISCC_SEARCH_SHARD_SIZE_SIMPRINTS` defaults
    lowered from `1024` MB to `512` MB. Smaller rotation threshold means active shards seal to immutable
    on-disk artifacts roughly twice as often, so the worst-case "active shard lost on crash" window shrinks
    by ~50% for high-volume simprint ingestion. Complementary to the `flush_interval` change above:
    `flush_interval` bounds loss by *mutation count* on the live active shard; `shard_size_*` bounds loss
    by *bytes sealed*. Trade-off: ~2× the number of sealed shard files on disk, marginally more I/O on
    rotation. Existing deployments need no migration: sealed shards from the old 1024 MB era stay
    untouched (they are never split or rewritten), and an active shard already above 512 MB simply
    rotates on its next `add()`. Expect heterogeneous shard sizes in directories that span the
    transition. Set explicitly to `1024` to restore the old behavior.
- Repo `compose.yaml` `stop_grace_period` raised from `90s` to `300s`. uvicorn shutdown is sequential —
    request drain (bounded by `--timeout-graceful-shutdown`, kept at `60s`) runs first, then the lifespan
    handler runs the HNSW flush with no uvicorn-side timeout. Docker's `stop_grace_period` is the only
    outer bound on the flush, so it must be `>= timeout_graceful_shutdown + expected_flush_duration`. The
    new `300s` covers `60s` drain plus `~240s` flush headroom; raise to `600s+` for indexes over 10M
    vectors. If your production compose file overrides `stop_grace_period`, recalculate using this formula.

### Fixed

- `UsearchIndexManager` and `LmdbIndexManager` now serialize first-load construction in
    `_get_or_load_index` with a `threading.Lock`. Previously, a concurrent burst against an uncached index
    (typical first-burst-after-restart) could race two threads into `lmdb.open()` on the same path,
    producing `lmdb.Error: The environment '...index.lmdb' is already open in this process` and a 500
    response. Once one thread populated the cache, subsequent requests were fine — so this only affected
    first bursts, but those are exactly when operators are paying attention.
- Deployment troubleshooting docs no longer instruct operators to delete `.usearch` files to trigger an
    auto-rebuild on restart. Auto-rebuild on startup was disabled in v0.1.0 to prevent OOM restart loops on
    large indexes; the docs now describe the explicit rebuild procedure from LMDB instead.

## [0.1.0] - 2026-04-16

Initial release of iscc-search.

### Added

- `IsccIndexProtocol` — backend-agnostic Protocol implemented by every index, so CLI, REST API,
    and library users share a single interface
- `MemoryIndex` backend — in-memory dict-based storage for tests and demos (`memory://`)
- `LmdbIndex` backend — LMDB-backed persistent storage with inverted prefix-search index
    (`lmdb:///path`)
- `UsearchIndex` backend — HNSW similarity search via `iscc-usearch` plus LMDB metadata storage,
    production-grade (`usearch:///path`)
- ISCC-SIMPRINT (granular feature) indexing with both exact LMDB search and approximate
    `ShardedIndex128`-based search with 20x oversampling and IDF-weighted scoring
- Similarity search across all ISCC-UNIT types (META, SEMANTIC-{TEXT,IMAGE,MIXED},
    CONTENT-{TEXT,IMAGE,AUDIO,VIDEO,MIXED}, DATA, INSTANCE) using NPHD metric
- `IsccBase`, `IsccID`, `IsccUnit`, `IsccCode`, `IsccItem` convenience models on top of `iscc-core`
- Composite multi-type asset matching with global score aggregation across ISCC-UNIT types
- Typer-based CLI with subcommands `add`, `get`, `search`, `serve`, `index`, `hub`, `datasets`
- Git-style multi-index management for the CLI (`~/.iscc-search/config.json`) with named local
    and remote indexes and an "active" index workflow
- `iscc-search hub` and `iscc-search datasets` commands for ingesting ISCC assets from
    HuggingFace Hub
- FastAPI REST server with endpoints for index management, asset add/get, and similarity search
    (POST/GET `/indexes/{name}/search`)
- Modular OpenAPI 3.0 specification (`openapi/openapi.yaml` + fragments) bundled into
    `openapi.json`, with auto-generated Pydantic v2 models in `iscc_search/schema.py`
- Optional API-key authentication for the REST server
- `/healthz` and `/readyz` Kubernetes-style probe endpoints
- Env-gated Sentry error tracking for the REST server
- `IsccSearchClient` HTTP client in `iscc_search/remote/` for talking to a remote server
- `SearchOptions` server configuration via `ISCC_SEARCH_*` environment variables and `.env`,
    with single-index 12-factor deployment model
- `get_index()` factory that parses `index_uri` and selects the matching backend
    (with Windows-specific URI path handling)
- Dockerfile and Docker Compose configuration for containerized deployments, plus published
    `ghcr.io/iscc/iscc-search:develop` images built on every push to `develop`
- Loguru-based structured logging configured via `iscc_search/log_config.json`
- Cross-platform support (Linux, macOS, Windows) for code, scripts, and dev tooling
- Python 3.11, 3.12, 3.13, and 3.14 support
- Documentation site built with Zensical (Material theme) following the Diataxis framework
- Comprehensive pytest test suite with 100% coverage requirement (parallelized via `pytest-xdist`)
- CI workflows for tests (Linux, macOS, Windows), Docker image publishing, and docs deployment

[0.1.0]: https://github.com/iscc/iscc-search/releases/tag/0.1.0
[0.2.0]: https://github.com/iscc/iscc-search/releases/tag/0.2.0
