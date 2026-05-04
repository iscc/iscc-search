# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Repo `compose.yaml` `stop_grace_period` raised from `90s` to `300s`. uvicorn shutdown is sequential —
    request drain (bounded by `--timeout-graceful-shutdown`, kept at `60s`) runs first, then the lifespan
    handler runs the HNSW flush with no uvicorn-side timeout. Docker's `stop_grace_period` is the only
    outer bound on the flush, so it must be `>= timeout_graceful_shutdown + expected_flush_duration`. The
    new `300s` covers `60s` drain plus `~240s` flush headroom; raise to `600s+` for indexes over 10M
    vectors. If your production compose file overrides `stop_grace_period`, recalculate using this formula.

### Fixed

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
