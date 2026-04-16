# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
