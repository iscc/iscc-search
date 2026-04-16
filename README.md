# iscc-search

[![Release](https://img.shields.io/github/v/release/iscc/iscc-search)](https://img.shields.io/github/v/release/iscc/iscc-search)
[![Tests](https://img.shields.io/github/actions/workflow/status/iscc/iscc-search/main.yml?branch=main)](https://github.com/iscc/iscc-search/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/iscc/iscc-search/branch/main/graph/badge.svg)](https://codecov.io/gh/iscc/iscc-search)
[![Commit activity](https://img.shields.io/github/commit-activity/m/iscc/iscc-search)](https://img.shields.io/github/commit-activity/m/iscc/iscc-search)
[![License](https://img.shields.io/github/license/iscc/iscc-search)](https://img.shields.io/github/license/iscc/iscc-search)

> [!WARNING]
> **BETA** - This project is under active development. The API is not yet stable and may change
> without notice. Use at your own risk.

High-performance similarity search engine for [ISCC](https://iscc.codes) (International Standard Content Code).
Ships as a Python package, a CLI, and a FastAPI REST server, with pluggable backends for in-memory, LMDB, and
HNSW-accelerated indexes.

- **Github repository**: <https://github.com/iscc/iscc-search/>
- **Documentation** <https://search.iscc.codes/>

> **Note:** [iscc-usearch](https://github.com/iscc/iscc-usearch) is a separate project - a patched
> fork of the [usearch](https://github.com/unum-cloud/usearch) vector search library that provides
> the NPHD metric and low-level vector indexes. iscc-search uses it internally as one of its
> backends. Most users only need to install iscc-search.

## Features

- REST API server (FastAPI) for indexing and searching ISCC assets
- CLI (`iscc-search`) for managing multiple local or remote indexes and ingesting assets
- Protocol-based backend abstraction with three implementations:
    - `memory://` — in-memory, no persistence (tests and demos)
    - `lmdb:///path` — LMDB-backed persistent storage with bidirectional prefix search
    - `usearch:///path` — HNSW + LMDB for high-performance approximate nearest neighbor search
- Variable-length ISCC-UNIT indexing using the NPHD metric (via
    [iscc-usearch](https://github.com/iscc/iscc-usearch))
- Granular ISCC-SIMPRINT search for fine-grained content matching
- Cross-platform (Linux, macOS, Windows)
- Python 3.10–3.13

## What is ISCC?

The [International Standard Content Code (ISCC)](https://iscc.codes) is a similarity-preserving content
identifier for digital media. ISCC codes are variable-length binary vectors that enable efficient similarity
search across different media types. This project provides the indexing and search engine for those codes.

## Installation

```bash
pip install iscc-search
```

For development:

```bash
git clone https://github.com/iscc/iscc-search.git
cd iscc-search
uv sync
```

## Quick Start

### Run the server

```bash
# Start the REST API server (development mode with auto-reload)
iscc-search serve --dev

# Or production mode
iscc-search serve --host 0.0.0.0 --port 8000
```

Interactive API docs are available at `http://localhost:8000/docs`.

### Use the CLI

```bash
# Register an index configuration (local or remote)
iscc-search index add my-index --uri usearch:///path/to/data
iscc-search index use my-index

# Add assets, search, retrieve
iscc-search add asset.json
iscc-search search asset.json
iscc-search get ISCC:KACYPXW557...
```

### Configure the server

The server reads its configuration from environment variables prefixed with `ISCC_SEARCH_` (or a `.env` file):

| Variable                   | Default          | Description                                                  |
| -------------------------- | ---------------- | ------------------------------------------------------------ |
| `ISCC_SEARCH_INDEX_URI`    | `usearch:///...` | Backend URI (`memory://`, `lmdb:///path`, `usearch:///path`) |
| `ISCC_SEARCH_HOST`         | `0.0.0.0`        | Server bind host                                             |
| `ISCC_SEARCH_PORT`         | `8000`           | Server bind port                                             |
| `ISCC_SEARCH_API_SECRET`   | *(unset)*        | Optional API key; when unset the API is public               |
| `ISCC_SEARCH_CORS_ORIGINS` | `*`              | Comma-separated CORS origins                                 |
| `ISCC_SEARCH_LOG_LEVEL`    | `info`           | Loguru log level                                             |

Additional knobs control HNSW parameters, shard sizes, match thresholds, and scoring — see
`iscc_search/options.py` or the [deployment guide](docs/deployment.md) for the full list.

## Architecture

iscc-search uses a protocol-based design so the CLI, REST API, and library users all talk to the same
`IsccIndexProtocol` interface regardless of backend:

```
  CLI / REST API / Remote client
              │
              ▼
     IsccIndexProtocol
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
  memory    lmdb      usearch
            (LMDB)    (HNSW + LMDB)
```

See [docs/architecture.md](docs/architecture.md) for the full picture.

## Development

This project uses [uv](https://docs.astral.sh/uv/) for package management and
[poethepoet](https://github.com/nat-n/poethepoet) for task automation.

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Common tasks

```bash
uv run poe build            # Rebuild schema.py + openapi.json and validate
uv run poe format           # Format code and markdown
uv run poe test             # Run tests with coverage (must stay at 100%)
uv run poe check-complexity # Radon complexity report
uv run poe precommit        # Run pre-commit hooks
uv run poe all              # Build, format, test, and complexity
```

### Running tests

```bash
# Run full test suite in parallel with coverage
uv run poe test

# Run a single test
uv run pytest tests/test_indexes_usearch_index.py::test_foo
```

## Technical Notes

### NPHD Metric

The Normalized Prefix Hamming Distance (NPHD) is a valid metric specifically designed for variable-length
prefix-compatible codes like ISCC. Unlike standard Hamming distance, NPHD:

- Correctly handles variable-length comparisons
- Normalizes over the common prefix length
- Satisfies all metric axioms (non-negativity, identity, symmetry, triangle inequality)

The implementation lives in the external [iscc-usearch](https://github.com/iscc/iscc-usearch) package, which
iscc-search depends on for its HNSW backend.

### Storage

- **LMDB** is used for durable key-value storage: ISCC entries, metadata, and the inverted prefix-search index.
- **usearch** (HNSW) is used for approximate nearest-neighbor search over ISCC-UNITs and ISCC-SIMPRINTS.
- Multi-worker deployments are **not** supported with the usearch backend — see
    [docs/deployment.md](docs/deployment.md) for details.

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please ensure:

- All tests pass (`uv run poe test`)
- Code is formatted (`uv run poe format`)
- Coverage remains at 100%
- Changes are documented

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
