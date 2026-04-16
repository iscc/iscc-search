# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**iscc-search** is a high-performance similarity search engine for ISCC (International Standard Content Code).
It ships as a Python package, a Typer-based CLI, and a FastAPI REST server. Indexes can be kept in memory,
persisted in LMDB, or accelerated with HNSW via the external `iscc-usearch` package. In active development
(v0.0.1).

## ISCC Specific Terminology

- **ISCC** - Any ISCC-CODE, ISCC-UNIT, or ISCC-ID
- **ISCC-HEADER** - Self-describing 2-byte header for V1 components (3 bytes for future versions). The first 12
    bits encode MainType, SubType, and Version. Additional bits encode Length for variable-length ISCCs.
- **ISCC-BODY** - Binary payload of an ISCC, similarity preserving compact binary code, hash or timestamp
    without HEADER
- **ISCC-DIGEST** - Binary representation of complete ISCC (ISCC-HEADER + ISCC-BODY).
- **ISCC-UNIT** - ISCC-HEADER + ISCC-BODY where the ISCC-BODY is calculated from a single algorithm
    - **META-UNIT** ISCC-UNIT that encodes syntactic/lexical **metadata** similarity
    - **SEMANTIC-UNIT** ISCC-UNITs that encode semantic/conceptual **content** similarity
        - Implemented SubTypes: TEXT, IMAGE, MIXED
    - **CONTENT-UNIT** ISCC-UNITs encode perceptual/syntactic/lexical/structural **content** similarity
        - Implemented SubTypes: TEXT, IMAGE, AUDIO, VIDEO, MIXED
    - **DATA-UNIT** ISCC-UNIT that encodes raw **data** similarity
    - **INSTANCE-UNIT** ISCC-UNIT identifies **data** like a checksum or cryptographic hash (depending on length)
- **ISCC-CODE** - ISCC-HEADER + ISCC-BODY where the ISCC-BODY is a sequence of multiple ISCC-UNIT BODYs
    - DATA and INSTANCE are the minimum required mandatory ISCC-UNITS for a valid ISCC-CODE
- **ISCC-ID** - Globally unique digital asset identifier (ISCC-HEADER + 52-bit timestamp + 12-bit server-id)
- **ISCC-SIMPRINT** - Headerless base64 encoded similarity hash that describes a content segment (granular
    feature)
- **ISCC-UNIT-TYPE**: Identifier for UNIT-TYPES that can be indexed together with meaningful similarity search
    - All ISCCs of the same type are stored in the same index regardless of length
    - The type is identified by the composite of MainType, SubType, Version
    - The type is encoded in the first 12 bits of the ISCC-HEADER
    - String representation example: CONTENT-TEXT-V0 (identified by the first 12 bits of an ISCC-UNIT)
    - Note: ISCC-UNIT-TYPE excludes the length segment from the header

## Development Commands

This project uses `uv` (modern Python package manager) and `poe` (poethepoet task runner):

```bash
# Show all available tasks
uv run poe --help

# OpenAPI build (regenerates schema.py from openapi.yaml and bundles openapi.json)
uv run poe build

# Formatting
uv run poe format-code      # Format Python code with ruff
uv run poe format-markdown  # Format markdown files
uv run poe format           # Format all files

# Testing
uv run poe test             # Run tests with coverage (fails if coverage < 100%)
uv run pytest tests/test_foo.py::test_function_name  # Run single test

# Complexity report
uv run poe check-complexity

# Pre-commit checks
uv run poe precommit

# Everything (build + format + test + complexity)
uv run poe all
```

Run the server locally: `uv run iscc-search serve --dev`.

## Architecture

iscc-search is organized around a protocol-based abstraction so CLI, REST API, and library users all talk to the
same `IsccIndexProtocol` regardless of the backend.

### Package Layout

- `iscc_search/` - Main package
    - `__init__.py` - Re-exports `SearchOptions` / `search_opts`; defines `PlatformDirs`
    - `options.py` - Server deployment configuration (`SearchOptions`, env vars prefixed `ISCC_SEARCH_`,
        `get_index()` factory that parses `index_uri`)
    - `config.py` - CLI multi-index management (persistent JSON at `~/.iscc-search/config.json`, git-style
        add/list/use/remove workflow)
    - `models.py` - `IsccBase`, `IsccID`, `IsccUnit`, `IsccCode`, `IsccItem` convenience classes on top of
        `iscc-core`
    - `schema.py` - **Auto-generated** Pydantic v2 models from `openapi/openapi.yaml` (do not hand-edit)
    - `processing.py` - Text processing helpers (tokenization, etc.)
    - `utils.py` - Shared utilities
    - `log_config.json` - Loguru logging configuration
    - `cli/` - Typer CLI (`add`, `get`, `search`, `serve`, `index` subcommands)
    - `server/` - FastAPI application (`assets`, `auth`, `indexes`, `search`, `playground`)
    - `remote/` - HTTP client (`IsccSearchClient`) for talking to a remote server
    - `protocols/` - `IsccIndexProtocol` Protocol definition
    - `indexes/` - Backend implementations (see below)
    - `openapi/` - Modular OpenAPI 3.0 spec (`openapi.yaml` + fragment files, bundled to `openapi.json`)
- `tests/` - Pytest test suite with 100% coverage requirement
- `docs/` - MkDocs documentation (Material theme)
- `scripts/bundle_openapi.py` - Bundles modular OpenAPI fragments into `openapi.json`
- `cauldron/` - Local untracked workspace for internal dev/ops info (see `CLAUDE.local.md`)

### Index Backends (`iscc_search/indexes/`)

All backends implement `IsccIndexProtocol`. `options.get_index()` selects one based on `ISCC_SEARCH_INDEX_URI`:

- `memory://` → `indexes/memory/` - In-memory dict-based, no persistence (tests, demos)
- `lmdb:///path` → `indexes/lmdb/` - LMDB-backed storage with inverted prefix-search index
- `usearch:///path` → `indexes/usearch/` - HNSW (via `iscc-usearch`) + LMDB for storage and metadata (production
    backend)

The `indexes/simprint/` subpackage implements ISCC-SIMPRINT (granular feature) indexing, with LMDB ops for exact
search and a usearch-backed approximate search path.

### Configuration Systems

There are **two** independent configuration systems — this is intentional and important:

1. **`options.py` (`SearchOptions`)** - Server deployment configuration
    - Consumed by `iscc-search serve` and the FastAPI app
    - Sourced from environment variables (`ISCC_SEARCH_*`) and `.env`
    - Single index per deployment, 12-factor style
1. **`config.py` (`AppConfig`)** - CLI multi-index management
    - Consumed by CLI data commands (`add`, `get`, `search`, `index ...`)
    - Sourced from persistent JSON at `~/.iscc-search/config.json`
    - Multiple named indexes (local or remote) with an "active" index, git-style workflow

Don't conflate them. The serve command uses `options.py`; CLI data commands use `config.py`.

### Key Dependencies

- `iscc-usearch>=0.6.1` - Provides `NphdIndex` and the NPHD metric (previously vendored into this repo as
    `nphd.py` / `metrics.py`; now an external package)
- `iscc-core>=1.2.1` - ISCC code generation and manipulation
- `iscc-sct>=0.1.3` - Semantic content-code generator
- `lmdb>=1.7.5` - Persistent key-value storage for entries, metadata, and inverted indexes
- `msgspec>=0.19.0` - Fast (de)serialization for simprint models
- `pysimdjson` - Fast JSON parsing
- `fastapi`, `uvicorn[standard]`, `pydantic-settings` - REST API server
- `typer`, `click`, `rich`, `tqdm` - CLI
- `loguru` - Logging
- `httpx` - Used by `remote/` client and by tests
- `platformdirs` - Cross-platform default data paths
- `simsimd` - SIMD-accelerated distance functions

## Code Standards

- **Python versions**: 3.11 to 3.14
- **Line length**: 120 characters
- **Type hints**: Use PEP 484 **type comments**, not annotations in function signatures (exception: FastAPI and
    Typer require annotations on route/command handlers — that's fine)
- **Imports**: Module level, absolute only
- **Linting**: Ruff with extensive rule sets
- **Testing**: pytest with **100% coverage requirement** (`fail_under = 100`)
- **Pre-commit**: Automated checks run on commit

@~/.claude/docs/python.md

## Code Style Preferences

1. **Prefer library functions over manual implementations**
    - Use `ic.decode_base64()` instead of manual base64 padding + `urlsafe_b64decode()`
    - Leverage domain-specific libraries (iscc-core, iscc-usearch, iscc-sct) for ISCC operations
1. **Use list comprehensions** over explicit append loops
1. **Thin wrappers over manual reconstruction** - CLI/API layers should serialize internal results directly
    rather than rebuilding schemas by hand

## Development Notes

1. **Auto-generated schema**: `iscc_search/schema.py` is generated from `iscc_search/openapi/openapi.yaml` via
    `uv run poe build-schema`. Don't hand-edit it — edit the YAML and rebuild.
1. **Coverage omits**: `cli/*`, `schema.py`, `protocols/*`, and `server/playground.py` are excluded from
    coverage (see `pyproject.toml [tool.coverage.run]`).
1. **Test fixtures**: Use fixtures from `tests/conftest.py` for ISCC code generation
1. **Type-only imports**: Ruff F401 doesn't recognize PEP 484 type-comment imports. Use the `TYPE_CHECKING`
    pattern:
    ```python
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from iscc_search.schema import IsccEntry  # noqa: F401
    ```
1. **Cross-platform**: All code must work on Linux, macOS, and Windows. `options.get_index()` contains
    Windows-specific URI path handling — preserve it when touching that code.
1. **Logging**: Use `loguru` (`from loguru import logger`). Log config is in `iscc_search/log_config.json`.
1. **Virtual environment**: Assume you are running inside an activated venv in the project directory — no need
    to prefix every command with `uv run`.

## Testing

- Tests live under `tests/` and follow the pattern `test_<module>_<submodule>.py`
- Run all: `uv run poe test` (parallel via `pytest-xdist`)
- Single test: `uv run pytest tests/test_indexes_lmdb_index.py::test_foo`
- **Coverage must stay at 100%** — adding new code means adding matching tests
- Prefer real data and fixtures over mocks. If a test requires heavy mocking, propose a refactor instead.

## External Library Documentation (DeepWiki)

Many libraries have nuanced details not in general training data. Use `mcp__deepwiki__ask_question` to verify
behavior before implementing:

- usearch → `unum-cloud/usearch`
- iscc-core → `iscc/iscc-core`
- lmdb (Python bindings) → `jnwatson/py-lmdb`
- lmdb (C library) → `LMDB/lmdb`
- simdjson → `TkTech/pysimdjson`
- fastapi → `fastapi/fastapi`
- typer → `fastapi/typer`
- rich → `Textualize/rich`
- uvicorn → `Kludex/uvicorn`
- loguru → `Delgan/loguru`
- msgspec → `jcrist/msgspec`

## Stability Policy

iscc-search is unreleased work-in-progress. We do **not** care about backwards compatibility and we do **not**
need to document breaking changes. Change anything that improves architecture, design, maintainability,
testability, or implementation.

@.claude/learnings.md

## Use the MAP

For a quick overview of the project structure, read `.claude/map.md` — it lists all directories, files,
functions, and classes. If the MAP looks out of date relative to the code you're reading, trust the code.

## Command Execution

- Never prefix commands with `cd <project-dir> &&`. The working directory is already the project root.
- Run commands directly (e.g., `gh issue list ...`, `uv run pytest`, not `cd /path && gh issue list ...`).
- Use absolute paths for files outside the project if needed.
