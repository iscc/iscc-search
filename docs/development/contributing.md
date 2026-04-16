---
icon: lucide/git-branch
description: Development setup, testing, and contribution guidelines.
---

# Contributing

## Prerequisites

You need the following installed:

- **Python 3.10-3.13**
- **[uv](https://docs.astral.sh/uv/)** - Python package manager
- **git**

## Development setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/iscc/iscc-search.git
cd iscc-search
uv sync
```

Verify the setup by running the test suite:

```bash
uv run poe test
```

## Available tasks

The project uses [poethepoet](https://poethepoet.naber.io/) as task runner.

| Task       | Command                       | Description                                               |
| ---------- | ----------------------------- | --------------------------------------------------------- |
| Build      | `uv run poe build`            | Regenerate schema.py from OpenAPI and bundle openapi.json |
| Format     | `uv run poe format`           | Format Python code with ruff                              |
| Test       | `uv run poe test`             | Run tests with coverage (100% required)                   |
| Complexity | `uv run poe check-complexity` | Check code complexity                                     |
| Pre-commit | `uv run poe precommit`        | Run pre-commit checks                                     |
| All        | `uv run poe all`              | Build + format + test + complexity                        |
| Docs serve | `uv run poe docs-serve`       | Serve documentation locally                               |
| Docs build | `uv run poe docs-build`       | Build documentation site                                  |

## Code standards

**Line length**: 120 characters.

**Type hints**: Use PEP 484 type comments on the first line below function definitions.
Do not use annotations in function signatures (exception: FastAPI and Typer handlers require them).

```python
def search_by_prefix(prefix, limit=100):
    # type: (bytes, int) -> list[bytes]
    """Search index entries by key prefix."""
    ...
```

**Imports**: Module-level, absolute only.

**Linting**: Ruff with extensive rule sets. Run `uv run poe format-code` to auto-fix.

**Principles**: YAGNI, SOLID, KISS, DRY. Prefer functional style with short, pure functions.

## Testing

Tests live in `tests/` and follow the pattern `test_<module>_<submodule>.py`. Coverage must
stay at 100% - adding code means adding matching tests.

```bash
# Run full suite with coverage
uv run poe test

# Run a single test
uv run pytest tests/test_indexes_lmdb_index.py::test_search_basic
```

Prefer real data and fixtures over mocks. Reusable fixtures live in `tests/conftest.py`.
If a test requires heavy mocking, propose a code refactor instead.

## Auto-generated files

`iscc_search/schema.py` is generated from `iscc_search/openapi/openapi.yaml`. Do not edit
it by hand. Edit the YAML source and rebuild:

```bash
uv run poe build-schema
```

## Cross-platform

All code must work on Linux, macOS, and Windows. Pay attention to path handling -
`options.get_index()` contains Windows-specific URI normalization that must be preserved.
