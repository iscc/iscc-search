# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**iscc-search** is a high-performance ISCC similarity search engine for ISCC (International Standard Content
Code). In active development (v0.0.1), it provides high-performance vector similarity search for variable-length
binary ISCC vectors using the Normalized Prefix Hamming Distance (NPHD) metric. Built on top of usearch for fast
approximate nearest neighbor search.

## Development Commands

This project uses `uv` (modern Python package manager) and `poe` (poethepoet task runner). All commands should
be run with `uv run poe <command>`:

```bash
# Show all available tasks
uv run poe --help

# Formatting
uv run poe format-code      # Format Python code with ruff
uv run poe format-markdown  # Format markdown files
uv run poe format           # Format all files (code + markdown)

# Testing
uv run poe test             # Run tests with coverage (fails if coverage < 100%)
uv run pytest tests/test_foo.py::test_function_name  # Run single test

# Pre-commit checks
uv run poe precommit        # Run pre-commit hooks

# All checks
uv run poe all              # Format, test, and all checks
```

## Architecture

The project structure follows standard Python packaging conventions:

- `iscc_search/` - Main package directory
    - `nphd.py` - NphdIndex class for ANNS with variable-length binary vectors
    - `metrics.py` - Custom NPHD metric implementation using Numba
- `tests/` - Test files using pytest
    - `conftest.py` - Shared fixtures for ISCC-IDs and ISCC-UNITs
    - `test_nphd.py` - Tests for NphdIndex, pad/unpad functions
    - `test_metrics.py` - Tests for NPHD metric calculation
    - `test_usearch_*.py` - Low-level usearch API tests
- `docs/` - MkDocs documentation with Material theme
- `scratch/` - Local debugging and one-off scripts (not tracked in git)

### Core Components

**NphdIndex** (`iscc_search/nphd.py`): Main index class that wraps usearch Index with:

- Automatic vector padding/unpadding for variable-length ISCC vectors
- NPHD metric configuration (binary vectors with length signal byte)
- Supports 64-256 bit vectors (configurable via `max_dim` parameter)

**Custom NPHD Metric** (`iscc_search/metrics.py`):

- Compiled Numba function for fast NPHD calculation
- Handles variable-length vectors via length signal in first byte
- Uses bit-packed binary representation (ScalarKind.B1)

### Key Dependencies

- `usearch==2.21.0` - Custom build from iscc.github.io (platform-specific wheels)
- `iscc-core>=1.2.1` - ISCC code generation and manipulation
- `numba>=0.60.0` - JIT compilation for custom metrics
- `platformdirs>=4.3.8` - Cross-platform directory paths

## Binary Vector Handling

- **ISCC vectors are binary**: 64, 128, 192, or 256 bits (8, 16, 24, or 32 bytes)
- **Usearch configuration**: `ndim` specifies bits (not bytes), use `dtype=ScalarKind.B1`
- **Storage calculations**: Always verify bit-to-byte conversions
- **Length signaling approach**: Default 264-bit vectors = 33 bytes (1 byte signal + 32 bytes max ISCC)
- **Vector padding**: Use `pad_vectors()` to add length prefix and pad to uniform size
- **Vector unpadding**: Use `unpad_vectors()` to extract original variable-length vectors

## NPHD Metric Properties

- NPHD (Normalized Prefix Hamming Distance) is a **valid metric** for prefix-compatible codes
- Satisfies all metric axioms: non-negativity, identity, symmetry, triangle inequality
- Correctly handles variable-length comparisons by normalizing over common prefix length
- Standard Hamming distance does NOT work because it treats all differences equally regardless of vector length

## Code Standards

- **Python versions**: 3.10 to 3.13
- **Line length**: 120 characters
- **Type checking**: Strict mypy configuration - all functions must use PEP 484 type comments
- **Type comments**: Use `# type: (arg_types) -> return_type` format (see @~/.claude/docs/python.md)
- **Linting**: Ruff with extensive rule sets for code quality, security, and style
- **Testing**: pytest with 100% coverage requirement (fail_under = 100)
- **Pre-commit**: Automated checks run on commit

@~/.claude/docs/python.md

## Development Notes

1. **Custom usearch build**: Project uses custom usearch wheels from iscc.github.io (not PyPI)
2. **Test fixtures**: Use fixtures from `tests/conftest.py` for ISCC code generation
3. **Numba JIT**: Functions decorated with `@njit` won't show in coverage - mark with `# pragma: no cover`
4. **Type hints**: Always use PEP 484 type comments, not function annotations
5. **Cross-platform**: Ensure all code works on Linux, macOS, and Windows
6. **Type-only imports**: Ruff F401 doesn't recognize imports used only in PEP 484 type comments as valid usage
    (known issue: [astral-sh/ruff#1619](https://github.com/astral-sh/ruff/issues/1619)). Workaround pattern:
    ```python
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from iscc_search.schema import IsccAsset  # noqa: F401
    ```
    This allows IDEs to resolve type references while preventing ruff from removing the imports

## Testing

- Write tests in `tests/` directory following pytest conventions
- Use shared fixtures from `tests/conftest.py` for ISCC code generation
- Run single test: `uv run pytest tests/test_foo.py::test_function_name`
- Run all tests with coverage: `uv run poe test`
- Coverage requirement is 100% - all code must be tested
- Mark JIT-compiled code with `# pragma: no cover` as it's not traceable
- Tests include: unit tests, integration tests, edge cases, and usearch API validation

## usearch Library Deep Dive

### Memory Reminder

**IMPORTANT**: When implementing NPHD or custom metrics, ALWAYS use `mcp__deepwiki__ask_question` with
repository `unum-cloud/usearch` to verify:

- Binary vector handling (`ScalarKind.B1`, bit packing)
- Custom metric implementation (`CompiledMetric`, Numba requirements)
- Index initialization parameters for binary data
- Performance implications of custom metrics

Common pitfalls to avoid:

- Confusing `ndim` (number of bits) with bytes
- Assuming built-in metrics work for variable-length vectors
- Not using bit-packing for binary data

### usearch Key Findings & Notes

*(This section will be updated as I learn more about usearch)*

#### Core Library Information

- Repository: `unum-cloud/usearch`
- Current project dependency: `usearch==2.21.0` (custom build)
- Primary use case: High-performance vector similarity search
- Language: C++ with Python bindings

#### Important Actionable Findings

- **Custom wheels**: Project uses custom usearch 2.21.0 builds hosted at iscc.github.io
- **Binary vectors**: Use `ScalarKind.B1` and set `ndim` to bit count (not bytes)
- **Custom metrics**: Use `CompiledMetric` with Numba `@cfunc` for custom distance functions
- **Length signaling**: First byte stores actual vector length for variable-length support
- **Batch operations**: Index supports batch add/search with multiple vectors
- **Virtual Environment**: Assume you are running within an activated virtual environment
- **Command prefix**: You don't need to prefix commands with `uv run` with an activated virtual environment
- **Working directory**: Assume your current working directory is the project directory

## Remember

For now ISCC-SEARCH is unreleased work in progress and we do not care about backwards compatibility neither do
we need to document breaking changes. We are free to change anything at any time if it helps to improve the
project architecture, desing, maintainability, testability, or implementation.

- Use loguru for logging
