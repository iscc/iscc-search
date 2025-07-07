# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**iscc-vdb** is an Embedded Vector Database for ISCC (International Standard Content Code). Currently in early
development (v0.0.1), it will provide high-performance vector similarity search functionality as part of the
ISCC ecosystem.

## Development Commands

This project uses `uv` (modern Python package manager) and `poe` (poethepoet task runner). All commands should
be run with `uv run poe <command>`:

```bash
# Initial setup
uv run poe install      # Install dependencies and pre-commit hooks

# Code quality
uv run poe check        # Run all checks (lock file, pre-commit, mypy)

# Testing
uv run poe test         # Run tests with coverage
tox                     # Test across Python 3.10-3.13

# Documentation
uv run poe docs         # Serve documentation locally at http://localhost:8000
uv run poe docs-test    # Test documentation build

# Building
uv run poe build        # Build package wheel
```

## Architecture

The project structure follows standard Python packaging conventions:

- `iscc_vdb/` - Main package directory (currently contains only placeholder code)
- `tests/` - Test files using pytest
- `docs/` - MkDocs documentation with Material theme
- `scripts/` - Project-related tools, benchmarks, and experiments (committed to repo)
- `scratch/` - Local debugging and one-off scripts (not tracked in git)
- Core dependency: `usearch>=2.18.0` - vector search library

## Binary Vector Handling

- **ISCC vectors are binary**: 64, 128, 192, or 256 bits (8, 16, 24, or 32 bytes)
- **Usearch configuration**: `ndim` specifies bits (not bytes), use `dtype=ScalarKind.B1`
- **Storage calculations**: Always verify bit-to-byte conversions
- **Length signalling approach**: 264-bit vectors = 33 bytes (1 byte signal + 32 bytes max ISCC)

## NPHD Metric Properties

- NPHD (Normalized Prefix Hamming Distance) is a **valid metric** for prefix-compatible codes
- Satisfies all metric axioms: non-negativity, identity, symmetry, triangle inequality
- Correctly handles variable-length comparisons by normalizing over common prefix length
- Standard Hamming distance does NOT work because it treats all differences equally regardless of vector length

## Code Standards

- **Python versions**: 3.10 to 3.13
- **Line length**: 120 characters
- **Type checking**: Strict mypy configuration - all functions must be typed
- **Linting**: Ruff with extensive rule sets for code quality, security, and style
- **Testing**: pytest with coverage reporting
- **Pre-commit**: Automated checks run on commit

@~/.claude/docs/python.md

## Development Notes

1. This is a greenfield project - the actual vector database implementation hasn't been started yet
2. All infrastructure is in place for a production-ready library
3. When implementing features, maintain compatibility with Python 3.10+
4. Use `usearch` library for vector operations
5. Follow existing patterns from the ISCC ecosystem

## Testing

- Write tests in `tests/` directory following pytest conventions
- Run single test: `uv run pytest tests/test_foo.py::test_function_name`
- Run with coverage: `uv run poe test`
- Ensure all tests pass before committing

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
- Current project dependency: `usearch>=2.18.0`
- Primary use case: High-performance vector similarity search
- Language: C++ with Python bindings

#### Important Actionable Findings

*(To be populated as I discover usearch-specific insights)*
