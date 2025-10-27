# iscc-search

[![Release](https://img.shields.io/github/v/release/iscc/iscc-search)](https://img.shields.io/github/v/release/iscc/iscc-search)
[![Build status](https://img.shields.io/github/actions/workflow/status/iscc/iscc-search/main.yml?branch=main)](https://github.com/iscc/iscc-search/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/iscc/iscc-search/branch/main/graph/badge.svg)](https://codecov.io/gh/iscc/iscc-search)
[![Commit activity](https://img.shields.io/github/commit-activity/m/iscc/iscc-search)](https://img.shields.io/github/commit-activity/m/iscc/iscc-search)
[![License](https://img.shields.io/github/license/iscc/iscc-search)](https://img.shields.io/github/license/iscc/iscc-search)

> [!WARNING]
> **This project is in early development and not ready for production use.**
>
> The API and features are subject to significant changes. Use at your own risk.

High-performance ISCC similarity search engine for variable-length binary [ISCC](https://iscc.codes) codes with
fast approximate nearest neighbor search.

- **Github repository**: <https://github.com/iscc/iscc-search/>
- **Documentation** <https://search.iscc.codes/>

## Features

- Fast approximate nearest neighbor search (ANNS) for variable-length binary vectors
- Custom NPHD (Normalized Prefix Hamming Distance) metric optimized for ISCC codes
- Support for 64-256 bit vectors (8-32 bytes)
- Built on [usearch](https://github.com/unum-cloud/usearch) with JIT-compiled Numba metrics
- Cross-platform support (Linux, macOS, Windows)
- Python 3.10-3.13 support

## What is ISCC?

The [International Standard Content Code (ISCC)](https://iscc.codes) is a similarity-preserving content
identifier for digital media. ISCC codes are variable-length binary vectors that enable efficient similarity
search across different media types. This library provides a specialized vector database for storing and
querying ISCC codes at scale.

## Installation

```bash
pip install iscc-search
```

For development installation:

```bash
git clone https://github.com/iscc/iscc-search.git
cd iscc-search
uv sync
```

## Quick Start

```python
from iscc_search import NphdIndex
import numpy as np

# Create index for up to 256-bit vectors
index = NphdIndex(max_dim=256)

# Add some binary vectors with integer keys
vectors = [
    np.array([18, 52, 86, 120], dtype=np.uint8),  # 32-bit vector
    np.array([171, 205, 239], dtype=np.uint8),  # 24-bit vector
    np.array([17, 34, 51, 68, 85], dtype=np.uint8),  # 40-bit vector
]
keys = [1, 2, 3]
index.add(keys, vectors)

# Search for similar vectors
query = np.array([18, 52, 86, 121], dtype=np.uint8)
matches = index.search(query, k=2)

print(f"Found {len(matches.keys)} matches")
print(f"Keys: {matches.keys}")
print(f"Distances: {matches.distances}")
```

## API Overview

### NphdIndex

The main index class for ANNS with variable-length binary vectors.

```python
NphdIndex(max_dim=256, **kwargs)
```

- `max_dim`: Maximum vector dimension in bits (default: 256)
- `**kwargs`: Additional arguments passed to usearch Index

#### Methods

- `add(keys, vectors)`: Add vectors with integer keys
- `search(query, k)`: Search for k nearest neighbors
- `get(keys)`: Retrieve vectors by keys
- `remove(keys)`: Remove vectors by keys

## Development

This project uses [uv](https://docs.astral.sh/uv/) for package management and
[poethepoet](https://github.com/nat-n/poethepoet) for task automation.

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Available Commands

```bash
uv run poe format-code      # Format Python code with ruff
uv run poe format-markdown  # Format markdown files
uv run poe format           # Format all files
uv run poe test             # Run tests with coverage (requires 100%)
uv run poe precommit        # Run pre-commit hooks
uv run poe all              # Format and test
```

### Running Tests

```bash
# Run all tests with coverage
uv run poe test

# Run specific test
uv run pytest tests/test_nphd.py::test_pad_vectors

# Run tests in watch mode
uv run pytest --watch
```

## Technical Details

### NPHD Metric

The Normalized Prefix Hamming Distance (NPHD) is a valid metric specifically designed for variable-length
prefix-compatible codes like ISCC. It normalizes the Hamming distance by the length of the common prefix,
enabling meaningful similarity comparisons between vectors of different lengths.

Unlike standard Hamming distance, NPHD:

- Correctly handles variable-length comparisons
- Normalizes over common prefix length
- Satisfies all metric axioms (non-negativity, identity, symmetry, triangle inequality)

### Binary Vector Format

Vectors are stored as packed binary arrays (`np.uint8`) with an internal length prefix:

- Each vector is prefixed with a length byte
- Vectors are padded to uniform size for efficient indexing
- `pad_vectors()` and `unpad_vectors()` handle conversions automatically

### Custom usearch Build

This project uses custom usearch 2.21.0 wheels with platform-specific builds hosted at iscc.github.io to ensure
consistent behavior across platforms.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please ensure:

- All tests pass (`uv run poe test`)
- Code is formatted (`uv run poe format`)
- Coverage remains at 100%
- Changes are documented

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

______________________________________________________________________

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
