---
icon: lucide/house
description: Similarity search engine for ISCC content codes (ISO 24138). Python library, CLI, and REST API.
---

# iscc-search

[![Tests](https://github.com/iscc/iscc-search/actions/workflows/tests.yml/badge.svg)](https://github.com/iscc/iscc-search/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-iscc%2Fiscc--search-blue.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PHBhdGggZD0iTTQgMTkuNXYtMTVBMi41IDIuNSAwIDAgMSA2LjUgMkgxOXYyMEg2LjVhMi41IDIuNSAwIDAgMS0yLjUtMi41eiIvPjxwYXRoIGQ9Ik04IDdoNiIvPjxwYXRoIGQ9Ik04IDExaDgiLz48cGF0aCBkPSJNOCAxNWg1Ii8+PC9zdmc+)](https://deepwiki.com/iscc/iscc-search)

!!! warning "BETA"
    This project is under active development. The API is not yet stable and may change without
    notice.

**Similarity search engine for ISCC content codes.**

iscc-search indexes ISCC codes and finds similar digital content. You provide ISCC codes (content
fingerprints defined by ISO 24138), and the engine returns ranked matches based on content similarity.

The project supports multiple storage backends - from in-memory indexes for testing to HNSW-accelerated
persistent stores for production. You can use it as a Python library, a command-line tool, or a REST API
server.

## Key capabilities

- **Variable-length code matching** - compares ISCC codes of different lengths using normalized prefix
  Hamming distance
- **Multiple backends** - in-memory (`memory://`), LMDB-backed (`lmdb://`), and HNSW-accelerated
  (`usearch://`)
- **REST API** - FastAPI server with OpenAPI documentation, health checks, and optional authentication
- **CLI** - manage indexes, add assets, and search from the terminal
- **Protocol-based abstraction** - all backends implement `IsccIndexProtocol`, so you can swap storage
  without changing application code

## Quick start

=== "uv"

    ```bash
    uv add iscc-search
    ```

=== "pip"

    ```bash
    pip install iscc-search
    ```

Use the Python API to create an index, add an asset, and search:

```python
import os

os.environ["ISCC_SEARCH_INDEX_URI"] = "memory://"

from iscc_search.options import get_index
from iscc_search.schema import IsccEntry, IsccIndex, IsccQuery

# Create index backend
index = get_index()
index.create_index(IsccIndex(name="demo"))

# Add an asset with an ISCC-CODE
asset = IsccEntry(iscc_code="ISCC:KACYPXW445FTYNJ3CYSXHAFJMA2HUWULUNRFE3BLHRSCXYH2M5AEGQY")
index.add_assets("demo", [asset])

# Search for similar content
query = IsccQuery(iscc_code="ISCC:KACYPXW445FTYNJ3CYSXHAFJMA2HUWULUNRFE3BLHRSCXYH2M5AEGQY")
results = index.search_assets("demo", query)

for match in results.global_matches:
    print(f"{match.iscc_id}  score={match.score}")

index.close()
```

## Documentation

<div class="grid cards" markdown>

- **[Tutorials](tutorials/getting-started.md)** - Learn the basics

    Hands-on guide from installation to your first search.

- **[How-to Guides](howto/index-backends.md)** - Solve specific problems

    Backend configuration, CLI usage, REST API, and deployment.

- **[Explanation](explanation/iscc-primer.md)** - Understand the concepts

    How ISCC works, system architecture, and similarity search internals.

- **[Reference](reference/api.md)** - Look up details

    API reference, configuration options, and agent documentation.

</div>

[Source code on GitHub :lucide-external-link:](https://github.com/iscc/iscc-search){ .md-button }
