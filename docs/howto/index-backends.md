---
icon: lucide/hard-drive
description: Choose and configure the right index backend for your workload.
---

# Index Backends

iscc-search supports three index backends. You select a backend by setting the `ISCC_SEARCH_INDEX_URI`
environment variable or passing a URI to the CLI configuration.

## Memory backend

The memory backend stores all data in-process dictionaries. Data is lost when the process exits. Use it for
tests and quick experiments.

```bash
export ISCC_SEARCH_INDEX_URI=memory://
```

```python
from iscc_search.indexes.memory import MemoryIndex

index = MemoryIndex()
```

No additional configuration is needed.

## LMDB backend

The LMDB backend persists assets in Lightning Memory-Mapped Database files. It supports bidirectional prefix
search on ISCC codes without HNSW indexing.

```bash
export ISCC_SEARCH_INDEX_URI=lmdb:///var/lib/iscc-search
```

Each index is stored as a single `.lmdb` file inside the configured directory:

```
/var/lib/iscc-search/
├── myindex.lmdb
└── another.lmdb
```

!!! warning "Single-writer only"

    LMDB supports multiple readers but only one writer at a time. Concurrent write attempts from
    separate processes will block until the lock is released.

## USearch backend

The usearch backend combines HNSW approximate nearest neighbor search (via `iscc-usearch`) with LMDB for
persistent storage. This is the production backend for similarity search.

```bash
export ISCC_SEARCH_INDEX_URI=usearch:///var/lib/iscc-search
```

Each index is a directory containing an LMDB file plus `.usearch` shard files for each ISCC-UNIT type:

```
/var/lib/iscc-search/
├── myindex/
│   ├── index.lmdb
│   ├── CONTENT_TEXT_V0.usearch
│   ├── SEMANTIC_TEXT_V0.usearch
│   └── DATA_NONE_V0.usearch
└── another/
    ├── index.lmdb
    └── ...
```

!!! warning "Single process only"

    The `.usearch` files have no multi-process coordination. Running multiple workers or processes against
    the same data directory **will corrupt your indexes**. Always run with a single worker process.

You can tune HNSW parameters via environment variables:

| Variable                                  | Default | Purpose                                  |
| ----------------------------------------- | ------- | ---------------------------------------- |
| `ISCC_SEARCH_HNSW_CONNECTIVITY_UNITS`     | 16      | Graph connectivity (M) for unit indexes  |
| `ISCC_SEARCH_HNSW_EXPANSION_ADD_UNITS`    | 128     | Build-time search depth (efConstruction) |
| `ISCC_SEARCH_HNSW_EXPANSION_SEARCH_UNITS` | 64      | Query-time search depth (ef)             |
| `ISCC_SEARCH_SHARD_SIZE_UNITS`            | 512     | Max shard file size in MB                |

## Choosing a backend

| Backend | URI               | Persistence | Search type     | Best for                     |
| ------- | ----------------- | ----------- | --------------- | ---------------------------- |
| Memory  | `memory://`       | None        | Exact           | Tests, demos, prototyping    |
| LMDB    | `lmdb:///path`    | Disk        | Prefix search   | Small datasets, exact lookup |
| USearch | `usearch:///path` | Disk        | HNSW similarity | Production similarity search |

For most production deployments, use the usearch backend. Use LMDB when you need prefix-based lookup without
similarity search. Use memory for automated tests.

For details on how the backends work internally, see the
[Architecture](../explanation/architecture.md) explanation.
