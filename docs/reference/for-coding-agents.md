---
icon: lucide/code
description: Dense reference optimized for AI coding agents working with the iscc-search codebase.
---

# For Coding Agents

Prescriptive reference for AI coding agents. Tables and code over prose. Terminology matches the
codebase exactly.

## Architecture Map

### File Layout

| Path | Contains |
|------|----------|
| `iscc_search/__init__.py` | Exports `SearchOptions`, `search_opts`, `PlatformDirs` |
| `iscc_search/options.py` | `SearchOptions` (Pydantic settings, env vars), `get_index()` factory |
| `iscc_search/config.py` | `AppConfig`, `ConfigManager` (CLI multi-index JSON config) |
| `iscc_search/models.py` | `IsccBase`, `IsccID`, `IsccUnit`, `IsccCode`, `IsccItem` |
| `iscc_search/schema.py` | **Auto-generated** Pydantic models from `openapi/openapi.yaml` |
| `iscc_search/processing.py` | Text tokenization utilities |
| `iscc_search/protocols/index.py` | `IsccIndexProtocol` (runtime-checkable Protocol) |
| `iscc_search/indexes/common.py` | Shared: serialization, ID encoding, validation, query normalization |
| `iscc_search/indexes/memory/index.py` | `MemoryIndex` (dict-based, no persistence) |
| `iscc_search/indexes/lmdb/index.py` | `LmdbIndex` (LMDB storage + inverted prefix search) |
| `iscc_search/indexes/lmdb/manager.py` | `LmdbIndexManager` (multi-index over .lmdb files) |
| `iscc_search/indexes/usearch/index.py` | `UsearchIndex` (HNSW + LMDB hybrid) |
| `iscc_search/indexes/usearch/manager.py` | `UsearchIndexManager` (multi-index over directories) |
| `iscc_search/indexes/simprint/lmdb_ops.py` | LMDB operations for simprints (pack/unpack, IDF, exact search) |
| `iscc_search/indexes/simprint/usearch_core.py` | `UsearchSimprintIndex` (ShardedIndex128 approximate search) |
| `iscc_search/indexes/simprint/models.py` | msgspec structs: `MatchedChunkRaw`, `SimprintMatchRaw` |
| `iscc_search/cli/` | Typer CLI: `add`, `get`, `search`, `serve`, `index` subcommands |
| `iscc_search/server/` | FastAPI app: endpoints for indexes, assets, search, auth, health |
| `iscc_search/remote/client.py` | `IsccSearchClient` (HTTP client for remote servers) |
| `iscc_search/openapi/` | Modular OpenAPI 3.0 spec (YAML fragments + bundled JSON) |

### Class Hierarchy

```
IsccIndexProtocol (typing.Protocol, runtime_checkable)
├── MemoryIndex          (indexes/memory/)
├── LmdbIndexManager     (indexes/lmdb/)
│   └── LmdbIndex        (per-index LMDB file)
└── UsearchIndexManager  (indexes/usearch/)
    └── UsearchIndex     (per-index directory)
        ├── ShardedNphdIndex      (from iscc-usearch, per unit-type)
        ├── UsearchSimprintIndex  (per simprint-type)
        └── LMDB env              (assets, metadata, instance, simprints)
```

### Schema Types (auto-generated, do not edit)

| Class | Purpose |
|-------|---------|
| `IsccIndex` | Index metadata (name, assets count, size) |
| `IsccEntry` | Asset record (iscc_id, iscc_code, units, simprints, metadata) |
| `IsccQuery` | Search query (iscc_code or units or simprints) |
| `IsccSearchResult` | Search response (query + global_matches + chunk_matches) |
| `IsccGlobalMatch` | Per-asset match (iscc_id, score, unit_scores) |
| `IsccAddResult` | Add response (iscc_id, status: created/updated) |
| `IsccSimprint` | Simprint with offset and size |
| `IsccChunk` | Chunk-level match detail |
| `TextQuery` | Text-based search query |

## Decision Dispatch

### Which backend?

| Scenario | Backend | URI | Why |
|----------|---------|-----|-----|
| Unit tests | Memory | `memory://` | No persistence, no deps |
| Exact prefix search | LMDB | `lmdb:///path` | Bidirectional prefix matching |
| Production similarity | USearch | `usearch:///path` | HNSW + LMDB hybrid |

### Which config system?

| Context | System | Source |
|---------|--------|--------|
| `iscc-search serve` | `options.py` (`SearchOptions`) | Env vars `ISCC_SEARCH_*` |
| `iscc-search add/get/search` | `config.py` (`ConfigManager`) | JSON `~/.iscc-search/config.json` |

### Which method for a task?

| Task | Protocol method | HTTP endpoint |
|------|----------------|---------------|
| Create index | `create_index(IsccIndex)` | `POST /indexes` |
| List indexes | `list_indexes()` | `GET /indexes` |
| Get index info | `get_index(name)` | `GET /indexes/{name}` |
| Delete index | `delete_index(name)` | `DELETE /indexes/{name}` |
| Add assets | `add_assets(name, [IsccEntry])` | `POST /indexes/{name}/assets` |
| Get asset | `get_asset(name, iscc_id)` | `GET /indexes/{name}/assets/{iscc_id}` |
| Search | `search_assets(name, IsccQuery, limit)` | `POST /indexes/{name}/search` |
| Health check | N/A | `GET /healthz` (liveness), `GET /readyz` (readiness) |

## Constraints and Invariants

### Index Names

Pattern: `^[a-z][a-z0-9]*$`. No underscores, hyphens, capitals, or special characters. Max
32 characters.

### ISCC-ID Format

- Total: 10 bytes (2-byte header + 8-byte body)
- Header encodes MainType=ID and realm_id (0 or 1)
- Body: 52-bit timestamp + 12-bit server-id
- Body stored as uint64 big-endian key in LMDB

### Realm Consistency

All assets in a single index must share the same `realm_id` (0 or 1). The first asset added
sets the realm. Subsequent assets with a different realm raise `ValueError`.

### LMDB Limits

| Backend | `max_dbs` | Reason |
|---------|-----------|--------|
| LmdbIndex | 16 | Assets + metadata + up to 14 unit-type indexes |
| UsearchIndex | 32 | Assets + metadata + instance + ~14 simprint types (2 DBs each) |

### Concurrency

| Backend | Readers | Writers | Multi-process |
|---------|---------|---------|---------------|
| Memory | Unlimited | Unlimited | No |
| LMDB | Multiple | Single (LMDB lock) | Reads only |
| USearch | Multiple | Single (`threading.RLock`) | No - corrupts `.usearch` files |

USearch production: single worker process, use FastAPI async for concurrency.

### MapFullError Auto-Retry

| Backend | Strategy | Limit |
|---------|----------|-------|
| LmdbIndex | Double `map_size` | Unbounded |
| UsearchIndex | Increment by `min(old_size, 1 GB)` | 10 retries, 1 TB max |

## Side Effects Catalog

### UsearchIndex.add_assets()

| Step | Mutation | Atomic with LMDB? |
|------|----------|--------------------|
| Store asset JSON in LMDB `__assets__` | Disk write | Yes (inside txn) |
| Add INSTANCE units to LMDB dupsort | Disk write | Yes (inside txn) |
| Store simprint data in LMDB | Disk write | Yes (inside txn) |
| LMDB transaction commits | Disk flush | - |
| Update ShardedNphdIndex (remove + add) | Memory + dirty flag | No |
| Update UsearchSimprintIndex | Memory + dirty flag | No |
| Auto-flush if dirty >= flush_interval | Disk write | No |

LMDB is the source of truth. Derived HNSW indexes can be rebuilt from LMDB if corrupted.

### UsearchIndex.flush()

Saves all dirty ShardedNphdIndex and UsearchSimprintIndex files to disk. Exception-safe: each
sub-index saved independently.

### UsearchIndex.close()

Calls `flush()`, then closes LMDB environment. Idempotent via `_closed` flag.

### Scoring Pipeline (UsearchIndex.search_assets)

```
Per-unit scores:
  INSTANCE → binary 1.0 (any prefix match)
  Similarity → 1.0 - NPHD distance

Aggregation:
  1. Filter: score >= match_threshold_units (default 0.75)
  2. Weight: score^confidence_exponent (default exponent 4)
  3. Average: sum(score^exponent) / sum(score)
  4. Self-exclude query asset by iscc_id
  5. Sort by score descending, take top limit
```

### Simprint Scoring

**Exact (LMDB)**: `score = coverage × quality`

- `coverage = unique_matched / total_query_simprints`
- `quality = min-max normalized inverse frequency`

**Approximate (UsearchSimprintIndex)**: IDF-weighted with oversampling

- Search with `limit × oversampling_factor` (default 20x) candidates
- Group by asset, IDF-weight per-query matches
- Unmatched query simprints penalize the score

## Task Recipes

### Add a new index backend

1. Create `iscc_search/indexes/yourbackend/` package
2. Implement all methods of `IsccIndexProtocol`
3. Add URI scheme handling in `options.get_index()`
4. Add tests matching pattern `test_indexes_yourbackend_*.py`
5. Update `docs/howto/index-backends.md` comparison table

### Add a new API endpoint

1. Add route handler in `iscc_search/server/` (new file or existing)
2. Add schema models in `iscc_search/openapi/` YAML fragments
3. Run `uv run poe build` to regenerate `schema.py` and `openapi.json`
4. Add tests in `tests/test_server*.py`

### Modify OpenAPI schema

1. Edit YAML fragments in `iscc_search/openapi/`
2. Run `uv run poe build-schema` to regenerate `iscc_search/schema.py`
3. Run `uv run poe build-openapi` to bundle `openapi.json`
4. Run `uv run poe build-validate` to check validity
5. Never hand-edit `schema.py`

### Add a new CLI command

1. Create handler in `iscc_search/cli/yourcommand.py`
2. Register with Typer app in `iscc_search/cli/__init__.py`
3. Use `get_active_index()` from `cli/common.py` for index access
4. Add tests in `tests/test_cli_*.py`

## Change Playbook

### If modifying `IsccIndexProtocol`

- Update all three backends: `MemoryIndex`, `LmdbIndexManager`, `UsearchIndexManager`
- Update `IsccSearchClient` in `remote/client.py`
- Update server route handlers in `server/`
- Update CLI commands if they use the changed method
- Update `docs/reference/api.md` (auto-generated, just rebuild)

### If modifying `SearchOptions`

- Check `server/__init__.py` for options usage
- Check `UsearchIndex.__init__()` and `search_assets()` for threshold/HNSW params
- Update `docs/reference/configuration.md` env var table
- Update `docs/howto/deployment.md` if production-relevant

### If modifying OpenAPI spec

- Edit YAML fragments in `iscc_search/openapi/`, not the bundled JSON
- Run `uv run poe build` (regenerates schema.py + bundles JSON + validates)
- Update any code using changed schema classes
- Never edit `schema.py` directly

### If modifying `indexes/common.py`

- Changes to `serialize_asset()` / `deserialize_asset()` affect ALL backends
- Changes to `validate_iscc_id()` affect all ID validation paths
- Changes to `normalize_query()` affect all search paths
- Run full test suite - these functions are called everywhere

### If modifying LMDB database schemas

- Changes to key/value format require data migration or re-index
- LMDB database names are string constants - grep for them
- `__assets__`, `__metadata__`, `__instance__` are reserved names
- Unit-type databases are created dynamically (e.g., `CONTENT_TEXT_V0`)

## Common Mistakes

### NEVER use multiple workers with usearch backend

Corrupts `.usearch` files silently. No file locking, no multi-process coordination.

```bash
# WRONG - data corruption
uvicorn iscc_search.server:app --workers 4

# CORRECT
uvicorn iscc_search.server:app
```

---

### NEVER hand-edit schema.py

It is auto-generated from `iscc_search/openapi/openapi.yaml`. Your changes will be
overwritten by `uv run poe build-schema`.

---

### NEVER mix realm_ids in one index

All assets must have the same realm_id (0 or 1). The first asset sets it. Adding an asset
with a different realm raises `ValueError`.

---

### ALWAYS call close() on index instances

`UsearchIndex.close()` flushes dirty HNSW indexes to disk. Skipping it loses unsaved data.
`LmdbIndex.close()` closes the LMDB environment. Leaked handles block other processes.

---

### ALWAYS use absolute imports

The codebase enforces absolute imports only. No relative imports (`from . import`).

---

### NEVER use type annotations in function signatures

Use PEP 484 type comments instead. Exception: FastAPI route handlers and Typer command
handlers require annotations.

```python
# WRONG
def search(query: IsccQuery, limit: int = 100) -> IsccSearchResult:

# CORRECT
def search(query, limit=100):
    # type: (IsccQuery, int) -> IsccSearchResult
```

---

### ALWAYS preserve Windows URI path handling in get_index()

`options.get_index()` has Windows-specific path normalization (strips leading `/` from
`/C:/path`). Do not simplify or remove this logic.

---

### NEVER assume derived indexes are in sync with LMDB

After `add_assets()`, LMDB commits first. HNSW updates happen outside the transaction. If the
process crashes between these steps, HNSW is stale. LMDB is always the source of truth.
Recovery: rebuild HNSW from LMDB data.

---

### ALWAYS run full test suite after changes to common.py

Functions in `indexes/common.py` (serialization, validation, normalization) are used by all
three backends. A change there can break any backend.

```bash
uv run poe test
```
