# ISCC-VDB Architecture

## Overview

ISCC-VDB implements a clean, protocol-based architecture that supports multiple index implementations through a
unified interface. The system uses Python's `typing.Protocol` to define an index abstraction that enables both
CLI and REST API frontends to work seamlessly with different storage implementations.

**Current Implementation Status**:

- MemoryIndex: Fully implemented (in-memory, no persistence, for testing)
- LmdbIndex: Fully implemented (LMDB-backed, production-ready with persistence)
- PostgresIndex: Planned for future development

## High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND LAYER                                 │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────────┐              ┌──────────────────────┐         │
│  │   CLI Package       │              │  Server Package      │         │
│  │   (cli/)            │              │  (server/)           │         │
│  │                     │              │                      │         │
│  │  - iscc-search index   │              │  - FastAPI app       │         │
│  │  - iscc-search add     │              │  - Route handlers    │         │
│  │  - iscc-search search  │              │  - Error handling    │         │
│  │  - iscc-search delete  │              │  - Sync endpoints    │         │
│  └──────────┬──────────┘              └──────────┬───────────┘         │
│             │                                    │                     │
│             └────────────────┬───────────────────┘                     │
│                              │                                         │
└──────────────────────────────┼─────────────────────────────────────────┘
                               │
┌──────────────────────────────┼─────────────────────────────────────────┐
│                      PROTOCOL LAYER                                    │
├──────────────────────────────┼─────────────────────────────────────────┤
│                              │                                         │
│                      ┌───────▼────────┐                                │
│                      │ IsccIndexProto │  (typing.Protocol)             │
│                      │                │                                │
│                      │ - list_indexes │  All methods synchronous       │
│                      │ - create_index │  Runtime checkable             │
│                      │ - get_index    │  Type-safe                     │
│                      │ - delete_index │                                │
│                      │ - add_assets   │                                │
│                      │ - get_asset    │                                │
│                      │ - search_assets│                                │
│                      │ - close        │                                │
│                      └───────┬────────┘                                │
│                              │                                         │
└──────────────────────────────┼─────────────────────────────────────────┘
                               │
                               │ implements
                               │
┌──────────────────────────────┼─────────────────────────────────────────┐
│                      INDEX LAYER                                       │
├──────────────────────────────┼─────────────────────────────────────────┤
│                              │                                         │
│    ┌─────────────────────────┼─────────────────────────┐               │
│    │                         │                         │               │
│    ▼                         ▼                         ▼               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │ LMDB Index       │  │ Postgres Index   │  │ Memory Index     │      │
│  │ (indexes/lmdb/)  │  │ (indexes/        │  │ (indexes/        │      │
│  │                  │  │  postgres/)      │  │  memory/)        │      │
│  │                  │  │                  │  │                  │      │
│  │ Production:      │  │ Planned:         │  │ Testing:         │      │
│  │ - LMDB store     │  │ - PG tables      │  │ - Dict-based     │      │
│  │ - Inverted       │  │ - pgvector       │  │ - No persistence │      │
│  │   per-unit index │  │   indexes        │  │ - Fast testing   │      │
│  │ - Bidirectional  │  │                  │  │                  │      │
│  │   prefix search  │  │                  │  │                  │      │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘      │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Protocol-Based Abstraction

The architecture uses Python's `typing.Protocol` to define the index interface. This provides:

- **Structural subtyping**: Index implementations don't inherit from a base class; they just implement the
    required methods
- **Static type checking**: Type checkers can verify index implementations
- **Runtime validation**: `@runtime_checkable` enables runtime type checking
- **Flexibility**: Easy to add new index implementations without modifying existing code

### 2. Synchronous API

All operations are synchronous for simplicity:

- **Easier implementation**: No async/await complexity
- **Direct database operations**: LMDB and usearch are inherently synchronous
- **Simpler testing**: Straightforward test setup without async fixtures
- **Postgres flexibility**: Can use psycopg2 (sync) initially, upgrade to asyncpg later if needed

### 3. Pydantic Settings-Based Configuration

Single `ISCC_VDB_INDEXES_URI` environment variable determines index implementation:

- **memory://**: `ISCC_VDB_INDEXES_URI=memory://` → MemoryIndex (in-memory, no persistence) **[IMPLEMENTED]**
- **Directory path**: `ISCC_VDB_INDEXES_URI=/path/to/index_data` → LmdbIndexManager (file-based persistence)
    **[IMPLEMENTED]**
- **Postgres DSN**: `ISCC_VDB_INDEXES_URI=postgresql://user:pass@host/db` → PostgresIndex **[PLANNED]**
- **Default**: Uses `platformdirs` to determine OS-appropriate user data directory → LmdbIndexManager
    **[IMPLEMENTED]**

Configuration uses Pydantic Settings for:

- Type validation and coercion
- Environment variable loading with `ISCC_VDB_` prefix
- `.env` file support
- Runtime override capability
- Clear documentation of all settings

### 4. Package-Per-Index

Each index implementation is a self-contained package:

```
indexes/
├── __init__.py          # Index factory and protocol definition
├── common.py            # Shared utilities (realm_id, normalization, validation)
├── lmdb/
│   ├── __init__.py      # LmdbIndexManager public API
│   ├── manager.py       # Protocol implementation managing multiple indexes
│   └── index.py         # Single LMDB index implementation
├── postgres/
│   ├── __init__.py      # PostgresIndex public API
│   ├── index.py         # Main index implementation
│   ├── schema.py        # Database schema definitions
│   └── queries.py       # SQL queries and operations
└── memory/
    ├── __init__.py      # MemoryIndex public API
    └── index.py         # In-memory index implementation
```

## Module Structure

```
iscc_search/
├── __init__.py
│
├── protocol.py              # IsccIndexProtocol definition
├── settings.py              # Pydantic settings and index factory
├── models.py                # Existing data models (IsccAsset, etc.)
├── schema.py                # Generated Pydantic models from OpenAPI
│
├── cli/
│   ├── __init__.py
│   ├── app.py              # Main Typer application
│   ├── commands.py         # CLI command implementations
│   └── utils.py            # CLI utilities
│
├── server/
│   ├── __init__.py
│   ├── app.py              # FastAPI application factory
│   ├── routes.py           # API route handlers
│   └── errors.py           # Error handlers and exceptions
│
├── indexes/
│   ├── __init__.py         # Index factory (get_index)
│   ├── common.py           # Shared utilities
│   ├── lmdb/               # LMDB-backed index (production)
│   ├── postgres/           # Postgres + pgvector index (planned)
│   └── memory/             # In-memory index (testing)
│
├── metrics.py               # NPHD metric implementation
├── nphd.py                  # NphdIndex for usearch
├── iscc.py                  # ISCC utilities
└── models.py                # Extended data models (IsccUnit, etc.)
```

## Core Components

### Schema Models (`schema.py`)

The schema is generated from OpenAPI specifications and defines the core data models:

**IsccAsset** - Represents an ISCC asset with metadata:

- `iscc_id` (str | None): Required when adding assets, optional for search queries
- `iscc_code` (str | None): Composite ISCC-CODE combining multiple ISCC-UNITs
- `units` (list[Unit] | None): List of individual ISCC-UNITs as canonical strings
- `metadata` (dict[str, Any] | None): Optional application-specific metadata stored opaquely

**IsccAddResult** - Result of adding an asset:

- `iscc_id` (str): The ISCC-ID of the asset that was added
- `status` (Status): Enum with values "created" or "updated"

**IsccIndex** - Index metadata:

- `name` (str): Short unique name matching pattern `^[a-z][a-z0-9]*$`
- `assets` (int | None): Number of ISCCs in the index (server-generated, read-only)
- `size` (int | None): Size of index in megabytes (server-generated, read-only)

**IsccSearchResult** - Search results:

- `query` (IsccAsset): The original query asset (may include auto-generated iscc_id)
- `metric` (Metric): Distance metric used (nphd, hamming, bitlength)
- `matches` (list[IsccMatch]): List of matched ISCC-IDs with scores

### Protocol Definition (`protocol.py`)

Defines `IsccIndexProtocol` as a runtime-checkable Protocol with methods:

- `list_indexes()` - List all available indexes with metadata
- `create_index(index)` - Create a new named index
- `get_index(name)` - Get index metadata by name
- `delete_index(name)` - Delete an index and all its data
- `add_assets(index_name, assets)` - Add assets to index (returns created/updated status)
- `get_asset(index_name, iscc_id)` - Get a specific asset by ISCC-ID
- `search_assets(index_name, query, limit)` - Search for similar assets
- `close()` - Close connections and cleanup resources

All methods are synchronous. Backends may use threading/connection pools internally.

### Settings and Configuration (`settings.py`)

**VdbSettings**: Pydantic settings class with:

- `indexes_uri` - Location for index data (path or DSN), defaults to OS user data directory
- Environment variable support (`ISCC_VDB_` prefix)
- `.env` file support
- `override()` method for runtime configuration changes

**get_index() Factory**: Parses `indexes_uri` and returns appropriate backend:

- `memory://` → MemoryIndex
- Directory path → LmdbIndexManager
- `postgresql://...` → PostgresIndex (planned)

### LMDB Index (`indexes/lmdb/`)

**LmdbIndexManager**: Protocol implementation managing multiple indexes in a base directory.

- Each index stored as separate `.lmdb` file (e.g., `myindex.lmdb`)
- Instance caching for performance
- Delegates operations to LmdbIndex instances

**LmdbIndex**: Single LMDB file containing:

- `__assets__` database: ISCC-ID → IsccAsset JSON
- `__metadata__` database: realm_id, created_at timestamps
- Per-unit-type databases: unit_body → [iscc_id_body, ...] (dupsort enabled)

**Key Features**:

- Bidirectional prefix search: matches both shorter and longer units
- Realm ID validation: ensures all assets in index have same realm
- Auto-resizing: doubles map_size on MapFullError
- Inverted index structure: efficient unit-based lookup

### Postgres Index (`indexes/postgres/`) - Planned

**PostgresIndex**: Protocol implementation using PostgreSQL + pgvector (not yet implemented).

**Planned Features**:

- Connection pooling for scalability
- `indexes` catalog table for multi-tenancy
- Per-index tables: `{name}_entries` for assets, per-unit-type tables for vectors
- pgvector extension for similarity search
- Horizontal scaling across multiple API instances

### Memory Index (`indexes/memory/`)

**MemoryIndex**: Simple dict-based in-memory storage for testing and development.

**Characteristics**:

- No persistence - data lost on process exit
- Simple exact-match search (no similarity)
- Dict storage: `{index_name: {assets: {iscc_id: IsccAsset}}}`
- Fast, zero dependencies, ideal for unit tests
- No `close()` cleanup needed

### FastAPI Server (`server/`)

**Application Factory** (`app.py`):

- `create_app()` initializes index from settings
- Index stored in `app.state` for request access
- Shutdown handler closes index resources
- Auto-generated OpenAPI documentation

**Route Handlers** (`routes.py`):

- RESTful endpoints mapping directly to protocol methods
- Proper HTTP status codes (201 Created, 404 Not Found, 409 Conflict)
- Exception translation to HTTPException
- Request validation via Pydantic models

### CLI Application (`cli/`)

**Commands**:

- `list` - List all indexes with metadata
- `create <name>` - Create new index
- `delete <name>` - Delete index (with confirmation)
- `add <index> <directory>` - Add assets from `*.iscc.json` files
- `search <index> <iscc_code>` - Search for similar assets

**Features**:

- Built with Typer for rich CLI experience
- JSON output for search results
- Error handling with appropriate exit codes
- Progress feedback for batch operations

## Usage Examples

### LMDB Index (Production)

```bash
# Set ISCC_VDB_INDEXES_URI to local path (or use default from platformdirs)
export ISCC_VDB_INDEXES_URI=/path/to/index_data

# CLI commands
iscc-search create myindex
iscc-search add myindex /data/
iscc-search search myindex "ISCC:..."
iscc-search list
iscc-search delete myindex

# Start API server with LMDB backend
uvicorn iscc_search.server.app:app --host 0.0.0.0 --port 8000
```

### Memory Index (Testing)

```bash
# Set ISCC_VDB_INDEXES_URI to memory:// for in-memory index
export ISCC_VDB_INDEXES_URI=memory://

# CLI works with in-memory storage (no persistence)
iscc-search create myindex
iscc-search add myindex /data/
iscc-search search myindex "ISCC:..."

# Start API server with in-memory index (for testing)
uvicorn iscc_search.server.app:app --host 0.0.0.0 --port 8000
```

### Postgres Index (Future)

```bash
# Set ISCC_VDB_INDEXES_URI to Postgres connection string
export ISCC_VDB_INDEXES_URI=postgresql://user:password@localhost/isccdb

# CLI and server work the same way across all backends
iscc-search create myindex
uvicorn iscc_search.server.app:app --host 0.0.0.0 --port 8000
```

## Key Benefits

1. **Clean Abstraction**: Protocol-based design enables index swapping without changing frontend code
2. **Simple Configuration**: Single `ISCC_VDB_INDEXES_URI` variable determines entire deployment topology
3. **Modular Packages**: Each index implementation is self-contained and independently testable
4. **Synchronous Simplicity**: No async complexity, straightforward implementation
5. **Zero Code Duplication**: CLI and API share identical index implementations
6. **Type Safety**: Protocol + Pydantic ensures compile-time and runtime validation
7. **Flexible Deployment**: LMDB (file-based), Memory (testing), or Postgres (future) configurations
8. **Easy Testing**: MemoryIndex for fast unit tests, LmdbIndex for production validation
9. **Future-Proof**: Add new index implementations without modifying existing code
10. **Developer Friendly**: Clear separation of concerns, easy to understand and extend

## Implementation Status

### Phase 1: Foundation ✓ **COMPLETED**

- Protocol definition (`IsccIndexProtocol`)
- Settings and configuration (`VdbSettings`, `get_index()`)
- Schema models from OpenAPI specification
- Index factory with URI parsing

### Phase 2: Memory Index ✓ **COMPLETED**

- Simple dict-based implementation for testing
- Full protocol compliance
- Comprehensive test coverage

### Phase 3: LMDB Index ✓ **COMPLETED**

- `LmdbIndexManager` for multi-index management
- `LmdbIndex` with inverted unit-type indexes
- Bidirectional prefix search for variable-length ISCC
- Realm ID validation
- Auto-resizing map_size handling
- Comprehensive test coverage

### Phase 4: Server Package ✓ **COMPLETED**

- FastAPI application factory
- RESTful route handlers
- OpenAPI documentation
- Error handling and validation

### Phase 5: CLI Package ✓ **COMPLETED**

- Typer-based commands
- Asset loading from JSON files
- Interactive confirmations
- JSON output formatting

### Phase 6: Future Work **PLANNED**

- PostgresIndex implementation with pgvector
- Performance benchmarks across backends
- Horizontal scaling documentation
- Advanced monitoring and observability

## Testing Strategy

### Unit Tests

- Protocol conformance tests for each index implementation
- Isolated tests with index-specific storage
- Mock-free testing using real backends (MemoryIndex for speed)

### Integration Tests

- Full stack with LmdbIndex (file-based, production-ready)
- Full stack with MemoryIndex (fast, ideal for CI/CD)
- Full stack with PostgresIndex (requires test database) - planned
- CLI → API → Index workflows

### End-to-End Tests

- Deployment scenarios across backends
- Configuration validation
- Performance benchmarks with real ISCC datasets

## Future Extensions

- **Redis Index**: Distributed in-memory index for hot data
- **S3 Index**: Cloud storage for massive archives
- **Hybrid Index**: Combine multiple index implementations with routing logic
- **HTTP Client Index**: Remote API client for distributed deployments
- **Async Support**: Add async variants of protocol methods
- **Batch Operations**: Optimize bulk add/search operations
- **Index Replication**: Sync indexes across implementations
- **Access Control**: Add authentication/authorization layer
