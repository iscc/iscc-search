# ISCC-VDB Architecture

## Overview

ISCC-VDB implements a clean, protocol-based architecture that supports multiple index implementations through a
unified interface. The system uses Python's `typing.Protocol` to define an index abstraction that enables both
CLI and REST API frontends to work seamlessly with different storage implementations (usearch, postgres,
in-memory).

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
│  │  - iscc-vdb index   │              │  - FastAPI app       │         │
│  │  - iscc-vdb add     │              │  - Route handlers    │         │
│  │  - iscc-vdb search  │              │  - Error handling    │         │
│  │  - iscc-vdb delete  │              │  - Sync endpoints    │         │
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
│                      │ - add_items    │                                │
│                      │ - search_items │                                │
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
│  │ Usearch Index    │  │ Postgres Index   │  │ Memory Index     │      │
│  │ (indexes/        │  │ (indexes/        │  │ (indexes/        │      │
│  │  usearch/)       │  │  postgres/)      │  │  memory/)        │      │
│  │                  │  │                  │  │                  │      │
│  │ Unified index:   │  │ Unified index:   │  │ In-memory index: │      │
│  │ - LMDB store     │  │ - PG tables      │  │ - Dict-based     │      │
│  │ - Per-UNIT       │  │ - pgvector       │  │ - No persistence │      │
│  │   usearch indexes│  │   indexes        │  │ - Fast testing   │      │
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

- **Directory path**: `ISCC_VDB_INDEXES_URI=/path/to/index_data` → UsearchIndex
- **Postgres DSN**: `ISCC_VDB_INDEXES_URI=postgresql://user:pass@host/db` → PostgresIndex
- **memory://**: `ISCC_VDB_INDEXES_URI=memory://` → MemoryIndex (in-memory, no persistence)
- **Default**: Uses `platformdirs` to determine OS-appropriate user data directory

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
├── usearch/
│   ├── __init__.py      # UsearchIndex public API
│   ├── index.py         # Main index implementation
│   ├── store.py         # LMDB storage layer (from existing store.py)
│   └── unit.py          # Usearch unit index management
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
iscc_vdb/
├── __init__.py
│
├── protocol.py              # IsccIndexProtocol definition
├── settings.py              # Pydantic settings and index factory
├── models.py                # Existing data models (IsccItem, etc.)
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
│   ├── __init__.py         # Index factory
│   ├── usearch/            # Local LMDB + Usearch index
│   ├── postgres/           # Postgres + pgvector index
│   └── memory/             # In-memory index (for testing)
│
├── metrics.py               # NPHD metric implementation
├── nphd.py                  # NphdIndex for usearch
├── iscc.py                  # ISCC utilities
├── lookup.py                # Lookup index (may be refactored into usearch index)
├── store.py                 # LMDB store (may be refactored into usearch index)
└── unit.py                  # Unit index (may be refactored into usearch index)
```

## Core Components

### Protocol Definition (`protocol.py`)

```python
from typing import Protocol, runtime_checkable
from iscc_vdb.schema import IsccIndex, IsccItem

@runtime_checkable
class IsccIndexProtocol(Protocol):
    """
    Protocol for ISCC index backends.

    All methods are synchronous. Backends are free to use
    threading, connection pools, etc. internally.
    """

    def list_indexes(self) -> list[IsccIndex]:
        """
        List all available indexes with metadata.

        :return: List of IsccIndex objects with name, items, and size
        """
        ...

    def create_index(self, index: IsccIndex) -> IsccIndex:
        """
        Create a new named index.

        :param index: IsccIndex with name (items and size ignored)
        :return: Created IsccIndex with initial metadata (items=0, size=0)
        :raises ValueError: If name is invalid
        :raises FileExistsError: If index already exists
        """
        ...

    def get_index(self, name: str) -> IsccIndex:
        """
        Get index metadata by name.

        :param name: Index name
        :return: IsccIndex with current metadata
        :raises FileNotFoundError: If index doesn't exist
        """
        ...

    def delete_index(self, name: str) -> None:
        """
        Delete an index and all its data.

        :param name: Index name
        :raises FileNotFoundError: If index doesn't exist
        """
        ...

    def add_items(
        self,
        index_name: str,
        items: list[IsccItem]
    ) -> int:
        """
        Add items to index.

        :param index_name: Target index name
        :param items: List of IsccItem objects to add
        :return: Number of items successfully added
        :raises FileNotFoundError: If index doesn't exist
        """
        ...

    def search_items(
        self,
        index_name: str,
        query: IsccItem,
        limit: int = 100
    ) -> list[dict]:
        """
        Search for similar items in index.

        :param index_name: Target index name
        :param query: IsccItem to search for
        :param limit: Maximum number of results
        :return: List of match dictionaries with scores and metadata
        :raises FileNotFoundError: If index doesn't exist
        """
        ...

    def close(self) -> None:
        """
        Close connections and cleanup resources.

        Should be called when backend is no longer needed.
        Safe to call multiple times.
        """
        ...
```

### Settings and Configuration (`settings.py`)

```python
from pathlib import Path
from urllib.parse import urlparse
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import iscc_vdb

class VdbSettings(BaseSettings):
    """
    Application settings for ISCC-VDB.

    Settings can be configured via:
    - Environment variables (prefixed with ISCC_VDB_)
    - .env file in the working directory
    - Direct instantiation with parameters
    - Runtime override using the override() method
    """

    indexes_uri: str = Field(
        iscc_vdb.dirs.user_data_dir,
        description="Location where index data is stored (local file path or DSN)",
    )

    model_config = SettingsConfigDict(
        env_prefix="ISCC_VDB_",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        validate_assignment=True,
    )

    def override(self, update=None):
        # type: (dict|None) -> VdbSettings
        """
        Returns an updated and validated deep copy of the current settings instance.

        :param update: Dictionary of field names and values to override.
        :return: New VdbSettings instance with updated and validated fields.
        """
        update = update or {}
        settings = self.model_copy(deep=True)
        for field, value in update.items():
            setattr(settings, field, value)
        return settings


# Global settings instance
vdb_settings = VdbSettings()


def get_index() -> IsccIndexProtocol:
    """
    Factory function to create index from settings.

    Parses indexes_uri to determine index type:
    - file:// or path → UsearchIndex
    - postgresql:// → PostgresIndex
    - memory:// → MemoryIndex

    :return: Index instance implementing IsccIndexProtocol
    :raises ValueError: If URI scheme is not supported
    """
    uri = vdb_settings.indexes_uri
    parsed = urlparse(uri)

    if parsed.scheme in ("", "file"):
        # Local file path
        from iscc_vdb.indexes.usearch import UsearchIndex
        path = parsed.path if parsed.scheme == "file" else uri
        return UsearchIndex(Path(path))

    elif parsed.scheme in ("postgresql", "postgres"):
        # Postgres connection
        from iscc_vdb.indexes.postgres import PostgresIndex
        return PostgresIndex(uri)

    elif parsed.scheme == "memory":
        # In-memory index
        from iscc_vdb.indexes.memory import MemoryIndex
        return MemoryIndex()

    else:
        raise ValueError(f"Unsupported ISCC_VDB_INDEXES_URI scheme: {parsed.scheme}")
```

### Usearch Index (`indexes/usearch/index.py`)

```python
from pathlib import Path
from iscc_vdb.protocol import IsccIndexProtocol
from iscc_vdb.schema import IsccIndex, IsccItem

class UsearchIndex:
    """
    Unified ISCC index using LMDB + per-UNIT-TYPE usearch indexes.

    Directory structure per index:
    /base/path/
    ├── index1/
    │   ├── store.mdb              # LMDB for entries and metadata
    │   ├── CONTENT_TEXT_V0.usearch
    │   ├── DATA_NONE_V0.usearch
    │   └── ...
    └── index2/
        └── ...
    """

    def __init__(self, base_path: Path):
        """
        Initialize UsearchIndex.

        :param base_path: Base directory for all indexes
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._indexes = {}  # Cache of loaded indexes

    def list_indexes(self) -> list[IsccIndex]:
        """List all indexes by scanning base_path directories."""
        indexes = []
        for path in self.base_path.iterdir():
            if path.is_dir() and (path / "store.mdb").exists():
                # Load index metadata
                index_data = self._load_index_metadata(path.name)
                indexes.append(index_data)
        return indexes

    def create_index(self, index: IsccIndex) -> IsccIndex:
        """Create new index directory and initialize LMDB."""
        index_path = self.base_path / index.name
        if index_path.exists():
            raise FileExistsError(f"Index '{index.name}' already exists")

        index_path.mkdir(parents=True)

        # Initialize LMDB store
        from iscc_vdb.indexes.usearch.store import IsccStore
        store = IsccStore(index_path / "store.mdb")
        store.put_metadata("__created__", time.time())
        store.close()

        return IsccIndex(name=index.name, items=0, size=0)

    def get_index(self, name: str) -> IsccIndex:
        """Get index metadata."""
        index_path = self.base_path / name
        if not index_path.exists():
            raise FileNotFoundError(f"Index '{name}' not found")

        return self._load_index_metadata(name)

    def delete_index(self, name: str) -> None:
        """Delete index directory and all data."""
        index_path = self.base_path / name
        if not index_path.exists():
            raise FileNotFoundError(f"Index '{name}' not found")

        # Close any open resources
        if name in self._indexes:
            self._indexes[name].close()
            del self._indexes[name]

        # Remove directory
        import shutil
        shutil.rmtree(index_path)

    def add_items(self, index_name: str, items: list[IsccItem]) -> int:
        """
        Add items to index.

        - Store in LMDB
        - Extract ISCC-UNITs
        - Add to per-UNIT-TYPE usearch indexes
        """
        idx = self._get_or_load_index(index_name)

        # Convert to IsccItem objects and add to store
        iscc_items = [IsccItem.from_dict(item) for item in items]

        # Store in LMDB
        store = idx["store"]
        iscc_ids = [int(IsccID(item.id_data)) for item in iscc_items]
        entries = [item.dict for item in iscc_items]
        added = store.add(iscc_ids, entries)

        # Index units in usearch
        for item in iscc_items:
            for unit_str in item.units:
                unit = IsccUnit(unit_str)
                unit_type = unit.unit_type

                # Get or create usearch index for this unit type
                unit_idx = idx["unit_indexes"].get(unit_type)
                if unit_idx is None:
                    unit_idx = self._create_unit_index(index_name, unit_type)
                    idx["unit_indexes"][unit_type] = unit_idx

                # Add to usearch
                unit_idx.add(int(IsccID(item.id_data)), unit.body)

        return added

    def search_items(
        self,
        index_name: str,
        query: IsccItem,
        limit: int = 100
    ) -> list[dict]:
        """
        Search across all UNIT-TYPE indexes and aggregate results.
        """
        idx = self._get_or_load_index(index_name)

        query_item = IsccItem.from_dict(query)

        # Aggregate matches across all unit types
        matches = {}  # iscc_id -> {unit_type -> score}

        for unit_str in query_item.units:
            unit = IsccUnit(unit_str)
            unit_type = unit.unit_type

            unit_idx = idx["unit_indexes"].get(unit_type)
            if unit_idx is None:
                continue

            # Search usearch index
            results = unit_idx.search(unit.body, limit=limit)

            for iscc_id, distance in results:
                # Convert distance to score (lower distance = higher score)
                score = len(unit) * (1.0 - distance)

                if iscc_id not in matches:
                    matches[iscc_id] = {}

                # Max score per unit_type
                matches[iscc_id][unit_type] = max(
                    matches[iscc_id].get(unit_type, 0),
                    score
                )

        # Build result list
        results = []
        store = idx["store"]

        for iscc_id, unit_scores in matches.items():
            total_score = sum(unit_scores.values())
            entry = store.get(iscc_id)

            results.append({
                "iscc_id": entry["iscc_id"],
                "score": total_score,
                "matches": unit_scores,
                "entry": entry
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def close(self) -> None:
        """Close all open indexes and stores."""
        for idx_data in self._indexes.values():
            idx_data["store"].close()
            for unit_idx in idx_data["unit_indexes"].values():
                unit_idx.close()
        self._indexes.clear()
```

### Postgres Index (`indexes/postgres/index.py`)

```python
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from iscc_vdb.protocol import IsccIndexProtocol
from iscc_vdb.schema import IsccIndex, IsccItem

class PostgresIndex:
    """
    Unified ISCC index using Postgres + pgvector.

    Database schema:
    - indexes: Catalog of all indexes (name, created_at, etc.)
    - {index_name}_entries: Items with ISCC-IDs and metadata
    - {index_name}_units_{unit_type}: Per-UNIT-TYPE vectors with pgvector indexes
    """

    def __init__(self, connection_string: str):
        """
        Initialize PostgresIndex with connection pool.

        :param connection_string: Postgres DSN
        """
        self.pool = SimpleConnectionPool(1, 10, connection_string)
        self._ensure_catalog()

    def _ensure_catalog(self):
        """Create indexes catalog table if it doesn't exist."""
        with self.pool.getconn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS indexes (
                        name TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                conn.commit()
            self.pool.putconn(conn)

    def list_indexes(self) -> list[IsccIndex]:
        """Query indexes catalog and compute metadata."""
        with self.pool.getconn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM indexes ORDER BY name")
                indexes = []
                for (name,) in cur.fetchall():
                    # Get item count
                    cur.execute(f"SELECT COUNT(*) FROM {name}_entries")
                    items = cur.fetchone()[0]

                    # Get approximate size
                    cur.execute(f"""
                        SELECT pg_total_relation_size('{name}_entries')
                    """)
                    size_bytes = cur.fetchone()[0]
                    size_mb = size_bytes // (1024 * 1024)

                    indexes.append(IsccIndex(
                        name=name,
                        items=items,
                        size=size_mb
                    ))
            self.pool.putconn(conn)

        return indexes

    def create_index(self, index: IsccIndex) -> IsccIndex:
        """Create index tables."""
        with self.pool.getconn() as conn:
            with conn.cursor() as cur:
                # Check if exists
                cur.execute(
                    "SELECT 1 FROM indexes WHERE name = %s",
                    (index.name,)
                )
                if cur.fetchone():
                    raise FileExistsError(f"Index '{index.name}' already exists")

                # Create catalog entry
                cur.execute(
                    "INSERT INTO indexes (name) VALUES (%s)",
                    (index.name,)
                )

                # Create entries table
                cur.execute(f"""
                    CREATE TABLE {index.name}_entries (
                        iscc_id BIGINT PRIMARY KEY,
                        data JSONB NOT NULL
                    )
                """)

                conn.commit()
            self.pool.putconn(conn)

        return IsccIndex(name=index.name, items=0, size=0)

    def close(self) -> None:
        """Close connection pool."""
        self.pool.closeall()

    # ... similar implementations for get_index, delete_index, add_items, search_items
```

### Memory Index (`indexes/memory/index.py`)

```python
from iscc_vdb.protocol import IsccIndexProtocol
from iscc_vdb.schema import IsccIndex, IsccItem

class MemoryIndex:
    """
    In-memory index implementing IsccIndexProtocol.

    Stores all data in memory using dictionaries. No persistence.
    Useful for testing and development.
    """

    def __init__(self):
        """
        Initialize MemoryIndex.
        """
        self._indexes = {}  # name -> {items: [], metadata: {}}

    def list_indexes(self) -> list[IsccIndex]:
        """List all in-memory indexes."""
        indexes = []
        for name, data in self._indexes.items():
            indexes.append(IsccIndex(
                name=name,
                items=len(data["items"]),
                size=0  # Memory indexes don't track size
            ))
        return indexes

    def create_index(self, index: IsccIndex) -> IsccIndex:
        """Create new in-memory index."""
        if index.name in self._indexes:
            raise FileExistsError(f"Index '{index.name}' already exists")

        self._indexes[index.name] = {
            "items": [],
            "metadata": {}
        }
        return IsccIndex(name=index.name, items=0, size=0)

    def get_index(self, name: str) -> IsccIndex:
        """Get index metadata."""
        if name not in self._indexes:
            raise FileNotFoundError(f"Index '{name}' not found")

        data = self._indexes[name]
        return IsccIndex(
            name=name,
            items=len(data["items"]),
            size=0
        )

    def delete_index(self, name: str) -> None:
        """Delete in-memory index."""
        if name not in self._indexes:
            raise FileNotFoundError(f"Index '{name}' not found")

        del self._indexes[name]

    def add_items(self, index_name: str, items: list[IsccItem]) -> int:
        """Add items to in-memory index."""
        if index_name not in self._indexes:
            raise FileNotFoundError(f"Index '{index_name}' not found")

        self._indexes[index_name]["items"].extend(items)
        return len(items)

    def search_items(
        self,
        index_name: str,
        query: IsccItem,
        limit: int = 100
    ) -> list[dict]:
        """Search for similar items (simple exact match for testing)."""
        if index_name not in self._indexes:
            raise FileNotFoundError(f"Index '{index_name}' not found")

        # Simple implementation for testing
        results = []
        for item in self._indexes[index_name]["items"]:
            if item.iscc_code == query.iscc_code:
                results.append({
                    "iscc_id": item.iscc_id,
                    "score": 1.0,
                    "matches": {},
                    "entry": item.dict()
                })
        return results[:limit]

    def close(self) -> None:
        """No-op for in-memory index."""
        pass
```

### FastAPI Server (`server/app.py`)

```python
from fastapi import FastAPI, Request
from iscc_vdb.settings import get_index
from iscc_vdb.server import routes

def create_app() -> FastAPI:
    """
    Create FastAPI application.

    Index implementation is determined by ISCC_VDB_INDEXES_URI environment variable.
    """
    app = FastAPI(
        title="ISCC-VDB API",
        version="0.1.0",
        description="Scalable Nearest Neighbor Search Multi-Index for ISCC"
    )

    # Initialize index from settings
    app.state.index = get_index()

    # Include routes
    app.include_router(routes.router)

    @app.on_event("shutdown")
    def shutdown():
        app.state.index.close()

    return app

# For uvicorn
app = create_app()
```

### FastAPI Routes (`server/routes.py`)

```python
from fastapi import APIRouter, Request, HTTPException, status
from iscc_vdb.schema import IsccIndex, IsccItem
from iscc_vdb.protocol import IsccIndexProtocol

router = APIRouter()

def get_index_impl(request: Request) -> IsccIndexProtocol:
    """Get index implementation from app state."""
    return request.app.state.index

@router.get("/indexes", response_model=list[IsccIndex])
def list_indexes(request: Request):
    """List all indexes."""
    index = get_index_impl(request)
    return index.list_indexes()

@router.post(
    "/indexes",
    response_model=IsccIndex,
    status_code=status.HTTP_201_CREATED
)
def create_index(index: IsccIndex, request: Request):
    """Create a new index."""
    idx = get_index_impl(request)
    try:
        return idx.create_index(index)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except FileExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )

@router.get("/indexes/{name}", response_model=IsccIndex)
def get_index(name: str, request: Request):
    """Get index metadata."""
    idx = get_index_impl(request)
    try:
        return idx.get_index(name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index '{name}' not found"
        )

@router.delete("/indexes/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_index(name: str, request: Request):
    """Delete an index."""
    idx = get_index_impl(request)
    try:
        idx.delete_index(name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index '{name}' not found"
        )

@router.post("/indexes/{name}/items")
def add_items(name: str, items: list[IsccItem], request: Request):
    """Add items to index."""
    idx = get_index_impl(request)
    try:
        added = idx.add_items(name, items)
        return {"added": added}
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index '{name}' not found"
        )

@router.post("/indexes/{name}/search")
def search_items(
    name: str,
    query: IsccItem,
    limit: int = 100,
    request: Request = None
):
    """Search for similar items."""
    idx = get_index_impl(request)
    try:
        return idx.search_items(name, query, limit)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index '{name}' not found"
        )
```

### CLI Application (`cli/app.py`)

```python
import typer
from pathlib import Path
from iscc_vdb.settings import get_index
from iscc_vdb.schema import IsccIndex, IsccItem

app = typer.Typer(name="iscc-vdb", help="ISCC Vector Database CLI")

@app.command()
def list():
    """List all indexes."""
    index = get_index()
    indexes = index.list_indexes()

    if not indexes:
        typer.echo("No indexes found")
        return

    for idx in indexes:
        typer.echo(f"{idx.name}: {idx.items} items, {idx.size} MB")

    index.close()

@app.command()
def create(name: str):
    """Create a new index."""
    index = get_index()

    try:
        result = index.create_index(IsccIndex(name=name))
        typer.echo(f"Created index: {result.name}")
    except FileExistsError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    finally:
        index.close()

@app.command()
def delete(name: str):
    """Delete an index."""
    index = get_index()

    # Confirm deletion
    confirm = typer.confirm(f"Delete index '{name}' and all its data?")
    if not confirm:
        typer.echo("Cancelled")
        index.close()
        return

    try:
        index.delete_index(name)
        typer.echo(f"Deleted index: {name}")
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    finally:
        index.close()

@app.command()
def add(index_name: str, directory: Path):
    """Add items from directory to index."""
    index = get_index()

    # Scan for *.iscc.json files
    json_files = list(directory.rglob("*.iscc.json"))
    typer.echo(f"Found {len(json_files)} *.iscc.json files")

    if not json_files:
        typer.echo("No files to add")
        index.close()
        return

    # Load items
    items = []
    from iscc_vdb.cli.utils import load_iscc_items
    for json_file in json_files:
        try:
            item = load_iscc_items(json_file)
            items.append(item)
        except Exception as e:
            typer.echo(f"Error loading {json_file}: {e}", err=True)

    # Add to index
    try:
        added = index.add_items(index_name, items)
        typer.echo(f"Added {added} items to {index_name}")
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    finally:
        index.close()

@app.command()
def search(
    index_name: str,
    iscc_code: str,
    limit: int = typer.Option(100, "--limit", "-n")
):
    """Search index for similar items."""
    index = get_index()

    try:
        query = IsccItem(iscc_code=iscc_code)
        results = index.search_items(index_name, query, limit)

        import json
        typer.echo(json.dumps(results, indent=2))
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    finally:
        index.close()

if __name__ == "__main__":
    app()
```

## Usage Examples

### Local Index (Usearch)

```bash
# Set ISCC_VDB_INDEXES_URI to local path (or use default from platformdirs)
export ISCC_VDB_INDEXES_URI=/path/to/index_data

# CLI commands
iscc-vdb create myindex
iscc-vdb add myindex /data/
iscc-vdb search myindex "ISCC:..."
iscc-vdb list
iscc-vdb delete myindex

# Start API server with local usearch index
uvicorn iscc_vdb.server.app:app --host 0.0.0.0 --port 8000
```

### Postgres Index

```bash
# Set ISCC_VDB_INDEXES_URI to Postgres connection string
export ISCC_VDB_INDEXES_URI=postgresql://user:password@localhost/isccdb

# CLI works the same
iscc-vdb create myindex
iscc-vdb add myindex /data/
iscc-vdb search myindex "ISCC:..."

# Start API server with Postgres index
uvicorn iscc_vdb.server.app:app --host 0.0.0.0 --port 8000
```

### Memory Index (Testing)

```bash
# Set ISCC_VDB_INDEXES_URI to memory:// for in-memory index
export ISCC_VDB_INDEXES_URI=memory://

# CLI works with in-memory storage (no persistence)
iscc-vdb create myindex
iscc-vdb add myindex /data/
iscc-vdb search myindex "ISCC:..."

# Start API server with in-memory index (for testing)
uvicorn iscc_vdb.server.app:app --host 0.0.0.0 --port 8000
```

### Mixed Deployment

```bash
# Server A: API with Postgres index (shared database)
# ISCC_VDB_INDEXES_URI=postgresql://shared-db/iscc
uvicorn iscc_vdb.server.app:app --host 0.0.0.0 --port 8000

# Server B: API with same Postgres index (horizontal scaling)
# ISCC_VDB_INDEXES_URI=postgresql://shared-db/iscc
uvicorn iscc_vdb.server.app:app --host 0.0.0.0 --port 8001

# Developer workstation: CLI with local usearch index
# ISCC_VDB_INDEXES_URI=/home/user/iscc-indexes
iscc-vdb add myindex /data/

# Testing environment: CLI with in-memory index
# ISCC_VDB_INDEXES_URI=memory://
iscc-vdb search test "ISCC:..."
```

## Key Benefits

1. **Clean Abstraction**: Protocol-based design enables index swapping without changing frontend code
2. **Simple Configuration**: Single `ISCC_VDB_INDEXES_URI` variable determines entire deployment topology
3. **Modular Packages**: Each index implementation is self-contained and independently testable
4. **Synchronous Simplicity**: No async complexity, straightforward implementation
5. **Zero Code Duplication**: CLI and API share identical index implementations
6. **Type Safety**: Protocol + Pydantic ensures compile-time and runtime validation
7. **Flexible Deployment**: Local, postgres, in-memory, or hybrid configurations
8. **Easy Testing**: Mock the protocol for unit tests, use different indexes for integration tests (especially
    MemoryIndex)
9. **Future-Proof**: Add new index implementations without modifying existing code
10. **Developer Friendly**: Clear separation of concerns, easy to understand and extend

## Implementation Strategy

### Phase 1: Foundation

1. Define `IsccIndexProtocol` in `protocol.py`
2. Create `settings.py` with Pydantic settings and index factory
3. Update OpenAPI spec with item endpoints
4. Regenerate `schema.py` from OpenAPI

### Phase 2: Usearch Index

1. Create `indexes/usearch/` package structure
2. Move/refactor existing `store.py`, `lookup.py`, `unit.py` into index package
3. Implement `UsearchIndex` with protocol methods
4. Write unit tests for index

### Phase 3: Server Package

1. Create `server/` package with FastAPI app
2. Implement route handlers using protocol
3. Add error handling and validation
4. Write integration tests

### Phase 4: CLI Package

1. Create `cli/` package with Typer app
2. Implement commands using protocol
3. Add utilities for file loading, formatting
4. Write CLI tests

### Phase 5: Additional Indexes

1. Implement `PostgresIndex` in `indexes/postgres/`
2. Implement `MemoryIndex` in `indexes/memory/`
3. Write index-specific tests
4. Update documentation

### Phase 6: Polish

1. Add comprehensive documentation
2. Create deployment guides for each index implementation
3. Add performance benchmarks
4. Optimize based on profiling

## Testing Strategy

### Unit Tests

- Mock `IsccIndexProtocol` for frontend tests (CLI, API routes)
- Test each index implementation in isolation with its own storage
- Test protocol conformance for each index implementation

### Integration Tests

- Test full stack with UsearchIndex (file-based, no external dependencies)
- Test full stack with PostgresIndex (requires test database)
- Test full stack with MemoryIndex (fast, in-memory, ideal for testing)
- Test CLI → API → Index workflows

### End-to-End Tests

- Test deployment scenarios (local, postgres, in-memory)
- Test mixed configurations (different indexes for different environments)
- Performance tests with real datasets

## Future Extensions

- **Redis Index**: Distributed in-memory index for hot data
- **S3 Index**: Cloud storage for massive archives
- **Hybrid Index**: Combine multiple index implementations with routing logic
- **HTTP Client Index**: Remote API client for distributed deployments
- **Async Support**: Add async variants of protocol methods
- **Batch Operations**: Optimize bulk add/search operations
- **Index Replication**: Sync indexes across implementations
- **Access Control**: Add authentication/authorization layer
