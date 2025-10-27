# ISCC-Search Spec - Draft

A Scalable Nearest Neighbor Search Multi-Index for the International Standard Content Code (ISCC)

## Overview

### ISCC Specific Terminology

- **ISCC** - Any ISCC-CODE, ISCC-UNIT, or ISCC-ID
- **ISCC-HEADER** - Self-describing 2-byte header for V1 components (3 bytes for future versions). The first 12
    bits encode MainType, SubType, and Version. Additional bits encode Length for variable-length ISCCs.
- **ISCC-BODY** - Binary payload of an ISCC, similarity preserving compact binary code, hash or timestamp
    without HEADER
- **ISCC-DIGEST** - Binary representation of complete ISCC (ISCC-HEADER + ISCC-BODY).
- **ISCC-UNIT** - ISCC-HEADER + ISCC-BODY where the ISCC-BODY is calculated from a single algorithm
- **ISCC-CODE** - ISCC-HEADER + ISCC-BODY where the ISCC-BODY is a sequence of multiple ISCC-UNIT BODYs
    - DATA and INSTANCE are the minimum required mandatory ISCC-UNITS for a valid ISCC-CODE
- **ISCC-ID** - Globally unique digital asset identifier (ISCC-HEADER + 52-bit timestamp + 12-bit server-id)
- **SIMPRINT** - Headerless base64 encoded similarity hash that describes a content segment (granular feature)
- **ISCC-UNIT-TYPE**: Identifier for UNIT-TYPES that can be indexed together with meaningful similarity search
    - All ISCCs of the same type are stored in the same index regardless of length
    - The type is identified by the composite of MainType, SubType, Version
    - The type is encoded in the first 12 bits of the ISCC-HEADER
    - String representation example: CONTENT-TEXT-V0 (identified by the first 12 bits of an ISCC-UNIT)
    - Note: ISCC-UNIT-TYPE excludes the length segment from the header

### ISCC Framework & Resources

The ISCC Framework consist of a collection of python libraries and applications published on GitHub:

- iscc/iscc-core - official python reference implementation of standardized low level codec and fingerprinting
    algorithms
- iscc/iscc-sdk - higher level content, detection, metadata extraction/embedding, content
    extraction/transformation, iscc code generation
- iscc/iscc-web - Rest api service for generating ISCC codes from media assets
- iscc/iscc-schema - ISCC metadata JSON schema and JSON-LD contexts
- iscc/iscc-crypto - cryptographic primitives for signing and timestamping iscc codes
- iscc/iscc-sct - library for generating semantic code text and granular semantic features
- iscc/iscc-sci - library for generating semantic code image
- iscc/iscc-ieps - Community driven specifications for the ISCC ecosystem

**Helpful Note**: These repositories are all available on deepwiki

## ISCC-Search Features

- ISCC-CODEs or extended ISCC-UNITs as bit-vectors for fast similarity search
- Distance Metric: Normalized Prefix Hamming Distance (NPHD)
- Highlevel API for indexing ISCCs
- ISCC-IDs are stored as 64-bit keys with ISCC-HEADER at index level for reconstruction. ISCC-ID strings include
    a 2-byte header + 64-bit body (52-bit timestamp + 12-bit server-id). The store uses the 64-bit body as the
    LMDB key; the header is reconstructed using the configured realm_id (0 or 1).
- Indexes ISCC-CODEs or lists of ISCC-UNITs in a Multi-Index (one per ISCC-UNIT-TYPE)

### Supported ISCC-UNITS

Overview of the different ISCC-UNITs that can be indexed for similarity search:

- **META** ISCC-UNIT encodes syntactic/lexical **metadata** similarity
- **SEMANTIC** ISCC-UNITs encode semantic/conceptual **content** similarity
    - SubTypes: TEXT, IMAGE
- **CONTENT** ISCC-UNITs encode perceptual/syntactic/structural **content** similarity
    - SubTypes: TEXT, IMAGE, AUDIO, VIDEO, MIXED
- **DATA** ISCC-UNIT encode raw **data** similarity

ISCC-UNIT for exact (prefix) match indexing:

- **INSTANCE** ISCC-UNIT identifies **data** like a checksum or cryptographic hash (depending on length)

## Interfaces

- python library
- command line tool
- REST API service

## Data Model

Index Entry:

- iscc_id - Unique ID for digital asset
- iscc_code - ISCC-CODE (composite of short ISCC-UNITs)
- units - List of exanded ISCC-UNITS
- features - Granular simprints (fingerprints) for digital asset content

## Architecture

### Storage Structure

```
{path}/
  primary.lmdb/          # IsccStore - primary entry storage (source of truth)
  instance/              # InstanceIndex - exact/prefix matching
  content-text-v0/       # UnitIndex - similarity search
  semantic-text-v0/      # UnitIndex - similarity search
  meta-none-v0/          # UnitIndex - similarity search
  ...
```

### IsccStore Class

**Module:** `iscc_search/store.py`

The `IsccStore` class provides durable LMDB-backed storage for ISCC entries and metadata. It serves as the
primary source of truth, with UnitIndex and InstanceIndex acting as derived indexes for search.

**LMDB Schema:**

- **Environment**: max_dbs=2, configurable map_size (auto-expands when full)
- **Entries database** (name: b'entries', integerkey=False):
    - Keys: 8-byte big-endian representation of 64-bit ISCC-IDs
    - Values: JSON-serialized dict with `{"iscc_id": str, "iscc_code": str, "units": list[str]}`
    - Entries may include other digital asset metadata that we store and retrieve transparently
- **Metadata database** (name: b'metadata'):
    - Keys: UTF-8 encoded strings (`__realm_id__`)
    - Values: JSON-serialized values
- **Custom metadata**: Users can optionally store additional key-value pairs in metadata database

**Time Ordering:**

ISCC-ID keys are stored as big-endian bytes. LMDB's lexicographic sorting of big-endian byte sequences preserves
chronological order (ISCC-IDs contain timestamps). Iteration yields entries in ascending timestamp order.

**Automatic Storage Management:**

The store automatically doubles `map_size` when LMDB reports the database is full, ensuring writes never fail
due to capacity limits.

**Key Features:**

- Atomic writes via LMDB transactions
- Efficient single-lookup retrieval by ISCC-ID
- Persistent storage of realm_id metadata
- Support for index rebuild from primary storage
- Optional extended metadata storage
- Automatic map_size expansion on capacity limits
- Batch operation support for add operations
- Flexible LMDB configuration via options dict

**LMDB Configuration:**

`IsccStore` uses durability-focused LMDB settings because it is the primary source of truth:

- `metasync=True`: Full durability with metadata flush to disk
- `sync=True`: Full ACID compliance with fsync on commit
- `writemap=False`: Safer, prevents corruption from bad writes

This differs from derived indexes like `InstanceIndex` and `UnitIndex`, which prioritize performance
(`metasync=False`, `sync=False`, `writemap=True`) since they can be rebuilt from `IsccStore` if corrupted.

Users can override these defaults via the `lmdb_options` parameter, but the durability settings are recommended
for production use where data integrity is critical.

**Constructor:**

```python
IsccStore(path, realm_id=0, lmdb_options=None)
```

**Parameters:**

- `path` (str | os.PathLike): Directory path for LMDB storage
- `realm_id` (int): ISCC realm ID for ISCC-ID reconstruction. Must be 0 or 1 for ID realms (default: 0)
- `lmdb_options` (dict | None): Optional LMDB configuration dict merged with defaults (default: None)

**Methods:**

```python
store.add(iscc_ids, entries) -> int
store.get(iscc_id) -> dict | None
store.delete(iscc_id) -> bool
store.iter_entries() -> Iterator[tuple[int, dict]]
store.get_metadata(key: str) -> Any
store.put_metadata(key: str, value: Any) -> None
store.set_mapsize(new_size: int) -> None
store.close() -> None
```

**Properties:**

```python
store.map_size -> int
```

**Method Details:**

### add() - Store ISCC entries

```python
add(iscc_ids, entries) -> int
```

**Parameters:**

- `iscc_ids` (int | str | list[int | str]): ISCC-ID(s) as integer(s) or string(s)
- `entries` (dict | list[dict]): Entry dict(s) to store

**Behavior:**

- Single values normalized to lists internally
- Number of ISCC-IDs must match number of entries
- Accepts ISCC-IDs as integers or strings (e.g., "ISCC:...")
- Automatically doubles map_size if database is full and retries

**Returns:** Number of entries successfully added

**Raises:**

- `ValueError`: If number of ISCC-IDs doesn't match number of entries

### get() - Retrieve entry by ISCC-ID

```python
get(iscc_id) -> dict | None
```

**Parameters:**

- `iscc_id` (int | str): ISCC-ID as integer or string

**Returns:** Entry dict or None if not found

### delete() - Remove entry by ISCC-ID

```python
delete(iscc_id) -> bool
```

**Parameters:**

- `iscc_id` (int | str): ISCC-ID as integer or string

**Returns:** True if deleted, False if not found

### iter_entries() - Iterate all entries

```python
iter_entries() -> Iterator[tuple[int, dict]]
```

**Returns:** Iterator yielding (iscc_id_int, entry_dict) tuples in ascending timestamp order

### set_mapsize() - Increase maximum database size

```python
set_mapsize(new_size) -> None
```

**Parameters:**

- `new_size` (int): New maximum size in bytes (must be larger than current)

**Raises:**

- `lmdb.Error`: If active transactions exist
- `ValueError`: If new_size would shrink database

## High-Level Core API (PYTHON, CLI, REST)

### IsccIndex Class

The IsccIndex manages multiple internal components:

- **IsccStore**: Primary LMDB storage (source of truth)
- **UnitIndex** instances: One per ISCC-UNIT-TYPE (META-NONE-V0, CONTENT-TEXT-V0, etc.)
- **InstanceIndex**: Exact/prefix matching for INSTANCE units. `add()` stores exact Instance-Code digests;
    `search()` supports prefix matching (and bidirectional expansion) over stored digests.

### Constructor

```python
IsccIndex(path=None, realm_id=0, max_dim=256, **kwargs)
```

**Parameters:**

- `path` (str | os.PathLike | None): Directory path for index storage (optional, creates in-memory index if
    None)
- `realm_id` (int): ISCC realm ID for ISCC-ID reconstruction. Must be 0 or 1 for ID realms. UnitIndex may infer
    realm from IDs (default: 0)
- `max_dim` (int): Maximum vector dimension in bits for UNIT indexes (default: 256)
- `**kwargs`: Additional arguments passed to underlying UnitIndex instances

**Behavior:**

- When `path` is provided: Creates durable index with full LMDB persistence
- When `path=None`: Creates non-persistent index. Implementations may back UnitIndex/InstanceIndex with
    temporary directories. `close()` may clean up such resources.
- In-memory mode is ideal for testing scenarios where durability is not required

**Returns:** IsccIndex instance

**Examples:**

```python
# Durable production index
idx = IsccIndex(path="./my_index")

# In-memory testing index (auto-cleanup on close)
idx = IsccIndex()
# ... use for tests ...
idx.close()  # Temp files automatically removed
```

### add() - Add ISCC entries to index

```python
add(entries) -> list[str]
```

**Parameters:**

- `entries` (dict | list[dict]): Single entry dict or list of entry dicts with optional fields:
    - `iscc_id` (str): ISCC-ID string (optional, auto-generated if omitted)
    - `iscc_code` (str): ISCC-CODE string to decompose and index
    - `units` (list[str]): Pre-decomposed ISCC-UNIT strings

**Behavior:**

- Single dict input normalized to list internally
- Each entry must provide at least one of `iscc_code` or `units`
- If `units` is present and not empty, `iscc_code` is ignored. If `units` is empty and `iscc_code` provided,
    decomposes. Otherwise raises ValueError.
- If `iscc_id` is omitted, auto-generates ISCC-IDs from current timestamp (52 bits) + server-id (12 bits,
    default 0). Preserves chronological ordering and uniqueness.
- Rejects duplicate `iscc_id` and returns the existing ID
- Decomposes ISCC-CODEs into units automatically (when `units` not provided)
- Writes entry to IsccStore first (atomic, source of truth)
- Routes each unit to appropriate UnitIndex based on ISCC-UNIT-TYPE
- Stores INSTANCE units in InstanceIndex for exact/prefix matching
- All units from one entry must share a consistent realm. Mixing different unit types is allowed (each goes to
    its own UnitIndex).
- All units for same entry share the same ISCC-ID
- Symmetric with `get()`: can add what `get()` returns

**Returns:** List of ISCC-ID strings (one per entry added)

**Raises:**

- `ValueError`: If entry has neither `iscc_code` nor `units`
- `ValueError`: If entry has `units` as empty list and no `iscc_code`
- `ValueError`: If units from same entry have inconsistent realms

**Examples:**

```python
# Single entry, auto-generate ISCC-ID
idx.add({"iscc_code": "ISCC:KACYPXW445FTYNJ3CYSXHAFJMA2HUWULUNRFE3BLHRSCXYH2M5AEGQY"})
# Returns: ["ISCC:IAACBFKZG52UU"]

# Batch with explicit ISCC-IDs
idx.add([
    {"iscc_id": "ISCC:IAACBFKZG52UU", "iscc_code": "ISCC:KAC..."},
    {"iscc_id": "ISCC:IAACBFKZG52UV", "iscc_code": "ISCC:KEC..."}
])

# Pre-decomposed units (batch)
idx.add([
    {"units": ["ISCC:GAA...", "ISCC:EAA...", "ISCC:IAA..."]},
    {"units": ["ISCC:GAB...", "ISCC:EAB...", "ISCC:IAB..."]}
])

# Copy from get() result
entry = idx.get("ISCC:IAACBFKZG52UU")
idx2.add(entry)  # Add to another index
```

### get() - Retrieve entries by ISCC-ID

```python
get(iscc_ids) -> list[dict] | dict | None
```

**Parameters:**

- `iscc_ids` (str | list[str]): ISCC-ID string(s) to lookup

**Behavior:**

- Single ISCC-ID: returns dict or None
- Multiple ISCC-IDs: returns list in input order with None placeholders for missing ids
- Retrieves entry directly from IsccStore (single lookup, fast)
- Always returns all fields (`iscc_id`, `iscc_code`, `units`); uses None for missing fields to preserve
    round-trip `add(get(x))`

**Returns:**

- Single query: `dict | None` with keys: `iscc_id`, `iscc_code`, `units` (None values for missing fields)
- Multiple queries: `list[dict | None]` in input order

**Examples:**

```python
# Single lookup
result = idx.get("ISCC:IAACBFKZG52UU")
# Returns: {
#     "iscc_id": "ISCC:IAACBFKZG52UU",
#     "iscc_code": "ISCC:KACYPXW445FTYNJ3CYSXHAFJMA2HUWULUNRFE3BLHRSCXYH2M5AEGQY",
#     "units": ["ISCC:GAA...", "ISCC:EAA...", "ISCC:IAA..."],
# }

# Batch lookup
results = idx.get(["ISCC:IAACBFKZG52UU", "ISCC:IAACBFKZG52UV"])
# Returns: [
#     {"iscc_id": "ISCC:IAACBFKZG52UU", "iscc_code": "...", "units": [...]},
#     {"iscc_id": "ISCC:IAACBFKZG52UV", "iscc_code": "...", "units": [...]},
# ]
```

### search() - Find similar entries

```python
search(iscc_codes=None, units=None, count=10, exact=False) -> IsccMatches | IsccBatchMatches
```

**Parameters:**

- `iscc_codes` (str | list[str] | None): ISCC-CODE string(s) to decompose and search
- `units` (list[str] | list\[list[str]\] | None): Pre-decomposed extended length ISCC-UNIT string(s) to search
- `count` (int): Maximum number of results per query (default: 10)
- `exact` (bool): Use exhaustive search instead of approximate (default: False)

**Behavior:**

- Either `iscc_codes` or `units` must be provided
- If `units` is provided, `iscc_codes` is ignored
- Decomposes ISCC-CODEs into units automatically
- Searches each unit in corresponding UnitIndex
- Searches INSTANCE units in InstanceIndex for exact matches
- Aggregates results across all unit types
- Results ranked by combined distance (weighted average across unit types)

**Returns:**

- Single query: `IsccMatches` object
- Multiple queries: `IsccBatchMatches` object

**IsccMatches attributes:**

- `keys` (ndarray[str]): Array of ISCC-ID strings
- `distances` (ndarray[float]): Array of combined distances
- `unit_matches` (dict): Per-unit-type UnitMatches objects
- `instance_matches` (list[str]): Exact instance code matches
- `visited_members` (int): Total graph nodes visited
- `computed_distances` (int): Total distance computations

**IsccMatches methods:**

- `__len__()`: Number of matches
- `__getitem__(index)`: Get IsccMatch object at index
- `to_list()`: Convert to list of (key, distance) tuples

**IsccBatchMatches attributes:**

- `keys` (ndarray[str, str]): 2D array of ISCC-ID strings (n_queries, k)
- `distances` (ndarray[float, float]): 2D array of distances
- `counts` (ndarray[int]): Number of valid results per query
- `unit_matches` (dict): Per-unit-type UnitBatchMatches objects
- `instance_matches` (list\[list[str]\]): Exact instance matches per query
- `visited_members` (int): Total graph nodes visited
- `computed_distances` (int): Total distance computations

**IsccBatchMatches methods:**

- `__len__()`: Number of queries
- `__getitem__(index)`: Get IsccMatches for query at index
- `to_list()`: Convert to list of lists of tuples
- `mean_recall(expected, count=None)`: Measure recall against expected results

**Examples:**

```python
# Single search
matches = idx.search(iscc_codes="ISCC:KACYPXW445FTYNJ...", count=5)
print(matches.keys)  # ['ISCC:IAACBFKZG52UU', ...]
print(matches.distances)  # [0.0, 0.15, 0.23, ...]
print(matches.instance_matches)  # ['ISCC:IAA...']

# Batch search
batch = idx.search(
    iscc_codes=["ISCC:KAC...", "ISCC:KEC..."],
    count=10
)
print(batch.keys.shape)  # (2, 10)
print(batch[0].keys)  # First query results

# Pre-decomposed units search
matches = idx.search(
    units=["ISCC:GAA...", "ISCC:EAA...", "ISCC:IAA..."],
    count=3,
    exact=True
)
```

### remove() - Remove entries by ISCC-ID

```python
remove(iscc_ids) -> int
```

**Parameters:**

- `iscc_ids` (str | list[str]): ISCC-ID string(s) to remove

**Behavior:**

- Removes entries from IsccStore
- Removes entries from all UnitIndex instances
- Removes entries from InstanceIndex
- Single string input normalized to list internally

**Returns:** Total number of mappings removed across all indexes

**Examples:**

```python
# Remove single entry
count = idx.remove("ISCC:IAACBFKZG52UU")
print(count)  # 3 (removed from META, CONTENT, INSTANCE indexes)

# Remove batch
count = idx.remove(["ISCC:IAACBFKZG52UU", "ISCC:IAACBFKZG52UV"])
```

### rebuild() - Rebuild derived indexes from primary storage

```python
rebuild() -> None
```

**Behavior:**

- Clears all UnitIndex instances
- Clears InstanceIndex
- Iterates through all entries in IsccStore
- Re-indexes all units and instance codes from primary storage
- Useful for index corruption recovery, parameter changes, or format upgrades

**Returns:** None

**Examples:**

```python
# Rebuild after corruption or to apply new parameters
idx = IsccIndex.restore(path)
idx.rebuild()
idx.save()
```

### save() - Persist index to disk

```python
save() -> None
```

**Behavior:**

- IsccStore persists automatically (LMDB transactions)
- Saves all UnitIndex instances to separate files
- InstanceIndex persists automatically (LMDB)

**Returns:** None

### load() - Load index from disk

```python
load() -> None
```

**Behavior:**

- Opens IsccStore (restores metadata: realm_id, max_dim)
- Loads all UnitIndex instances from files
- Opens InstanceIndex LMDB environment

**Returns:** None

### view() - Memory-map index from disk

```python
view() -> None
```

**Behavior:**

- Opens IsccStore (restores metadata: realm_id, max_dim)
- Memory-maps UnitIndex instances (read-only)
- Opens InstanceIndex LMDB environment (read-only)

**Returns:** None

### copy() - Create a copy of the index

```python
copy() -> IsccIndex
```

**Returns:** New IsccIndex with same configuration and data

### close() - Close index and release resources

```python
close() -> None
```

**Behavior:**

- Closes IsccStore LMDB environment
- Closes all UnitIndex instances
- Closes InstanceIndex LMDB environment

**Returns:** None

### Static Methods

#### restore() - Restore index from saved files

```python
IsccIndex.restore(path, view=False, **kwargs) -> IsccIndex | None
```

**Parameters:**

- `path` (str | os.PathLike): Directory path to restore from
- `view` (bool): If True, memory-map instead of loading (default: False)
- `**kwargs`: Additional arguments passed to IsccIndex constructor

**Returns:** Restored IsccIndex or None if path is invalid

### Properties

```python
@property
def size() -> int
```

**Returns:** Total number of unique ISCC-IDs in index

```python
@property
def unit_types() -> list[str]
```

**Returns:** List of indexed ISCC-UNIT-TYPEs (e.g., ["META-NONE-V0", "CONTENT-TEXT-V0"])

## CLI Interface

All Python API methods exposed as CLI commands:

```bash
# Add single entry with auto-generated ISCC-ID
iscc-search add --iscc-code "ISCC:KAC..."

# Add entry with explicit ISCC-ID
iscc-search add --iscc-id "ISCC:IAA..." --iscc-code "ISCC:KAC..."

# Add entry with pre-decomposed units
iscc-search add --units "ISCC:GAA..." "ISCC:EAA..." "ISCC:IAA..."

# Add from JSON file (batch)
iscc-search add --file entries.json

# Get entry
iscc-search get "ISCC:IAACBFKZG52UU"

# Search
iscc-search search --iscc-code "ISCC:KAC..." --count 10
iscc-search search --units "ISCC:GAA..." "ISCC:EAA..." --count 5 --exact

# Remove
iscc-search remove "ISCC:IAACBFKZG52UU"

# Index management
iscc-search save
iscc-search rebuild  # Rebuild indexes from primary storage
iscc-search info  # Show index stats (size, unit_types, etc.)
```

## REST API Interface

RESTful endpoints mapping to Python API:

```
POST   /add        - Add entries (body: {entries: dict | list[dict]})
GET    /get/{id}   - Get single entry
POST   /get        - Get multiple entries (body: {iscc_ids: list[str]})
POST   /search     - Search index (body: {iscc_codes, units, count, exact})
DELETE /remove/{id} - Remove single entry
POST   /remove     - Remove multiple entries (body: {iscc_ids: list[str]})
POST   /rebuild    - Rebuild indexes from primary storage
GET    /info       - Get index info (size, unit_types, etc.)
POST   /save       - Persist index
```

**Response Format:**

```json
{
  "status": "success",
  "data": { ... },
  "error": null
}
```

**POST /add Examples:**

```json
// Single entry
{
  "entries": {
    "iscc_code": "ISCC:KACYPXW445FTYNJ3CYSXHAFJMA2HUWULUNRFE3BLHRSCXYH2M5AEGQY"
  }
}

// Batch with mixed formats
{
  "entries": [
    {"iscc_id": "ISCC:IAA...", "iscc_code": "ISCC:KAC..."},
    {"units": ["ISCC:GAA...", "ISCC:EAA...", "ISCC:IAA..."]}
  ]
}
```
