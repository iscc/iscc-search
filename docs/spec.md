# ISCC-VDB Spec

A Nearest Neighbor Search Index for ISCCs

## Overview

### Terms and Definitions

- **ISCC** - Any ISCC-CODE, ISCC-UNIT, or ISCC-ID
- **ISCC-HEADER** - Self describing 3-byte header section of all ISCCs designating MainType, SubType, Version,
    Length
- **ISCC-BODY** - Actual payload of an ISCC, similarity preserving compact binary code, hash or timestamp
- **ISCC-UNIT** - ISCC-HEADER + ISCC-BODY where the ISCC-BODY is calculated from a single algorithm
- **ISCC-CODE** - ISCC-HEADER + ISCC-BODY where the ISCC-BODY is a sequence of multiple ISCC-UNIT BODYs
    - DATA and INSTANCE are the minimum required mandatory ISCC-UNITS for a valid ISCC-CODE
- **ISCC-ID** - Globally unique digital asset idendifier (ISCC-HEADER + 52-bit timestamp + 12-bit server-id)
- **SIMPRINT** - Headerless base64 encoded similarity hash that describes a content segment (granular feature)
- ISCC-UNIT-TYPE: Identifier for UNIT-TYPES that can be indexed together with meaningful similarity search
    - All ISCCs of the same type are stored in the same index regardless of length
    - The type is identified by the composite of MainType, SubType, Version
    - The typs is encoded in the first 12 bits of the ISCC-HEADER
    - String representation example: CONTENT-TEXT-V0 (identified by the first 12 bits of an ISCC-UNIT)

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

## ISCC-VDB Features

- ISCC-CODEs or extended ISCC-UNITs as bit-vectors for fast similarity search
- Distance Metric: Normalized Prefix Hamming Distance (NPHD)
- Highlevel API for indexing ISCCs
- ISCC-IDs as stored as 64-bit keys with ISCC-HEADER at index level for reconstruction
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

- iscc_id - Unique Digital (Optional)
- iscc_code - ISCC-CODE
- units - List of ISCC-UNITS
- features

## High-Level Core API (PYTHON, CLI, REST)

### IsccIndex Class

The IsccIndex manages multiple internal indexes:

- One UnitIndex per ISCC-UNIT-TYPE (META-NONE-V0, CONTENT-TEXT-V0, etc.)
- One InstanceIndex for exact/prefix matching

### Constructor

```python
IsccIndex(path, realm_id=0, max_dim=256, **kwargs)
```

**Parameters:**

- `path` (str | os.PathLike): Directory path for index storage
- `realm_id` (int): ISCC realm ID (0-1) for ISCC-ID reconstruction (default: 0)
- `max_dim` (int): Maximum vector dimension in bits for UNIT indexes (default: 256)
- `**kwargs`: Additional arguments passed to underlying UnitIndex instances

**Returns:** IsccIndex instance

### add() - Add ISCC entries to index

```python
add(iscc_ids=None, iscc_codes=None, units=None) -> list[str]
```

**Parameters:**

- `iscc_ids` (str | list[str] | None): ISCC-ID string(s) or None for auto-generation
- `iscc_codes` (str | list[str] | None): ISCC-CODE string(s) to decompose and index
- `units` (list[str] | list\[list[str]\] | None): Pre-decomposed ISCC-UNIT string(s)

**Behavior:**

- Either `iscc_codes` or `units` must be provided (not both)
- If `units` is provided, `iscc_codes` is ignored
- If `iscc_ids` is None, auto-generates sequential ISCC-IDs starting from 0
- Single string inputs are normalized to lists internally
- Decomposes ISCC-CODEs into units automatically
- Routes each unit to appropriate UnitIndex based on ISCC-UNIT-TYPE
- Stores INSTANCE units in InstanceIndex for exact matching
- All units for same entry share the same ISCC-ID

**Returns:** List of ISCC-ID strings (one per entry added)

**Raises:**

- `ValueError`: If neither iscc_codes nor units provided
- `ValueError`: If number of iscc_ids doesn't match number of entries

**Examples:**

```python
# Auto-generate ISCC-IDs
idx.add(iscc_codes="ISCC:KACYPXW445FTYNJ3CYSXHAFJMA2HUWULUNRFE3BLHRSCXYH2M5AEGQY")
# Returns: ["ISCC:IAACBFKZG52UU"]

# Explicit ISCC-IDs
idx.add(
    iscc_ids=["ISCC:IAACBFKZG52UU", "ISCC:IAACBFKZG52UV"],
    iscc_codes=["ISCC:KAC...", "ISCC:KEC..."]
)

# Pre-decomposed units (batch)
idx.add(
    units=[
        ["ISCC:GAA...", "ISCC:EAA...", "ISCC:IAA..."],
        ["ISCC:GAB...", "ISCC:EAB...", "ISCC:IAB..."]
    ]
)
```

### get() - Retrieve entries by ISCC-ID

```python
get(iscc_ids) -> list[dict] | dict | None
```

**Parameters:**

- `iscc_ids` (str | list[str]): ISCC-ID string(s) to lookup

**Behavior:**

- Single ISCC-ID: returns dict or None
- Multiple ISCC-IDs: returns list of dicts (with None for missing)
- Retrieves units from all UnitIndex instances
- Retrieves instance codes from InstanceIndex

**Returns:**

- Single query: `dict | None` with keys: `iscc_id`, `iscc_code`, `units`
- Multiple queries: `list[dict | None]`

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

### save() - Persist index to disk

```python
save() -> None
```

**Behavior:**

- Saves all UnitIndex instances to separate files
- Saves InstanceIndex LMDB environment
- Saves index metadata (realm_id, max_dim, unit types)

**Returns:** None

### load() - Load index from disk

```python
load() -> None
```

**Behavior:**

- Loads all UnitIndex instances from files
- Opens InstanceIndex LMDB environment
- Restores index metadata

**Returns:** None

### view() - Memory-map index from disk

```python
view() -> None
```

**Behavior:**

- Memory-maps UnitIndex instances (read-only)
- Opens InstanceIndex LMDB environment
- Restores index metadata

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
# Add entries
iscc-vdb add --iscc-code "ISCC:KAC..." [--iscc-id "ISCC:IAA..."]
iscc-vdb add --units "ISCC:GAA..." "ISCC:EAA..." "ISCC:IAA..."

# Get entry
iscc-vdb get "ISCC:IAACBFKZG52UU"

# Search
iscc-vdb search --iscc-code "ISCC:KAC..." --count 10
iscc-vdb search --units "ISCC:GAA..." "ISCC:EAA..." --count 5 --exact

# Remove
iscc-vdb remove "ISCC:IAACBFKZG52UU"

# Index management
iscc-vdb save
iscc-vdb info  # Show index stats (size, unit_types, etc.)
```

## REST API Interface

RESTful endpoints mapping to Python API:

```
POST   /add        - Add entries (body: {iscc_ids, iscc_codes, units})
GET    /get/{id}   - Get single entry
POST   /get        - Get multiple entries (body: {iscc_ids})
POST   /search     - Search index (body: {iscc_codes, units, count, exact})
DELETE /remove/{id} - Remove single entry
POST   /remove     - Remove multiple entries (body: {iscc_ids})
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
