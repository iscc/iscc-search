# ISCC-Search CLI Documentation

## Overview

The ISCC-Search CLI provides a command-line interface for managing ISCC indexes, adding assets, and searching
for similar content. The CLI is built on top of the protocol-based architecture and works seamlessly with all
index backends (memory, LMDB, PostgreSQL).

NOTE: Does the memory backend make any sense for a CLI application? All data is stored in memory, and lost when
the process is terminated. It would only make sense if MemoryIndex can be saved and restored.

## Design Principles

1. **Protocol-First**: CLI operates through the `IsccIndexProtocol`, supporting all index backends
2. **Stateful Context**: Selected current active index information is persisted, eliminating repetitive `--index` flags
3. **Smart Defaults**: Auto-select index when only one exists, sensible default behaviors
4. **Bulk Operations**: Efficient handling of large datasets with progress feedback
5. **Output Flexibility**: Multiple output formats (table, JSON, compact) for different use cases
6. **Excellent DX**: Clear error messages, confirmations for destructive operations, colored output

NODE: If no index exists, the CLI will automatically create a new UsearchIndex with the name `default` and
set it as the current active index.

## State Management

The CLI maintains a persistent state file using `platformdirs` to store:

- **Current active index**: The default index for commands that require one
- **Backend URI**: Current `ISCC_SEARCH_INDEX_URI` setting
- **CLI preferences**: Output format, color settings, etc.

NOTE: We might have an architecture/design issue here. Our current design allows us to manage multiple indexes of the
same backend type, but due to the global singleton `ISCC_SEARCH_INDEX_URI` we cannot easily support multiple indexes
of different backend types. The FastAPI REST-API server should also be able to manage multiple indexes of different
backend types. We painted ourselves into a corner here. We need to rethink our design and make sure we can support
multiple indexes of different backend types before our first release.

**State file location**:

- Linux: `~/.local/share/iscc-search/state.json`
- macOS: `~/Library/Application Support/iscc-search/state.json`
- Windows: `%LOCALAPPDATA%\iscc-search\state.json`

NOTE: Dynamically derived depending by platformdirs depending on OS.

## Command Reference

### Index Management

#### `iscc-search index list`

List all available indexes with their metadata.

**Usage**:

```bash
iscc-search index list
iscc-search index list --output json
```

**Output**:

```
NAME      ASSETS    SIZE      SELECTED
──────────────────────────────────────
default   150000    42 MB     *
myindex   5000      2 MB
```

**JSON Output**:

```json
[
  {
    "name": "default",
    "assets": 150000,
    "size": 42,
    "selected": true
  },
  {
    "name": "myindex",
    "assets": 5000,
    "size": 2,
    "selected": false
  }
]
```

NOTE:`iscc-search index` without subcommand should show help for subcommands.
NOTE: An `iscc-search index select` command that allows for interactive selection would be nice

**Options**:

- `--output-format, -o`: Output format (table, json, compact) [default: table]

---

NOTE: I am not sure if we need to support compcat output format? What would that look like?

#### `iscc-search index create <name>`

Create a new ISCC index.

**Usage**:

```bash
iscc-search index create myindex
iscc-search index create myindex --use
```

**Arguments**:

- `name`: Index name (pattern: `^[a-z][a-z0-9]*$`, max 32 chars)

**Options**:

- `--use`: Automatically set as current index after creation
- `--output-format, -o`: Output format (table, json, compact) [default: table]

**Output**:

```
✓ Created index 'myindex'
  Assets: 0
  Size: 0 MB
```

**Exit Codes**:

- `0`: Success
- `1`: Invalid name format
- `2`: Index already exists

---

NOTE: we don't need a `--use` option. Instead, we should always set the new index as the active index. The
`--output-format` option is ambiguous here. Does it set/store the output format to be used by default when the new
index is active, or does it only determine the output format of this one command? Any ideas on how to improve this?


#### `iscc-search index delete <name>`

Delete an ISCC index and all its data.

**Usage**:

```bash
iscc-search index delete myindex
iscc-search index delete myindex --yes
```

**Arguments**:

- `name`: Index name to delete

**Options**:

- `--yes, -y`: Skip confirmation prompt
- `--output, -o`: Output format (table, json, compact) [default: table]

**Behavior**:

- Prompts for confirmation unless `--yes` is specified
- If deleting current index, clears current index state
- Cannot delete if index is currently being used by API server

**Output**:

```
⚠ This will permanently delete index 'myindex' and all its data.
  Continue? [y/N]: y
✓ Deleted index 'myindex'
```

**Exit Codes**:

- `0`: Success
- `1`: Invalid name
- `2`: Index not found
- `3`: User cancelled

---

#### `iscc-search index use <name>`

Set the active index for subsequent commands.

NOTE: I would prefer `iscc-search index select <name>` instead

**Usage**:

```bash
iscc-search index use myindex
```

**Arguments**:

- `name`: Index name to set as current

**Output**:

```
✓ Now using index 'myindex'
```

**Exit Codes**:

- `0`: Success
- `1`: Invalid name
- `2`: Index not found

---

#### `iscc-search index show [name]`

Show detailed metadata for an index.

**Usage**:

```bash
iscc-search index show              # Show current index
iscc-search index show myindex      # Show specific index
iscc-search index show --output json
```

**Arguments**:

- `name`: Index name to show (optional, defaults to current index)

**Options**:

- `--output, -o`: Output format (table, json, compact) [default: table]

**Output**:

```
Index: myindex
──────────────────────
Assets:     150000
Size:       42 MB
Created:    2025-10-29 10:30:15
Backend:    lmdb:///path/to/data
```

**JSON Output**:

```json
{
  "name": "myindex",
  "assets": 150000,
  "size": 42,
  "created": "2025-10-29T10:30:15Z",
  "backend": "lmdb:///path/to/data"
}
```

**Exit Codes**:

- `0`: Success
- `1`: No current index and no name specified
- `2`: Index not found

---

### Asset Management

#### `iscc-search add <path>`

Add ISCC assets to the current index from JSON files.

**Usage**:

```bash
# Add single file
iscc-search add asset.iscc.json

# Add all *.iscc.json files from directory to active index
iscc-search add /path/to/assets/*.iscc.json

```

**Arguments**:

- `path`: Path to `.iscc.json` file or directory with glob selected files

**Options**:

- `--output-format, -o`: Output format (table, json, compact) [default: table]

**Field Mapping**:

JSON files use the following field mapping:

| JSON Field | ISCC Schema Field | Required | Notes                         |
| ---------- | ----------------- | -------- | ----------------------------- |
| `iscc`     | `iscc_code`       | No       | Composite ISCC-CODE           |
| `iscc_id`  | `iscc_id`         | Yes      | ISCC-ID identifier            |
| `units`    | `units`           | No       | Array of ISCC-UNITs           |
| `*`        | `metadata`        | No       | All other fields as JSON blob |

**Important**: Files must contain either `iscc_code` (as `iscc` field) or `units` array. The `iscc_id` field
is required for all assets.

NOTE: for the ISCC Schema Field `iscc_code` we should first look for `iscc_code` and fallback to `iscc`

**Performance**:

- Uses `simdjson` for fast parsing of large JSON files
- Extracts only required fields (`iscc`, `iscc_id`, `units`) without loading/decoding the entire json file
- Batch processing with progress indicator
- Parallel file reading (if backend supports concurrent writes)

**Output**:

```
Scanning for *.iscc.json files...
Found 1,523 files

Adding assets to 'myindex'...
[████████████████████████████████] 1523/1523 100%

✓ Added 1,523 assets
  Created: 1,450
  Updated: 73
  Errors: 0
  Duration: 12.3s
```

**JSON Output**:

```json
{
  "files_scanned": 1523,
  "assets_added": 1523,
  "created": 1450,
  "updated": 73,
  "errors": 0,
  "duration_seconds": 12.3,
  "results": [
    {
      "file": "/path/to/asset1.iscc.json",
      "iscc_id": "ISCC:MAIGIIFJRDGEQQAA",
      "status": "created"
    }
  ]
}
```

**Exit Codes**:

- `0`: Success (all files processed)
- `1`: No current index
- `2`: Path not found
- `3`: No `.iscc.json` files found
- `4`: Validation errors (unless `--continue-on-error`)

---

#### `iscc-search get <iscc-id>`

Retrieve a specific asset by its ISCC-ID.

**Usage**:

```bash
iscc-search get ISCC:MAIGIIFJRDGEQQAA
iscc-search get ISCC:MAIGIIFJRDGEQQAA --index myindex
iscc-search get ISCC:MAIGIIFJRDGEQQAA --output json
```

**Arguments**:

- `iscc-id`: ISCC-ID to retrieve (format: `ISCC:[A-Z2-7]{16}`)

**Options**:

- `--index, -i`: Override current index
- `--output, -o`: Output format (table, json, compact) [default: json]

**Output (JSON)**:

```json
{
  "iscc_id": "ISCC:MAIGIIFJRDGEQQAA",
  "iscc_code": "ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY",
  "units": [
    "ISCC:AADYCMZIOY36XXGZ5B5BME7EIPPXRFKYQZ7VXKI7V55AEQQE67A33BY",
    "ISCC:EED7ZPIEYNACCLXXZSS2LIM6JVXDYGCG2QSMC7DCPER4MYJPJATIM4Y"
  ],
  "metadata": {
    "title": "Example Document",
    "creator": "John Doe"
  }
}
```

**Table Output**:

```
ISCC-ID:    ISCC:MAIGIIFJRDGEQQAA
ISCC-CODE:  ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY

Units:
  - ISCC:AADYCMZIOY36XXGZ5B5BME7EIPPXRFKYQZ7VXKI7V55AEQQE67A33BY
  - ISCC:EED7ZPIEYNACCLXXZSS2LIM6JVXDYGCG2QSMC7DCPER4MYJPJATIM4Y

Metadata:
  title:   Example Document
  creator: John Doe
```

**Exit Codes**:

- `0`: Success
- `1`: No current index
- `2`: Invalid ISCC-ID format
- `3`: Asset not found

---

### Search

#### `iscc-search search <query>`

Search for similar ISCC assets using an ISCC-CODE or JSON file.

**Usage**:

```bash
# Search by ISCC-CODE
iscc-search search ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY

# Search by ISCC-CODE from file
iscc-search search --file query.iscc.json

# Limit results
iscc-search search ISCC:KEC... --limit 20

# Specify index
iscc-search search ISCC:KEC... --index myindex
```

**Arguments**:

- `query`: ISCC-CODE to search for (format: `ISCC:[A-Z2-7]{16,}`)

**Options**:

- `--file, -f`: Read query from `.iscc.json` file instead of command line
- `--index, -i`: Override current index
- `--limit, -l`: Maximum number of results [default: 10]
- `--min-score, -s`: Minimum similarity score threshold
- `--output, -o`: Output format (table, json, compact) [default: table]

**Output**:

```
Query: ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY
Metric: bitlength
Results: 3 matches

ISCC-ID                  SCORE  MATCHES
─────────────────────────────────────────────────────────────────
ISCC:MAIGIIFJRDGEQQAA    448    CONTENT_TEXT_V0 (256)
                                DATA_NONE_V0 (128)
                                INSTANCE_NONE_V0 (64)

ISCC:MAIGXXFZRDGEQQBB    384    CONTENT_TEXT_V0 (256)
                                DATA_NONE_V0 (128)

ISCC:MAIGZZZZRDGEQQCC    256    CONTENT_TEXT_V0 (256)
```

**JSON Output**:

```json
{
  "query": {
    "iscc_code": "ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY",
    "units": [
      "ISCC:AADYCMZIOY36XXGZ5B5BME7EIPPXRFKYQZ7VXKI7V55AEQQE67A33BY",
      "ISCC:EED7ZPIEYNACCLXXZSS2LIM6JVXDYGCG2QSMC7DCPER4MYJPJATIM4Y"
    ]
  },
  "metric": "bitlength",
  "matches": [
    {
      "iscc_id": "ISCC:MAIGIIFJRDGEQQAA",
      "score": 448,
      "matches": {
        "CONTENT_TEXT_V0": 256,
        "DATA_NONE_V0": 128,
        "INSTANCE_NONE_V0": 64
      }
    }
  ]
}
```

**Exit Codes**:

- `0`: Success (matches found)
- `1`: No current index
- `2`: Invalid ISCC-CODE format
- `3`: No matches found

---

### Global Commands

#### `iscc-search info`

Show current configuration and system information.

**Usage**:

```bash
iscc-search info
iscc-search info --output json
```

**Output**:

```
ISCC-Search v0.1.0

Configuration:
  Backend:       lmdb:///path/to/data
  State file:    ~/.local/share/iscc-search/state.json
  Current index: myindex

Current Index:
  Name:     myindex
  Assets:   150000
  Size:     42 MB

Available Indexes: 2
  - default (150000 assets)
  - myindex (5000 assets) *
```

**JSON Output**:

```json
{
  "version": "0.1.0",
  "backend": "lmdb:///path/to/data",
  "state_file": "~/.local/share/iscc-search/state.json",
  "current_index": {
    "name": "myindex",
    "assets": 150000,
    "size": 42
  },
  "available_indexes": ["default", "myindex"]
}
```

---

#### `iscc-search --help`

Show help for all commands or a specific command.

**Usage**:

```bash
iscc-search --help
iscc-search index --help
iscc-search add --help
```

---

#### `iscc-search --version`

Show version information.

**Usage**:

```bash
iscc-search --version
```

**Output**:

```
iscc-search version 0.1.0
```

---

## Global Options

These options apply to all commands:

- `--index, -i <name>`: Override current index for this command
- `--output, -o <format>`: Output format (table, json, compact)
- `--no-color`: Disable colored output
- `--verbose, -v`: Enable verbose logging
- `--quiet, -q`: Suppress non-error output
- `--help, -h`: Show help message
- `--version`: Show version information

---

## Configuration

### Environment Variables

The CLI respects the same environment variables as the API server:

- `ISCC_SEARCH_INDEX_URI`: Index backend URI (memory://, lmdb://, postgresql://)

**Examples**:

```bash
# Use LMDB backend
export ISCC_SEARCH_INDEX_URI=lmdb:///path/to/data

# Use in-memory backend (no persistence)
export ISCC_SEARCH_INDEX_URI=memory://

# Use PostgreSQL backend (future)
export ISCC_SEARCH_INDEX_URI=postgresql://user:pass@host/db
```

### Configuration File

Optional `.env` file support in current directory or `~/.config/iscc-search/.env`:

```env
ISCC_SEARCH_INDEX_URI=lmdb:///home/user/iscc-data
```

---

## Exit Codes

The CLI uses standard exit codes:

| Code | Meaning                            |
| ---- | ---------------------------------- |
| 0    | Success                            |
| 1    | General error (invalid args, etc.) |
| 2    | Resource not found                 |
| 3    | User cancelled operation           |
| 4    | Validation error                   |

---

## Examples

### Basic Workflow

```bash
# Show info
iscc-search info

# Create index
iscc-search index create myindex --use

# Add assets from directory
iscc-search add /data/iscc-files/

# Search for similar content
iscc-search search ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY

# Get specific asset
iscc-search get ISCC:MAIGIIFJRDGEQQAA

# List all indexes
iscc-search index list

# Switch to different index
iscc-search index use default

# Delete old index
iscc-search index delete myindex
```

### Advanced Usage

```bash
# Batch add with JSON output for logging
iscc-search add /data/iscc-files/ --output json > add-results.json

# Search with high precision threshold
iscc-search search ISCC:KEC... --min-score 400 --limit 50

# Dry run to validate files before adding
iscc-search add /data/new-batch/ --dry-run

# Add to specific index without changing current
iscc-search add /data/files/ --index staging

# Recursive scan with error handling
iscc-search add /data/nested/ --recursive --continue-on-error

# Export search results for processing
iscc-search search ISCC:KEC... --output json | jq '.matches[].iscc_id'
```

### Working with Multiple Indexes

```bash
# Create development and production indexes
iscc-search index create dev --use
iscc-search index create prod

# Add test data to dev
iscc-search add /data/test-set/

# Add production data to prod
iscc-search add /data/prod-set/ --index prod

# Compare search results across indexes
iscc-search search ISCC:KEC... --index dev
iscc-search search ISCC:KEC... --index prod

# Switch to production
iscc-search index use prod
```

---

## Implementation Notes

### Technology Stack

- **CLI Framework**: Typer (built on Click)
- **Output Formatting**: Rich (for tables, colors, progress bars)
- **JSON Parsing**: simdjson-python (for high-performance parsing)
- **State Management**: platformdirs + JSON file
- **Backend Communication**: Direct protocol calls (IsccIndexProtocol)

### Performance Considerations

1. **Bulk Operations**:
    - Use batch API calls when available
    - Show progress indicators for long-running operations
    - Parallel file reading with controlled concurrency

2. **JSON Parsing**:
    - Use simdjson for large files (can be >100MB)
    - Extract only required fields to minimize memory
    - Stream processing for very large directories

3. **State Management**:
    - Cache index list to avoid repeated backend calls
    - Lazy load state file only when needed
    - Atomic writes for state updates

### Error Handling

- Clear, actionable error messages with suggestions
- Graceful degradation when optional features unavailable
- Proper cleanup on interruption (Ctrl+C)
- Detailed error output with `--verbose`

### Testing Strategy

- Unit tests for all commands using MemoryIndex backend
- Integration tests with LMDB backend
- CLI output snapshot tests
- Performance benchmarks for bulk operations

---

## Future Enhancements

- **Export/Import**: `iscc-search export/import` for backup/restore
- **Stats**: `iscc-search stats` for detailed index analytics
- **Reindex**: `iscc-search reindex` to rebuild index from assets
- **Validate**: `iscc-search validate` to check index integrity
- **Watch**: `iscc-search watch <dir>` to auto-add new files
- **Shell Completion**: Generate completion scripts for bash/zsh/fish
- **Interactive Mode**: TUI for exploring indexes and results
