---
icon: lucide/settings
description: Environment variables and configuration options for iscc-search.
---

# Configuration

iscc-search uses two independent configuration systems. The server uses `SearchOptions` for
deployment. The CLI uses `AppConfig` for multi-index management.

## Server Configuration (SearchOptions)

`SearchOptions` configures the `iscc-search serve` command and the FastAPI app. All variables
use the `ISCC_SEARCH_` prefix and follow 12-factor app principles.

You can set these through environment variables, a `.env` file in the working directory, or
direct instantiation in code.

| Variable                                      | Default                | Description                                        |
| --------------------------------------------- | ---------------------- | -------------------------------------------------- |
| `ISCC_SEARCH_INDEX_URI`                       | `usearch://{data_dir}` | Backend URI (`memory://`, `lmdb://`, `usearch://`) |
| `ISCC_SEARCH_API_SECRET`                      | `None`                 | API authentication secret (unset = public access)  |
| `ISCC_SEARCH_CORS_ORIGINS`                    | `*`                    | Comma-separated allowed origins                    |
| `ISCC_SEARCH_HOST`                            | `0.0.0.0`              | Server bind address                                |
| `ISCC_SEARCH_PORT`                            | `8000`                 | Server bind port                                   |
| `ISCC_SEARCH_WORKERS`                         | `None`                 | Number of worker processes                         |
| `ISCC_SEARCH_SHARD_SIZE_UNITS`                | `512`                  | Max shard size for unit indexes (MB)               |
| `ISCC_SEARCH_SHARD_SIZE_SIMPRINTS`            | `512`                  | Max shard size for simprint indexes (MB)           |
| `ISCC_SEARCH_HNSW_EXPANSION_ADD_UNITS`        | `128`                  | Build-time search depth for unit HNSW              |
| `ISCC_SEARCH_HNSW_EXPANSION_SEARCH_UNITS`     | `64`                   | Query-time search depth for unit HNSW              |
| `ISCC_SEARCH_HNSW_CONNECTIVITY_UNITS`         | `16`                   | Graph connectivity (M) for unit HNSW               |
| `ISCC_SEARCH_HNSW_EXPANSION_ADD_SIMPRINTS`    | `16`                   | Build-time search depth for simprint HNSW          |
| `ISCC_SEARCH_HNSW_EXPANSION_SEARCH_SIMPRINTS` | `512`                  | Query-time search depth for simprint HNSW          |
| `ISCC_SEARCH_HNSW_CONNECTIVITY_SIMPRINTS`     | `8`                    | Graph connectivity (M) for simprint HNSW           |
| `ISCC_SEARCH_MATCH_THRESHOLD_UNITS`           | `0.75`                 | Minimum score for unit matches (0.0-1.0)           |
| `ISCC_SEARCH_MATCH_THRESHOLD_SIMPRINTS`       | `0.75`                 | Minimum score for simprint matches (0.0-1.0)       |
| `ISCC_SEARCH_CONFIDENCE_EXPONENT`             | `4`                    | Exponent for confidence-weighted scoring           |
| `ISCC_SEARCH_OVERSAMPLING_FACTOR`             | `20`                   | Oversampling multiplier for simprint search        |
| `ISCC_SEARCH_FLUSH_INTERVAL`                  | `100000`               | Auto-flush after N mutations (0 = disabled)        |
| `ISCC_SEARCH_LOG_LEVEL`                       | `info`                 | Log level (`debug`, `info`, `warning`, `error`)    |
| `ISCC_SEARCH_SENTRY_DSN`                      | `None`                 | Sentry error tracking DSN                          |
| `ISCC_SEARCH_SENTRY_TRACES_SAMPLE_RATE`       | `0.05`                 | Sentry performance sampling rate (0.0-1.0)         |

### .env file support

Place a `.env` file in the working directory. iscc-search loads it automatically at startup.
Environment variables take precedence over `.env` values.

```bash
ISCC_SEARCH_INDEX_URI=usearch:///data/iscc-index
ISCC_SEARCH_API_SECRET=my-secret-key
ISCC_SEARCH_LOG_LEVEL=debug
```

### Runtime override

In code, you can create a modified copy of the options with `override()`:

```python
from iscc_search.options import search_opts

custom = search_opts.override({"port": 9000, "log_level": "debug"})
```

## CLI Configuration (AppConfig)

CLI data commands (`add`, `get`, `search`) use a persistent JSON config stored at
`~/.iscc-search/config.json`. This provides a git-like workflow for managing multiple
named indexes.

### Workflow

```bash
# Add a local index
iscc-search index add myindex

# Add a remote index
iscc-search index add production --remote https://api.example.com --api-key SECRET

# List configured indexes
iscc-search index list

# Switch active index
iscc-search index use production

# Remove an index
iscc-search index remove myindex
```

### Config file structure

Each index configuration includes a name, type (`local` or `remote`), and type-specific
fields.

```json
{
  "active_index": "production",
  "indexes": {
    "default": {
      "type": "local",
      "path": "/home/user/.local/share/iscc-search"
    },
    "production": {
      "type": "remote",
      "url": "https://api.example.com",
      "api_key": "my-secret"
    }
  }
}
```

Local indexes store `path` (directory on disk). Remote indexes store `url` and optional
`api_key`. The first index added becomes the active index automatically.

## Two systems, separate purposes

These configuration systems are intentionally separate.

- **`SearchOptions`** drives the server. One index per deployment, configured through
    environment variables. Used by `iscc-search serve`.
- **`AppConfig`** drives the CLI. Multiple named indexes with an active selection, persisted
    in JSON. Used by `iscc-search add`, `search`, `get`.

See [Architecture](../explanation/architecture.md) for how these systems fit into the
overall design.
