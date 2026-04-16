---
icon: lucide/terminal
description: Accomplish common tasks with the iscc-search command-line interface.
---

# Use the CLI

The `iscc-search` CLI manages indexes, adds assets, searches for similar content, and starts the REST API
server. All data commands operate on the **active index** unless you override it with `--index`.

## Manage indexes

Register a local index:

```bash
iscc-search index add myindex --local
```

Register a local index with a custom data path:

```bash
iscc-search index add myindex --local --path /data/iscc
```

Register a remote index pointing to a running iscc-search server:

```bash
iscc-search index add production --remote https://search.example.com
```

Register a remote index with an API key:

```bash
iscc-search index add production --remote https://search.example.com --api-key your-secret
```

List all configured indexes:

```bash
iscc-search index list
```

Switch the active index:

```bash
iscc-search index use production
```

Remove an index from configuration:

```bash
iscc-search index remove staging
```

Remove an index and delete its local data:

```bash
iscc-search index remove old-local --delete-data
```

## Add assets

Add assets from a directory of JSON files:

```bash
iscc-search add /path/to/assets/
```

The command looks for `*.iscc.json` files first, then falls back to `*.json`. Each file must contain at least
an `iscc_code` or `iscc` field.

Add assets with a glob pattern:

```bash
iscc-search add /data/corpus/*.iscc.json
```

Add a single file:

```bash
iscc-search add asset.iscc.json
```

Control batch size and truncate simprints:

```bash
iscc-search add --batch-size 500 --simprint-bits 128 /data/assets/
```

Target a specific index instead of the active one:

```bash
iscc-search add --index production /data/assets/
```

## Retrieve assets

Fetch full asset details by ISCC-ID:

```bash
iscc-search get ISCC:MAIGIIFJRDGEQQAA
```

Target a specific index:

```bash
iscc-search get ISCC:MAIGIIFJRDGEQQAA --index production
```

## Search for similar assets

Search by ISCC-CODE:

```bash
iscc-search search ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY
```

Limit the number of results:

```bash
iscc-search search ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY --limit 10
```

Search a specific index:

```bash
iscc-search search ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY --index production
```

## Start the server

Start in production mode:

```bash
iscc-search serve
```

Start in development mode with auto-reload:

```bash
iscc-search serve --dev
```

Use a custom host and port:

```bash
iscc-search serve --host 127.0.0.1 --port 9000
```

## Check the version

```bash
iscc-search version
```
