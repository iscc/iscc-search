---
icon: lucide/server
description: Run the REST API server and interact with it over HTTP.
---

# Use the REST API

The iscc-search REST API lets you manage indexes, add assets, and search for similar content over HTTP.

## Start the server

Start in production mode:

```bash
iscc-search serve
```

Start in development mode with auto-reload:

```bash
iscc-search serve --dev
```

Configure the server via environment variables:

```bash
export ISCC_SEARCH_INDEX_URI=usearch:///var/lib/iscc-search
export ISCC_SEARCH_HOST=0.0.0.0
export ISCC_SEARCH_PORT=8000
export ISCC_SEARCH_LOG_LEVEL=info
```

See the [deployment guide](deployment.md) for production configuration.

## Authentication

By default, the API is public. Set `ISCC_SEARCH_API_SECRET` to require authentication:

```bash
export ISCC_SEARCH_API_SECRET=your-secret-key
```

When enabled, include the secret in the `X-API-Key` header on every request:

```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8000/indexes
```

## Index management

Create an index:

```bash
curl -X POST http://localhost:8000/indexes \
  -H "Content-Type: application/json" \
  -d '{"name": "myindex"}'
```

List all indexes:

```bash
curl http://localhost:8000/indexes
```

Get metadata for a specific index:

```bash
curl http://localhost:8000/indexes/myindex
```

Delete an index and all its data:

```bash
curl -X DELETE http://localhost:8000/indexes/myindex
```

## Asset operations

Add assets to an index. The request body is an array of asset objects. Each must include `iscc_id` and at
least one of `iscc_code` or `units`:

```bash
curl -X POST http://localhost:8000/indexes/myindex/assets \
  -H "Content-Type: application/json" \
  -d '[
    {
      "iscc_id": "ISCC:MAIGIIFJRDGEQQAA",
      "iscc_code": "ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY"
    }
  ]'
```

Retrieve an asset by ISCC-ID:

```bash
curl http://localhost:8000/indexes/myindex/assets/ISCC:MAIGIIFJRDGEQQAA
```

## Search

Search for similar assets using POST with a JSON body:

```bash
curl -X POST http://localhost:8000/indexes/myindex/search \
  -H "Content-Type: application/json" \
  -d '{"iscc_code": "ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY"}'
```

Limit the number of results with a query parameter:

```bash
curl -X POST "http://localhost:8000/indexes/myindex/search?limit=5" \
  -H "Content-Type: application/json" \
  -d '{"iscc_code": "ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY"}'
```

Search using GET with a query parameter:

```bash
curl "http://localhost:8000/indexes/myindex/search?iscc_code=ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY"
```

## Health checks

iscc-search exposes two health endpoints for orchestrators and load balancers.

**Liveness probe** - returns 200 if the process is alive. Does not check index state:

```bash
curl http://localhost:8000/healthz
```

```json
{"status": "ok"}
```

**Readiness probe** - returns 200 only when the index is initialized and operational:

```bash
curl http://localhost:8000/readyz
```

=== "Ready"

    ```json
    {"status": "ready"}
    ```

=== "Not ready"

    ```json
    {"status": "not_ready", "reason": "index_not_initialized"}
    ```

    Returns HTTP 503 when not ready.

## Interactive documentation

The server hosts interactive API documentation at `/docs`, powered by Stoplight Elements. Open
`http://localhost:8000/docs` in your browser to explore endpoints, view schemas, and send test requests.
