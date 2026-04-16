---
icon: lucide/rocket
description: Deploy iscc-search safely in production with correct configuration.
---

# Deployment

This guide covers the settings and precautions required to run iscc-search reliably in production.

## Single worker only

!!! warning "Data corruption risk"

    The usearch backend has **no multi-process coordination**. Running multiple workers against the same
    data directory will corrupt your indexes. This is not recoverable without a full re-index.

Always run with exactly one worker process. FastAPI's async/await handles concurrent connections within that
single process.

=== "Correct"

    ```bash
    uvicorn iscc_search.server:app --host 0.0.0.0 --port 8000
    ```

=== "Wrong - will corrupt data"

    ```bash
    uvicorn iscc_search.server:app --host 0.0.0.0 --port 8000 --workers 4
    ```

Do not set `ISCC_SEARCH_WORKERS` to a value greater than 1 when using the usearch backend.

## Docker quick start

```yaml title="compose.yaml"
services:
  iscc-search:
    image: ghcr.io/iscc/iscc-search:latest
    container_name: iscc-search-api
    ports:
      - "8000:8000"
    volumes:
      - iscc-data:/data
    environment:
      - ISCC_SEARCH_INDEX_URI=usearch:///data
      - ISCC_SEARCH_FLUSH_INTERVAL=100000
      # - ISCC_SEARCH_API_SECRET=your-secret-key
      # - ISCC_SEARCH_CORS_ORIGINS=https://example.com
    restart: unless-stopped
    stop_grace_period: 120s
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

volumes:
  iscc-data:
    driver: local
```

Start it:

```bash
docker compose up -d
```

## Graceful shutdown

On shutdown, iscc-search flushes all dirty HNSW indexes to disk. Large indexes can take tens of seconds to
save. You must give the process enough time to complete.

The timing chain works like this:

1. **uvicorn** `--timeout-graceful-shutdown 60` - stops accepting connections, waits up to 60s for the
   lifespan handler to flush indexes.
2. **Docker** `stop_grace_period: 120s` - sends SIGTERM, then waits 120s before SIGKILL. Must be longer
   than uvicorn's timeout.

The Dockerfile sets `--timeout-graceful-shutdown 60` by default. Set `stop_grace_period` in your compose file
to at least double that value.

## Environment variables

| Variable | Default | Production recommendation |
|---|---|---|
| `ISCC_SEARCH_INDEX_URI` | `usearch://` + platform dir | `usearch:///data` (explicit path) |
| `ISCC_SEARCH_API_SECRET` | None (public) | Set a strong secret |
| `ISCC_SEARCH_CORS_ORIGINS` | `*` | Restrict to your domains |
| `ISCC_SEARCH_FLUSH_INTERVAL` | 0 (disabled) | `100000` (flush every 100K mutations) |
| `ISCC_SEARCH_LOG_LEVEL` | `info` | `info` or `warning` |
| `ISCC_SEARCH_SENTRY_DSN` | None | Set for error tracking |
| `ISCC_SEARCH_HOST` | `0.0.0.0` | `0.0.0.0` |
| `ISCC_SEARCH_PORT` | 8000 | 8000 |

!!! tip "Flush interval"

    `FLUSH_INTERVAL=0` means indexes are only saved on graceful shutdown. If the process is killed (OOM,
    SIGKILL, power loss), all mutations since the last save are lost. Setting a non-zero value reduces the
    blast radius of hard crashes.

## Sizing profiles

| Profile | Assets | RAM | CPU | Notes |
|---|---|---|---|---|
| Sandbox | up to 100K | 4 GB | 2 cores | Development and testing |
| Validation | up to 500K | 64 GB | 4 cores | Pre-production validation |
| Launch | up to 1M | 128 GB | 8 cores | Initial production deployment |
| Growth | 5M+ | 256 GB+ | 16+ cores | Large-scale production |

HNSW indexes are memory-mapped. RAM requirements grow with index size. Monitor RSS and adjust limits.

## Health probes

Use the built-in health endpoints with your orchestrator.

**Liveness** (`/healthz`) - always returns 200 if the process responds. Use for restart decisions:

```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 30
```

**Readiness** (`/readyz`) - returns 200 only when the index is initialized and `list_indexes()` succeeds.
Use for traffic routing:

```yaml
readinessProbe:
  httpGet:
    path: /readyz
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

## Horizontal scaling

Each iscc-search instance must have its own data volume. Shared volumes between instances will corrupt data.

Run independent instances behind a load balancer. Each instance holds a separate copy of the index (or a
partition of the data).

```nginx title="nginx upstream example"
upstream iscc_search {
    server iscc-search-1:8000;
    server iscc-search-2:8000;
    server iscc-search-3:8000;
}

server {
    listen 443 ssl;
    location / {
        proxy_pass http://iscc_search;
    }
}
```

Feed the same data to each instance, or shard by content type. Writes must be routed to the correct instance.

## Production checklist

Before going live, verify the following:

- [ ] `ISCC_SEARCH_INDEX_URI` points to a persistent volume
- [ ] Worker count is 1 (or unset)
- [ ] `ISCC_SEARCH_FLUSH_INTERVAL` is non-zero (e.g., `100000`)
- [ ] `ISCC_SEARCH_API_SECRET` is set
- [ ] `ISCC_SEARCH_CORS_ORIGINS` is restricted to your domains
- [ ] `stop_grace_period` is at least 2x the uvicorn graceful shutdown timeout
- [ ] Health probes are configured in your orchestrator
- [ ] Each instance has its own data volume (no sharing)
- [ ] Resource limits (CPU, memory) match your sizing profile
- [ ] Sentry DSN is configured for error tracking
- [ ] Backups are scheduled for the data volume

## Troubleshooting

### Index corruption

**Symptom**: Server crashes on startup, search returns errors, or asset counts are wrong.

**Cause**: Multiple processes wrote to the same data directory, or the process was killed during a write
(SIGKILL, OOM).

**Fix**: Stop the server. Delete the corrupted `.usearch` files from the index directory. Restart the server.
It will rebuild HNSW indexes from the LMDB data on next access. Re-index if LMDB is also corrupted.

### Slow shutdown

**Symptom**: Container takes a long time to stop or is killed by Docker after the grace period.

**Cause**: Large HNSW indexes need time to flush to disk. The grace period is too short.

**Fix**: Increase `stop_grace_period` in your compose file. For indexes over 1M assets, use 300s or more.
Monitor shutdown logs to find the actual flush duration.

### Out of memory

**Symptom**: Process is killed by the OOM killer. Container restarts repeatedly.

**Cause**: HNSW indexes are memory-mapped. Index size exceeds available RAM.

**Fix**: Increase the memory limit in your container configuration. Refer to the sizing profiles table above.
Consider sharding data across multiple instances.
