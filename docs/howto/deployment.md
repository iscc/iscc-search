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
      - 8000:8000
    volumes:
      - iscc-data:/data
    environment:
      - ISCC_SEARCH_INDEX_URI=usearch:///data
      # - ISCC_SEARCH_API_SECRET=your-secret-key
      # - ISCC_SEARCH_CORS_ORIGINS=https://example.com
    restart: unless-stopped
    stop_grace_period: 300s
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

On shutdown, iscc-search flushes all dirty HNSW indexes to disk. Large indexes can take a few minutes to
save. You must give the process enough time to complete the **drain phase plus the flush phase**.

uvicorn's shutdown is **strictly sequential**:

1. **uvicorn** stops accepting new connections (immediate).
1. **uvicorn** waits for in-flight requests to complete, bounded by `--timeout-graceful-shutdown`.
1. **uvicorn** runs the FastAPI lifespan handler, which calls `index.close()` to flush HNSW shards.
    This step has **no timeout in uvicorn** — only Docker's `stop_grace_period` can stop it.
1. **Docker** sends SIGKILL once `stop_grace_period` elapses since the initial SIGTERM.

This means:

```
stop_grace_period >= timeout_graceful_shutdown + expected_flush_duration + buffer
```

If `stop_grace_period` equals `timeout_graceful_shutdown`, a slow request can consume the entire grace
window and Docker SIGKILLs the process the moment the lifespan flush tries to start, **losing all dirty
HNSW state**. This is a real failure mode, not a theoretical one.

Defaults:

- Dockerfile: `--timeout-graceful-shutdown 60` (drain timeout)
- compose.yaml: `stop_grace_period 300s` (60s drain + 240s flush headroom)

For very large indexes (10M+ vectors), raise `stop_grace_period` to `600s` or more — the drain timeout
stays at `60s` because it bounds request latency, not flush latency.

## Environment variables

| Variable                     | Default                     | Production recommendation                             |
| ---------------------------- | --------------------------- | ----------------------------------------------------- |
| `ISCC_SEARCH_INDEX_URI`      | `usearch://` + platform dir | `usearch:///data` (explicit path)                     |
| `ISCC_SEARCH_API_SECRET`     | None (public)               | Set a strong secret                                   |
| `ISCC_SEARCH_CORS_ORIGINS`   | `*`                         | Restrict to your domains                              |
| `ISCC_SEARCH_FLUSH_INTERVAL` | `100000`                    | Keep at default, or raise for higher write throughput |
| `ISCC_SEARCH_LOG_LEVEL`      | `info`                      | `info` or `warning`                                   |
| `ISCC_SEARCH_SENTRY_DSN`     | None                        | Set for error tracking                                |
| `ISCC_SEARCH_HOST`           | `0.0.0.0`                   | `0.0.0.0`                                             |
| `ISCC_SEARCH_PORT`           | 8000                        | 8000                                                  |

!!! tip "Flush interval"

    The default `FLUSH_INTERVAL=100000` auto-flushes derived HNSW indexes every 100,000 mutations, capping
    data loss on hard crashes (OOM, SIGKILL, power loss). Setting it to `0` disables auto-flush and means
    indexes are only saved on graceful shutdown — faster ingestion but unbounded loss on crash. Raise the
    value for slightly higher write throughput at the cost of a larger loss window.

## Sizing profiles

| Profile    | Assets     | RAM     | CPU       | Notes                         |
| ---------- | ---------- | ------- | --------- | ----------------------------- |
| Sandbox    | up to 100K | 4 GB    | 2 cores   | Development and testing       |
| Validation | up to 500K | 64 GB   | 4 cores   | Pre-production validation     |
| Launch     | up to 1M   | 128 GB  | 8 cores   | Initial production deployment |
| Growth     | 5M+        | 256 GB+ | 16+ cores | Large-scale production        |

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
- [ ] `ISCC_SEARCH_FLUSH_INTERVAL` is non-zero (default `100000`)
- [ ] `ISCC_SEARCH_API_SECRET` is set
- [ ] `ISCC_SEARCH_CORS_ORIGINS` is restricted to your domains
- [ ] `stop_grace_period` is `>= timeout_graceful_shutdown + expected_flush_duration` (default `300s`
    covers ~60s drain + ~240s flush; raise for indexes over 10M vectors)
- [ ] Health probes are configured in your orchestrator
- [ ] Each instance has its own data volume (no sharing)
- [ ] Resource limits (CPU, memory) match your sizing profile
- [ ] Sentry DSN is configured for error tracking
- [ ] Backups are scheduled for the data volume

## Troubleshooting

### Derived indexes out of sync

**Symptom**: Boot logs show `ShardedNphdIndex 'X' out of sync: expected N vectors, found M` or
`UsearchSimprintIndex 'X' out of sync: expected N, found M`. Search results are stale or empty for affected
unit/simprint types. Asset counts in `/indexes` reflect LMDB and may be larger than what searches return.

**Cause**: The process was killed (SIGKILL, OOM, host crash, `stop_grace_period` too short) before the
lifespan handler could flush dirty HNSW shards to disk. LMDB is the source of truth and survives unclean
exits; derived HNSW shards do not unless they were saved by `flush_interval` rotation, shard-size rotation,
or graceful `close()`.

**Fix**: Stop the server, then run an explicit rebuild from the intact LMDB data. Auto-rebuild on startup is
intentionally disabled because rebuilding large indexes can OOM the container.

```python
# One-shot rebuild from a Python REPL or script (until the CLI command lands)
from iscc_search.indexes.usearch.index import UsearchIndex

idx = UsearchIndex("/path/to/index-dir")
for unit_type in ("META_NONE_V0", "DATA_NONE_V0", "CONTENT_TEXT_V0", "SEMANTIC_TEXT_V0"):
    idx._rebuild_nphd_index(unit_type)
for sp_type in ("CONTENT_TEXT_V0", "SEMANTIC_TEXT_V0"):
    idx._rebuild_simprint_index(sp_type)
idx.close()
```

Restart the server. To prevent recurrence, ensure `ISCC_SEARCH_FLUSH_INTERVAL` is set to a non-zero value
(default `100000`) and `stop_grace_period` is sized as
`timeout_graceful_shutdown + expected_flush_duration + buffer` (default `60s + 240s = 300s`).

### LMDB corruption

**Symptom**: Server crashes on startup with `lmdb.Error` reading `index.lmdb`, or asset retrieval returns
malformed data.

**Cause**: Disk corruption, killed mid-write at the LMDB layer (very rare — LMDB uses MVCC and is
crash-safe by design), or downgrading the LMDB version with on-disk format incompatibility.

**Fix**: Restore `index.lmdb` from backup, then run the rebuild procedure above to regenerate derived
indexes. Re-ingest if no backup is available.

### Slow shutdown

**Symptom**: Container takes a long time to stop or is killed by Docker after the grace period.

**Cause**: Large HNSW indexes need time to flush to disk. The grace period is too short.

**Fix**: Raise `stop_grace_period` in your compose file. Keep `--timeout-graceful-shutdown` at the default
`60s` (it bounds request drain, not flush). Use the formula
`stop_grace_period = 60s + measured_flush_duration + 60s buffer`. Monitor shutdown logs
(`Saved ShardedNphdIndex`, `Saved UsearchSimprintIndex`) to measure actual flush duration.

### Out of memory

**Symptom**: Process is killed by the OOM killer. Container restarts repeatedly.

**Cause**: HNSW indexes are memory-mapped. Index size exceeds available RAM.

**Fix**: Increase the memory limit in your container configuration. Refer to the sizing profiles table above.
Consider sharding data across multiple instances.
