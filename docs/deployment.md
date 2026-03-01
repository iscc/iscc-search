# Production Deployment

## ⚠️ Critical: Data Corruption Risks

**NEVER use multiple workers with usearch backend** - it WILL corrupt your indexes.

```bash
# ❌ WRONG - corrupts data
uvicorn iscc_search.server:app --workers 4

# ✅ CORRECT - single worker
uvicorn iscc_search.server:app
```

**Why:**

- `.usearch` files have no file locking or multi-process coordination
- Instance cache doesn't synchronize between processes
- Improper shutdown (SIGKILL) loses unsaved indexes

**Solutions:**

- Use single worker with FastAPI async/await (handles 1000s concurrent connections)
- Scale horizontally with separate data directories (one per instance)
- Ensure graceful shutdown: Docker `stop_grace_period > 60s`

## Quick Start

```bash
# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop gracefully (IMPORTANT: waits for index save)
docker-compose stop

# Backup data
docker cp iscc-search-api:/data ./backup
```

## Essential Configuration

### Graceful Shutdown (Required)

**Dockerfile:**

```dockerfile
CMD ["uvicorn", "iscc_search.server:app",
     "--timeout-graceful-shutdown", "60"]
```

**docker-compose.yml:**

```yaml
stop_grace_period: 90s  # Must be > uvicorn timeout
```

**Timing:** `Docker grace (90s) > Uvicorn timeout (60s) + buffer`

### Environment Variables

| Variable                     | Default           | Required | Description                                 |
| ---------------------------- | ----------------- | -------- | ------------------------------------------- |
| `ISCC_SEARCH_INDEX_URI`      | `usearch:///data` | No       | Backend URI                                 |
| `ISCC_SEARCH_CORS_ORIGINS`   | `*`               | No       | Comma-separated origins                     |
| `ISCC_SEARCH_API_SECRET`     | None              | No       | API authentication                          |
| `ISCC_SEARCH_FLUSH_INTERVAL` | `0`               | No       | Auto-flush after N mutations (0 = disabled) |
| `ISCC_SEARCH_LOG_LEVEL`      | `info`            | No       | Log level (debug, info, warning, error)     |
| `ISCC_SEARCH_HOST`           | `0.0.0.0`         | No       | Server bind address                         |
| `ISCC_SEARCH_PORT`           | `8000`            | No       | Server bind port                            |

See `.env.example` for the full list of tunable parameters (HNSW, shard sizes, thresholds, scoring).

**Example:**

```yaml
environment:
  ISCC_SEARCH_INDEX_URI: "usearch:///data"
  ISCC_SEARCH_CORS_ORIGINS: "https://example.com"
  ISCC_SEARCH_FLUSH_INTERVAL: "10000"
```

**Flush interval:** When loading data in large batches, set `ISCC_SEARCH_FLUSH_INTERVAL` to a value
larger than your batch size (e.g. `10000` for 1000-entry batches) to avoid excessive disk I/O during
ingestion. Indexes are always flushed on graceful shutdown regardless of this setting.

### Resource Limits

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
```

**Memory:** Usearch indexes load entirely into memory (~2-4GB for small indexes) **CPU:** Single worker uses 1-2
cores effectively with async I/O

## Index Comparison

| Backend           | Multi-Process | Use Case          | Status  |
| ----------------- | ------------- | ----------------- | ------- |
| `memory://`       | ❌ No         | Testing only      | Ready   |
| `lmdb:///path`    | ⚠️ Reads only | Prefix matching   | Ready   |
| `usearch:///path` | ❌ No         | Similarity search | Ready   |
| `postgres://`     | ✅ Yes        | Future            | Planned |

**Recommended:** Use `usearch://` backend with single worker for production.

## Scaling Patterns

### Horizontal Scaling (Multiple Instances)

✅ **Correct** - separate data volumes:

```yaml
services:
  iscc-search-1:
    volumes:
      - iscc-data-1:/data

  iscc-search-2:
    volumes:
      - iscc-data-2:/data
```

❌ **Wrong** - shared volume corrupts data:

```yaml
services:
  iscc-search-1:
    volumes:
      - iscc-data:/data  # DANGER

  iscc-search-2:
    volumes:
      - iscc-data:/data  # CORRUPTION
```

### Load Balancing

Use nginx/traefik to distribute across independent instances:

```nginx
upstream iscc_search {
    server iscc-search-1:8000;
    server iscc-search-2:8000;
}
```

## Production Checklist

- [ ] Single worker configured (no `--workers`)
- [ ] Graceful shutdown timeouts set (60s uvicorn, 90s docker)
- [ ] Data volume mounted at `/data`
- [ ] CORS origins configured (not `*`)
- [ ] API secret set if needed
- [ ] Health checks passing
- [ ] Backup strategy tested
- [ ] Load tested with single worker

## Troubleshooting

### Index Corruption

**Symptoms:** Inconsistent results, startup errors **Cause:** Multiple workers or SIGKILL shutdown **Fix:**
Restore from backup, verify single worker config

### Slow Shutdown

**Symptoms:** Docker stop takes full grace period **Cause:** Large indexes need time to save **Fix:** Increase
`stop_grace_period` in docker-compose.yml

### Out of Memory

**Symptoms:** Container killed, OOM errors **Cause:** Indexes too large for allocated memory **Fix:** Increase
memory limits or shard across instances

## Monitoring

Watch logs for:

- `Saved NphdIndex for unit_type` - index saved successfully
- `Closed simprint index` - clean shutdown
- `MapFullError` - LMDB auto-resize (normal)

## Sandbox Deployment (search.iscc.id)

The sandbox runs on a single Ubuntu 22.04 VPS (2 CPU, 4 GB RAM, 78 GB disk) behind a Caddy reverse
proxy that handles TLS automatically.

**Server layout:**

```
/opt/iscc-search/
  docker-compose.yml    # Service definition
  data/                 # Bind-mounted index data (/data inside container)
/opt/caddy/
  docker-compose.yml    # Caddy reverse proxy (auto-TLS for search.iscc.id)
```

**Compose file (`/opt/iscc-search/docker-compose.yml`):**

```yaml
services:
  iscc-search:
    image: ghcr.io/iscc/iscc-search:develop
    container_name: iscc-search
    restart: unless-stopped
    volumes:
      - ./data:/data
    networks:
      - caddy
    labels:
      caddy: search.iscc.id
      caddy.reverse_proxy: "{{upstreams 8000}}"
      caddy.encode: gzip
    environment:
      ISCC_SEARCH_INDEX_URI: "usearch:///data"
      ISCC_SEARCH_CORS_ORIGINS: "*"
      ISCC_SEARCH_FLUSH_INTERVAL: "500000"
      ISCC_SEARCH_SHARD_SIZE_UNITS: "256"
      ISCC_SEARCH_SHARD_SIZE_SIMPRINTS: "256"
    stop_grace_period: 90s
    healthcheck:
      test:
        [
          "CMD",
          "python",
          "-c",
          "import urllib.request; urllib.request.urlopen('http://localhost:8000/').read()",
        ]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 3G
        reservations:
          cpus: "1.0"
          memory: 2G

networks:
  caddy:
    external: true
    name: caddy
```

**CI/CD:** Pushes to the `develop` branch trigger the Docker Build and Publish workflow, which runs
tests, builds the image, and publishes to `ghcr.io/iscc/iscc-search:develop`.

**Updating the sandbox:**

```bash
ssh root@search.iscc.id
cd /opt/iscc-search
docker compose pull
docker compose up -d
docker compose logs -f
```

**Wiping indexes for a fresh start:**

```bash
cd /opt/iscc-search
docker compose down
rm -rf data/*
docker compose up -d
```
