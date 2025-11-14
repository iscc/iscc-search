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

| Variable                   | Default           | Required | Description             |
| -------------------------- | ----------------- | -------- | ----------------------- |
| `ISCC_SEARCH_INDEX_URI`    | `usearch:///data` | No       | Backend URI             |
| `ISCC_SEARCH_CORS_ORIGINS` | `*`               | No       | Comma-separated origins |
| `ISCC_SEARCH_API_SECRET`   | None              | No       | API authentication      |

**Example:**

```yaml
environment:
  - ISCC_SEARCH_INDEX_URI=usearch:///data
  - ISCC_SEARCH_CORS_ORIGINS=https://example.com
```

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
