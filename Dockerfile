# syntax=docker/dockerfile:1

# Multi-stage build: installs a pre-built wheel (produced by CI) into a minimal runtime image.
# The wheel must exist at dist/*.whl before `docker build`. The CI workflow builds it via `uv build`
# and copies it into the build context. Building locally requires running `uv build` first.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.15 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_CACHE=1

# Install the pre-built wheel into an isolated venv.
COPY dist/*.whl /tmp/
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python /tmp/*.whl

# Pre-download iscc-sct model weights so container start is fast and offline-capable.
RUN /app/.venv/bin/python -c "import iscc_sct.code_semantic_text as sct; sct.model()"


FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /root/.local/share/iscc-sct /root/.local/share/iscc-sct

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

RUN mkdir -p /data
ENV ISCC_SEARCH_INDEX_URI=usearch:///data

# Single worker only: usearch indexes have no multi-process coordination.
#
# Shutdown timing: uvicorn drains in-flight requests for up to
# --timeout-graceful-shutdown seconds, THEN runs the FastAPI lifespan handler
# (which calls index.close() to flush HNSW shards). The lifespan handler is
# UNBOUNDED in uvicorn — only Docker's stop_grace_period can stop it.
#
# Therefore: compose stop_grace_period must be >= this drain timeout PLUS the
# expected flush duration. Default 60s drain assumes typical request latency
# under a minute; combine with a generous stop_grace_period (>=300s) on the
# compose side to leave room for the flush.
CMD ["uvicorn", "iscc_search.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--timeout-graceful-shutdown", "60", \
     "--log-config", "/app/.venv/lib/python3.12/site-packages/iscc_search/log_config.json"]
