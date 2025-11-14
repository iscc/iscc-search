# syntax=docker/dockerfile:1

# Multi-stage build for minimal production image
# Stage 1: Builder - creates the virtual environment
FROM python:3.12-slim AS builder

# Install uv - version pinned for reproducibility
COPY --from=ghcr.io/astral-sh/uv:0.5.15 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

# Disable uv cache since we won't mount it in production
ENV UV_NO_CACHE=1

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies only (not the project itself yet)
# This layer is cached unless dependencies change
RUN uv sync --locked --no-install-project --no-dev

# Copy the rest of the application code
COPY . .

# Install the project as non-editable
# This creates compiled bytecode and optimized install
RUN uv sync --locked --no-editable --no-dev


# Stage 2: Runtime - minimal production image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy only the virtual environment from builder
# This excludes source code, build files, and other artifacts
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --from=builder /app/iscc_search /app/iscc_search

# Copy OpenAPI schema files (needed by the app)
COPY --from=builder /app/iscc_search/openapi /app/iscc_search/openapi

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose FastAPI default port
EXPOSE 8000

# Create directory for LMDB data persistence
RUN mkdir -p /data

# Set default index URI to use /data volume for persistence
ENV ISCC_SEARCH_INDEX_URI=usearch:///data

# Run uvicorn in production mode with single worker (REQUIRED for data integrity)
# - Single worker only: usearch indexes have no multi-process coordination
# - Graceful shutdown: 60s timeout allows large indexes to save cleanly
# - Concurrency: FastAPI async/await handles concurrent connections with single worker
CMD ["uvicorn", "iscc_search.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--timeout-graceful-shutdown", "60"]
