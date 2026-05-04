"""Configuration options for ISCC-Search.

Provides configuration management using Pydantic settings with support for:
- Environment variables with ISCC_SEARCH_ prefix
- .env file loading
- Runtime settings override
- Type validation and defaults

**RELATIONSHIP WITH config.py**:

iscc-search has TWO independent configuration systems:

1. **options.py (this file)** - Server deployment configuration
   - Consumer: API server (iscc-search serve)
   - Source: Environment variables (ISCC_SEARCH_*)
   - Scope: Single index per deployment
   - Pattern: 12-factor app principles

2. **config.py** - CLI multi-index management
   - Consumer: CLI commands (add, search, get)
   - Source: Persistent JSON file (~/.iscc-search/config.json)
   - Scope: Multiple named indexes with "active" concept
   - Pattern: Git-like workflow (add/list/use/remove)

These systems are SEPARATE and serve different purposes. The serve command uses
options.py while CLI data commands use config.py.
"""

from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import iscc_search

load_dotenv()

__all__ = [
    "SearchOptions",
    "search_opts",
    "get_index",
]


class SearchOptions(BaseSettings):
    """
    Application options for ISCC-Search.

    Options can be configured via:
    - Environment variables (prefixed with ISCC_SEARCH_)
    - .env file in the working directory
    - Direct instantiation with parameters
    - Runtime override using the override() method
    """

    index_uri: str = Field(
        f"usearch:///{Path(iscc_search.dirs.user_data_dir).as_posix()}",
        description="ISCC_SEARCH_INDEX_URI - URI specifying index backend (memory://, lmdb://, usearch://)",
    )

    api_secret: str | None = Field(
        None,
        description="ISCC_SEARCH_API_SECRET - Optional API secret for authentication (if unset, API is public)",
    )

    cors_origins: str = Field(
        "*",
        description="ISCC_SEARCH_CORS_ORIGINS - CORS allowed origins (comma-separated, or '*' for all)",
    )

    host: str = Field(
        "0.0.0.0",
        description="ISCC_SEARCH_HOST - Host to bind server to",
    )

    port: int = Field(
        8000,
        description="ISCC_SEARCH_PORT - Port to bind server to",
    )

    workers: int | None = Field(
        None,
        description="ISCC_SEARCH_WORKERS - Number of worker processes (production only)",
    )

    # Shard sizes (in MB)
    shard_size_units: int = Field(
        512,
        ge=1,
        description="ISCC_SEARCH_SHARD_SIZE_UNITS - Maximum shard file size for unit indexes in MB",
    )

    shard_size_simprints: int = Field(
        512,
        ge=1,
        description="ISCC_SEARCH_SHARD_SIZE_SIMPRINTS - Maximum shard file size for simprint indexes in MB",
    )

    # HNSW parameters for unit indexes
    hnsw_expansion_add_units: int = Field(
        128,
        ge=1,
        description="ISCC_SEARCH_HNSW_EXPANSION_ADD_UNITS - Build-time search depth for unit HNSW indexes",
    )

    hnsw_expansion_search_units: int = Field(
        64,
        ge=1,
        description="ISCC_SEARCH_HNSW_EXPANSION_SEARCH_UNITS - Query-time search depth for unit HNSW indexes",
    )

    hnsw_connectivity_units: int = Field(
        16,
        ge=1,
        description="ISCC_SEARCH_HNSW_CONNECTIVITY_UNITS - Graph connectivity (M) for unit HNSW indexes",
    )

    # HNSW parameters for simprint indexes
    hnsw_expansion_add_simprints: int = Field(
        16,
        ge=1,
        description="ISCC_SEARCH_HNSW_EXPANSION_ADD_SIMPRINTS - Build-time search depth for simprint HNSW indexes",
    )

    hnsw_expansion_search_simprints: int = Field(
        512,
        ge=1,
        description="ISCC_SEARCH_HNSW_EXPANSION_SEARCH_SIMPRINTS - Query-time search depth for simprint HNSW indexes",
    )

    hnsw_connectivity_simprints: int = Field(
        8,
        ge=1,
        description="ISCC_SEARCH_HNSW_CONNECTIVITY_SIMPRINTS - Graph connectivity (M) for simprint HNSW indexes",
    )

    # Match thresholds
    match_threshold_units: float = Field(
        0.75,
        ge=0.0,
        le=1.0,
        description="ISCC_SEARCH_MATCH_THRESHOLD_UNITS - Minimum score for unit similarity matches (0.0-1.0)",
    )

    match_threshold_simprints: float = Field(
        0.75,
        ge=0.0,
        le=1.0,
        description="ISCC_SEARCH_MATCH_THRESHOLD_SIMPRINTS - Minimum score for simprint matches (0.0-1.0)",
    )

    # Scoring
    confidence_exponent: int = Field(
        4,
        ge=1,
        description="ISCC_SEARCH_CONFIDENCE_EXPONENT - Exponent for confidence-weighted score aggregation",
    )

    oversampling_factor: int = Field(
        20,
        ge=1,
        description="ISCC_SEARCH_OVERSAMPLING_FACTOR - Oversampling multiplier for simprint search diversity",
    )

    # Flush control
    flush_interval: int = Field(
        100000,
        ge=0,
        description="ISCC_SEARCH_FLUSH_INTERVAL - Auto-flush sub-indexes after N dirty key "
        "mutations (0 = disabled). Only safe with a single writer process.",
    )

    # Logging
    log_level: str = Field(
        "info",
        description="ISCC_SEARCH_LOG_LEVEL - Log level for the server (debug, info, warning, error, critical)",
    )

    # Error tracking
    sentry_dsn: str | None = Field(
        None,
        description="ISCC_SEARCH_SENTRY_DSN - Sentry DSN for error tracking (disabled when unset)",
    )

    sentry_traces_sample_rate: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="ISCC_SEARCH_SENTRY_TRACES_SAMPLE_RATE - Sentry performance sampling rate (0.0-1.0)",
    )

    @property
    def cors_origins_list(self):
        # type: () -> list[str]
        """
        Split comma-separated CORS origins string into a list.

        :return: List of allowed origin strings
        """
        return [origin.strip() for origin in self.cors_origins.split(",")]

    model_config = SettingsConfigDict(
        env_prefix="ISCC_SEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_assignment=True,
    )

    def override(self, update=None):
        # type: (dict|None) -> SearchOptions
        """
        Return an updated and validated deep copy of the current options instance.

        :param update: Dictionary of field names and values to override.
        :return: New SearchOptions instance with updated and validated fields.
        """

        update = update or {}  # sets {} if update is None

        options = self.model_copy(deep=True)
        # We need update fields individually so validation gets triggered
        for field, value in update.items():
            setattr(options, field, value)
        return options


search_opts = SearchOptions()


def get_index():
    # type: () -> IsccIndexProtocol
    """
    Factory function to create index from options.

    Parses index_uri to determine index type and returns appropriate
    implementation. Supported URI schemes:
    - memory:// → MemoryIndex (in-memory, no persistence)
    - lmdb:///path → LmdbIndexManager (LMDB-backed, production-ready)
    - usearch:///path → UsearchIndexManager (HNSW + LMDB, high-performance)

    :return: Index instance implementing IsccIndexProtocol
    :raises ValueError: If URI scheme is not supported or missing
    """
    from iscc_search.protocols.index import IsccIndexProtocol  # noqa: F401
    import sys

    uri = search_opts.index_uri

    # Handle memory:// scheme (no path needed)
    if uri == "memory://" or uri.startswith("memory://"):
        from iscc_search.indexes.memory import MemoryIndex

        return MemoryIndex()

    # Require explicit URI scheme (reject plain paths)
    if "://" not in uri:
        supported = ["memory://", "lmdb:///path", "usearch:///path"]
        raise ValueError(
            f"ISCC_SEARCH_INDEX_URI requires explicit scheme, got: '{uri}'. Supported schemes: {', '.join(supported)}"
        )

    # Parse as URI
    parsed = urlparse(uri)

    # Handle lmdb:// scheme
    if parsed.scheme == "lmdb":
        from iscc_search.indexes.lmdb import LmdbIndexManager

        # Handle path normalization after URI parsing
        path = parsed.path
        if sys.platform == "win32" and path.startswith("/") and len(path) > 2 and path[2] == ":":
            path = path[1:]  # pragma: no cover - Remove leading '/' from '/C:/path' on Windows
        elif path.startswith("//"):  # pragma: no cover - Unix-specific path handling
            # Unix paths with double leading slash from URIs like lmdb:////tmp/path
            path = path[1:]  # Strip one leading '/' to normalize path

        return LmdbIndexManager(path)

    # Handle usearch:// scheme
    if parsed.scheme == "usearch":
        from iscc_search.indexes.usearch import UsearchIndexManager

        # Handle path normalization after URI parsing
        path = parsed.path
        if sys.platform == "win32" and path.startswith("/") and len(path) > 2 and path[2] == ":":
            path = path[1:]  # pragma: no cover - Remove leading '/' from '/C:/path' on Windows
        elif path.startswith("//"):  # pragma: no cover - Unix-specific path handling
            # Unix paths with double leading slash from URIs like usearch:////tmp/path
            path = path[1:]  # Strip one leading '/' to normalize path

        return UsearchIndexManager(path)

    # Reject unsupported URI schemes
    supported = ["memory://", "lmdb://", "usearch://"]
    raise ValueError(f"Unsupported ISCC_SEARCH_INDEX_URI scheme: '{uri}'. Supported schemes: {', '.join(supported)}.")
