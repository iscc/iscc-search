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
from pydantic import Field, field_validator
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
        description="ISCC_SEARCH_INDEX_URI - URI specifying index backend (memory://, lmdb://, usearch://, postgres://)",
    )

    api_secret: str | None = Field(
        None,
        description="ISCC_SEARCH_API_SECRET - Optional API secret for authentication (if unset, API is public)",
    )

    cors_origins: list[str] = Field(
        ["*"],
        description="ISCC_SEARCH_CORS_ORIGINS - CORS allowed origins (comma-separated in env var, or '*' for all)",
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

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        # type: (str|list[str]) -> list[str]
        """
        Parse CORS origins from environment variable or config.

        Accepts either a list (from direct instantiation) or a comma-separated
        string (from ISCC_SEARCH_CORS_ORIGINS environment variable).

        :param v: CORS origins as list or comma-separated string
        :return: List of allowed origin strings
        """
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

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

    Future implementations:
    - postgresql:// → PostgresIndex (planned)

    :return: Index instance implementing IsccIndexProtocol
    :raises ValueError: If URI scheme is not supported or missing
    :raises NotImplementedError: If URI scheme is planned but not yet implemented
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
        supported = ["memory://", "lmdb:///path", "usearch:///path (planned)"]
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
    supported = ["memory://", "lmdb://", "usearch://", "postgres:// (planned)"]
    raise ValueError(
        f"Unsupported ISCC_SEARCH_INDEX_URI scheme: '{uri}'. "
        f"Supported schemes: {', '.join(supported)}. "
        f"PostgreSQL URIs are planned for future implementation."
    )
