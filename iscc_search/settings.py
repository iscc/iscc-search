"""
Server runtime settings for ISCC-Search API.

**PURPOSE**: Configure the FastAPI server deployment via environment variables.

**SCOPE**: Single index backend per server deployment. This is used exclusively
by the API server (iscc-search serve) to configure runtime behavior.

**USAGE**:
- Deploying the API server with uvicorn
- Configuring server behavior (CORS, authentication)
- Setting which index backend to use (memory://, lmdb://, usearch://)

**CONFIGURATION SOURCE**: Environment variables with ISCC_SEARCH_ prefix

**NOT FOR**: CLI multi-index workflows. See iscc_search.config for that.

---

**RELATIONSHIP WITH config.py**:

iscc-search has TWO independent configuration systems:

1. **settings.py (this file)** - Server deployment configuration
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
settings.py while CLI data commands use config.py.

---

Provides configuration management using Pydantic settings with support for:
- Environment variables with ISCC_SEARCH_ prefix
- .env file loading
- Runtime settings override
- Type validation and defaults
"""

from pathlib import Path
from urllib.parse import urlparse
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import iscc_search


__all__ = [
    "SearchSettings",
    "search_settings",
    "get_index",
]


class SearchSettings(BaseSettings):
    """
    Application settings for ISCC-Search.

    Settings can be configured via:
    - Environment variables (prefixed with ISCC_SEARCH_)
    - .env file in the working directory
    - Direct instantiation with parameters
    - Runtime override using the override() method

    Attributes:
        index_uri: URI specifying index backend and location. Supported schemes:
                   - memory:// → In-memory index (no persistence)
                   - lmdb:///path → LMDB index at directory path
                   - usearch:///path → Usearch index with HNSW + LMDB (high-performance)
                   - postgres:///connection → PostgreSQL index (planned)
        api_secret: Optional API secret for authentication (if unset, API is public)
        cors_origins: List of allowed CORS origins (default: ["*"] for all origins)
    """

    index_uri: str = Field(
        f"usearch:///{Path(iscc_search.dirs.user_data_dir).as_posix()}",
        description="URI specifying index backend (memory://, lmdb://, usearch://, postgres://)",
    )

    api_secret: str | None = Field(
        None,
        description="Optional API secret for authentication (if unset, API is public)",
    )

    cors_origins: list[str] = Field(
        ["*"],
        description="CORS allowed origins (comma-separated in env var, or '*' for all)",
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
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        validate_assignment=True,
    )

    def override(self, update=None):
        # type: (dict|None) -> SearchSettings
        """
        Returns an updated and validated deep copy of the current settings instance.

        :param update: Dictionary of field names and values to override.
        :return: New SearchSettings instance with updated and validated fields.
        """

        update = update or {}  # sets {} if update is None

        settings = self.model_copy(deep=True)
        # We need update fields individually so validation gets triggered
        for field, value in update.items():
            setattr(settings, field, value)
        return settings


search_settings = SearchSettings()


def get_index():
    # type: () -> IsccIndexProtocol
    """
    Factory function to create index from settings.

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

    uri = search_settings.index_uri

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
