"""
Settings management for ISCC-Search.

Provides configuration management using Pydantic settings with support for:
- Environment variables with ISCC_SEARCH_ prefix
- .env file loading
- Runtime settings override
- Type validation and defaults
"""

from urllib.parse import urlparse
from pydantic import Field
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
    """

    index_uri: str = Field(
        iscc_search.dirs.user_data_dir,
        description="URI specifying index backend (memory://, lmdb://, usearch://, postgres://)",
    )

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
    from iscc_search.protocol import IsccIndexProtocol  # noqa: F401
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

        # Handle Windows paths: urlparse('/C:/path') → need to strip leading '/'
        path = parsed.path
        if sys.platform == "win32" and path.startswith("/") and len(path) > 2 and path[2] == ":":
            path = path[1:]  # Remove leading '/' from '/C:/path'

        return LmdbIndexManager(path)

    # Handle usearch:// scheme
    if parsed.scheme == "usearch":
        from iscc_search.indexes.usearch import UsearchIndexManager

        # Handle Windows paths: urlparse('/C:/path') → need to strip leading '/'
        path = parsed.path
        if sys.platform == "win32" and path.startswith("/") and len(path) > 2 and path[2] == ":":
            path = path[1:]  # pragma: no cover - Windows-specific path handling

        return UsearchIndexManager(path)

    # Reject unsupported URI schemes
    supported = ["memory://", "lmdb://", "usearch://", "postgres:// (planned)"]
    raise ValueError(
        f"Unsupported ISCC_SEARCH_INDEX_URI scheme: '{uri}'. "
        f"Supported schemes: {', '.join(supported)}. "
        f"PostgreSQL URIs are planned for future implementation."
    )
