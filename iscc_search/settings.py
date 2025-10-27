"""
Settings management for ISCC-VDB.

Provides configuration management using Pydantic settings with support for:
- Environment variables with ISCC_VDB_ prefix
- .env file loading
- Runtime settings override
- Type validation and defaults
"""

from urllib.parse import urlparse
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import iscc_search


__all__ = [
    "VdbSettings",
    "vdb_settings",
    "get_index",
]


class VdbSettings(BaseSettings):
    """
    Application settings for ISCC-VDB.

    Settings can be configured via:
    - Environment variables (prefixed with ISCC_VDB_)
    - .env file in the working directory
    - Direct instantiation with parameters
    - Runtime override using the override() method

    Attributes:
        indexes_uri: Location where index data is stored (local file path, DSN, or URI scheme)
    """

    indexes_uri: str = Field(
        iscc_search.dirs.user_data_dir,
        description="Location where index data is stored (local file path or DSN)",
    )

    model_config = SettingsConfigDict(
        env_prefix="ISCC_VDB_",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        validate_assignment=True,
    )

    def override(self, update=None):
        # type: (dict|None) -> VdbSettings
        """
        Returns an updated and validated deep copy of the current settings instance.

        :param update: Dictionary of field names and values to override.
        :return: New VdbSettings instance with updated and validated fields.
        """

        update = update or {}  # sets {} if update is None

        settings = self.model_copy(deep=True)
        # We need update fields individually so validation gets triggered
        for field, value in update.items():
            setattr(settings, field, value)
        return settings


vdb_settings = VdbSettings()


def get_index():
    # type: () -> IsccIndexProtocol
    """
    Factory function to create index from settings.

    Parses indexes_uri to determine index type and returns appropriate
    implementation. Currently supports:
    - memory:// → MemoryIndex (in-memory, no persistence)
    - file:// or path → LmdbIndexManager (LMDB-backed, production-ready)

    Future implementations:
    - postgresql:// → PostgresIndex (not yet implemented)

    :return: Index instance implementing IsccIndexProtocol
    :raises ValueError: If URI scheme is not supported
    """
    from iscc_search.protocol import IsccIndexProtocol  # noqa: F401
    import os

    uri = vdb_settings.indexes_uri

    # Handle memory:// scheme
    if uri == "memory://" or uri.startswith("memory://"):
        from iscc_search.indexes.memory import MemoryIndex

        return MemoryIndex()

    # Detect file paths (Windows: C:\..., Unix: /..., relative: ...)
    # Check if it's an absolute path before parsing as URI
    if os.path.isabs(uri) or "://" not in uri:
        # It's a file path (absolute or relative)
        from iscc_search.indexes.lmdb import LmdbIndexManager

        return LmdbIndexManager(uri)

    # Parse as URI
    parsed = urlparse(uri)

    if parsed.scheme == "file":
        # file:// URI
        from iscc_search.indexes.lmdb import LmdbIndexManager

        return LmdbIndexManager(parsed.path)

    # Reject unsupported URI schemes to prevent silent data loss
    supported = ["memory://", "file path", "file://"]
    raise ValueError(
        f"Unsupported ISCC_VDB_INDEXES_URI: '{uri}'. "
        f"Currently supported schemes: {', '.join(supported)}. "
        f"PostgreSQL URIs are not yet implemented."
    )
