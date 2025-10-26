"""
Settings management for ISCC-VDB.

Provides configuration management using Pydantic settings with support for:
- Environment variables with ISCC_VDB_ prefix
- .env file loading
- Runtime settings override
- Type validation and defaults
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import iscc_vdb


__all__ = [
    "VdbSettings",
    "vdb_settings",
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
        iscc_vdb.dirs.user_data_dir,
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
