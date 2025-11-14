"""
CLI multi-index configuration management for ISCC-Search.

**PURPOSE**: Manage multiple named indexes for CLI workflows with persistent configuration.

**SCOPE**: Multi-index management for CLI commands. Allows users to work with multiple
indexes (local and remote) and switch between them using an "active" index concept.

**USAGE**:
- CLI commands: add, search, get
- Managing multiple indexes: iscc-search index add/list/use/remove
- Working with remote ISCC-Search servers
- Multi-environment workflows (dev/staging/production)

**CONFIGURATION SOURCE**: Persistent JSON file at ~/.iscc-search/config.json

**NOT FOR**: Server deployment configuration. See iscc_search.settings for that.

---

**RELATIONSHIP WITH settings.py**:

iscc-search has TWO independent configuration systems:

1. **settings.py** - Server deployment configuration
   - Consumer: API server (iscc-search serve)
   - Source: Environment variables (ISCC_SEARCH_*)
   - Scope: Single index per deployment
   - Pattern: 12-factor app principles

2. **config.py (this file)** - CLI multi-index management
   - Consumer: CLI commands (add, search, get)
   - Source: Persistent JSON file (~/.iscc-search/config.json)
   - Scope: Multiple named indexes with "active" concept
   - Pattern: Git-like workflow (add/list/use/remove)

These systems are SEPARATE and serve different purposes. The serve command uses
settings.py while CLI data commands use config.py.

---

Features:
- Register multiple indexes (local or remote)
- Track active index used by default for all commands
- Auto-discover existing local indexes
- CRUD operations on index configurations
- JSON persistence at ~/.iscc-search/config.json
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import iscc_search
from loguru import logger

if TYPE_CHECKING:
    from iscc_search.protocols.index import IsccIndexProtocol  # noqa: F401


__all__ = [
    "IndexConfig",
    "LocalIndexConfig",
    "RemoteIndexConfig",
    "Config",
    "ConfigManager",
    "get_config_manager",
]


class IndexConfig:
    """Base class for index configurations."""

    def __init__(self, name, type_):
        # type: (str, str) -> None
        """
        Initialize index configuration.

        :param name: Index name (unique identifier)
        :param type_: Index type ("local" or "remote")
        """
        self.name = name
        self.type = type_

    def to_dict(self):
        # type: () -> dict
        """
        Convert to dictionary for JSON serialization.

        :return: Dictionary representation
        """
        return {"type": self.type}

    @staticmethod
    def from_dict(name, data):
        # type: (str, dict) -> IndexConfig
        """
        Create IndexConfig from dictionary.

        :param name: Index name
        :param data: Dictionary with config data
        :return: IndexConfig instance (LocalIndexConfig or RemoteIndexConfig)
        """
        type_ = data.get("type")
        if type_ == "local":
            return LocalIndexConfig(name=name, path=data.get("path"))
        elif type_ == "remote":
            return RemoteIndexConfig(
                name=name,
                url=data.get("url"),
                api_key=data.get("api_key"),
            )
        else:
            raise ValueError(f"Unknown index type: {type_}")


class LocalIndexConfig(IndexConfig):
    """Configuration for local index."""

    def __init__(self, name, path=None):
        # type: (str, str|None) -> None
        """
        Initialize local index configuration.

        :param name: Index name
        :param path: Path to index directory (defaults to data dir)
        """
        super().__init__(name, "local")
        self.path = path or str(Path(iscc_search.dirs.user_data_dir).as_posix())

    def to_dict(self):
        # type: () -> dict
        """
        Convert to dictionary for JSON serialization.

        :return: Dictionary with type and path
        """
        return {"type": self.type, "path": self.path}


class RemoteIndexConfig(IndexConfig):
    """Configuration for remote index."""

    def __init__(self, name, url, api_key=None):
        # type: (str, str, str|None) -> None
        """
        Initialize remote index configuration.

        :param name: Index name (also used as target index name on server)
        :param url: Base URL of remote server (e.g., "https://api.example.com")
        :param api_key: Optional API key for authentication
        """
        super().__init__(name, "remote")
        self.url = url.rstrip("/")  # Normalize URL
        self.api_key = api_key

    def to_dict(self):
        # type: () -> dict
        """
        Convert to dictionary for JSON serialization.

        :return: Dictionary with type, url, and optional api_key
        """
        result = {"type": self.type, "url": self.url}
        if self.api_key:
            result["api_key"] = self.api_key
        return result


class Config:
    """Configuration data container."""

    def __init__(self, active_index=None, indexes=None):
        # type: (str|None, dict[str, IndexConfig]|None) -> None
        """
        Initialize configuration.

        :param active_index: Name of currently active index
        :param indexes: Dictionary mapping index names to IndexConfig instances
        """
        self.active_index = active_index
        self.indexes = indexes or {}

    def to_dict(self):
        # type: () -> dict
        """
        Convert to dictionary for JSON serialization.

        :return: Dictionary with active_index and indexes
        """
        return {
            "active_index": self.active_index,
            "indexes": {name: cfg.to_dict() for name, cfg in self.indexes.items()},
        }

    @staticmethod
    def from_dict(data):
        # type: (dict) -> Config
        """
        Create Config from dictionary.

        :param data: Dictionary with config data
        :return: Config instance
        """
        active_index = data.get("active_index")
        indexes_data = data.get("indexes", {})
        indexes = {name: IndexConfig.from_dict(name, cfg) for name, cfg in indexes_data.items()}
        return Config(active_index=active_index, indexes=indexes)


class ConfigManager:
    """Manager for persistent configuration."""

    def __init__(self, config_path=None):
        # type: (str|Path|None) -> None
        """
        Initialize configuration manager.

        :param config_path: Path to config file (defaults to ~/.iscc-search/config.json)
        """
        if config_path is None:
            config_path = Path(iscc_search.dirs.user_data_dir) / "config.json"
        self.config_path = Path(config_path)
        self._config = None  # type: Config|None

    def load(self):
        # type: () -> Config
        """
        Load configuration from file.

        If config file doesn't exist, creates default config with "default" local index.
        Auto-discovers existing local indexes in data directory.

        :return: Config instance
        """
        if not self.config_path.exists():
            logger.info(f"No config found at {self.config_path}, creating default configuration")
            self._config = self._create_default_config()
            self.save()
            return self._config

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._config = Config.from_dict(data)

            # Auto-discover local indexes
            self._auto_discover_local_indexes()

            return self._config
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            logger.info("Creating default configuration")
            self._config = self._create_default_config()
            self.save()
            return self._config

    def save(self):
        # type: () -> None
        """
        Save configuration to file.

        Creates config directory if it doesn't exist.
        """
        if self._config is None:
            raise ValueError("No config loaded")

        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self._config.to_dict(), f, indent=2)

        logger.debug(f"Saved config to {self.config_path}")

    def get_active(self):
        # type: () -> IndexConfig|None
        """
        Get active index configuration.

        :return: Active IndexConfig or None if no active index
        """
        if self._config is None:
            self.load()

        if self._config.active_index is None:
            return None

        return self._config.indexes.get(self._config.active_index)

    def set_active(self, name):
        # type: (str) -> None
        """
        Set active index by name.

        :param name: Index name to set as active
        :raises KeyError: If index name not found in configuration
        """
        if self._config is None:
            self.load()

        if name not in self._config.indexes:
            raise KeyError(f"Index '{name}' not found in configuration")

        self._config.active_index = name
        self.save()
        logger.info(f"Set active index to '{name}'")

    def add_index(self, index_config):
        # type: (IndexConfig) -> None
        """
        Add or update index configuration.

        If index with same name exists, it will be replaced.

        :param index_config: IndexConfig instance to add
        """
        if self._config is None:
            self.load()

        self._config.indexes[index_config.name] = index_config

        # If this is the first index, make it active
        if self._config.active_index is None:
            self._config.active_index = index_config.name

        self.save()
        logger.info(f"Added index '{index_config.name}' ({index_config.type})")

    def remove_index(self, name):
        # type: (str) -> None
        """
        Remove index from configuration.

        If removing active index, sets active to None.

        :param name: Index name to remove
        :raises KeyError: If index name not found
        """
        if self._config is None:
            self.load()

        if name not in self._config.indexes:
            raise KeyError(f"Index '{name}' not found in configuration")

        del self._config.indexes[name]

        # If removing active index, clear active
        if self._config.active_index == name:
            # Set new active index to first available, or None if no indexes left
            if self._config.indexes:
                self._config.active_index = next(iter(self._config.indexes))
                logger.info(f"Active index changed to '{self._config.active_index}'")
            else:
                self._config.active_index = None
                logger.info("No indexes remaining")

        self.save()
        logger.info(f"Removed index '{name}' from configuration")

    def list_indexes(self):
        # type: () -> list[tuple[str, IndexConfig, bool]]
        """
        List all configured indexes.

        :return: List of tuples (name, IndexConfig, is_active)
        """
        if self._config is None:
            self.load()

        result = []
        for name, cfg in self._config.indexes.items():
            is_active = name == self._config.active_index
            result.append((name, cfg, is_active))

        return result

    def _create_default_config(self):
        # type: () -> Config
        """
        Create default configuration with "default" local index.

        :return: Config instance with default settings
        """
        default_index = LocalIndexConfig(name="default")
        return Config(active_index="default", indexes={"default": default_index})

    def _auto_discover_local_indexes(self):
        # type: () -> None
        """
        Auto-discover existing local indexes in data directory.

        Scans for index.lmdb files and registers them if not already in config.
        Does not override existing configurations.
        """
        data_dir = Path(iscc_search.dirs.user_data_dir)

        if not data_dir.exists():
            return

        # Scan for index.lmdb files
        for lmdb_path in data_dir.rglob("index.lmdb"):
            index_dir = lmdb_path.parent
            index_name = index_dir.name if index_dir != data_dir else "default"

            # Skip if already configured
            if index_name in self._config.indexes:
                continue

            # Register discovered index
            logger.debug(f"Auto-discovered local index '{index_name}' at {index_dir}")
            discovered_config = LocalIndexConfig(name=index_name, path=str(index_dir.as_posix()))
            self._config.indexes[index_name] = discovered_config


# Singleton instance
_config_manager = None  # type: ConfigManager|None


def get_config_manager():
    # type: () -> ConfigManager
    """
    Get singleton ConfigManager instance.

    :return: ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
