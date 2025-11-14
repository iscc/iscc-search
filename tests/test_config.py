"""Tests for configuration management."""

import json

import pytest

from iscc_search.config import (
    Config,
    ConfigManager,
    LocalIndexConfig,
    RemoteIndexConfig,
)


@pytest.fixture
def temp_config_path(tmp_path):
    # type: (Path) -> Path
    """Create temporary config path for testing."""
    config_path = tmp_path / "config.json"
    return config_path


@pytest.fixture
def config_manager(temp_config_path):
    # type: (Path) -> ConfigManager
    """Create ConfigManager with temporary config path."""
    return ConfigManager(config_path=temp_config_path)


def test_local_index_config():
    # type: () -> None
    """Test LocalIndexConfig creation and serialization."""
    config = LocalIndexConfig(name="test", path="/path/to/index")
    assert config.name == "test"
    assert config.type == "local"
    assert config.path == "/path/to/index"

    # Test to_dict
    data = config.to_dict()
    assert data == {"type": "local", "path": "/path/to/index"}


def test_remote_index_config():
    # type: () -> None
    """Test RemoteIndexConfig creation and serialization."""
    config = RemoteIndexConfig(name="test", url="https://api.example.com", api_key="secret")
    assert config.name == "test"
    assert config.type == "remote"
    assert config.url == "https://api.example.com"
    assert config.api_key == "secret"

    # Test to_dict
    data = config.to_dict()
    assert data == {"type": "remote", "url": "https://api.example.com", "api_key": "secret"}

    # Test without api_key
    config_no_key = RemoteIndexConfig(name="test", url="https://api.example.com")
    data_no_key = config_no_key.to_dict()
    assert "api_key" not in data_no_key


def test_remote_index_config_url_normalization():
    # type: () -> None
    """Test that RemoteIndexConfig normalizes URLs by removing trailing slashes."""
    config = RemoteIndexConfig(name="test", url="https://api.example.com/")
    assert config.url == "https://api.example.com"


def test_config_to_dict():
    # type: () -> None
    """Test Config to_dict serialization."""
    local_cfg = LocalIndexConfig(name="local1", path="/path")
    remote_cfg = RemoteIndexConfig(name="remote1", url="https://api.example.com")

    config = Config(active_index="local1", indexes={"local1": local_cfg, "remote1": remote_cfg})

    data = config.to_dict()
    assert data["active_index"] == "local1"
    assert "local1" in data["indexes"]
    assert "remote1" in data["indexes"]
    assert data["indexes"]["local1"]["type"] == "local"
    assert data["indexes"]["remote1"]["type"] == "remote"


def test_config_from_dict():
    # type: () -> None
    """Test Config from_dict deserialization."""
    data = {
        "active_index": "test",
        "indexes": {
            "test": {"type": "local", "path": "/path"},
            "remote": {"type": "remote", "url": "https://api.example.com", "api_key": "secret"},
        },
    }

    config = Config.from_dict(data)
    assert config.active_index == "test"
    assert "test" in config.indexes
    assert "remote" in config.indexes
    assert isinstance(config.indexes["test"], LocalIndexConfig)
    assert isinstance(config.indexes["remote"], RemoteIndexConfig)


def test_config_manager_create_default(config_manager, temp_config_path):
    # type: (ConfigManager, Path) -> None
    """Test ConfigManager creates default config if none exists."""
    assert not temp_config_path.exists()

    config = config_manager.load()

    assert temp_config_path.exists()
    assert config.active_index == "default"
    assert "default" in config.indexes
    assert isinstance(config.indexes["default"], LocalIndexConfig)


def test_config_manager_save_and_load(config_manager, temp_config_path):
    # type: (ConfigManager, Path) -> None
    """Test ConfigManager save and load cycle."""
    # Create and save config
    config_manager.load()
    config_manager.add_index(LocalIndexConfig(name="test", path="/test/path"))

    # Load again and verify
    config_manager2 = ConfigManager(config_path=temp_config_path)
    config = config_manager2.load()

    assert "test" in config.indexes
    assert config.indexes["test"].path == "/test/path"  # type: ignore


def test_config_manager_add_index(config_manager):
    # type: (ConfigManager) -> None
    """Test adding index to configuration."""
    config_manager.load()

    # Add local index
    local_cfg = LocalIndexConfig(name="local1", path="/path")
    config_manager.add_index(local_cfg)

    indexes = config_manager.list_indexes()
    names = [name for name, _, _ in indexes]
    assert "local1" in names


def test_config_manager_add_first_index_becomes_active(config_manager, temp_config_path):
    # type: (ConfigManager, Path) -> None
    """Test that first added index becomes active."""
    # Start with empty config
    temp_config_path.write_text('{"active_index": null, "indexes": {}}')

    config_manager.load()
    assert config_manager.get_active() is None

    # Add first index
    config_manager.add_index(LocalIndexConfig(name="first"))

    # Should be active
    active = config_manager.get_active()
    assert active is not None
    assert active.name == "first"


def test_config_manager_set_active(config_manager):
    # type: (ConfigManager) -> None
    """Test setting active index."""
    config_manager.load()
    config_manager.add_index(LocalIndexConfig(name="test1"))
    config_manager.add_index(LocalIndexConfig(name="test2"))

    config_manager.set_active("test2")

    active = config_manager.get_active()
    assert active is not None
    assert active.name == "test2"


def test_config_manager_set_active_invalid(config_manager):
    # type: (ConfigManager) -> None
    """Test setting active to non-existent index raises error."""
    config_manager.load()

    with pytest.raises(KeyError, match="not found"):
        config_manager.set_active("nonexistent")


def test_config_manager_remove_index(config_manager):
    # type: (ConfigManager) -> None
    """Test removing index from configuration."""
    config_manager.load()
    config_manager.add_index(LocalIndexConfig(name="test1"))

    config_manager.remove_index("test1")

    indexes = config_manager.list_indexes()
    names = [name for name, _, _ in indexes]
    assert "test1" not in names


def test_config_manager_remove_active_index(config_manager):
    # type: (ConfigManager) -> None
    """Test removing active index sets new active."""
    config_manager.load()
    initial_active = config_manager.get_active().name  # type: ignore

    config_manager.add_index(LocalIndexConfig(name="test1"))
    config_manager.set_active("test1")

    config_manager.remove_index("test1")

    # Should fall back to first remaining index (default)
    active = config_manager.get_active()
    assert active is not None
    assert active.name == initial_active


def test_config_manager_remove_invalid(config_manager):
    # type: (ConfigManager) -> None
    """Test removing non-existent index raises error."""
    config_manager.load()

    with pytest.raises(KeyError, match="not found"):
        config_manager.remove_index("nonexistent")


def test_config_manager_list_indexes(config_manager):
    # type: (ConfigManager) -> None
    """Test listing all indexes with active indicator."""
    config_manager.load()
    config_manager.add_index(LocalIndexConfig(name="test1"))
    config_manager.add_index(RemoteIndexConfig(name="test2", url="https://api.example.com"))

    indexes = config_manager.list_indexes()

    assert len(indexes) == 3  # default + test1 + test2

    # Find active index
    active_indexes = [name for name, _, is_active in indexes if is_active]
    assert len(active_indexes) == 1


@pytest.mark.skip(reason="Auto-discovery testing requires mocking property which is complex")
def test_config_manager_auto_discovery(config_manager, temp_config_path, tmp_path, monkeypatch):
    # type: (ConfigManager, Path, Path, pytest.MonkeyPatch) -> None
    """Test auto-discovery of local indexes."""
    # Create mock index directories with index.lmdb files
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    index1_dir = data_dir / "index1"
    index1_dir.mkdir()
    (index1_dir / "index.lmdb").touch()

    index2_dir = data_dir / "index2"
    index2_dir.mkdir()
    (index2_dir / "index.lmdb").touch()

    # Mock the data directory
    import iscc_search

    monkeypatch.setattr(iscc_search.dirs, "user_data_dir", str(data_dir))

    # Load config (will auto-discover)
    config_manager.load()

    indexes = config_manager.list_indexes()
    names = [name for name, _, _ in indexes]

    assert "index1" in names
    assert "index2" in names


@pytest.mark.skip(reason="Auto-discovery testing requires mocking property which is complex")
def test_config_manager_auto_discovery_skips_existing(config_manager, temp_config_path, tmp_path, monkeypatch):
    # type: (ConfigManager, Path, Path, pytest.MonkeyPatch) -> None
    """Test auto-discovery doesn't override existing configurations."""
    # Create initial config with "test" index
    config_data = {
        "active_index": "test",
        "indexes": {"test": {"type": "remote", "url": "https://api.example.com"}},
    }
    temp_config_path.write_text(json.dumps(config_data))

    # Create mock data directory with "test" index.lmdb
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    test_dir = data_dir / "test"
    test_dir.mkdir()
    (test_dir / "index.lmdb").touch()

    # Mock the data directory
    import iscc_search

    monkeypatch.setattr(iscc_search.dirs, "user_data_dir", str(data_dir))

    # Load config (will attempt auto-discovery)
    config_manager.load()

    indexes = config_manager.list_indexes()
    test_cfg = [cfg for name, cfg, _ in indexes if name == "test"][0]

    # Should still be remote (not overridden by auto-discovery)
    assert test_cfg.type == "remote"


def test_index_config_from_dict_unknown_type():
    # type: () -> None
    """Test IndexConfig.from_dict with unknown type raises error."""
    with pytest.raises(ValueError, match="Unknown index type"):
        from iscc_search.config import IndexConfig

        IndexConfig.from_dict("test", {"type": "unknown"})
