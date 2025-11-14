"""Tests for CLI index management commands."""

import pytest

from iscc_search.cli.index import remove_command
from iscc_search.config import ConfigManager, LocalIndexConfig


@pytest.fixture
def temp_config_path(tmp_path):
    # type: (Path) -> Path
    """Create temporary config path for testing."""
    return tmp_path / "config.json"


@pytest.fixture
def config_manager(temp_config_path, monkeypatch):
    # type: (Path, pytest.MonkeyPatch) -> ConfigManager
    """Create ConfigManager with temporary config path and mock get_config_manager."""
    manager = ConfigManager(config_path=temp_config_path)

    # Mock get_config_manager to return our test config manager
    def mock_get_config_manager():
        # type: () -> ConfigManager
        return manager

    monkeypatch.setattr("iscc_search.cli.index.get_config_manager", mock_get_config_manager)
    return manager


@pytest.fixture
def mock_index_structure(tmp_path):
    # type: (Path) -> tuple[Path, Path, Path]
    """
    Create mock index directory structure.

    Returns base_path, index1_path, index2_path
    """
    base_path = tmp_path / "indexes"
    base_path.mkdir()

    # Create first index directory with files
    index1_path = base_path / "index1"
    index1_path.mkdir()
    (index1_path / "index.lmdb").mkdir()
    (index1_path / "CONTENT_TEXT_V0.usearch").touch()
    (index1_path / "DATA_NONE_V0.usearch").touch()

    # Create second index directory with files
    index2_path = base_path / "index2"
    index2_path.mkdir()
    (index2_path / "index.lmdb").mkdir()
    (index2_path / "SEMANTIC_TEXT_V0.usearch").touch()

    return base_path, index1_path, index2_path


def test_remove_command_with_delete_data_removes_correct_subdirectory(config_manager, mock_index_structure):
    # type: (ConfigManager, tuple[Path, Path, Path]) -> None
    """Test that remove --delete-data deletes only the target index subdirectory."""
    base_path, index1_path, index2_path = mock_index_structure

    # Register both indexes with same base path
    config_manager.load()
    config_manager.add_index(LocalIndexConfig(name="index1", path=str(base_path)))
    config_manager.add_index(LocalIndexConfig(name="index2", path=str(base_path)))

    # Verify both index directories exist
    assert index1_path.exists()
    assert index2_path.exists()
    assert (index1_path / "CONTENT_TEXT_V0.usearch").exists()
    assert (index2_path / "SEMANTIC_TEXT_V0.usearch").exists()

    # Remove index1 with delete_data
    remove_command(name="index1", delete_data=True)

    # Verify index1 directory is deleted
    assert not index1_path.exists()

    # Verify index2 directory still exists (not affected)
    assert index2_path.exists()
    assert (index2_path / "SEMANTIC_TEXT_V0.usearch").exists()

    # Verify base path still exists
    assert base_path.exists()


def test_remove_command_without_delete_data_keeps_files(config_manager, mock_index_structure):
    # type: (ConfigManager, tuple[Path, Path, Path]) -> None
    """Test that remove without --delete-data keeps index files."""
    base_path, index1_path, _ = mock_index_structure

    config_manager.load()
    config_manager.add_index(LocalIndexConfig(name="index1", path=str(base_path)))

    # Verify index exists
    assert index1_path.exists()

    # Remove without delete_data
    remove_command(name="index1", delete_data=False)

    # Verify index directory still exists
    assert index1_path.exists()
    assert (index1_path / "CONTENT_TEXT_V0.usearch").exists()


def test_remove_command_with_delete_data_handles_missing_directory(config_manager, tmp_path):
    # type: (ConfigManager, Path) -> None
    """Test that remove --delete-data handles gracefully when directory doesn't exist."""
    base_path = tmp_path / "nonexistent_base"

    config_manager.load()
    config_manager.add_index(LocalIndexConfig(name="ghost", path=str(base_path)))

    # Should not raise error when directory doesn't exist
    remove_command(name="ghost", delete_data=True)

    # Config should still be removed
    indexes = config_manager.list_indexes()
    names = [name for name, _, _ in indexes]
    assert "ghost" not in names
