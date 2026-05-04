"""Tests for CLI index management commands."""

import iscc_core as ic
import pytest

from iscc_search.cli.index import rebuild_command, remove_command
from iscc_search.config import ConfigManager, LocalIndexConfig
from iscc_search.indexes.usearch import UsearchIndexManager
from iscc_search.schema import IsccEntry, IsccIndex


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


# Rebuild command tests


@pytest.fixture
def populated_usearch_index(tmp_path, sample_iscc_ids):
    # type: (Path, list[str]) -> tuple[Path, str]
    """
    Create a populated UsearchIndex on disk that can be re-opened by tests.

    Returns the base_path and the index name.
    """
    base_path = tmp_path / "indexes"
    mgr = UsearchIndexManager(base_path)
    name = "rebuilder"
    mgr.create_index(IsccIndex(name=name))

    content_unit = ic.gen_text_code_v0("Test content for CLI rebuild")["iscc"]
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
    )
    mgr.add_assets(name, [asset])
    mgr.close()
    return base_path, name


@pytest.fixture
def patched_active_index(monkeypatch, populated_usearch_index):
    # type: (pytest.MonkeyPatch, tuple) -> tuple[Path, str]
    """
    Patch get_active_index to return a fresh UsearchIndexManager on the populated path.

    Each call gets a new manager (matching real CLI behavior where each command instantiates
    one). The manager is closed by the rebuild_command itself.
    """
    base_path, name = populated_usearch_index

    def fake_get_active_index(index_name=None):
        # type: (str | None) -> tuple
        target = index_name or name
        return UsearchIndexManager(base_path), target

    monkeypatch.setattr("iscc_search.cli.index.get_active_index", fake_get_active_index)
    return base_path, name


def test_rebuild_requires_at_least_one_target(patched_active_index):
    """Bare `index rebuild` (no flags) should exit with an error."""
    with pytest.raises(SystemExit) as exc_info:
        rebuild_command(unit_type=[], simprint_type=[], all_types=False, index_name=None)
    assert exc_info.value.code == 1


def test_rebuild_all_with_explicit_types_is_rejected(patched_active_index):
    """`--all` combined with `--unit-type` should exit with an error (ambiguous intent)."""
    with pytest.raises(SystemExit) as exc_info:
        rebuild_command(
            unit_type=["CONTENT_TEXT_V0"],
            simprint_type=[],
            all_types=True,
            index_name=None,
        )
    assert exc_info.value.code == 1


def test_rebuild_all_succeeds_on_local_usearch(patched_active_index):
    """`--all` on a populated usearch index rebuilds every tracked NPHD type."""
    rebuild_command(unit_type=[], simprint_type=[], all_types=True, index_name=None)


def test_rebuild_specific_unit_type(patched_active_index):
    """`--unit-type X` rebuilds only X."""
    rebuild_command(
        unit_type=["CONTENT_TEXT_V0"],
        simprint_type=[],
        all_types=False,
        index_name=None,
    )


def test_rebuild_unknown_unit_type_prints_nothing_rebuilt(patched_active_index, capsys):
    """`--unit-type` typos should not be reported as successful rebuilds."""
    rebuild_command(
        unit_type=["CONTENT_TXT_V0"],
        simprint_type=[],
        all_types=False,
        index_name=None,
    )

    captured = capsys.readouterr()
    assert "nothing rebuilt" in captured.out
    assert "Rebuilt NPHD" not in captured.out


def test_rebuild_with_no_tracked_types_prints_yellow(monkeypatch, tmp_path):
    """An empty tracked-types result (--all on an empty index) should NOT exit nonzero."""
    base_path = tmp_path / "indexes"
    mgr = UsearchIndexManager(base_path)
    mgr.create_index(IsccIndex(name="empty"))
    mgr.close()

    def fake_get_active_index(index_name=None):
        # type: (str | None) -> tuple
        return UsearchIndexManager(base_path), "empty"

    monkeypatch.setattr("iscc_search.cli.index.get_active_index", fake_get_active_index)

    rebuild_command(unit_type=[], simprint_type=[], all_types=True, index_name=None)


def test_rebuild_rejects_remote_index(monkeypatch):
    """Rebuild against a non-UsearchIndexManager (e.g. remote client) must error out."""

    class FakeRemote:
        def close(self):  # pragma: no cover - never reached because we error before close
            pass

    def fake_get_active_index(index_name=None):
        # type: (str | None) -> tuple
        return FakeRemote(), "remote-name"

    monkeypatch.setattr("iscc_search.cli.index.get_active_index", fake_get_active_index)

    with pytest.raises(SystemExit) as exc_info:
        rebuild_command(unit_type=[], simprint_type=[], all_types=True, index_name=None)
    assert exc_info.value.code == 1


def test_rebuild_propagates_index_not_found(monkeypatch, tmp_path):
    """If get_active_index returns a manager but the index doesn't exist, exit nonzero."""
    base_path = tmp_path / "indexes"
    base_path.mkdir()

    def fake_get_active_index(index_name=None):
        # type: (str | None) -> tuple
        return UsearchIndexManager(base_path), "ghost"

    monkeypatch.setattr("iscc_search.cli.index.get_active_index", fake_get_active_index)

    with pytest.raises(SystemExit) as exc_info:
        rebuild_command(unit_type=[], simprint_type=[], all_types=True, index_name=None)
    assert exc_info.value.code == 1


def test_rebuild_handles_get_active_index_error(monkeypatch):
    """A ValueError from get_active_index (e.g. no active index configured) exits nonzero."""

    def fake_get_active_index(index_name=None):
        # type: (str | None) -> tuple
        raise ValueError("No active index configured")

    monkeypatch.setattr("iscc_search.cli.index.get_active_index", fake_get_active_index)

    with pytest.raises(SystemExit) as exc_info:
        rebuild_command(unit_type=[], simprint_type=[], all_types=True, index_name=None)
    assert exc_info.value.code == 1
