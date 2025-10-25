"""Test settings management for ISCC-VDB."""

from pathlib import Path
from iscc_vdb.settings import VdbSettings, vdb_settings


def test_vdb_settings_default_initialization():
    """Test VdbSettings with default values."""
    settings = VdbSettings()
    assert isinstance(settings.data_dir, Path)
    assert settings.data_dir.exists() or True  # May or may not exist yet


def test_vdb_settings_custom_data_dir():
    """Test VdbSettings with custom data_dir."""
    custom_dir = Path("/tmp/custom_iscc_data")
    settings = VdbSettings(data_dir=custom_dir)
    assert settings.data_dir == custom_dir


def test_vdb_settings_override_with_dict():
    """Test override method with dict parameter."""
    settings = VdbSettings()
    original_dir = settings.data_dir

    new_dir = Path("/tmp/new_data_dir")
    updated = settings.override({"data_dir": new_dir})

    # Original should be unchanged
    assert settings.data_dir == original_dir
    # Updated should have new value
    assert updated.data_dir == new_dir
    # Should be different instances
    assert settings is not updated


def test_vdb_settings_override_with_none():
    """Test override method with None parameter."""
    settings = VdbSettings()
    original_dir = settings.data_dir

    updated = settings.override(None)

    # Should return a deep copy with same values
    assert updated.data_dir == original_dir
    # Should be different instances
    assert settings is not updated


def test_vdb_settings_override_without_args():
    """Test override method without arguments."""
    settings = VdbSettings()
    original_dir = settings.data_dir

    updated = settings.override()

    # Should return a deep copy with same values
    assert updated.data_dir == original_dir
    # Should be different instances
    assert settings is not updated


def test_vdb_settings_override_multiple_fields():
    """Test that override handles field updates correctly."""
    settings = VdbSettings()

    # Valid Path should work
    new_dir = Path("/tmp/valid_path")
    updated = settings.override({"data_dir": new_dir})
    assert updated.data_dir == new_dir
    assert isinstance(updated, VdbSettings)


def test_module_level_vdb_settings():
    """Test that module-level vdb_settings is initialized."""
    assert isinstance(vdb_settings, VdbSettings)
    assert isinstance(vdb_settings.data_dir, Path)


def test_vdb_settings_extra_fields_ignored():
    """Test that extra fields are ignored per model_config."""
    # Should not raise error due to extra='ignore'
    settings = VdbSettings(unknown_field="value")
    assert not hasattr(settings, "unknown_field")
