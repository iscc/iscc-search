"""Embedded Vector Database for ISCC."""

from platformdirs import PlatformDirs
from importlib import metadata

__package_name__ = "iscc-vdb"
__author__ = "iscc"
__version__ = metadata.version(__package_name__)
dirs = PlatformDirs(appname=__package_name__, appauthor=__author__)

from iscc_vdb.settings import VdbSettings, vdb_settings  # noqa: E402
from iscc_vdb.nphd import NphdIndex  # noqa: E402
from iscc_vdb.unit import UnitIndex  # noqa: E402
from iscc_vdb.instance import InstanceIndex  # noqa: E402

__all__ = ["NphdIndex", "UnitIndex", "InstanceIndex", "VdbSettings", "vdb_settings"]
