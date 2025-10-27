"""High-performance ISCC similarity search engine."""

from platformdirs import PlatformDirs
from importlib import metadata

__package_name__ = "iscc-search"
__author__ = "iscc"
__version__ = metadata.version(__package_name__)
dirs = PlatformDirs(appname=__package_name__, appauthor=__author__)

from iscc_search.settings import SearchSettings, search_settings  # noqa: E402
from iscc_search.nphd import NphdIndex  # noqa: E402
from iscc_search.unit import UnitIndex  # noqa: E402
from iscc_search.instance import InstanceIndex  # noqa: E402

__all__ = ["NphdIndex", "UnitIndex", "InstanceIndex", "SearchSettings", "search_settings"]
