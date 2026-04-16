"""High-performance ISCC similarity search engine."""

from platformdirs import PlatformDirs
from importlib import metadata

__package_name__ = "iscc-search"
__author__ = "iscc"
__version__ = metadata.version(__package_name__)
dirs = PlatformDirs(appname=__package_name__, appauthor=__author__)

from iscc_search.options import SearchOptions, search_opts  # noqa: E402

__all__ = ["SearchOptions", "search_opts"]
