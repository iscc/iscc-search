"""
LMDB Index Package.

Provides LMDB-backed index implementation for production use.

Exports:
- LmdbIndexManager: Protocol implementation for managing multiple indexes
"""

from iscc_vdb.indexes.lmdb.manager import LmdbIndexManager

__all__ = ["LmdbIndexManager"]
