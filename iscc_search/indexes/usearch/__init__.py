"""
Usearch Index Package.

Provides high-performance HNSW-based index implementation using NphdIndex
with LMDB for metadata and asset storage.

Hybrid architecture:
- LMDB: Metadata, assets, INSTANCE exact-matching (dupsort)
- NphdIndex: Similarity search for META, CONTENT, DATA units

Exports:
- UsearchIndexManager: Protocol implementation for managing multiple indexes
"""

from iscc_search.indexes.usearch.manager import UsearchIndexManager

__all__ = ["UsearchIndexManager"]
