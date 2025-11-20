"""
Sharded Usearch Index - Support Larger than Ram Usearch Indexes

Problem:
    Usearch indexes need to be fully in memory to be able to add new entries.

Solution:
    A) Have a small writable index in RAM and flush them periodically to disk.
    B) Have a large read-only index that queries the smaller shards in parallel.
"""


class ShardedUsearchIndex:
    """
    Drop-In replacement for stanbard Usearch Index with support for larger than Ram indexes.

    Wraps the usearch Index class with a fully compatabile interface and manages subindexes.
    Subindexes are one writabel in-memory index (load) and multiple read-only (views) on-disk shards
    joined via usearch Indexes implementation.
    """
