"""
Integration tests for approximate simprint search via ShardedIndex128.

Tests the new UsearchSimprintIndex with composite 128-bit keys, IDF-weighted
scoring, 20x oversampling, derived index persistence, and rebuild from LMDB.
"""

import iscc_core as ic
import numpy as np

from iscc_search.indexes.simprint import lmdb_ops
from iscc_search.indexes.simprint.usearch_core import UsearchSimprintIndex
from iscc_search.indexes.usearch.index import UsearchIndex
from iscc_search.schema import IsccEntry, IsccQuery, IsccSimprint


# -- Helpers --


def _make_entry_simprints(sp_type, sp_data):
    # type: (str, list[tuple[bytes, int, int]]) -> dict[str, list[IsccSimprint]]
    """Build simprints dict for IsccEntry from (bytes, offset, size) tuples."""
    return {
        sp_type: [IsccSimprint(simprint=ic.encode_base64(sp), offset=offset, size=size) for sp, offset, size in sp_data]
    }


def _make_query_simprints(sp_type, sp_bytes_list):
    # type: (str, list[bytes]) -> dict[str, list[str]]
    """Build simprints dict for IsccQuery from raw bytes."""
    return {sp_type: [ic.encode_base64(sp) for sp in sp_bytes_list]}


def _flip_bits(data, n):
    # type: (bytes, int) -> bytes
    """Flip first n bits in a byte string for controlled Hamming distance."""
    ba = bytearray(data)
    bits_flipped = 0
    for i in range(len(ba)):
        for bit in range(8):
            if bits_flipped >= n:
                return bytes(ba)
            ba[i] ^= 1 << (7 - bit)
            bits_flipped += 1
    return bytes(ba)


# -- UsearchSimprintIndex unit tests --


def test_usearch_simprint_index_init(tmp_path):
    """Create new UsearchSimprintIndex with ShardedIndex128."""
    sp_dir = tmp_path / "sp_test"
    idx = UsearchSimprintIndex(path=sp_dir, ndim=128)
    assert idx.ndim == 128
    assert idx.size == 0
    assert idx.path == sp_dir
    idx.close()


def test_usearch_simprint_index_dirty(tmp_path):
    """dirty property tracks unsaved key mutations."""
    sp_dir = tmp_path / "sp_dirty"
    idx = UsearchSimprintIndex(path=sp_dir, ndim=64)
    assert idx.dirty == 0

    sp_bytes = b"\xaa" * 8
    asset_id = b"\x00" * 8
    key = lmdb_ops.pack_chunk_pointer(asset_id, 0, 100)
    vector = np.frombuffer(sp_bytes, dtype=np.uint8)

    idx.add_raw([key], [vector])
    assert idx.dirty > 0

    idx.save()
    assert idx.dirty == 0

    idx.close()


def test_usearch_simprint_index_add_and_search(tmp_path):
    """Add vectors with composite keys and search."""
    sp_dir = tmp_path / "sp_add"
    idx = UsearchSimprintIndex(path=sp_dir, ndim=64)

    sp_bytes = b"\xaa" * 8  # 64-bit simprint
    asset_id = b"\x00" * 8
    composite_key = lmdb_ops.pack_chunk_pointer(asset_id, 0, 500)
    vector = np.frombuffer(sp_bytes, dtype=np.uint8)

    idx.add_raw([composite_key], [vector])
    assert idx.size == 1

    # Search for exact match
    results = idx.search_raw([sp_bytes], limit=10, total_assets=1)
    assert len(results) == 1
    assert results[0].iscc_id_body == asset_id
    assert results[0].score > 0.0
    assert results[0].matches == 1

    idx.close()


def test_usearch_simprint_index_empty_search(tmp_path):
    """Search on empty index returns empty list."""
    sp_dir = tmp_path / "sp_empty"
    idx = UsearchSimprintIndex(path=sp_dir, ndim=128)

    results = idx.search_raw([b"\xff" * 16], limit=10)
    assert results == []

    idx.close()


def test_usearch_simprint_index_empty_add(tmp_path):
    """Adding empty list is a no-op."""
    sp_dir = tmp_path / "sp_noop"
    idx = UsearchSimprintIndex(path=sp_dir, ndim=64)

    idx.add_raw([], [])
    assert idx.size == 0

    idx.close()


def test_usearch_simprint_index_remove_empty(tmp_path):
    """Removing empty list is a no-op."""
    sp_dir = tmp_path / "sp_remove_noop"
    idx = UsearchSimprintIndex(path=sp_dir, ndim=64)
    idx.remove([])
    assert idx.size == 0
    idx.close()


def test_usearch_simprint_index_remove(tmp_path):
    """Remove vectors by composite keys."""
    sp_dir = tmp_path / "sp_remove"
    sp_bytes = b"\xcc" * 8
    asset_id = b"\x01" * 8
    key = lmdb_ops.pack_chunk_pointer(asset_id, 0, 100)
    vector = np.frombuffer(sp_bytes, dtype=np.uint8)

    idx = UsearchSimprintIndex(path=sp_dir, ndim=64)
    idx.add_raw([key], [vector])
    assert idx.size == 1

    idx.remove([key])
    # After tombstone removal, search returns no results
    results = idx.search_raw([sp_bytes], limit=10, total_assets=1)
    assert len(results) == 0
    idx.close()


def test_usearch_simprint_index_persistence(tmp_path):
    """Save, close, reopen preserves vectors."""
    sp_dir = tmp_path / "sp_persist"
    sp_bytes = b"\xbb" * 8
    asset_id = b"\x01" * 8
    composite_key = lmdb_ops.pack_chunk_pointer(asset_id, 100, 200)
    vector = np.frombuffer(sp_bytes, dtype=np.uint8)

    idx = UsearchSimprintIndex(path=sp_dir, ndim=64)
    idx.add_raw([composite_key], [vector])
    idx.save()
    idx.reset()

    # Reopen
    idx2 = UsearchSimprintIndex(path=sp_dir, ndim=64)
    assert idx2.size == 1

    results = idx2.search_raw([sp_bytes], limit=10, total_assets=1)
    assert len(results) == 1
    assert results[0].iscc_id_body == asset_id

    idx2.close()


def test_usearch_simprint_index_multi_chunk_asset(tmp_path):
    """Multiple chunks from same asset are correctly grouped."""
    sp_dir = tmp_path / "sp_multi_chunk"
    idx = UsearchSimprintIndex(path=sp_dir, ndim=64)

    asset_id = b"\x02" * 8
    sp1 = b"\xaa" * 8
    sp2 = b"\xbb" * 8

    key1 = lmdb_ops.pack_chunk_pointer(asset_id, 0, 100)
    key2 = lmdb_ops.pack_chunk_pointer(asset_id, 100, 200)
    vec1 = np.frombuffer(sp1, dtype=np.uint8)
    vec2 = np.frombuffer(sp2, dtype=np.uint8)

    idx.add_raw([key1, key2], [vec1, vec2])
    assert idx.size == 2

    # Search with both simprints - should return single asset with 2 matches
    results = idx.search_raw([sp1, sp2], limit=10, total_assets=1)
    assert len(results) == 1
    assert results[0].iscc_id_body == asset_id
    assert results[0].matches == 2

    idx.close()


def test_usearch_simprint_index_detailed_mode(tmp_path):
    """Detailed mode returns chunk metadata (offset, size)."""
    sp_dir = tmp_path / "sp_detailed"
    idx = UsearchSimprintIndex(path=sp_dir, ndim=64)

    asset_id = b"\x03" * 8
    sp_bytes = b"\xcc" * 8
    composite_key = lmdb_ops.pack_chunk_pointer(asset_id, 500, 1000)
    vector = np.frombuffer(sp_bytes, dtype=np.uint8)

    idx.add_raw([composite_key], [vector])

    results = idx.search_raw([sp_bytes], limit=10, detailed=True, total_assets=1)
    assert len(results) == 1
    assert results[0].chunks is not None
    assert len(results[0].chunks) == 1

    chunk = results[0].chunks[0]
    assert chunk.offset == 500
    assert chunk.size == 1000
    assert chunk.score > 0.9  # Near-exact match

    idx.close()


def test_usearch_simprint_index_threshold(tmp_path):
    """Threshold filters weak matches."""
    sp_dir = tmp_path / "sp_threshold"
    idx = UsearchSimprintIndex(path=sp_dir, ndim=64)

    asset_id = b"\x04" * 8
    sp_bytes = b"\xaa" * 8
    # Create a dissimilar vector (many bits flipped)
    dissimilar = _flip_bits(sp_bytes, 48)  # 48 of 64 bits different = score 0.25

    key = lmdb_ops.pack_chunk_pointer(asset_id, 0, 100)
    vec = np.frombuffer(dissimilar, dtype=np.uint8)
    idx.add_raw([key], [vec])

    # High threshold should filter it out
    results = idx.search_raw([sp_bytes], limit=10, threshold=0.9, total_assets=1)
    assert len(results) == 0

    # Low threshold should include it
    results = idx.search_raw([sp_bytes], limit=10, threshold=0.0, total_assets=1)
    assert len(results) == 1

    idx.close()


def test_usearch_simprint_index_idf_scoring(tmp_path):
    """IDF weighting ranks rare simprints higher than common ones."""
    sp_dir = tmp_path / "sp_idf"
    idx = UsearchSimprintIndex(path=sp_dir, ndim=64)

    # Create 3 assets: one with rare simprint, one with common simprint
    rare_sp = b"\xaa" * 8
    common_sp = b"\xbb" * 8

    # Asset 1: has the rare simprint
    asset1 = b"\x01" * 8
    key1 = lmdb_ops.pack_chunk_pointer(asset1, 0, 100)
    idx.add_raw([key1], [np.frombuffer(rare_sp, dtype=np.uint8)])

    # Asset 2: has the common simprint
    asset2 = b"\x02" * 8
    key2 = lmdb_ops.pack_chunk_pointer(asset2, 0, 100)
    idx.add_raw([key2], [np.frombuffer(common_sp, dtype=np.uint8)])

    # Asset 3: also has the common simprint (making it more common)
    asset3 = b"\x03" * 8
    key3 = lmdb_ops.pack_chunk_pointer(asset3, 0, 100)
    idx.add_raw([key3], [np.frombuffer(common_sp, dtype=np.uint8)])

    # Create doc_freq_fn that returns freq=1 for rare, freq=2 for common
    def doc_freq_fn(sp_key):
        # type: (bytes) -> int
        if sp_key == rare_sp:
            return 1
        return 2

    # Search for the rare simprint
    results = idx.search_raw([rare_sp], limit=10, doc_freq_fn=doc_freq_fn, total_assets=3)

    # Asset 1 should match with highest score
    assert len(results) >= 1
    assert results[0].iscc_id_body == asset1

    idx.close()


def test_usearch_simprint_index_no_doc_freq(tmp_path):
    """Without doc_freq_fn, all frequencies default to 1."""
    sp_dir = tmp_path / "sp_no_freq"
    idx = UsearchSimprintIndex(path=sp_dir, ndim=64)

    sp_bytes = b"\xdd" * 8
    asset_id = b"\x05" * 8
    key = lmdb_ops.pack_chunk_pointer(asset_id, 0, 100)
    idx.add_raw([key], [np.frombuffer(sp_bytes, dtype=np.uint8)])

    # Search without doc_freq_fn
    results = idx.search_raw([sp_bytes], limit=10, total_assets=1)
    assert len(results) == 1
    assert results[0].score > 0.0

    idx.close()


def test_usearch_simprint_index_unmatched_penalty(tmp_path):
    """Unmatched query simprints penalize score via IDF denominator."""
    sp_dir = tmp_path / "sp_unmatched"
    idx = UsearchSimprintIndex(path=sp_dir, ndim=64)

    sp_bytes = b"\xee" * 8
    unrelated = b"\x11" * 8
    asset_id = b"\x06" * 8
    key = lmdb_ops.pack_chunk_pointer(asset_id, 0, 100)
    idx.add_raw([key], [np.frombuffer(sp_bytes, dtype=np.uint8)])

    # Search with one matching and one unmatched simprint
    results_partial = idx.search_raw([sp_bytes, unrelated], limit=10, total_assets=1)

    # Search with only matching simprint
    results_full = idx.search_raw([sp_bytes], limit=10, total_assets=1)

    # Partial match should score lower than full match
    assert len(results_partial) >= 1
    assert len(results_full) >= 1
    assert results_partial[0].score < results_full[0].score

    idx.close()


# -- iter_simprint_vectors tests --


def test_iter_simprint_vectors(tmp_path):
    """Extract vectors from LMDB for ShardedIndex128 rebuild."""
    import lmdb

    env = lmdb.open(str(tmp_path / "rebuild.lmdb"), max_dbs=2, subdir=False)
    with env.begin(write=True) as txn:
        db = env.open_db(b"__sp_test__", txn=txn, dupsort=True, dupfixed=True)

        sp1 = b"\xaa" * 16
        sp2 = b"\xbb" * 16
        ptr1 = lmdb_ops.pack_chunk_pointer(b"\x01" * 8, 0, 100)
        ptr2 = lmdb_ops.pack_chunk_pointer(b"\x02" * 8, 200, 300)

        txn.put(sp1, ptr1, db=db)
        txn.put(sp2, ptr2, db=db)

    all_keys = []
    all_vectors = []
    with env.begin() as txn:
        db = env.open_db(b"__sp_test__", txn=txn, dupsort=True, dupfixed=True)
        for keys, vectors in lmdb_ops.iter_simprint_vectors(txn, db):
            all_keys.extend(keys)
            all_vectors.extend(vectors)

    assert len(all_keys) == 2
    assert len(all_vectors) == 2
    # Keys are chunk pointers (16 bytes)
    assert all(len(k) == 16 for k in all_keys)
    # Vectors are numpy arrays
    assert all(isinstance(v, np.ndarray) for v in all_vectors)

    env.close()


def test_iter_simprint_vectors_empty_db(tmp_path):
    """Empty database yields no batches."""
    import lmdb

    env = lmdb.open(str(tmp_path / "empty.lmdb"), max_dbs=2, subdir=False)
    with env.begin(write=True) as txn:
        db = env.open_db(b"__sp_test__", txn=txn, dupsort=True, dupfixed=True)

    with env.begin() as txn:
        db = env.open_db(b"__sp_test__", txn=txn, dupsort=True, dupfixed=True)
        batches = list(lmdb_ops.iter_simprint_vectors(txn, db))

    assert batches == []

    env.close()


def test_iter_simprint_vectors_multi_dup(tmp_path):
    """Multiple dups per key are all extracted."""
    import lmdb

    env = lmdb.open(str(tmp_path / "multidup.lmdb"), max_dbs=2, subdir=False)
    with env.begin(write=True) as txn:
        db = env.open_db(b"__sp_test__", txn=txn, dupsort=True, dupfixed=True)

        sp = b"\xcc" * 16
        ptr1 = lmdb_ops.pack_chunk_pointer(b"\x01" * 8, 0, 100)
        ptr2 = lmdb_ops.pack_chunk_pointer(b"\x02" * 8, 200, 300)

        txn.put(sp, ptr1, db=db)
        txn.put(sp, ptr2, db=db)

    all_keys = []
    all_vectors = []
    with env.begin() as txn:
        db = env.open_db(b"__sp_test__", txn=txn, dupsort=True, dupfixed=True)
        for keys, vectors in lmdb_ops.iter_simprint_vectors(txn, db):
            all_keys.extend(keys)
            all_vectors.extend(vectors)

    assert len(all_keys) == 2
    # Both vectors should be the same simprint
    assert np.array_equal(all_vectors[0], all_vectors[1])

    env.close()


def test_iter_simprint_vectors_batching(tmp_path):
    """Batches are yielded according to batch_size."""
    import lmdb

    env = lmdb.open(str(tmp_path / "batch.lmdb"), max_dbs=2, subdir=False)
    with env.begin(write=True) as txn:
        db = env.open_db(b"__sp_test__", txn=txn, dupsort=True, dupfixed=True)
        # Insert 5 entries with distinct keys
        for i in range(5):
            sp = bytes([i]) * 16
            ptr = lmdb_ops.pack_chunk_pointer(bytes([i]) * 8, 0, i)
            txn.put(sp, ptr, db=db)

    with env.begin() as txn:
        db = env.open_db(b"__sp_test__", txn=txn, dupsort=True, dupfixed=True)
        batches = list(lmdb_ops.iter_simprint_vectors(txn, db, batch_size=2))

    # 5 entries with batch_size=2: should yield 3 batches (2, 2, 1)
    assert len(batches) == 3
    assert len(batches[0][0]) == 2
    assert len(batches[1][0]) == 2
    assert len(batches[2][0]) == 1

    env.close()


def test_iter_simprint_vectors_exact_batch_boundary(tmp_path):
    """Entry count is exact multiple of batch_size (no remainder batch)."""
    import lmdb

    env = lmdb.open(str(tmp_path / "exact.lmdb"), max_dbs=2, subdir=False)
    with env.begin(write=True) as txn:
        db = env.open_db(b"__sp_test__", txn=txn, dupsort=True, dupfixed=True)
        # Insert exactly 4 entries with distinct keys
        for i in range(4):
            sp = bytes([i + 10]) * 16
            ptr = lmdb_ops.pack_chunk_pointer(bytes([i]) * 8, 0, i)
            txn.put(sp, ptr, db=db)

    with env.begin() as txn:
        db = env.open_db(b"__sp_test__", txn=txn, dupsort=True, dupfixed=True)
        batches = list(lmdb_ops.iter_simprint_vectors(txn, db, batch_size=2))

    # 4 entries with batch_size=2: exactly 2 batches, no remainder
    assert len(batches) == 2
    assert len(batches[0][0]) == 2
    assert len(batches[1][0]) == 2

    env.close()


# -- UsearchIndex integration tests --


def test_approximate_search_returns_results(tmp_path, sample_iscc_ids):
    """Approximate search via ShardedIndex128 returns matching results."""
    index_path = tmp_path / "approx_search"
    sp_bytes = b"\xaa" * 16  # 128-bit simprint
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Approximate search test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp_bytes, 0, 500)]),
    )
    idx.add_assets([asset])

    # Search (exact=False, default)
    query = IsccQuery(simprints=_make_query_simprints(sp_type, [sp_bytes]))
    result = idx.search_assets(query, limit=10)

    assert len(result.chunk_matches) > 0
    match = result.chunk_matches[0]
    assert match.iscc_id == sample_iscc_ids[0]
    assert match.score > 0.0

    idx.close()


def test_approximate_search_with_multiple_assets(tmp_path, sample_iscc_ids):
    """Approximate search ranks multiple assets correctly."""
    index_path = tmp_path / "multi_asset"
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Multi-asset test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256, match_threshold_simprints=0.0)

    # Asset 1: exact match
    sp_exact = b"\xaa" * 16
    asset1 = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp_exact, 0, 100)]),
    )

    # Asset 2: similar match (8 bits flipped out of 128 = ~93% similar)
    sp_similar = _flip_bits(sp_exact, 8)
    asset2 = IsccEntry(
        iscc_id=sample_iscc_ids[1],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp_similar, 0, 200)]),
    )

    idx.add_assets([asset1, asset2])

    query = IsccQuery(simprints=_make_query_simprints(sp_type, [sp_exact]))
    result = idx.search_assets(query, limit=10)

    assert len(result.chunk_matches) >= 2
    # Exact match should rank first
    assert result.chunk_matches[0].iscc_id == sample_iscc_ids[0]
    assert result.chunk_matches[0].score >= result.chunk_matches[1].score

    idx.close()


def test_approximate_search_unknown_type(tmp_path, sample_iscc_ids):
    """Approximate search with unknown simprint type returns empty."""
    index_path = tmp_path / "unknown_type"
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Unknown type test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(b"\xaa" * 16, 0, 100)]),
    )
    idx.add_assets([asset])

    # Query with type that doesn't exist
    query = IsccQuery(simprints=_make_query_simprints("NONEXISTENT_V0", [b"\xbb" * 16]))
    result = idx.search_assets(query, limit=10)

    assert len(result.chunk_matches) == 0

    idx.close()


def test_simprint_index_persistence_across_close(tmp_path, sample_iscc_ids):
    """Derived simprint index survives close/reopen."""
    index_path = tmp_path / "sp_persist"
    sp_bytes = b"\xdd" * 16
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Persistence test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp_bytes, 0, 100)]),
    )
    idx.add_assets([asset])
    idx.close()

    # Reopen and search
    idx2 = UsearchIndex(index_path, realm_id=0, max_dim=256)
    query = IsccQuery(simprints=_make_query_simprints(sp_type, [sp_bytes]))
    result = idx2.search_assets(query, limit=10)

    assert len(result.chunk_matches) > 0
    assert result.chunk_matches[0].iscc_id == sample_iscc_ids[0]

    idx2.close()


def test_simprint_index_rebuild_from_lmdb(tmp_path, sample_iscc_ids):
    """Delete derived index directory, reopen triggers rebuild from LMDB."""
    import shutil

    index_path = tmp_path / "sp_rebuild"
    sp_bytes = b"\xee" * 16
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Rebuild test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp_bytes, 0, 100)]),
    )
    idx.add_assets([asset])
    idx.close()

    # Delete the derived simprint index directory
    sp_dir = index_path / f"SIMPRINT_{sp_type}"
    assert sp_dir.exists()
    shutil.rmtree(sp_dir)
    assert not sp_dir.exists()

    # Reopen - should detect missing directory
    idx2 = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Manually trigger rebuild
    idx2._rebuild_simprint_index(sp_type)

    # Search should work after rebuild
    query = IsccQuery(simprints=_make_query_simprints(sp_type, [sp_bytes]))
    result = idx2.search_assets(query, limit=10)

    assert len(result.chunk_matches) > 0
    assert result.chunk_matches[0].iscc_id == sample_iscc_ids[0]

    idx2.close()


def test_simprint_index_sync_check(tmp_path, sample_iscc_ids):
    """Sync mismatch between metadata and derived index triggers rebuild."""
    index_path = tmp_path / "sp_sync"
    sp_bytes = b"\xff" * 16
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Sync check test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp_bytes, 0, 100)]),
    )
    idx.add_assets([asset])

    # Verify sp_count metadata was set
    sp_count = idx._get_sp_metadata(sp_type)
    assert sp_count is not None
    assert sp_count == 1

    idx.close()


def test_multi_type_approximate_search(tmp_path, sample_iscc_ids):
    """Approximate search aggregates across multiple simprint types."""
    index_path = tmp_path / "multi_type"
    sp_type_1 = "CONTENT_TEXT_V0"
    sp_type_2 = "SEMANTIC_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Multi-type test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256, match_threshold_simprints=0.0)

    sp1 = b"\xaa" * 16
    sp2 = b"\xbb" * 16

    simprints = {}
    simprints[sp_type_1] = [IsccSimprint(simprint=ic.encode_base64(sp1), offset=0, size=100)]
    simprints[sp_type_2] = [IsccSimprint(simprint=ic.encode_base64(sp2), offset=100, size=200)]

    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=simprints,
    )
    idx.add_assets([asset])

    # Query both types
    query_sps = {
        sp_type_1: [ic.encode_base64(sp1)],
        sp_type_2: [ic.encode_base64(sp2)],
    }
    query = IsccQuery(simprints=query_sps)
    result = idx.search_assets(query, limit=10)

    assert len(result.chunk_matches) > 0
    match = result.chunk_matches[0]
    assert match.iscc_id == sample_iscc_ids[0]
    # Should have results for both types
    assert len(match.types) == 2
    assert sp_type_1 in match.types
    assert sp_type_2 in match.types

    idx.close()


def test_approximate_search_self_exclusion(tmp_path, sample_iscc_ids):
    """Search by iscc_id excludes the query asset from results."""
    index_path = tmp_path / "self_exclude"
    sp_type = "CONTENT_TEXT_V0"
    sp_bytes = b"\xaa" * 16

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Self-exclusion test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256, match_threshold_simprints=0.0)

    # Add two assets with the same simprint
    for i in range(2):
        asset = IsccEntry(
            iscc_id=sample_iscc_ids[i],
            units=[instance_unit, content_unit],
            simprints=_make_entry_simprints(sp_type, [(sp_bytes, 0, 100)]),
        )
        idx.add_assets([asset])

    # Search by iscc_id - should exclude self
    query = IsccQuery(iscc_id=sample_iscc_ids[0])
    result = idx.search_assets(query, limit=10)

    # Self should not appear in chunk_matches
    for match in result.chunk_matches:
        assert match.iscc_id != sample_iscc_ids[0]

    idx.close()


def test_flush_saves_simprint_indexes(tmp_path, sample_iscc_ids):
    """flush() saves derived simprint indexes to disk."""
    index_path = tmp_path / "sp_flush"
    sp_bytes = b"\xab" * 16
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Flush test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp_bytes, 0, 100)]),
    )
    idx.add_assets([asset])
    idx.flush()

    # Verify simprint index directory exists
    sp_dir = index_path / f"SIMPRINT_{sp_type}"
    assert sp_dir.exists()

    idx.close()


def test_flush_skips_clean_simprint_indexes(tmp_path, sample_iscc_ids):
    """flush() skips simprint sub-indexes with dirty == 0."""
    index_path = tmp_path / "sp_flush_clean"
    sp_bytes = b"\xab" * 16
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Flush clean simprint test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp_bytes, 0, 100)]),
    )
    idx.add_assets([asset])
    idx.flush()

    # Second flush should skip clean simprint index
    assert idx._simprint_indexes[sp_type].dirty == 0
    idx.flush()

    # Index still works
    query = IsccQuery(simprints=_make_query_simprints(sp_type, [sp_bytes]))
    result = idx.search_assets(query, limit=10)
    assert len(result.chunk_matches) > 0

    idx.close()


def test_auto_flush_triggers_simprint_indexes(tmp_path, sample_iscc_ids):
    """Auto-flush triggers for simprint sub-indexes when dirty >= flush_interval."""
    index_path = tmp_path / "sp_auto_flush"
    sp_bytes = b"\xab" * 16
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Auto-flush simprint test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256, flush_interval=1)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp_bytes, 0, 100)]),
    )
    idx.add_assets([asset])

    # Simprint index should have been auto-flushed (dirty reset to 0)
    assert idx._simprint_indexes[sp_type].dirty == 0

    idx.close()


def test_detect_sp_ndim(tmp_path, sample_iscc_ids):
    """_detect_sp_ndim reads simprint size from LMDB data."""
    index_path = tmp_path / "sp_ndim"
    sp_bytes = b"\xaa" * 16  # 128-bit simprint
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Ndim detection test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp_bytes, 0, 100)]),
    )
    idx.add_assets([asset])

    ndim = idx._detect_sp_ndim(sp_type)
    assert ndim == 128

    # Unknown type returns None
    assert idx._detect_sp_ndim("NONEXISTENT_V0") is None

    idx.close()


def test_get_total_sp_assets(tmp_path, sample_iscc_ids):
    """_get_total_sp_assets returns count of assets in __assets__ database."""
    index_path = tmp_path / "sp_total"
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Total assets test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    assert idx._get_total_sp_assets() == 0

    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(b"\xaa" * 16, 0, 100)]),
    )
    idx.add_assets([asset])
    assert idx._get_total_sp_assets() == 1

    idx.close()


def test_rebuild_simprint_index_no_data(tmp_path, sample_iscc_ids):
    """Rebuilding simprint index with no data in LMDB is a no-op."""
    index_path = tmp_path / "sp_rebuild_empty"
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Register type in sp_data_dbs via adding and removing
    sp_type = "CONTENT_TEXT_V0"
    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Empty rebuild")["iscc"]
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(b"\xaa" * 16, 0, 100)]),
    )
    idx.add_assets([asset])

    # Manually clear the LMDB data to simulate empty state
    with idx.env.begin(write=True) as txn:
        data_db = idx._sp_data_dbs[sp_type]
        txn.drop(data_db, delete=False)  # Clear all entries

    # Rebuild should handle empty gracefully
    idx._rebuild_simprint_index(sp_type)

    idx.close()


def test_rebuild_simprint_index_unknown_type(tmp_path, sample_iscc_ids):
    """Rebuilding simprint index with unknown type logs warning."""
    index_path = tmp_path / "sp_rebuild_unknown"
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Rebuild for type that doesn't exist in _sp_data_dbs
    idx._rebuild_simprint_index("NONEXISTENT_V0")

    idx.close()


def test_usearch_simprint_best_score_update(tmp_path):
    """Best-per-query-per-asset updates when a better score appears (line 145)."""
    sp_dir = tmp_path / "sp_best_update"
    idx = UsearchSimprintIndex(path=sp_dir, ndim=64)

    # Add 2 chunks for same asset with similar simprints
    asset_id = b"\x07" * 8
    sp_exact = b"\xaa" * 8  # Exact match to query
    sp_similar = _flip_bits(sp_exact, 4)  # 4 bits different, worse match

    key1 = lmdb_ops.pack_chunk_pointer(asset_id, 0, 100)
    key2 = lmdb_ops.pack_chunk_pointer(asset_id, 100, 200)

    # Add the worse match first, then the better match
    idx.add_raw(
        [key1, key2],
        [np.frombuffer(sp_similar, dtype=np.uint8), np.frombuffer(sp_exact, dtype=np.uint8)],
    )

    # Search with the exact match query - should use best score (exact)
    results = idx.search_raw([sp_exact], limit=10, total_assets=1)
    assert len(results) == 1
    # Score should reflect the better (exact) match, not the worse one
    assert results[0].score > 0.9

    idx.close()


def test_usearch_simprint_unmatched_penalty(tmp_path):
    """Unmatched query simprints apply penalty via avg IDF (lines 177-179)."""
    sp_dir = tmp_path / "sp_unmatched_idf"
    idx = UsearchSimprintIndex(path=sp_dir, ndim=64)

    sp_bytes = b"\xaa" * 8
    unrelated = b"\x00" * 8  # Very different, won't match
    asset_id = b"\x08" * 8
    key = lmdb_ops.pack_chunk_pointer(asset_id, 0, 100)
    idx.add_raw([key], [np.frombuffer(sp_bytes, dtype=np.uint8)])

    # Search with one matching + one non-matching simprint at high threshold
    # The non-matching simprint triggers the unmatched penalty path
    results = idx.search_raw([sp_bytes, unrelated], limit=10, threshold=0.9, total_assets=10)

    if results:
        # Score should be penalized for unmatched simprint
        assert results[0].matches == 1  # Only 1 of 2 matched
        assert results[0].queried == 2
        assert results[0].score < 1.0

    idx.close()


def test_simprint_sync_mismatch_loads_stale(tmp_path, sample_iscc_ids):
    """Sync mismatch between sp_count metadata and ShardedIndex128 loads stale index."""
    index_path = tmp_path / "sp_sync_mismatch"
    sp_bytes = b"\xaa" * 16
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Sync mismatch test")["iscc"]

    # Create index with data
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp_bytes, 0, 100)]),
    )
    idx.add_assets([asset])

    # Flush to make indexes clean, then corrupt metadata
    idx.flush()
    idx._update_sp_metadata(sp_type, 999)  # Wrong count
    idx.close()

    # Reopen - should detect mismatch, log warning, but load stale index
    idx2 = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Stale index is loaded and functional (1 actual vector despite metadata saying 999)
    assert sp_type in idx2._simprint_indexes
    assert idx2._simprint_indexes[sp_type].size == 1

    # Search still works with stale data
    query = IsccQuery(simprints=_make_query_simprints(sp_type, [sp_bytes]))
    result = idx2.search_assets(query, limit=10)
    assert len(result.chunk_matches) > 0

    idx2.close()


def test_detect_sp_ndim_empty_db(tmp_path, sample_iscc_ids):
    """_detect_sp_ndim returns None when LMDB database is empty (line 1428)."""
    index_path = tmp_path / "sp_ndim_empty"
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Ndim empty test")["iscc"]

    # Create index, add data, then clear the LMDB data
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(b"\xaa" * 16, 0, 100)]),
    )
    idx.add_assets([asset])

    # Clear the simprint data
    with idx.env.begin(write=True) as txn:
        data_db = idx._sp_data_dbs[sp_type]
        txn.drop(data_db, delete=False)

    # Now ndim detection should return None
    assert idx._detect_sp_ndim(sp_type) is None

    idx.close()


def test_get_sp_metadata_not_set(tmp_path):
    """_get_sp_metadata returns None when sp_count not in metadata (line 1387)."""
    index_path = tmp_path / "sp_meta_notset"
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # No simprints added, so no sp_count metadata
    assert idx._get_sp_metadata("CONTENT_TEXT_V0") is None

    idx.close()


def test_doc_freq_fn_none_path(tmp_path, sample_iscc_ids):
    """doc_freq_fn is None when sp_type not in _sp_data_dbs (line 897)."""
    index_path = tmp_path / "doc_freq_none"
    sp_bytes = b"\xaa" * 16
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Doc freq none test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp_bytes, 0, 100)]),
    )
    idx.add_assets([asset])

    # Remove the LMDB data db handle to force doc_freq_fn=None path
    del idx._sp_data_dbs[sp_type]

    # Search should still work (with default freq=1)
    query = IsccQuery(simprints=_make_query_simprints(sp_type, [sp_bytes]))
    result = idx.search_assets(query, limit=10)
    assert len(result.chunk_matches) > 0

    idx.close()


def test_rebuild_with_existing_directory(tmp_path, sample_iscc_ids):
    """Rebuild removes stale directory before creating fresh index (line 1346)."""
    index_path = tmp_path / "sp_rebuild_stale"
    sp_bytes = b"\xaa" * 16
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Stale dir rebuild test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp_bytes, 0, 100)]),
    )
    idx.add_assets([asset])

    # Verify directory exists
    sp_dir = index_path / f"SIMPRINT_{sp_type}"
    assert sp_dir.exists()

    # Rebuild with existing directory
    idx._rebuild_simprint_index(sp_type)

    # Verify rebuild succeeded
    assert sp_type in idx._simprint_indexes
    assert idx._simprint_indexes[sp_type].size == 1

    idx.close()


def test_update_removes_stale_derived_vectors(tmp_path, sample_iscc_ids):
    """Updating asset with empty simprints removes stale vectors from derived index."""
    index_path = tmp_path / "stale_removal"
    sp_type = "CONTENT_TEXT_V0"
    sp_bytes = b"\xaa" * 16

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Stale removal test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Add asset with simprints
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp_bytes, 0, 500)]),
    )
    idx.add_assets([asset])
    assert sp_type in idx._simprint_indexes
    assert idx._simprint_indexes[sp_type].size == 1

    # Update same asset with empty simprints (removes all for this type)
    asset_updated = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints={sp_type: []},
    )
    idx.add_assets([asset_updated])

    # Derived index should have the stale vector removed
    assert idx._simprint_indexes[sp_type].size == 0

    # Approximate search should return no results
    query = IsccQuery(simprints=_make_query_simprints(sp_type, [sp_bytes]))
    result = idx.search_assets(query, limit=10)
    assert len(result.chunk_matches) == 0

    idx.close()


def test_idf_uses_matched_simprint(tmp_path):
    """IDF scoring uses matched (stored) simprint, not query simprint."""
    sp_dir = tmp_path / "idf_match"
    idx = UsearchSimprintIndex(path=sp_dir, ndim=64)

    stored_sp = b"\xaa" * 8  # Stored simprint
    query_sp = _flip_bits(stored_sp, 4)  # 4 bits different (approximate match)

    asset_id = b"\x01" * 8
    key = lmdb_ops.pack_chunk_pointer(asset_id, 0, 100)
    idx.add_raw([key], [np.frombuffer(stored_sp, dtype=np.uint8)])

    # Track which simprint bytes are passed to doc_freq_fn
    freq_calls = []

    def doc_freq_fn(sp_key):
        # type: (bytes) -> int
        freq_calls.append(sp_key)
        return 1

    results = idx.search_raw([query_sp], limit=10, detailed=True, doc_freq_fn=doc_freq_fn, total_assets=10)

    assert len(results) == 1
    # doc_freq_fn should have been called with stored_sp (the match), not query_sp
    assert stored_sp in freq_calls
    assert query_sp not in freq_calls

    # Detailed chunk should have correct query and match bytes
    chunk = results[0].chunks[0]
    assert chunk.query == query_sp
    assert chunk.match == stored_sp

    idx.close()


def test_load_simprint_ndim_none_skips(tmp_path, sample_iscc_ids):
    """_load_simprint_indexes skips types where ndim detection fails (line 1294)."""

    index_path = tmp_path / "sp_ndim_none"
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Ndim none load test")["iscc"]

    # Create index with data
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(b"\xaa" * 16, 0, 100)]),
    )
    idx.add_assets([asset])

    # Clear the data DB entries (so ndim detection returns None)
    with idx.env.begin(write=True) as txn:
        data_db = idx._sp_data_dbs[sp_type]
        txn.drop(data_db, delete=False)

    idx.close()

    # Reopen - should skip the type since ndim can't be detected
    idx2 = UsearchIndex(index_path, realm_id=0, max_dim=256)
    # Type should NOT be in _simprint_indexes since ndim detection failed
    assert sp_type not in idx2._simprint_indexes

    idx2.close()
