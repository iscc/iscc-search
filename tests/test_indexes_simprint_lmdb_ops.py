"""
Tests for pure LMDB simprint operations (lmdb_ops module).

Tests pack/unpack roundtrips, IDF calculation, document frequency counting,
and exact simprint search using real LMDB transactions.
"""

import math
import struct

import lmdb
import pytest

from iscc_search.indexes.simprint.lmdb_ops import (
    CHUNK_POINTER_BYTES,
    MAX_OFFSET,
    MAX_SIZE,
    _calculate_coverage_quality_score,
    calculate_idf,
    count_doc_freq,
    delete_asset_simprints,
    pack_chunk_pointer,
    search_simprints_exact,
    unpack_chunk_pointer,
)


# --- pack/unpack roundtrip tests ---


def test_pack_unpack_roundtrip():
    """Pack and unpack should produce original values."""
    body = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    packed = pack_chunk_pointer(body, 100, 200)
    assert len(packed) == CHUNK_POINTER_BYTES
    out_body, out_offset, out_size = unpack_chunk_pointer(packed)
    assert out_body == body
    assert out_offset == 100
    assert out_size == 200


def test_pack_unpack_zero_values():
    """Pack/unpack with zero offset and size."""
    body = b"\x00" * 8
    packed = pack_chunk_pointer(body, 0, 0)
    out_body, out_offset, out_size = unpack_chunk_pointer(packed)
    assert out_body == body
    assert out_offset == 0
    assert out_size == 0


def test_pack_unpack_max_values():
    """Pack/unpack with maximum offset and size."""
    body = b"\xff" * 8
    packed = pack_chunk_pointer(body, MAX_OFFSET, MAX_SIZE)
    out_body, out_offset, out_size = unpack_chunk_pointer(packed)
    assert out_body == body
    assert out_offset == MAX_OFFSET
    assert out_size == MAX_SIZE


def test_pack_invalid_body_length():
    """Pack with wrong body length raises ValueError."""
    with pytest.raises(ValueError, match="8 bytes"):
        pack_chunk_pointer(b"\x00" * 7, 0, 0)


def test_pack_offset_exceeds_max():
    """Pack with offset > MAX_OFFSET raises ValueError."""
    with pytest.raises(ValueError, match="Offset"):
        pack_chunk_pointer(b"\x00" * 8, MAX_OFFSET + 1, 0)


def test_pack_size_exceeds_max():
    """Pack with size > MAX_SIZE raises ValueError."""
    with pytest.raises(ValueError, match="Size"):
        pack_chunk_pointer(b"\x00" * 8, 0, MAX_SIZE + 1)


def test_unpack_invalid_length():
    """Unpack with wrong data length raises ValueError."""
    with pytest.raises(ValueError, match="Expected 16 bytes"):
        unpack_chunk_pointer(b"\x00" * 15)


# --- IDF calculation tests ---


def test_calculate_idf_normal():
    """IDF with typical values."""
    result = calculate_idf(freq=10, total_assets=1000)
    expected = math.log(1000 / (1 + 10))
    assert result == pytest.approx(expected)


def test_calculate_idf_zero_freq():
    """IDF with zero document frequency."""
    result = calculate_idf(freq=0, total_assets=100)
    expected = math.log(100 / 1)
    assert result == pytest.approx(expected)


def test_calculate_idf_zero_total():
    """IDF with zero total assets returns 0.0."""
    assert calculate_idf(freq=5, total_assets=0) == 0.0


def test_calculate_idf_negative_total():
    """IDF with negative total assets returns 0.0."""
    assert calculate_idf(freq=5, total_assets=-10) == 0.0


# --- count_doc_freq tests ---


@pytest.fixture
def lmdb_sp_env(tmp_path):
    """Create a temporary LMDB environment with dupsort database for testing."""
    env = lmdb.open(
        str(tmp_path / "test.lmdb"),
        max_dbs=2,
        subdir=False,
        map_size=10 * 1024 * 1024,
    )
    db = env.open_db(b"simprints", dupsort=True, dupfixed=True)
    yield env, db
    env.close()


def test_delete_asset_simprints_empty_db(lmdb_sp_env):
    """Delete from empty database returns 0."""
    env, db = lmdb_sp_env
    body = b"\x01" * 8
    with env.begin(write=True) as txn:
        assert delete_asset_simprints(txn, db, body) == 0


def test_delete_asset_simprints_removes_target(lmdb_sp_env):
    """Delete removes all entries for target asset, preserves others."""
    env, db = lmdb_sp_env
    body_a = b"\x01" * 8
    body_b = b"\x02" * 8
    sp1 = b"\xaa" * 8
    sp2 = b"\xbb" * 8

    with env.begin(write=True) as txn:
        # Asset A has sp1 and sp2
        txn.put(sp1, pack_chunk_pointer(body_a, 0, 100), db=db)
        txn.put(sp2, pack_chunk_pointer(body_a, 200, 300), db=db)
        # Asset B has sp1
        txn.put(sp1, pack_chunk_pointer(body_b, 0, 150), db=db)

    # Delete asset A entries
    with env.begin(write=True) as txn:
        deleted = delete_asset_simprints(txn, db, body_a)
        assert deleted == 2

    # Verify only asset B's entry remains
    with env.begin() as txn:
        cursor = txn.cursor(db)
        entries = list(cursor.iternext(keys=True, values=True))
        assert len(entries) == 1
        assert entries[0][1][:8] == body_b


def test_delete_asset_simprints_no_match(lmdb_sp_env):
    """Delete with non-matching body returns 0, preserves all entries."""
    env, db = lmdb_sp_env
    body_a = b"\x01" * 8
    body_missing = b"\xff" * 8
    sp = b"\xaa" * 8

    with env.begin(write=True) as txn:
        txn.put(sp, pack_chunk_pointer(body_a, 0, 100), db=db)

    with env.begin(write=True) as txn:
        assert delete_asset_simprints(txn, db, body_missing) == 0

    with env.begin() as txn:
        assert txn.stat(db=db)["entries"] == 1


def test_count_doc_freq_not_found(lmdb_sp_env):
    """Key not in database returns 0."""
    env, db = lmdb_sp_env
    with env.begin() as txn:
        assert count_doc_freq(txn, db, b"\xaa" * 8) == 0


def test_count_doc_freq_single(lmdb_sp_env):
    """Single asset with one chunk returns 1."""
    env, db = lmdb_sp_env
    sp_key = b"\xaa" * 8
    body = b"\x01" * 8
    chunk_ptr = pack_chunk_pointer(body, 0, 100)

    with env.begin(write=True) as txn:
        txn.put(sp_key, chunk_ptr, db=db)

    with env.begin() as txn:
        assert count_doc_freq(txn, db, sp_key) == 1


def test_count_doc_freq_multiple_assets(lmdb_sp_env):
    """Multiple assets with same simprint returns correct count."""
    env, db = lmdb_sp_env
    sp_key = b"\xbb" * 8

    with env.begin(write=True) as txn:
        for i in range(5):
            body = struct.pack(">Q", i + 1)
            chunk_ptr = pack_chunk_pointer(body, i * 100, 50)
            txn.put(sp_key, chunk_ptr, db=db)

    with env.begin() as txn:
        assert count_doc_freq(txn, db, sp_key) == 5


def test_count_doc_freq_same_asset_multiple_chunks(lmdb_sp_env):
    """Same asset with multiple chunks counts as 1."""
    env, db = lmdb_sp_env
    sp_key = b"\xcc" * 8
    body = b"\x01" * 8

    with env.begin(write=True) as txn:
        # Same body, different offsets - but dupdata=False means only one is stored
        txn.put(sp_key, pack_chunk_pointer(body, 0, 100), db=db)
        txn.put(sp_key, pack_chunk_pointer(body, 100, 200), db=db)

    with env.begin() as txn:
        freq = count_doc_freq(txn, db, sp_key)
        # dupfixed+dupsort stores both entries (different value bytes), but same asset body
        assert freq == 1


def test_count_doc_freq_dup_limit(lmdb_sp_env):
    """Dup limit caps the number of dups scanned."""
    env, db = lmdb_sp_env
    sp_key = b"\xdd" * 8

    with env.begin(write=True) as txn:
        for i in range(10):
            body = struct.pack(">Q", i + 1)
            chunk_ptr = pack_chunk_pointer(body, 0, 100)
            txn.put(sp_key, chunk_ptr, db=db)

    with env.begin() as txn:
        # With dup_limit=5, we only scan 5 entries
        freq = count_doc_freq(txn, db, sp_key, dup_limit=5)
        assert freq == 5


# --- search_simprints_exact tests ---


def test_search_exact_empty_query(lmdb_sp_env):
    """Empty query returns empty results."""
    env, db = lmdb_sp_env
    with env.begin() as txn:
        results = search_simprints_exact(txn, db, [], total_assets=0, limit=10, threshold=0.0, detailed=True)
    assert results == []


def test_search_exact_no_match(lmdb_sp_env):
    """Query with no matching simprints returns empty results."""
    env, db = lmdb_sp_env
    with env.begin() as txn:
        results = search_simprints_exact(txn, db, [b"\xff" * 8], total_assets=0, limit=10, threshold=0.0, detailed=True)
    assert results == []


def test_search_exact_single_match(lmdb_sp_env):
    """Single matching simprint returns one result."""
    env, db = lmdb_sp_env
    sp_key = b"\xaa" * 8
    body = b"\x01" * 8
    chunk_ptr = pack_chunk_pointer(body, 0, 500)

    with env.begin(write=True) as txn:
        txn.put(sp_key, chunk_ptr, db=db)

    with env.begin() as txn:
        results = search_simprints_exact(txn, db, [sp_key], total_assets=1, limit=10, threshold=0.0, detailed=True)

    assert len(results) == 1
    assert results[0].iscc_id_body == body
    assert results[0].score == 1.0  # 1/1 coverage, single match = quality 1.0
    assert results[0].queried == 1
    assert results[0].matches == 1
    assert results[0].chunks is not None
    assert len(results[0].chunks) == 1
    assert results[0].chunks[0].offset == 0
    assert results[0].chunks[0].size == 500


def test_search_exact_multi_asset(lmdb_sp_env):
    """Multiple assets matching same simprint returns sorted results."""
    env, db = lmdb_sp_env
    sp1 = b"\xaa" * 8
    sp2 = b"\xbb" * 8

    body_a = b"\x01" * 8
    body_b = b"\x02" * 8

    with env.begin(write=True) as txn:
        # Asset A matches both simprints
        txn.put(sp1, pack_chunk_pointer(body_a, 0, 100), db=db)
        txn.put(sp2, pack_chunk_pointer(body_a, 100, 200), db=db)
        # Asset B matches only first simprint
        txn.put(sp1, pack_chunk_pointer(body_b, 0, 150), db=db)

    with env.begin() as txn:
        results = search_simprints_exact(txn, db, [sp1, sp2], total_assets=2, limit=10, threshold=0.0, detailed=True)

    assert len(results) == 2
    # Asset A has 2/2 coverage, Asset B has 1/2 coverage
    # Both sorted by score descending, then by iscc_id_body for ties
    assert results[0].iscc_id_body == body_a
    assert results[0].score >= results[1].score
    assert results[0].matches == 2
    assert results[1].matches == 1


def test_search_exact_threshold(lmdb_sp_env):
    """Threshold filters out low-scoring matches."""
    env, db = lmdb_sp_env
    sp1 = b"\xaa" * 8
    sp2 = b"\xbb" * 8

    body_a = b"\x01" * 8
    body_b = b"\x02" * 8

    with env.begin(write=True) as txn:
        # Asset A: matches both sp1 and sp2 (high coverage)
        txn.put(sp1, pack_chunk_pointer(body_a, 0, 100), db=db)
        txn.put(sp2, pack_chunk_pointer(body_a, 100, 200), db=db)
        # Asset B: matches only sp1 of 2 queried (coverage = 0.5)
        txn.put(sp1, pack_chunk_pointer(body_b, 0, 150), db=db)

    with env.begin() as txn:
        # Use threshold that filters Asset B (coverage 0.5) but keeps Asset A
        # Asset A: coverage=1.0, quality depends on freq distribution
        # Asset B: coverage=0.5, quality=1.0 (single match), score=0.5
        results_all = search_simprints_exact(
            txn, db, [sp1, sp2], total_assets=2, limit=10, threshold=0.0, detailed=True
        )
        assert len(results_all) == 2

        # With threshold=0.5 (exclusive), only Asset A should pass since its score >= 0.5
        # Asset B has score=0.5, exactly at threshold, so it should also pass
        results_filtered = search_simprints_exact(
            txn, db, [sp1, sp2], total_assets=2, limit=10, threshold=0.51, detailed=True
        )
        # Asset B (score=0.5) filtered out
        assert len(results_filtered) < len(results_all)


def test_search_exact_limit(lmdb_sp_env):
    """Limit caps the number of results returned."""
    env, db = lmdb_sp_env
    sp = b"\xaa" * 8

    with env.begin(write=True) as txn:
        for i in range(10):
            body = struct.pack(">Q", i + 1)
            txn.put(sp, pack_chunk_pointer(body, 0, 100), db=db)

    with env.begin() as txn:
        results = search_simprints_exact(txn, db, [sp], total_assets=10, limit=3, threshold=0.0, detailed=True)

    assert len(results) == 3


def test_search_exact_detailed_false(lmdb_sp_env):
    """When detailed=False, chunks should be None."""
    env, db = lmdb_sp_env
    sp = b"\xaa" * 8
    body = b"\x01" * 8

    with env.begin(write=True) as txn:
        txn.put(sp, pack_chunk_pointer(body, 0, 100), db=db)

    with env.begin() as txn:
        results = search_simprints_exact(txn, db, [sp], total_assets=1, limit=10, threshold=0.0, detailed=False)

    assert len(results) == 1
    assert results[0].chunks is None


def test_calculate_coverage_quality_score_empty():
    """Empty matches returns 0.0 score."""
    assert _calculate_coverage_quality_score([], {}, 5) == 0.0


def test_calculate_coverage_quality_score_duplicate_query_simprint():
    """Same query simprint appearing multiple times keeps best (lowest) frequency."""
    sp = b"\xaa" * 8
    matches = [
        (sp, sp, 0, 100),  # First occurrence, will get freq=5
        (sp, sp, 200, 300),  # Second occurrence, will get freq=5 too
    ]
    doc_frequencies = {sp: 5}
    score = _calculate_coverage_quality_score(matches, doc_frequencies, 1)
    # coverage = 1/1 = 1.0, single unique query simprint -> quality = 1.0
    assert score == 1.0


def test_search_exact_dup_limit(lmdb_sp_env):
    """Dup limit in search_simprints_exact caps per-simprint scanning."""
    env, db = lmdb_sp_env
    sp = b"\xaa" * 8

    with env.begin(write=True) as txn:
        for i in range(10):
            body = struct.pack(">Q", i + 1)
            txn.put(sp, pack_chunk_pointer(body, 0, 100), db=db)

    with env.begin() as txn:
        # With dup_limit=3, only 3 duplicates scanned per simprint
        results = search_simprints_exact(
            txn, db, [sp], total_assets=10, limit=10, threshold=0.0, detailed=True, dup_limit=3
        )

    assert len(results) == 3


def test_search_exact_same_simprint_different_freqs(lmdb_sp_env):
    """Same query simprint matched by multiple assets with different frequencies."""
    env, db = lmdb_sp_env
    sp1 = b"\xaa" * 8
    sp2 = b"\xbb" * 8

    body_a = b"\x01" * 8
    body_b = b"\x02" * 8

    with env.begin(write=True) as txn:
        # sp1 in both assets (freq=2), sp2 only in asset A (freq=1)
        txn.put(sp1, pack_chunk_pointer(body_a, 0, 100), db=db)
        txn.put(sp2, pack_chunk_pointer(body_a, 100, 200), db=db)
        txn.put(sp1, pack_chunk_pointer(body_b, 0, 150), db=db)

    with env.begin() as txn:
        results = search_simprints_exact(txn, db, [sp1, sp2], total_assets=2, limit=10, threshold=0.0, detailed=True)

    # Both assets should be returned
    assert len(results) == 2
    # Asset A: 2 matches, Asset B: 1 match
    result_a = next(r for r in results if r.iscc_id_body == body_a)
    result_b = next(r for r in results if r.iscc_id_body == body_b)
    assert result_a.matches == 2
    assert result_b.matches == 1


def test_search_exact_same_freq_all_matches(lmdb_sp_env):
    """All matched simprints have equal frequency (tests equal-freq quality branch)."""
    env, db = lmdb_sp_env
    sp1 = b"\xaa" * 8
    sp2 = b"\xbb" * 8

    body_a = b"\x01" * 8
    body_b = b"\x02" * 8

    with env.begin(write=True) as txn:
        # Both simprints appear in both assets (freq=2 for both)
        txn.put(sp1, pack_chunk_pointer(body_a, 0, 100), db=db)
        txn.put(sp2, pack_chunk_pointer(body_a, 100, 200), db=db)
        txn.put(sp1, pack_chunk_pointer(body_b, 0, 150), db=db)
        txn.put(sp2, pack_chunk_pointer(body_b, 100, 250), db=db)

    with env.begin() as txn:
        results = search_simprints_exact(txn, db, [sp1, sp2], total_assets=2, limit=10, threshold=0.0, detailed=True)

    # Both assets match both simprints -> coverage=1.0, quality=1.0 (equal freqs)
    assert len(results) == 2
    for r in results:
        assert r.score == 1.0  # coverage=1.0 * quality=1.0
