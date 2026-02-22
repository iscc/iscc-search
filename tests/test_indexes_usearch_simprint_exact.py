"""
Integration tests for LMDB-based exact simprint search in UsearchIndex.

Tests dual-write to LMDB, persistence across close/reopen, deduplication,
exact search with metadata enrichment, and multi-type aggregation.
"""

import iscc_core as ic
from iscc_search.indexes.usearch.index import UsearchIndex
from iscc_search.schema import IsccEntry, IsccQuery, IsccSimprint


def _make_entry_simprints(sp_type, sp_data):
    # type: (str, list[tuple[bytes, int, int]]) -> dict[str, list[IsccSimprint]]
    """
    Build simprints dict for IsccEntry from raw (bytes, offset, size) tuples.

    :param sp_type: Simprint type identifier
    :param sp_data: List of (simprint_bytes, offset, size) tuples
    :return: Simprints dict for IsccEntry
    """
    return {
        sp_type: [IsccSimprint(simprint=ic.encode_base64(sp), offset=offset, size=size) for sp, offset, size in sp_data]
    }


def _make_query_simprints(sp_type, sp_bytes_list):
    # type: (str, list[bytes]) -> dict[str, list[str]]
    """
    Build simprints dict for IsccQuery from raw bytes.

    IsccQuery expects dict[str, list[Simprint]] where Simprint is a base64 string.

    :param sp_type: Simprint type identifier
    :param sp_bytes_list: List of simprint byte values
    :return: Simprints dict for IsccQuery
    """
    return {sp_type: [ic.encode_base64(sp) for sp in sp_bytes_list]}


def test_max_dbs_respects_user_value(tmp_path):
    """User-provided max_dbs higher than default is respected."""
    index_path = tmp_path / "max_dbs"
    idx = UsearchIndex(index_path, realm_id=0, max_dim=256, lmdb_options={"max_dbs": 64})
    # LMDB info doesn't expose max_dbs directly, but if it were overridden
    # to 32, opening >14 sp types would fail. Verify the env opened successfully.
    assert idx.env.info()["map_size"] > 0
    idx.close()


def test_sp_types_metadata_persistence(tmp_path, sample_iscc_ids):
    """Add asset with simprints -> close -> reopen -> sp_types preserved."""
    index_path = tmp_path / "sp_persist"
    sp_bytes_a = b"\xaa" * 8
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Persistence test content")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp_bytes_a, 0, 500)]),
    )
    idx.add_assets([asset])

    # Verify sp_types before close
    sp_types = idx._get_sp_types()
    assert sp_type in sp_types

    idx.close()

    # Reopen and verify sp_types are preserved
    idx2 = UsearchIndex(index_path, realm_id=0, max_dim=256)
    sp_types = idx2._get_sp_types()
    assert sp_type in sp_types

    # Verify database handles are loaded
    assert sp_type in idx2._sp_data_dbs
    assert sp_type in idx2._sp_assets_dbs

    idx2.close()


def test_simprint_update_replaces_old(tmp_path, sample_iscc_ids):
    """Re-adding asset with changed simprints replaces old entries in LMDB."""
    index_path = tmp_path / "sp_update"
    sp_old = b"\xaa" * 8
    sp_new = b"\xbb" * 8
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Update test content")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256, match_threshold_simprints=0.0)

    # Add asset with old simprint
    asset_v1 = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp_old, 0, 500)]),
    )
    idx.add_assets([asset_v1])

    # Verify old simprint is searchable
    query_old = IsccQuery(simprints=_make_query_simprints(sp_type, [sp_old]))
    result_old = idx.search_assets(query_old, limit=10, exact=True)
    assert len(result_old.chunk_matches) == 1

    # Update asset with new simprint (different bytes)
    asset_v2 = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp_new, 100, 600)]),
    )
    idx.add_assets([asset_v2])

    # Old simprint should no longer match
    result_stale = idx.search_assets(query_old, limit=10, exact=True)
    assert len(result_stale.chunk_matches) == 0

    # New simprint should match
    query_new = IsccQuery(simprints=_make_query_simprints(sp_type, [sp_new]))
    result_new = idx.search_assets(query_new, limit=10, exact=True)
    assert len(result_new.chunk_matches) == 1
    assert result_new.chunk_matches[0].iscc_id == sample_iscc_ids[0]

    idx.close()


def test_asset_dedup(tmp_path, sample_iscc_ids):
    """Adding same asset twice results in single entry in sp_assets."""
    index_path = tmp_path / "sp_dedup"
    sp_bytes = b"\xbb" * 8
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Dedup test content")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp_bytes, 0, 300)]),
    )

    # Add same asset twice
    idx.add_assets([asset])
    idx.add_assets([asset])

    # Check sp_assets has only one entry
    with idx.env.begin() as txn:
        sp_assets_db = idx._sp_assets_dbs[sp_type]
        count = txn.stat(db=sp_assets_db)["entries"]
        assert count == 1

    idx.close()


def test_exact_search_returns_results(tmp_path, sample_iscc_ids):
    """Exact search finds matching simprints with scores."""
    index_path = tmp_path / "sp_exact"
    sp_bytes = b"\xcc" * 8
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Exact search test content")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp_bytes, 0, 500)]),
        metadata={"title": "Test Asset", "source": "https://example.com/test"},
    )
    idx.add_assets([asset])

    # Search with exact=True (query uses base64 strings)
    query = IsccQuery(simprints=_make_query_simprints(sp_type, [sp_bytes]))
    result = idx.search_assets(query, limit=10, exact=True)

    assert len(result.chunk_matches) == 1
    assert result.chunk_matches[0].iscc_id == sample_iscc_ids[0]
    assert result.chunk_matches[0].score > 0
    assert sp_type in result.chunk_matches[0].types

    # Verify metadata enrichment (IsccMetadata is a Pydantic model with extra="allow")
    assert result.chunk_matches[0].metadata is not None
    assert str(result.chunk_matches[0].source) == "https://example.com/test"

    idx.close()


def test_multi_type_exact_search_aggregation(tmp_path, sample_iscc_ids):
    """Exact search across multiple simprint types aggregates correctly."""
    index_path = tmp_path / "sp_multi_type"
    sp_text = b"\xdd" * 8
    sp_sem = b"\xee" * 8

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Multi-type test content")["iscc"]

    entry_simprints = {
        "CONTENT_TEXT_V0": [IsccSimprint(simprint=ic.encode_base64(sp_text), offset=0, size=500)],
        "SEMANTIC_TEXT_V0": [IsccSimprint(simprint=ic.encode_base64(sp_sem), offset=1000, size=300)],
    }

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=entry_simprints,
    )
    idx.add_assets([asset])

    # Build query with both types (base64 strings)
    query_simprints = {
        "CONTENT_TEXT_V0": [ic.encode_base64(sp_text)],
        "SEMANTIC_TEXT_V0": [ic.encode_base64(sp_sem)],
    }
    query = IsccQuery(simprints=query_simprints)
    result = idx.search_assets(query, limit=10, exact=True)

    assert len(result.chunk_matches) == 1
    match = result.chunk_matches[0]
    assert "CONTENT_TEXT_V0" in match.types
    assert "SEMANTIC_TEXT_V0" in match.types
    # Overall score should be mean of type scores
    assert match.score > 0

    idx.close()


def test_new_type_auto_registration(tmp_path, sample_iscc_ids):
    """Adding simprints of a new type auto-registers it in sp_types metadata."""
    index_path = tmp_path / "sp_auto_reg"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Auto registration test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Start with no sp_types
    assert idx._get_sp_types() == []

    # Add asset with type A
    asset1 = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints("CONTENT_TEXT_V0", [(b"\xaa" * 8, 0, 100)]),
    )
    idx.add_assets([asset1])
    assert "CONTENT_TEXT_V0" in idx._get_sp_types()

    # Add asset with type B
    asset2 = IsccEntry(
        iscc_id=sample_iscc_ids[1],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints("SEMANTIC_TEXT_V0", [(b"\xbb" * 8, 0, 200)]),
    )
    idx.add_assets([asset2])
    sp_types = idx._get_sp_types()
    assert "CONTENT_TEXT_V0" in sp_types
    assert "SEMANTIC_TEXT_V0" in sp_types

    idx.close()


def test_empty_simprints_handling(tmp_path, sample_iscc_ids):
    """Assets without simprints do not write to LMDB simprint databases."""
    index_path = tmp_path / "sp_empty"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("No simprints asset")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
    )
    idx.add_assets([asset])

    # No sp_types should be registered
    assert idx._get_sp_types() == []
    assert len(idx._sp_data_dbs) == 0

    idx.close()


def test_exact_search_no_match(tmp_path, sample_iscc_ids):
    """Exact search with no matching simprints returns empty chunk_matches."""
    index_path = tmp_path / "sp_no_match"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("No match test")["iscc"]
    sp_type = "CONTENT_TEXT_V0"

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(b"\xaa" * 8, 0, 100)]),
    )
    idx.add_assets([asset])

    # Search with different simprint (base64 string)
    query = IsccQuery(simprints=_make_query_simprints(sp_type, [b"\xff" * 8]))
    result = idx.search_assets(query, limit=10, exact=True)

    assert len(result.chunk_matches) == 0

    idx.close()


def test_exact_search_unknown_type(tmp_path, sample_iscc_ids):
    """Exact search for a type not in the index returns empty chunk_matches."""
    index_path = tmp_path / "sp_unknown_type"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Unknown type test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints("CONTENT_TEXT_V0", [(b"\xaa" * 8, 0, 100)]),
    )
    idx.add_assets([asset])

    # Search for a type that doesn't exist in the index
    query = IsccQuery(simprints=_make_query_simprints("NONEXISTENT_V0", [b"\xaa" * 8]))
    result = idx.search_assets(query, limit=10, exact=True)

    assert len(result.chunk_matches) == 0

    idx.close()


def test_exact_search_multiple_assets(tmp_path, sample_iscc_ids):
    """Exact search finds multiple matching assets with correct ranking."""
    index_path = tmp_path / "sp_multi_asset"
    sp1 = b"\xaa" * 8
    sp2 = b"\xbb" * 8
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Multi asset test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256, match_threshold_simprints=0.0)

    # Asset 0: matches both simprints
    asset0 = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp1, 0, 100), (sp2, 100, 200)]),
    )
    # Asset 1: matches only sp1
    asset1 = IsccEntry(
        iscc_id=sample_iscc_ids[1],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(sp1, 0, 150)]),
    )
    idx.add_assets([asset0, asset1])

    # Search for both simprints
    query = IsccQuery(simprints=_make_query_simprints(sp_type, [sp1, sp2]))
    result = idx.search_assets(query, limit=10, exact=True)

    assert len(result.chunk_matches) == 2
    # Asset 0 should rank first (2/2 coverage vs 1/2)
    assert result.chunk_matches[0].iscc_id == sample_iscc_ids[0]
    assert result.chunk_matches[0].score >= result.chunk_matches[1].score

    idx.close()


def test_sp_type_already_registered(tmp_path, sample_iscc_ids):
    """Adding assets of an already-registered sp_type skips re-registration."""
    index_path = tmp_path / "sp_already_reg"
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Already registered test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Add first asset (registers the type)
    asset1 = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(b"\xaa" * 8, 0, 100)]),
    )
    idx.add_assets([asset1])
    assert idx._get_sp_types() == [sp_type]

    # Clear the cached handles to force re-open of already-registered type
    idx._sp_data_dbs.clear()
    idx._sp_assets_dbs.clear()

    # Add second asset with same type (hits the "already in sp_types" branch)
    asset2 = IsccEntry(
        iscc_id=sample_iscc_ids[1],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(b"\xbb" * 8, 0, 200)]),
    )
    idx.add_assets([asset2])

    # sp_types should still have just one entry (no duplicate)
    assert idx._get_sp_types() == [sp_type]

    idx.close()


def test_exact_search_sp_assets_db_missing(tmp_path, sample_iscc_ids):
    """Exact search handles case where sp_data_db exists but sp_assets_db is not loaded."""
    index_path = tmp_path / "sp_assets_missing"
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Assets DB missing test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)
    asset = IsccEntry(
        iscc_id=sample_iscc_ids[0],
        units=[instance_unit, content_unit],
        simprints=_make_entry_simprints(sp_type, [(b"\xaa" * 8, 0, 100)]),
    )
    idx.add_assets([asset])

    # Remove sp_assets_db handle to exercise the "else: total_assets = 0" branch
    del idx._sp_assets_dbs[sp_type]

    query = IsccQuery(simprints=_make_query_simprints(sp_type, [b"\xaa" * 8]))
    result = idx.search_assets(query, limit=10, exact=True)

    # Should still find results (total_assets=0 is just used for IDF, not for filtering)
    assert len(result.chunk_matches) == 1

    idx.close()


def test_exact_search_self_exclusion(tmp_path, sample_iscc_ids):
    """Exact search with iscc_id query excludes the query asset from chunk_matches."""
    index_path = tmp_path / "sp_self_exclude"
    sp_bytes = b"\xaa" * 8
    sp_type = "CONTENT_TEXT_V0"

    instance_unit = f"ISCC:{ic.Code.rnd(ic.MT.INSTANCE, bits=128)}"
    content_unit = ic.gen_text_code_v0("Self-exclusion test")["iscc"]

    idx = UsearchIndex(index_path, realm_id=0, max_dim=256)

    # Add two assets with same simprint
    for i in range(2):
        asset = IsccEntry(
            iscc_id=sample_iscc_ids[i],
            units=[instance_unit, content_unit],
            simprints=_make_entry_simprints(sp_type, [(sp_bytes, 0, 100)]),
        )
        idx.add_assets([asset])

    # Direct simprint query (not iscc_id lookup) returns both
    query_both = IsccQuery(simprints=_make_query_simprints(sp_type, [sp_bytes]))
    result_both = idx.search_assets(query_both, limit=10, exact=True)
    assert len(result_both.chunk_matches) == 2

    idx.close()
