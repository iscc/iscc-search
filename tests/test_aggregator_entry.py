"""Tests for log-record to IsccEntry conversion (pure core)."""

import iscc_core as ic
from iscc_search.aggregator.entry import expand_gateway, record_to_entry
from iscc_search.models import IsccCode
from conftest import IDP_DATAHASH, IDP_ISCC_CODE


TESTNET_ID = ic.gen_iscc_id(timestamp=1750000000000000, hub_id=0, realm_id=0)["iscc"]
MAINNET_ID = ic.gen_iscc_id(timestamp=1750000000000000, hub_id=0, realm_id=1)["iscc"]
DECOMPOSED_UNITS = [str(unit) for unit in IsccCode(IDP_ISCC_CODE).units]


def test_declaration_ok(make_log_record):
    """A declaration converts to an IsccEntry with decomposed units and no metadata."""
    record = make_log_record(iscc_id=TESTNET_ID)
    entry, reason = record_to_entry(record, "testnet")
    assert reason == "ok"
    assert entry.iscc_id == TESTNET_ID
    assert entry.iscc_code == IDP_ISCC_CODE
    assert entry.units == DECOMPOSED_UNITS
    assert len(entry.units) >= 2  # iscc_code always yields DATA+INSTANCE minimum (U3 resolution)
    assert entry.metadata is None


def test_declaration_units_merged_and_deduped(make_log_record):
    """Extra note.units merge after the decomposed units without duplicates."""
    extra = "ISCC:AADWN77F73NA44D6X3N4VEUAPOW5HJKGK5JKLNGLNFPOESXWYDVDVUQ"
    record = make_log_record(iscc_id=TESTNET_ID, units=[DECOMPOSED_UNITS[0], extra])
    entry, reason = record_to_entry(record, "testnet")
    assert reason == "ok"
    assert entry.units == DECOMPOSED_UNITS + [extra]


def test_declaration_units_dedup_within_note(make_log_record):
    """Duplicate extra units within note.units collapse to a single entry."""
    extra = "ISCC:AADWN77F73NA44D6X3N4VEUAPOW5HJKGK5JKLNGLNFPOESXWYDVDVUQ"
    record = make_log_record(iscc_id=TESTNET_ID, units=[extra, extra])
    entry, reason = record_to_entry(record, "testnet")
    assert reason == "ok"
    assert entry.units == DECOMPOSED_UNITS + [extra]


def test_declaration_gateway_plain_url(make_log_record):
    """A plain gateway URL is stored unchanged."""
    record = make_log_record(iscc_id=TESTNET_ID, gateway="https://registry.example.com/metadata")
    entry, reason = record_to_entry(record, "testnet")
    assert reason == "ok"
    assert entry.metadata == {"gateway": "https://registry.example.com/metadata"}


def test_declaration_gateway_template_expanded(make_log_record):
    """URI-template variables are expanded to concrete values at ingestion."""
    record = make_log_record(iscc_id=TESTNET_ID, gateway="https://example.com/{iscc_id}/{iscc_code}/{datahash}")
    entry, reason = record_to_entry(record, "testnet")
    assert reason == "ok"
    iscc_id_clean = TESTNET_ID.removeprefix("ISCC:").lower()
    iscc_code_clean = IDP_ISCC_CODE.removeprefix("ISCC:").lower()
    assert entry.metadata == {"gateway": f"https://example.com/{iscc_id_clean}/{iscc_code_clean}/{IDP_DATAHASH}"}


def test_deletion_skipped(make_log_record):
    """A deletion record is skipped with its own reason (D2)."""
    record = make_log_record(iscc_id=TESTNET_ID, deletion=True)
    assert record_to_entry(record, "testnet") == (None, "deletion")


def test_unknown_schema_skipped(make_log_record):
    """Future schema version bumps are unknown note types, skipped not rejected."""
    bumped_note = make_log_record(iscc_id=TESTNET_ID, note_schema="http://purl.org/iscc/schema/iscc-note-0.9.0.json")
    assert record_to_entry(bumped_note, "testnet") == (None, "unknown_schema")
    bumped_delete = make_log_record(
        iscc_id=TESTNET_ID, deletion=True, note_schema="http://purl.org/iscc/schema/iscc-note-delete-0.9.0.json"
    )
    assert record_to_entry(bumped_delete, "testnet") == (None, "unknown_schema")


def test_malformed_records(make_log_record):
    """Unparseable or structurally broken records classify as malformed."""
    assert record_to_entry(b"not json", "testnet") == (None, "malformed")
    assert record_to_entry(b'{"iscc_id": "ISCC:MAIGG6O2AW3AAAAA"}', "testnet") == (None, "malformed")
    assert record_to_entry(b'{"note": {"no_schema": true}}', "testnet") == (None, "malformed")
    assert record_to_entry(b'"just a string"', "testnet") == (None, "malformed")
    # Declaration with an invalid iscc_code fails during conversion
    bad_code = make_log_record(iscc_id=TESTNET_ID, iscc_code="ISCC:INVALID!")
    assert record_to_entry(bad_code, "testnet") == (None, "malformed")


def test_malformed_extra_units(make_log_record):
    """Undecodable extra units classify as malformed instead of failing the whole index batch."""
    record = make_log_record(iscc_id=TESTNET_ID, units=["not-a-valid-unit"])
    assert record_to_entry(record, "testnet") == (None, "malformed")
    # units accidentally a string instead of a list (iterates as characters)
    record = make_log_record(iscc_id=TESTNET_ID, units=DECOMPOSED_UNITS[0])
    assert record_to_entry(record, "testnet") == (None, "malformed")


def test_non_id_iscc_id_malformed(make_log_record):
    """An iscc_id whose MainType is not ID is malformed, even if its subtype bits match the realm."""
    meta_unit = ic.gen_meta_code_v0("Spoofed Title")["iscc"]  # META unit: subtype bits == realm 0
    record = make_log_record(iscc_id=meta_unit)
    assert record_to_entry(record, "testnet") == (None, "malformed")


def test_realm_mismatch(make_log_record):
    """Records minted for the other network's realm are skipped."""
    mainnet_record = make_log_record(iscc_id=MAINNET_ID)
    assert record_to_entry(mainnet_record, "testnet") == (None, "realm_mismatch")
    testnet_record = make_log_record(iscc_id=TESTNET_ID)
    assert record_to_entry(testnet_record, "mainnet") == (None, "realm_mismatch")


def test_expand_gateway_operator_forms():
    """The schema-admitted {/var} and {.var} operator forms expand correctly."""
    url = expand_gateway("https://example.com{/iscc_id}", "ISCC:MAIGG6O2AW3AAAAA", "ISCC:KACW", "1e20ff")
    assert url == "https://example.com/maigg6o2aw3aaaaa"
    url = expand_gateway("https://example.com/h{.datahash}", "ISCC:MAIGG6O2AW3AAAAA", "ISCC:KACW", "1e20ff")
    assert url == "https://example.com/h.1e20ff"
    url = expand_gateway("https://example.com/{iscc_code}", "ISCC:MAIGG6O2AW3AAAAA", "ISCC:KACW", "1e20ff")
    assert url == "https://example.com/kacw"
