"""
Log-record to IsccEntry conversion for the IDP aggregator (pure, Sans-IO).

Discriminates log entries on the pinned ``note.$schema`` values and converts
declarations into IsccEntry objects ready for ``add_assets``. Unknown note
types are skipped, never rejected (IDP forward-compatibility rule: a future
schema version bump is an unknown note type with unreviewed field semantics).
"""

import json
from iscc_search.aggregator import NETWORKS
from iscc_search.indexes.common import validate_iscc_id
from iscc_search.models import IsccCode, IsccID, IsccUnit
from iscc_search.schema import IsccEntry

# Pinned note schema URIs — match exactly, do not semver-wildcard (see module docstring).
DECLARATION_SCHEMA = "http://purl.org/iscc/schema/iscc-note-0.8.0.json"
DELETION_SCHEMA = "http://purl.org/iscc/schema/iscc-note-delete-0.8.0.json"

# Skip-reason vocabulary returned by record_to_entry ("ok" = converted).
REASONS = ("ok", "deletion", "unknown_schema", "malformed", "realm_mismatch")


def expand_gateway(template, iscc_id, iscc_code, datahash):
    # type: (str, str, str, str) -> str
    """
    Expand IDP gateway URI-template variables into a concrete URL.

    Matches iscc-hub's expansion semantics: {iscc_id} and {iscc_code}
    substitute the lowercase prefix-less base32 form (the iscc: URI body per
    ISO 24138, no "ISCC:"), {datahash} the lowercase hex multihash. The
    schema-admitted operator forms {/var} and {.var} expand to "/value" and
    ".value". A plain URL passes through unchanged.

    :param template: Gateway URL or RFC 6570 URI template from the note
    :param iscc_id: Canonical ISCC-ID of the declaration
    :param iscc_code: Canonical ISCC-CODE of the declaration
    :param datahash: BLAKE3 multihash of the declared content
    :return: Concrete gateway URL
    """
    values = {
        "iscc_id": iscc_id.removeprefix("ISCC:").lower(),
        "iscc_code": iscc_code.removeprefix("ISCC:").lower(),
        "datahash": datahash,
    }
    for var, value in values.items():
        template = template.replace("{" + var + "}", value)
        template = template.replace("{/" + var + "}", "/" + value)
        template = template.replace("{." + var + "}", "." + value)
    return template


def record_to_entry(record, network):
    # type: (bytes, str) -> tuple[IsccEntry | None, str]
    """
    Convert a log-entry record into an IsccEntry, classifying skips.

    Declarations index the ISCC-CODE decomposed into units merged (deduped)
    with any extra note.units; the optional gateway is template-expanded at
    ingestion and stored as the only metadata field. The function never
    raises — the caller keeps per-reason counters and does its own logging.

    :param record: JCS-canonical log-entry JSON bytes ({$schema, iscc_id, note})
    :param network: Deployment network ("testnet" or "mainnet") for realm checks
    :return: (entry, "ok") for a declaration, else (None, reason) with reason
        one of "deletion", "unknown_schema", "malformed", "realm_mismatch"
    """
    try:
        parsed = json.loads(record)
        note = parsed["note"]
        note_schema = note["$schema"]
    except (ValueError, KeyError, TypeError):
        return None, "malformed"
    if note_schema == DELETION_SCHEMA:
        return None, "deletion"
    if note_schema != DECLARATION_SCHEMA:
        return None, "unknown_schema"
    try:
        iscc_id = parsed["iscc_id"]
        validate_iscc_id(iscc_id)
        if IsccID(iscc_id).realm_id != NETWORKS[network]["realm"]:
            return None, "realm_mismatch"
        iscc_code = note["iscc_code"]
        units = [str(unit) for unit in IsccCode(iscc_code).units]
        # IsccUnit(...).unit_type raises for undecodable units, so a bad extra unit
        # is classified "malformed" here instead of failing the whole batch in add_assets
        extra = [unit for unit in note.get("units", []) if IsccUnit(unit).unit_type]
        units = list(dict.fromkeys(units + extra))
        metadata = None
        if note.get("gateway"):
            metadata = {"gateway": expand_gateway(note["gateway"], iscc_id, iscc_code, note["datahash"])}
        entry = IsccEntry(iscc_id=iscc_id, iscc_code=iscc_code, units=units, metadata=metadata)
    except Exception:
        return None, "malformed"
    return entry, "ok"
