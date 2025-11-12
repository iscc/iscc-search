"""
Common utilities for ISCC index implementations.

Provides reusable functions for:
- IsccEntry serialization and deserialization
- ISCC-ID and ISCC-UNIT parsing and reconstruction
- Index name and ISCC format validation
- Consistent error handling
"""

import re
import json
import iscc_core as ic
from iscc_search.schema import IsccEntry
from iscc_search.models import IsccUnit, IsccCode


# Validation patterns
INDEX_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*$")


def serialize_asset(asset):
    # type: (IsccEntry) -> bytes
    """
    Serialize IsccEntry to JSON bytes for storage.

    Excludes simprints field to avoid duplication - simprints are stored
    separately in the simprint index for efficient chunk-level search.

    :param asset: IsccEntry instance to serialize
    :return: UTF-8 encoded JSON bytes
    """
    # Exclude simprints to avoid storing them twice (they're in simprint index)
    asset_dict = asset.model_dump(mode="json", exclude_none=True, exclude={"simprints"})
    return json.dumps(asset_dict, separators=(",", ":")).encode("utf-8")


def deserialize_asset(data):
    # type: (bytes) -> IsccEntry
    """
    Deserialize JSON bytes to IsccEntry.

    :param data: UTF-8 encoded JSON bytes
    :return: IsccEntry instance
    :raises ValueError: If JSON is invalid or doesn't match schema
    """
    asset_dict = json.loads(data.decode("utf-8"))
    return IsccEntry(**asset_dict)


def extract_iscc_id_body(iscc_id):
    # type: (str) -> bytes
    """
    Extract 8-byte body from ISCC-ID canonical string.

    ISCC-ID format: "ISCC:" + base32(2-byte header + 8-byte body)
    This function decodes the full 10 bytes and returns bytes [2:10] (the body).

    :param iscc_id: ISCC-ID canonical string (e.g., "ISCC:MAAQ...")
    :return: 8-byte ISCC-ID body (digest)
    :raises ValueError: If ISCC-ID format is invalid
    """
    validate_iscc_id(iscc_id)
    # Decode to get full 10 bytes (2-byte header + 8-byte body)
    code_bytes = ic.decode_base32(iscc_id.split(":")[-1])
    return code_bytes[2:]  # Return only body (skip header)


def extract_realm_id(iscc_id):
    # type: (str) -> int
    """
    Extract realm ID (0 or 1) from ISCC-ID header.

    The realm is encoded in the header's first 2 bytes.

    :param iscc_id: ISCC-ID canonical string
    :return: Realm ID (0 or 1)
    :raises ValueError: If ISCC-ID format is invalid
    """
    validate_iscc_id(iscc_id)
    code_bytes = ic.decode_base32(iscc_id.split(":")[-1])
    # Decode returns: maintype, subtype, version, length, body
    _mt, realm, _vs, _len, _body = ic.decode_header(code_bytes)
    return realm


def reconstruct_iscc_id(body, realm_id):
    # type: (bytes, int) -> str
    """
    Reconstruct ISCC-ID canonical string from 8-byte body and realm ID.

    Creates proper ISCC-ID header for the given realm and combines with body.

    :param body: 8-byte ISCC-ID body (digest)
    :param realm_id: Realm ID (0 or 1)
    :return: ISCC-ID canonical string (e.g., "ISCC:MAAQ...")
    :raises ValueError: If realm_id is not 0 or 1, or body is not 8 bytes
    """
    if realm_id not in (0, 1):
        raise ValueError(f"Invalid realm_id {realm_id}, must be 0 or 1")
    if len(body) != 8:
        raise ValueError(f"ISCC-ID body must be 8 bytes, got {len(body)}")

    # Create header for ISCC-ID with proper realm
    header = ic.encode_header(ic.MT.ID, realm_id, ic.VS.V1, 0)
    return "ISCC:" + ic.encode_base32(header + body)


def extract_unit_body(unit):
    # type: (str) -> bytes
    """
    Extract body bytes from ISCC-UNIT canonical string.

    ISCC-UNIT format: "ISCC:" + base32(2-byte header + N-byte body)
    Body length varies: 8, 16, 24, or 32 bytes (for 64, 128, 192, 256-bit units).

    :param unit: ISCC-UNIT canonical string
    :return: Variable-length body bytes
    :raises ValueError: If UNIT format is invalid
    """
    # Use IsccUnit for validation and parsing
    iscc_unit = IsccUnit(unit)
    return iscc_unit.body


def get_unit_type(unit):
    # type: (str) -> str
    """
    Extract unit type string from ISCC-UNIT.

    Unit type format: "{MAINTYPE}_{SUBTYPE}_V{VERSION}"
    Example: "CONTENT_TEXT_V0", "DATA_NONE_V0"

    :param unit: ISCC-UNIT canonical string
    :return: Unit type string
    :raises ValueError: If UNIT format is invalid
    """
    iscc_unit = IsccUnit(unit)
    return iscc_unit.unit_type


def validate_index_name(name):
    # type: (str) -> None
    """
    Validate index name matches required pattern.

    Pattern: ^[a-z][a-z0-9]*$
    - Starts with lowercase letter
    - Followed by zero or more lowercase letters or digits
    - No special characters, underscores, or hyphens

    :param name: Index name to validate
    :raises ValueError: If name doesn't match pattern
    """
    if not INDEX_NAME_PATTERN.match(name):
        raise ValueError(
            f"Invalid index name: '{name}'. "
            f"Must match pattern ^[a-z][a-z0-9]*$ "
            f"(start with lowercase letter, followed by lowercase letters/digits only)"
        )


def validate_iscc_id(iscc_id, expected_realm=None):
    # type: (str, int | None) -> None
    """
    Validate ISCC-ID format and optionally check realm.

    Checks:
    - Starts with "ISCC:"
    - Valid base32 encoding
    - Correct length (10 bytes = 2-byte header + 8-byte body)
    - Main type is ID
    - Length field is 0 (64-bit ISCC-ID v1)
    - Realm matches expected_realm (if provided)

    :param iscc_id: ISCC-ID string to validate
    :param expected_realm: Optional realm ID (0 or 1) to validate against
    :raises ValueError: If ISCC-ID format is invalid or realm doesn't match
    """
    if not iscc_id or not iscc_id.startswith("ISCC:"):
        raise ValueError(f"Invalid ISCC-ID format: '{iscc_id}' (must start with 'ISCC:')")

    try:
        code_bytes = ic.decode_base32(iscc_id.split(":")[-1])
    except Exception as e:
        raise ValueError(f"Invalid ISCC-ID base32 encoding: {e}")

    if len(code_bytes) != 10:
        raise ValueError(
            f"Invalid ISCC-ID length: {len(code_bytes)} bytes (expected 10 bytes = 2-byte header + 8-byte body)"
        )

    # Validate main type is ID
    # Decode returns: maintype, subtype, version, length, body
    mt, realm, _vs, length_field, _body = ic.decode_header(code_bytes)
    if mt != ic.MT.ID:
        raise ValueError(f"Invalid ISCC-ID main type: {mt} (expected {ic.MT.ID})")

    # Validate length field is 0 (64-bit ISCC-ID v1)
    if length_field != 0:
        raise ValueError(
            f"Invalid ISCC-ID length field: {length_field} (expected 0 for 64-bit ISCC-ID v1). "
            f"ISCC-ID '{iscc_id}' appears to be malformed."
        )

    # Validate realm if expected_realm provided
    if expected_realm is not None and realm != expected_realm:
        raise ValueError(
            f"Realm mismatch: ISCC-ID '{iscc_id}' has realm={realm}, "
            f"but expected realm={expected_realm}. "
            f"Cannot query assets from different realm."
        )


def normalize_query(query):
    # type: (IsccQuery) -> IsccQuery
    """
    Normalize query to ensure consistent search behavior across backends.

    Indexes need both representations when possible:
    - LMDB searches by units (more bits, better precision)
    - Memory searches by iscc_code (simple exact matching)

    This function performs bidirectional normalization:

    1. If query has both units and iscc_code → keep both
    2. If query has only units → try to derive iscc_code (if units form valid code)
    3. If query has only iscc_code → derive units from code
    4. If query has only simprints → return as is (simprints-only query)
    5. If query has neither units, iscc_code, nor simprints → raise error

    Note: Not all unit combinations form valid ISCC-CODEs. Units-only queries
    that don't form valid codes (e.g., missing DATA/INSTANCE) will work with
    LMDB backend but return no matches with memory backend.

    :param query: Query object (may have iscc_code, units, simprints, or combination)
    :return: Query with units and/or iscc_code populated (simprints passed through)
    :raises ValueError: If query has neither iscc_code, units, nor simprints
    """
    import iscc_core as ic

    # Case 1: Has both - return as is
    if query.units and query.iscc_code:
        return query

    # Case 2: Has units but no iscc_code - try to derive code from units
    if query.units and not query.iscc_code:
        # Attempt to compose ISCC-CODE from ISCC-UNITs
        # Always use wide=True (handles short/long units transparently)
        try:
            iscc_code = ic.gen_iscc_code_v0(query.units, wide=True)["iscc"]
            return query.model_copy(update={"iscc_code": iscc_code})
        except ValueError:
            # Units don't form a valid ISCC-CODE (e.g., missing DATA/INSTANCE)
            # Return query with units only - works for LMDB backend
            return query

    # Case 3: Has iscc_code but no units - derive units from code
    if query.iscc_code and not query.units:
        # Decompose ISCC-CODE into ISCC-UNITs
        code = IsccCode(query.iscc_code)
        units = [str(unit) for unit in code.units]
        return query.model_copy(update={"units": units})

    # Case 4: Has simprints but no units/iscc_code - return as is (simprints-only query)
    if query.simprints:
        return query

    # Case 5: Neither units nor iscc_code nor simprints - cannot search
    raise ValueError("Query must have 'iscc_code', 'units', or 'simprints' for search")
