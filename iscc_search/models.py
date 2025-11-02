"""
# Types and convenience classes for handling ISCCs

## Terms and Definitions

- **ISCC** - Any ISCC-CODE, ISCC-UNIT, or ISCC-ID
- **ISCC-HEADER** - Self-describing 2-byte header for V1 components (3 bytes for future versions). The first 12
    bits encode MainType, SubType, and Version. Additional bits encode Length for variable-length ISCCs.
- **ISCC-BODY** - Actual payload of an ISCC, similarity preserving compact binary code, hash or timestamp
- **ISCC-DIGEST** - Binary representation of complete ISCC (ISCC-HEADER + ISCC-BODY).
- **ISCC-SEQUENCE** - Binary sequence of ISCC-DIGESTS
- **ISCC-UNIT** - ISCC-HEADER + ISCC-BODY where the ISCC-BODY is calculated from a single algorithm
- **ISCC-CODE** - ISCC-HEADER + ISCC-BODY where the ISCC-BODY is a sequence of multiple ISCC-UNIT BODYs
    - DATA and INSTANCE are the minimum required mandatory ISCC-UNITS for a valid ISCC-CODE
- **ISCC-ID** - Globally unique digital asset identifier (ISCC-HEADER + 52-bit timestamp + 12-bit hub_id)
- **SIMPRINT** - Headerless base64 encoded similarity hash that describes a content segment (granular feature)
- **UNIT-TYPE**: Identifier for ISCC-UNIT types that can be indexed together with meaningful similarity search
"""

import time
from functools import cached_property, cache
from random import randint
from typing import TypedDict
import iscc_core as ic
import numpy as np
import msgspec


def new_iscc_id():
    # type: () -> bytes
    """
    Generate a new random ISCC-ID digest.

    Creates a 10-byte ISCC-ID using current timestamp (52 bits) and random server ID (12 bits).
    Uses REALM-0 for non-authoritative identifiers.

    :return: Complete ISCC-ID digest (2-byte header + 8-byte body)
    """
    timestamp = time.time_ns() // 1000
    identifier = (timestamp << 12) | randint(0, 4095)
    body = identifier.to_bytes(8, byteorder="big")
    return ic.encode_header(ic.MT.ID, ic.ST_ID_REALM.REALM_0, ic.VS.V1, 0) + body


def split_iscc_sequence(data):
    # type: (bytes) -> list[bytes]
    """
    Split a sequence of concatenated ISCC-DIGESTS.

    :param data: Concatenated ISCC-DIGESTS (variable-length)
    :return: List of individual ISCC-DIGEST bytes
    :raises: ValueError if ISCC-SEQUENCE parsing fails
    """
    units = []
    offset = 0
    try:
        while offset < len(data):
            mt, st, vs, ln, body = ic.decode_header(data[offset:])
            ln_bits = ic.decode_length(mt, ln)
            unit_len = 2 + (ln_bits // 8)  # header (2 bytes) + body
            units.append(data[offset : offset + unit_len])
            offset += unit_len
    except Exception as e:
        raise ValueError(f"Invalid ISCC-SEQUENCE: {e}")
    return units


class IsccBase:
    """
    Base class for ISCC objects providing common properties and methods.

    Handles conversion between different ISCC representations (string, bytes)
    and provides access to ISCC components (header, body, fields).
    """

    def __init__(self, iscc):
        # type: (str | bytes) -> None
        """
        Initialize ISCC object from string or binary representation.

        :param iscc: ISCC in canonical string format (with or without "ISCC:" prefix) or binary digest
        :raises TypeError: If iscc is not str or bytes
        """
        if isinstance(iscc, str):
            self.digest = ic.decode_base32(iscc.removeprefix("ISCC:"))
        elif isinstance(iscc, bytes):
            self.digest = iscc
        else:
            raise TypeError("`iscc` must be str, bytes")

    @property
    def body(self):
        # type: () -> bytes
        """
        Extract ISCC-BODY bytes (payload without header).

        :return: ISCC-BODY as raw bytes
        """
        return self.digest[2:]

    @cached_property
    def fields(self):
        # type: () -> ic.IsccTuple
        """
        Decode ISCC header into structured fields.

        :return: IsccTuple with MainType, SubType, Version, Length, and Body
        """
        return ic.decode_header(self.digest)

    @cached_property
    def iscc_type(self):
        # type: () -> str
        """
        Get human-readable ISCC type identifier.

        :return: ISCC type string in format "MAINTYPE-SUBTYPE-VERSION" (e.g., "CONTENT-TEXT-V1")
        """
        mtype = ic.MT(self.fields[0])
        stype = ic.SUBTYPE_MAP[(self.fields[0], self.fields[2])](self.fields[1])
        version = ic.VS(self.fields[2])
        return f"{mtype.name}_{stype.name}_{version.name}"

    def __str__(self):
        # type: () -> str
        """
        Get canonical ISCC string representation.

        :return: ISCC in canonical format with "ISCC:" prefix and base32 encoding
        """
        return f"ISCC:{ic.encode_base32(self.digest)}"

    def __len__(self):
        # type: () -> int
        """
        Get ISCC-BODY bit-length.

        :return: Number of bits in ISCC-BODY (64, 128, 192, or 256)
        """
        return len(self.digest[2:]) * 8

    def __bytes__(self):
        # type: () -> bytes
        """
        Get binary ISCC-DIGEST representation.

        :return: Complete ISCC-DIGEST as bytes (ISCC-HEADER + ISCC-BODY)
        """
        return self.digest


class IsccID(IsccBase):
    """
    ISCC-ID: Globally unique digital asset identifier.

    Combines ISCC-HEADER with 52-bit timestamp and 12-bit server-id for
    unique identification of digital assets across distributed systems.
    """

    # Pre-computed headers for valid realm IDs (only REALM_0 and REALM_1 are defined)
    _iscc_id_headers = (
        ic.encode_header(ic.MT.ID, 0, ic.VS.V1, 0),  # REALM_0
        ic.encode_header(ic.MT.ID, 1, ic.VS.V1, 0),  # REALM_1
    )

    @cache
    def __int__(self):
        """
        Convert ISCC-ID to integer representation.

        WARNING: Integer representation does not include ISCC-HEADER information.
        Use as a 64-bit integer database ID and keep track of the REALM-ID for reconstruction.

        :return: Integer representation of complete ISCC-ID digest
        """
        return int.from_bytes(self.body, "big", signed=False)

    @property
    def realm_id(self):
        # type: () -> int
        """
        Extract REALM-ID from ISCC-ID header.

        :return: Realm identifier (0 for REALM_0, 1 for REALM_1)
        """
        return self.fields[1]

    @classmethod
    def from_int(cls, iscc_id, realm_id):
        # type: (int, int) -> IsccID
        """
        Construct ISCC-ID from integer and realm identifier.

        :param iscc_id: Integer representation of ISCC-ID body (8 bytes)
        :param realm_id: Realm identifier for ISCC-HEADER SubType (0 or 1)
        :return: New IsccID instance
        """
        return cls(cls._iscc_id_headers[realm_id] + iscc_id.to_bytes(8, "big", signed=False))

    @classmethod
    def from_body(cls, body, realm_id):
        # type: (bytes, int) -> IsccID
        """
        Construct ISCC-ID from body bytes and realm identifier.

        :param body: ISCC-ID body bytes (8 bytes)
        :param realm_id: Realm identifier for ISCC-HEADER SubType (0 or 1)
        :return: New IsccID instance
        """
        return cls(cls._iscc_id_headers[realm_id] + body)

    @classmethod
    def random(cls):
        # type: () -> IsccID
        """
        Create a new random ISCC-ID.

        Uses REALM-ID 0 for non-authoritative ISCC-IDs with current timestamp and random server ID.

        :return: New IsccID instance with random identifier
        """
        return cls(new_iscc_id())


class IsccUnit(IsccBase):
    """
    ISCC-UNIT: Single-algorithm ISCC component.

    An ISCC-UNIT combines ISCC-HEADER with ISCC-BODY calculated from a single
    algorithm. Multiple ISCC-UNITs can be combined to form an ISCC-CODE.
    """

    @property
    def unit_type(self):
        # type: () -> str
        """
        Get ISCC-UNIT type identifier.

        :return: ISCC type string (alias for iscc_type property)
        """
        return self.iscc_type

    def __array__(self, dtype=np.uint8, copy=None):
        # type: (DTypeLike, bool | None) -> NDArray
        """
        Return numpy array from ISCC-BODY bytes.

        :param dtype: NumPy dtype for the array
        :param copy: If True, always copy. If False, never copy (view only). If None, copy only if needed.
        :return: NumPy array representation of ISCC-BODY
        """
        arr = np.frombuffer(self.body, dtype=dtype)
        if copy:
            return arr.copy()
        return arr


class IsccCode(IsccBase):
    """
    ISCC-CODE: Composite ISCC combining multiple ISCC-UNITs.

    An ISCC-CODE combines multiple ISCC-UNIT bodies into a single identifier.
    Minimum requirement: DATA and INSTANCE units. Can include META, SEMANTIC,
    and CONTENT units depending on the content type.
    """

    @cached_property
    def units(self):
        # type: () -> list[IsccUnit]
        """
        Decompose ISCC-CODE into constituent ISCC-UNITs.

        Parses the ISCC-CODE body and reconstructs individual ISCC-UNITs with
        their headers. Handles both standard codes and WIDE subtype codes.

        :return: List of IsccUnit objects contained in this ISCC-CODE
        """
        units = []
        raw_code = self.digest
        while raw_code:
            mt, st, vs, ln, body = ic.decode_header(raw_code)
            # standard ISCC-UNIT with tail continuation
            if mt != ic.MT.ISCC:
                ln_bits = ic.decode_length(mt, ln)
                unit_digest = ic.encode_header(mt, st, vs, ln) + body[: ln_bits // 8]
                units.append(IsccUnit(unit_digest))
                raw_code = body[ln_bits // 8 :]
                continue
            # ISCC-CODE
            main_types = ic.decode_units(ln)

            # Special case for WIDE subtype (128-bit Data + 128-bit Instance)
            if st == ic.ST_ISCC.WIDE:
                data_ln = ic.encode_length(ic.MT.DATA, 128)
                instance_ln = ic.encode_length(ic.MT.INSTANCE, 128)
                data_code_digest = ic.encode_header(ic.MT.DATA, ic.ST.NONE, vs, data_ln) + body[:16]
                instance_code_digest = ic.encode_header(ic.MT.INSTANCE, ic.ST.NONE, vs, instance_ln) + body[16:32]
                units.extend([IsccUnit(data_code_digest), IsccUnit(instance_code_digest)])
                break

            # rebuild dynamic units (META, SEMANTIC, CONTENT)
            for idx, mtype in enumerate(main_types):
                stype = ic.ST.NONE if mtype == ic.MT.META else st
                ln = ic.encode_length(mtype, 64)
                unit_digest = ic.encode_header(mtype, stype, vs, ln) + body[idx * 8 : (idx + 1) * 8]
                units.append(IsccUnit(unit_digest))

            # rebuild static units (DATA, INSTANCE)
            data_ln = ic.encode_length(ic.MT.DATA, 64)
            instance_ln = ic.encode_length(ic.MT.INSTANCE, 64)
            data_code_digest = ic.encode_header(ic.MT.DATA, ic.ST.NONE, vs, data_ln) + body[-16:-8]
            instance_code_digest = ic.encode_header(ic.MT.INSTANCE, ic.ST.NONE, vs, instance_ln) + body[-8:]
            units.extend([IsccUnit(data_code_digest), IsccUnit(instance_code_digest)])
            break

        return units


class IsccItemDict(TypedDict, total=False):
    """Dictionary representation of an ISCC item with ID, code, and units as strings."""

    iscc_id: str
    iscc_code: str
    units: list[str]


class IsccItem(msgspec.Struct, frozen=True, array_like=True):
    """
    Minimal ISCC container for efficient indexing.

    Stores only binary representations (id and units). String representations
    and derived values are computed on-demand (no caching for memory efficiency).

    :param id_data: ISCC-ID digest (10 bytes: 2-byte header + 8-byte body)
    :param units_data: Sequence of ISCC-UNIT digests
    """

    id_data: bytes
    units_data: bytes

    @classmethod
    def new(cls, iscc_id, iscc_code=None, units=None):
        # type: (str|bytes, str|bytes|None, list[str|bytes] | None) -> IsccItem
        """
        Create a new IsccItem from ISCC-ID and either ISCC-CODE or units.

        :param iscc_id: ISCC-ID as string or binary digest
        :param iscc_code: Optional ISCC-CODE as string or binary digest
        :param units: Optional list of ISCC-UNITs as strings or binary digests
        :return: New IsccItem instance
        :raises ValueError: If neither iscc_code nor units is provided
        """
        if units:
            units_data = b"".join(IsccUnit(u).digest for u in units)
        elif iscc_code:
            units_data = b"".join(u.digest for u in IsccCode(iscc_code).units)
        else:
            raise ValueError("Either iscc_code or units must be provided")
        return IsccItem(IsccID(iscc_id).digest, units_data)

    @classmethod
    def from_dict(cls, data):
        # type: (IsccItemDict) -> IsccItem
        """
        Create IsccItem from dictionary, generating random ISCC-ID if missing.

        :param data: Dictionary with optional iscc_id and either iscc_code or units
        :return: New IsccItem instance
        :raises ValueError: If neither iscc_code nor units is provided
        """
        iscc_id = data.get("iscc_id")
        if iscc_id is None:
            iscc_id = str(IsccID.random())

        iscc_code = data.get("iscc_code")
        units = data.get("units")

        return cls.new(iscc_id, iscc_code=iscc_code, units=units)

    @property
    def iscc_id(self):
        # type: () -> str
        """ISCC-ID as canonical string."""
        return f"ISCC:{ic.encode_base32(self.id_data)}"

    @property
    def iscc_code(self):
        # type: () -> str
        """ISCC-CODE computed from units (wide format)."""
        return ic.gen_iscc_code_v0(self.units, wide=True)["iscc"]

    @property
    def units(self):
        # type: () -> list[str]
        """ISCC-UNITs as list of canonical strings."""
        return [f"ISCC:{ic.encode_base32(u)}" for u in split_iscc_sequence(self.units_data)]

    @property
    def dict(self):
        # type: () -> IsccItemDict
        """
        Convert IsccItem to dictionary representation.

        :return: Dictionary with iscc_id, iscc_code, and units as canonical strings
        """
        return dict(
            iscc_id=self.iscc_id,
            iscc_code=self.iscc_code,
            units=self.units,
        )

    @property
    def json(self):
        # type: () -> bytes
        """
        Serialize IsccItem to JSON bytes.

        :return: JSON-encoded representation of IsccItem dictionary
        """
        return msgspec.json.encode(self.dict)
