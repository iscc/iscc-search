"""
# Types and convenience classes for handling ISCCs

## Terms and Definitions

- **ISCC** - Any ISCC-CODE, ISCC-UNIT, or ISCC-ID
- **ISCC-HEADER** - Self-describing 2-byte header for V1 components (3 bytes for future versions). The first 12
    bits encode MainType, SubType, and Version. Additional bits encode Length for variable-length ISCCs.
- **ISCC-BODY** - Actual payload of an ISCC, similarity preserving compact binary code, hash or timestamp
- **ISCC-DIGEST** - Binary representation of complete ISCC (ISCC-HEADER + ISCC-BODY).
- **ISCC-UNIT** - ISCC-HEADER + ISCC-BODY where the ISCC-BODY is calculated from a single algorithm
- **ISCC-CODE** - ISCC-HEADER + ISCC-BODY where the ISCC-BODY is a sequence of multiple ISCC-UNIT BODYs
    - DATA and INSTANCE are the minimum required mandatory ISCC-UNITS for a valid ISCC-CODE
- **ISCC-ID** - Globally unique digital asset identifier (ISCC-HEADER + 52-bit timestamp + 12-bit server-id)
- **SIMPRINT** - Headerless base64 encoded similarity hash that describes a content segment (granular feature)
- **UNIT-TYPE**: Identifier for ISCC-UNIT types that can be indexed together with meaningful similarity search
"""

from functools import cached_property, cache
import iscc_core as ic
import numpy as np
from numpy.typing import NDArray, DTypeLike


class IsccBase:
    def __init__(self, iscc):
        # type: (str | bytes) -> None
        if isinstance(iscc, str):
            self.digest = ic.decode_base32(iscc.removeprefix("ISCC:"))
        elif isinstance(iscc, bytes):
            self.digest = iscc
        else:
            raise TypeError("`iscc` must be str, bytes")

    @property
    def body(self):
        # type: () -> bytes
        return self.digest[2:]

    @cached_property
    def fields(self):
        # type: () -> ic.IsccTuple
        return ic.decode_header(self.digest)

    @cached_property
    def iscc_type(self):
        # type: () -> str
        mtype = ic.MT(self.fields[0])
        stype = ic.SUBTYPE_MAP[(self.fields[0], self.fields[2])](self.fields[1])
        version = ic.VS(self.fields[2])
        return f"{mtype.name}-{stype.name}-{version.name}"

    @cache
    def __str__(self):
        # type: () -> str
        """Canongical ISCC"""
        return f"ISCC:{ic.encode_base32(self.digest)}"

    @cache
    def __len__(self):
        # type: () -> int
        """ISCC-BODY bit-length"""
        return len(self.digest[2:]) * 8

    def __bytes__(self):
        # type: () -> bytes
        """ISCC-DIGEST bytes"""
        return self.digest


class IsccID(IsccBase):
    @cache
    def __int__(self):
        """
        WARNING: Integer representation does not include ISCC-HEADER information.
        """
        return int.from_bytes(self.digest, "big", signed=False)

    @classmethod
    def from_int(cls, iscc_id, realm_id):
        # type: (int, int) -> IsccID
        """Build ISCC-ID with REALM-ID"""
        return cls(ic.encode_header(ic.MT.ID, realm_id, ic.VS.V1, 0) + iscc_id.to_bytes(8, "big", signed=False))


class IsccUnit(IsccBase):
    @property
    def unit_type(self):
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
    @cached_property
    def units(self):
        # type: () -> list[IsccUnit]
        """List of ISCC-UNITS in ISCC-CODE."""
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
