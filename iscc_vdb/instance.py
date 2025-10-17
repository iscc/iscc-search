"""
An index for variable length ISCC Instance-Codes.

Basic Datastructure

{
<instance_code>: set(<iscc_id>, <iscc_id>, ...),
...
}

Where:
- <instance_code> is a variable length digest (Body of ISCC Instance-Code).
- <iscc_id> is an 8-byte digest of the ISCC-ID body.

InstanceIndex methods accept multiple representations of ISCC-IDs and Instance-Codes:

ISCC-ID as string - decode: iscc_core.decode_base32(iscc_id.removeprefix("ISCC:"))[2:]
ISCC-ID as bytes - decode:  iscc_id

Instance-Code as string - decode: iscc_core.decode_base32(iscc_code.removeprefix("ISCC:"))[2:]
Instance-Code as bytes - decode: iscc_code

For return values reconstruct ISCC-ID based on Realm-ID

Lists of ISCC-IDs and Instance-Codes are also supported for batch operations.
"""

import os
from typing import Protocol


IsccIds: str | list[str] | bytes | list[bytes]
InstanceCodes: str | list[str] | bytes | list[bytes]


class PInstanceIndex(Protocol):
    """
    Maps variable length Instance-Codes to ISCC-IDs.
    """

    path: os.PathLike
    realm_id: int

    def add(self, iscc_ids, instance_codes):
        # type: (IsccIds, InstanceCodes) -> int
        """
        Add ISCC-ID under key Instance-Code.

        :param iscc_ids: ISCC-ID string or digest or list of ISCC-ID strings or digests
        :param instance_codes: Instance-Code string or digest or list of Instance-Code strings or digests
        :return: Number of new keys added
        """

    def get(self, instance_codes):
        # type: (InstanceCodes) -> list[str]
        """
        Return all ISCC-IDS for a given instance_code (exact match).

        :param instance_codes: Instance-Code string or digest or list of Instance-Code strings or digests
        :return: List of matching ISCC-IDs or empty list if not found.
        """

    def search(self, instance_codes):
        # type: (InstanceCodes) -> dict[str, list[str]]
        """
        Return all ISCC-IDS for all instance_codes with a matching prefix.

        :param instance_codes: Instance-Code string or digest or list of Instance-Code strings or digests
        :return: Dict with ISCC-IDs as keys and sets of matched ISCC-CODEs.
        """
