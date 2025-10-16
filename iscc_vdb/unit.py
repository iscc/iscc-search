"""
Scalable ANNS search for variable-length ISCC-UNITs.
"""

from typing import Any

from iscc_vdb.nphd import NphdIndex


class UnitIndex(NphdIndex):
    """Fast approximate nearest neighbor search for variable-length ISCC-UNITs.

    Instead of integer keys and uint8-vectors, we accept ISCC-IDs as keys and ISCC-UNITs as vectors.
    """

    def __init__(self, unit_type=None, max_dim=256, realm_id=None, **kwargs):
        # type: (str | None, int, int | None, Any) -> None
        """Create a new ISCC-UNIT index.

        :param unit_type: ISCC type string (e.g. 'META-NONE-V0') or None for auto-detection
        :param max_dim: Maximum vector dimension in bits (default 256)
        :param realm_id: ISCC realm ID (0-15) or None for auto-detection
        :param kwargs: Additional arguments passed to NphdIndex
        """
        super().__init__(max_dim=max_dim, **kwargs)
        self.unit_type = unit_type
        self.realm_id = realm_id
