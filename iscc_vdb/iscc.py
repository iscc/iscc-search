"""Scalable multi-index ANNS search for ISCCs."""

import os
from iscc_vdb.instance import InstanceIndex


class IsccIndex:
    """Multi-index ANNS search for ISCCs.

    Manages multiple internal indexes:
    - One UnitIndex per ISCC-UNIT-TYPE (META_NONE_V0, CONTENT_TEXT_V0, etc.)
    - One InstanceIndex for exact/prefix matching
    """

    def __init__(self, *, path=None, realm_id=0, max_dim=256, **kwargs):
        # type: (str | os.PathLike | None, int, int, Any) -> None
        """Create or open ISCC multi-index.

        :param path: Directory path for index storage (optional)
        :param realm_id: ISCC realm ID (0-1) for ISCC-ID reconstruction
        :param max_dim: Maximum vector dimension in bits for UNIT indexes
        :param kwargs: Additional arguments passed to underlying UnitIndex instances
        """
        self.path = os.fspath(path) if path is not None else None
        self.realm_id = realm_id
        self.max_dim = max_dim
        self.unit_index_kwargs = kwargs

        # Dictionary to hold UnitIndex instances (created on-demand)
        self.unit_indexes = {}  # type: dict[str, UnitIndex]

        # Create directory structure and InstanceIndex only if path is provided
        if self.path is not None:
            os.makedirs(self.path, exist_ok=True)
            instance_path = os.path.join(self.path, "instance")
            self.instance_index = InstanceIndex(instance_path, realm_id=realm_id)
        else:
            self.instance_index = None
