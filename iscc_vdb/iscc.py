"""Scalable multi-index ANNS search for ISCCs."""

import os

from iscc_vdb.instance import InstanceIndex


class IsccIndex:
    """Multi-index ANNS search for ISCCs.

    Manages multiple internal indexes:
    - One UnitIndex per ISCC-UNIT-TYPE (META-NONE-V0, CONTENT-TEXT-V0, etc.)
    - One InstanceIndex for exact/prefix matching
    """

    def __init__(self, path, realm_id=0, max_dim=256, **kwargs):
        # type: (str | os.PathLike, int, int, Any) -> None
        """Create or open ISCC multi-index.

        :param path: Directory path for index storage
        :param realm_id: ISCC realm ID (0-1) for ISCC-ID reconstruction
        :param max_dim: Maximum vector dimension in bits for UNIT indexes
        :param kwargs: Additional arguments passed to underlying UnitIndex instances
        """
        self.path = os.fspath(path)
        self.realm_id = realm_id
        self.max_dim = max_dim
        self.unit_index_kwargs = kwargs

        # Create directory structure
        os.makedirs(self.path, exist_ok=True)

        # Dictionary to hold UnitIndex instances (created on-demand)
        self.unit_indexes = {}  # type: dict[str, UnitIndex]

        # Create InstanceIndex for exact/prefix matching
        instance_path = os.path.join(self.path, "instance")
        self.instance_index = InstanceIndex(instance_path, realm_id=realm_id)
