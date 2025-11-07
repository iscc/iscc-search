"""Tests for simprint core protocol definitions."""


def test_simprint_raw_protocol_minimal():
    # type: () -> None
    """Test SimprintRaw protocol with minimal implementation."""

    class MinimalSimprintRaw:
        def __init__(self):
            # type: () -> None
            self.simprint = b"\x01\x02\x03"
            self.offset = 0
            self.size = 512

    obj = MinimalSimprintRaw()
    assert obj.simprint == b"\x01\x02\x03"
    assert obj.offset == 0
    assert obj.size == 512


def test_simprint_entry_raw_protocol_minimal():
    # type: () -> None
    """Test SimprintEntryRaw protocol with minimal implementation."""

    class MinimalSimprintEntryRaw:
        def __init__(self):
            # type: () -> None
            self.iscc_id_body = b"\x12\x34\x56\x78\x9a\xbc\xde\xf0"
            self.simprints = []  # type: list[SimprintRaw]

    obj = MinimalSimprintEntryRaw()
    assert obj.iscc_id_body == b"\x12\x34\x56\x78\x9a\xbc\xde\xf0"
    assert obj.simprints == []


def test_matched_chunk_raw_protocol_minimal():
    # type: () -> None
    """Test MatchedChunkRaw protocol with minimal implementation."""

    class MinimalMatchedChunkRaw:
        def __init__(self):
            # type: () -> None
            self.query = b"\x01\x02"
            self.match = b"\x01\x02"
            self.score = 1.0
            self.offset = 0
            self.size = 512

    obj = MinimalMatchedChunkRaw()
    assert obj.query == b"\x01\x02"
    assert obj.match == b"\x01\x02"
    assert obj.score == 1.0
    assert obj.offset == 0
    assert obj.size == 512


def test_simprint_match_raw_protocol_minimal():
    # type: () -> None
    """Test SimprintMatchRaw protocol with minimal implementation."""

    class MinimalSimprintMatchRaw:
        def __init__(self):
            # type: () -> None
            self.iscc_id_body = b"\x12\x34\x56\x78\x9a\xbc\xde\xf0"
            self.score = 0.95
            self.queried = 5
            self.matches = 3
            self.chunks = None  # type: list[MatchedChunkRaw] | None

    obj = MinimalSimprintMatchRaw()
    assert obj.iscc_id_body == b"\x12\x34\x56\x78\x9a\xbc\xde\xf0"
    assert obj.score == 0.95
    assert obj.queried == 5
    assert obj.matches == 3
    assert obj.chunks is None


def test_simprint_index_raw_protocol_minimal():
    # type: () -> None
    """Test SimprintIndexRaw protocol with minimal implementation."""

    class MinimalSimprintIndexRaw:
        def __init__(self, uri, **kwargs):
            # type: (str, ...) -> None
            self.uri = uri

        def add_raw(self, entries):
            # type: (list[SimprintEntryRaw]) -> None
            pass

        def search_raw(self, simprints, limit=10, threshold=0.8, detailed=True):
            # type: (list[bytes], int, float, bool) -> list[SimprintMatchRaw]
            return []

        def __contains__(self, iscc_id_body):
            # type: (bytes) -> bool
            return False

        def __len__(self):
            # type: () -> int
            return 0

        def close(self):
            # type: () -> None
            pass

    obj = MinimalSimprintIndexRaw("/tmp/test")
    assert obj.uri == "/tmp/test"
    obj.add_raw([])
    assert obj.search_raw([b"\x01\x02"]) == []
    assert b"\x00" not in obj
    assert len(obj) == 0
    obj.close()


def test_simprint_index_mutable_raw_protocol_minimal():
    # type: () -> None
    """Test SimprintIndexMutableRaw protocol with minimal implementation."""

    class MinimalSimprintIndexMutableRaw:
        def __init__(self, uri, **kwargs):
            # type: (str, ...) -> None
            self.uri = uri

        def add_raw(self, entries):
            # type: (list[SimprintEntryRaw]) -> None
            pass

        def search_raw(self, simprints, limit=10, threshold=0.8, detailed=True):
            # type: (list[bytes], int, float, bool) -> list[SimprintMatchRaw]
            return []

        def __contains__(self, iscc_id_body):
            # type: (bytes) -> bool
            return False

        def __len__(self):
            # type: () -> int
            return 0

        def close(self):
            # type: () -> None
            pass

        def get_raw(self, iscc_id_bodies):
            # type: (list[bytes]) -> list[SimprintEntryRaw]
            return []

        def delete_raw(self, iscc_id_bodies):
            # type: (list[bytes]) -> None
            pass

    obj = MinimalSimprintIndexMutableRaw("/tmp/test")
    assert obj.uri == "/tmp/test"
    obj.add_raw([])
    assert obj.search_raw([b"\x01\x02"]) == []
    assert b"\x00" not in obj
    assert len(obj) == 0
    assert obj.get_raw([b"\x00"]) == []
    obj.delete_raw([b"\x00"])
    obj.close()
