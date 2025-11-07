"""Tests for simprint multi-type protocol definitions."""


def test_simprint_entry_multi_protocol_minimal():
    # type: () -> None
    """Test SimprintEntryMulti protocol with minimal implementation."""

    class MinimalSimprintEntryMulti:
        def __init__(self):
            # type: () -> None
            self.iscc_id = b"\x00\x10\x12\x34\x56\x78\x9a\xbc\xde\xf0"
            self.simprints = {}  # type: dict[str, list[SimprintRaw]]

    obj = MinimalSimprintEntryMulti()
    assert obj.iscc_id == b"\x00\x10\x12\x34\x56\x78\x9a\xbc\xde\xf0"
    assert obj.simprints == {}


def test_type_match_result_protocol_minimal():
    # type: () -> None
    """Test TypeMatchResult protocol with minimal implementation."""

    class MinimalTypeMatchResult:
        def __init__(self):
            # type: () -> None
            self.score = 0.92
            self.queried = 3
            self.matches = 2
            self.chunks = None  # type: list[MatchedChunkRaw] | None

    obj = MinimalTypeMatchResult()
    assert obj.score == 0.92
    assert obj.queried == 3
    assert obj.matches == 2
    assert obj.chunks is None


def test_simprint_match_multi_protocol_minimal():
    # type: () -> None
    """Test SimprintMatchMulti protocol with minimal implementation."""

    class MinimalSimprintMatchMulti:
        def __init__(self):
            # type: () -> None
            self.iscc_id = b"\x00\x10\x12\x34\x56\x78\x9a\xbc\xde\xf0"
            self.score = 0.89
            self.types = {}  # type: dict[str, TypeMatchResult]

    obj = MinimalSimprintMatchMulti()
    assert obj.iscc_id == b"\x00\x10\x12\x34\x56\x78\x9a\xbc\xde\xf0"
    assert obj.score == 0.89
    assert obj.types == {}


def test_simprint_index_multi_protocol_minimal():
    # type: () -> None
    """Test SimprintIndexMulti protocol with minimal implementation."""

    class MinimalSimprintIndexMulti:
        def __init__(self, uri, **kwargs):
            # type: (str, ...) -> None
            self.uri = uri

        def add_raw_multi(self, entries):
            # type: (list[SimprintEntryMulti]) -> None
            pass

        def search_raw_multi(self, simprints, limit=10, threshold=0.8, detailed=True):
            # type: (dict[str, list[bytes]], int, float, bool) -> list[SimprintMatchMulti]
            return []

        def get_indexed_types(self):
            # type: () -> list[str]
            return []

        def __contains__(self, iscc_id):
            # type: (bytes) -> bool
            return False

        def close(self):
            # type: () -> None
            pass

    obj = MinimalSimprintIndexMulti("/tmp/test")
    assert obj.uri == "/tmp/test"
    obj.add_raw_multi([])
    assert obj.search_raw_multi({"CONTENT_TEXT_V0": [b"\x01\x02"]}) == []
    assert obj.get_indexed_types() == []
    assert b"\x00\x10\x12\x34\x56\x78\x9a\xbc\xde\xf0" not in obj
    obj.close()


def test_simprint_index_mutable_multi_protocol_minimal():
    # type: () -> None
    """Test SimprintIndexMutableMulti protocol with minimal implementation."""

    class MinimalSimprintIndexMutableMulti:
        def __init__(self, uri, **kwargs):
            # type: (str, ...) -> None
            self.uri = uri

        def add_raw_multi(self, entries):
            # type: (list[SimprintEntryMulti]) -> None
            pass

        def search_raw_multi(self, simprints, limit=10, threshold=0.8, detailed=True):
            # type: (dict[str, list[bytes]], int, float, bool) -> list[SimprintMatchMulti]
            return []

        def get_indexed_types(self):
            # type: () -> list[str]
            return []

        def __contains__(self, iscc_id):
            # type: (bytes) -> bool
            return False

        def close(self):
            # type: () -> None
            pass

        def get_raw_multi(self, iscc_ids):
            # type: (list[bytes]) -> list[SimprintEntryMulti]
            return []

        def delete_raw_multi(self, iscc_ids):
            # type: (list[bytes]) -> None
            pass

    obj = MinimalSimprintIndexMutableMulti("/tmp/test")
    assert obj.uri == "/tmp/test"
    obj.add_raw_multi([])
    assert obj.search_raw_multi({"CONTENT_TEXT_V0": [b"\x01\x02"]}) == []
    assert obj.get_indexed_types() == []
    assert b"\x00\x10\x12\x34\x56\x78\x9a\xbc\xde\xf0" not in obj
    assert obj.get_raw_multi([b"\x00\x10\x12\x34\x56\x78\x9a\xbc\xde\xf0"]) == []
    obj.delete_raw_multi([b"\x00\x10\x12\x34\x56\x78\x9a\xbc\xde\xf0"])
    obj.close()
