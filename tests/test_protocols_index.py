"""Test ISCC Index Protocol definition and runtime checking."""

from iscc_search.protocols.index import IsccIndexProtocol
from iscc_search.schema import IsccAddResult, IsccEntry, IsccIndex, IsccSearchResult, Metric


def test_protocol_is_runtime_checkable():
    """Test that IsccIndexProtocol is runtime checkable."""

    # Create a minimal valid implementation
    class ValidIndex:
        def list_indexes(self):
            # type: () -> list[IsccIndex]
            return []

        def create_index(self, index):
            # type: (IsccIndex) -> IsccIndex
            return index

        def get_index(self, name):
            # type: (str) -> IsccIndex
            return IsccIndex(name=name)

        def delete_index(self, name):
            # type: (str) -> None
            pass

        def add_assets(self, index_name, assets):
            # type: (str, list[IsccEntry]) -> list[IsccAddResult]
            return []

        def get_asset(self, index_name, iscc_id):
            # type: (str, str) -> IsccEntry
            return IsccEntry(iscc_id=iscc_id)

        def search_assets(self, index_name, query, limit=100):
            # type: (str, IsccEntry, int) -> IsccSearchResult
            return IsccSearchResult(query=query, metric=Metric.bitlength, global_matches=[])

        def close(self):
            # type: () -> None
            pass

    valid_index = ValidIndex()
    assert isinstance(valid_index, IsccIndexProtocol)


def test_protocol_rejects_incomplete_implementation():
    """Test that protocol rejects objects missing required methods."""

    class IncompleteIndex:
        def list_indexes(self):
            # type: () -> list[IsccIndex]
            return []

        # Missing other required methods

    incomplete_index = IncompleteIndex()
    assert not isinstance(incomplete_index, IsccIndexProtocol)


def test_protocol_rejects_non_index_objects():
    """Test that protocol rejects objects that are clearly not indexes."""

    class NotAnIndex:
        def some_method(self):
            pass

    not_an_index = NotAnIndex()
    assert not isinstance(not_an_index, IsccIndexProtocol)


def test_protocol_rejects_empty_object():
    """Test that protocol rejects empty objects."""

    class Empty:
        pass

    empty = Empty()
    assert not isinstance(empty, IsccIndexProtocol)


def test_protocol_accepts_complete_implementation():
    """Test that protocol accepts objects with all required methods."""

    class CompleteIndex:
        def list_indexes(self):
            return [IsccIndex(name="test", assets=0, size=0)]

        def create_index(self, index):
            return IsccIndex(name=index.name, assets=0, size=0)

        def get_index(self, name):
            return IsccIndex(name=name, assets=100, size=10)

        def delete_index(self, name):
            return None

        def add_assets(self, index_name, assets):
            return [IsccAddResult(iscc_id="ISCC:MAIGIIFJRDGEQQAA", status="created")]

        def get_asset(self, index_name, iscc_id):
            return IsccEntry(iscc_id=iscc_id)

        def search_assets(self, index_name, query, limit=100):
            return IsccSearchResult(
                query=IsccEntry(iscc_id="ISCC:MAIGIIFJRDGEQQAA", units=["ISCC:AADYCMZIOY36XXGZ"]),
                metric=Metric.nphd,
                global_matches=[],
            )

        def close(self):
            pass

    complete_index = CompleteIndex()
    assert isinstance(complete_index, IsccIndexProtocol)


def test_protocol_method_signatures():
    """Test that protocol methods have expected signatures."""

    class SignatureTestIndex:
        def list_indexes(self):
            return []

        def create_index(self, index):
            return index

        def get_index(self, name):
            return IsccIndex(name=name)

        def delete_index(self, name):
            pass

        def add_assets(self, index_name, assets):
            return []

        def get_asset(self, index_name, iscc_id):
            return IsccEntry(iscc_id=iscc_id)

        def search_assets(self, index_name, query, limit=100):
            return IsccSearchResult(query=query, metric=Metric.bitlength, global_matches=[])

        def close(self):
            pass

    test_index = SignatureTestIndex()
    assert isinstance(test_index, IsccIndexProtocol)

    # Verify method existence
    assert hasattr(test_index, "list_indexes")
    assert hasattr(test_index, "create_index")
    assert hasattr(test_index, "get_index")
    assert hasattr(test_index, "delete_index")
    assert hasattr(test_index, "add_assets")
    assert hasattr(test_index, "get_asset")
    assert hasattr(test_index, "search_assets")
    assert hasattr(test_index, "close")


def test_protocol_with_wrong_method_names():
    """Test that objects with wrong method names are rejected."""

    class WrongMethodNames:
        def list_indices(self):  # Wrong name (should be list_indexes)
            return []

        def create(self, index):  # Wrong name (should be create_index)
            return index

        def get(self, name):  # Wrong name (should be get_index)
            return None

        def delete(self, name):  # Wrong name (should be delete_index)
            pass

        def add(self, index_name, assets):  # Wrong name (should be add_assets)
            return []

        def search(self, index_name, query, limit=100):  # Wrong name (should be search_assets)
            return []

        def cleanup(self):  # Wrong name (should be close)
            pass

    wrong_names = WrongMethodNames()
    assert not isinstance(wrong_names, IsccIndexProtocol)


def test_protocol_partial_implementation():
    """Test protocol rejection with various levels of partial implementation."""

    class OnlyListIndexes:
        def list_indexes(self):
            return []

    assert not isinstance(OnlyListIndexes(), IsccIndexProtocol)

    class MissingClose:
        def list_indexes(self):
            return []

        def create_index(self, index):
            return index

        def get_index(self, name):
            return IsccIndex(name=name)

        def delete_index(self, name):
            pass

        def add_assets(self, index_name, assets):
            return []

        def search_assets(self, index_name, query, limit=100):
            return IsccSearchResult(query=query, metric=Metric.bitlength, global_matches=[])

        # Missing close() method

    assert not isinstance(MissingClose(), IsccIndexProtocol)


def test_protocol_with_extra_methods():
    """Test that protocol accepts implementations with extra methods."""

    class IndexWithExtras:
        def list_indexes(self):
            return []

        def create_index(self, index):
            return index

        def get_index(self, name):
            return IsccIndex(name=name)

        def delete_index(self, name):
            pass

        def add_assets(self, index_name, assets):
            return []

        def get_asset(self, index_name, iscc_id):
            return IsccEntry(iscc_id=iscc_id)

        def search_assets(self, index_name, query, limit=100):
            return IsccSearchResult(query=query, metric=Metric.bitlength, global_matches=[])

        def close(self):
            pass

        # Extra methods (should be allowed)
        def some_extra_method(self):
            return "extra"

        def another_helper(self, x):
            return x * 2

    index_with_extras = IndexWithExtras()
    assert isinstance(index_with_extras, IsccIndexProtocol)
    assert index_with_extras.some_extra_method() == "extra"
    assert index_with_extras.another_helper(5) == 10
