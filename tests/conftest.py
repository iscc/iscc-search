"""Test fixtures for IsccUnitIndex testing."""

import iscc_core as ic
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sample_iscc_ids():
    # type: () -> list[str]
    """Generate a list of sample ISCC-IDs for testing."""
    ids = []
    for i in range(10):
        # Generate ISCC-IDs with different hub_ids (timestamp in microseconds)
        iscc_id = ic.gen_iscc_id(timestamp=1000000 + i, hub_id=i, realm_id=0)["iscc"]
        ids.append(iscc_id)
    return ids


@pytest.fixture
def sample_meta_units():
    # type: () -> list[str]
    """Generate META type ISCC-UNITs with various bit lengths."""
    units = []
    for bit_length in [64, 128, 192, 256]:
        unit = ic.Code.rnd(ic.MT.META, bits=bit_length)
        units.append(f"ISCC:{unit}")
    return units


@pytest.fixture
def sample_semantic_units():
    # type: () -> list[str]
    """Generate SEMANTIC type ISCC-UNITs with various subtypes and bit lengths."""
    units = []
    subtypes = [ic.ST_CC.TEXT, ic.ST_CC.IMAGE, ic.ST_CC.MIXED]
    for st in subtypes:
        for bit_length in [64, 128]:
            unit = ic.Code.rnd(ic.MT.SEMANTIC, st, bits=bit_length)
            units.append(f"ISCC:{unit}")
    return units


@pytest.fixture
def sample_content_units():
    # type: () -> list[str]
    """Generate CONTENT type ISCC-UNITs with various subtypes."""
    units = []
    subtypes = [ic.ST_CC.TEXT, ic.ST_CC.IMAGE, ic.ST_CC.AUDIO, ic.ST_CC.VIDEO, ic.ST_CC.MIXED]
    for st in subtypes:
        unit = ic.Code.rnd(ic.MT.CONTENT, st, bits=128)
        units.append(f"ISCC:{unit}")
    return units


@pytest.fixture
def sample_data_units():
    # type: () -> list[str]
    """Generate DATA type ISCC-UNITs with various bit lengths."""
    units = []
    for bit_length in [64, 128, 192, 256]:
        unit = ic.Code.rnd(ic.MT.DATA, bits=bit_length)
        units.append(f"ISCC:{unit}")
    return units


@pytest.fixture
def mixed_unit_types():
    # type: () -> list[str]
    """Generate ISCC-UNITs of different MainTypes (should fail when mixed)."""
    return [
        f"ISCC:{ic.Code.rnd(ic.MT.META, bits=64)}",
        f"ISCC:{ic.Code.rnd(ic.MT.SEMANTIC, ic.ST_CC.TEXT, bits=128)}",
        f"ISCC:{ic.Code.rnd(ic.MT.CONTENT, ic.ST_CC.IMAGE, bits=192)}",
        f"ISCC:{ic.Code.rnd(ic.MT.DATA, bits=256)}",
    ]


@pytest.fixture
def invalid_iscc_ids():
    # type: () -> list[str]
    """Generate invalid ISCC-ID strings for error testing."""
    return [
        "INVALID:ABCD1234",  # Wrong prefix
        "ISCC:",  # Empty after prefix
        "ISCC:INVALIDBASE32!@#",  # Invalid base32 characters
        "NotAnISCC",  # No prefix at all
        "",  # Empty string
    ]


@pytest.fixture
def invalid_iscc_units():
    # type: () -> list[str]
    """Generate invalid ISCC-UNIT strings for error testing."""
    return [
        "INVALID:UNIT1234",  # Wrong prefix
        "ISCC:",  # Empty after prefix
        "ISCC:SHORTUNIT",  # Too short to be valid
        "NotAUnit",  # No prefix at all
        "",  # Empty string
    ]


@pytest.fixture
def iscc_id_key_pairs():
    # type: () -> list[tuple[str, int]]
    """Generate pairs of ISCC-IDs and their corresponding integer keys."""
    pairs = []
    for i in range(5):
        iscc_id = ic.gen_iscc_id(timestamp=2000000 + i, hub_id=i, realm_id=0)["iscc"]
        # Extract the key from the ISCC-ID
        body = ic.iscc_decode(iscc_id)[-1]  # Get the digest/body
        key = int.from_bytes(body, "big", signed=False)
        pairs.append((iscc_id, key))
    return pairs


@pytest.fixture
def unit_type_meta():
    # type: () -> tuple[typing.Any, typing.Any, typing.Any]
    """Return Unit-Type tuple for META units."""
    return (ic.MT.META, ic.ST.NONE, ic.VS.V0)


@pytest.fixture
def unit_type_semantic_text():
    # type: () -> tuple[typing.Any, typing.Any, typing.Any]
    """Return Unit-Type tuple for SEMANTIC TEXT units."""
    return (ic.MT.SEMANTIC, ic.ST_CC.TEXT, ic.VS.V0)


@pytest.fixture
def unit_type_content_image():
    # type: () -> tuple[typing.Any, typing.Any, typing.Any]
    """Return Unit-Type tuple for CONTENT IMAGE units."""
    return (ic.MT.CONTENT, ic.ST_CC.IMAGE, ic.VS.V0)


@pytest.fixture
def unit_type_data():
    # type: () -> tuple[typing.Any, typing.Any, typing.Any]
    """Return Unit-Type tuple for DATA units."""
    return (ic.MT.DATA, ic.ST.NONE, ic.VS.V0)


@pytest.fixture
def similar_units():
    # type: () -> tuple[str, str, str]
    """Generate similar ISCC-UNITs for search testing."""
    # Create base vector
    base_bytes = bytes([255, 170, 85, 0] * 4)  # 16 bytes / 128 bits

    # Create similar vector (1 bit different)
    similar_bytes = bytearray(base_bytes)
    similar_bytes[0] = 254  # Change 1 bit

    # Create dissimilar vector (many bits different)
    dissimilar_bytes = bytes([0, 85, 170, 255] * 4)  # Inverted pattern

    # Convert to ISCC-UNITs
    base_unit = f"ISCC:{ic.encode_component(ic.MT.META, ic.ST.NONE, ic.VS.V0, 128, base_bytes)}"
    similar_unit = f"ISCC:{ic.encode_component(ic.MT.META, ic.ST.NONE, ic.VS.V0, 128, bytes(similar_bytes))}"
    dissimilar_unit = f"ISCC:{ic.encode_component(ic.MT.META, ic.ST.NONE, ic.VS.V0, 128, dissimilar_bytes)}"

    return base_unit, similar_unit, dissimilar_unit


@pytest.fixture
def temp_index_path(tmp_path):
    # type: (typing.Any) -> Iterator[typing.Any]
    """Provide a temporary path for index saving/loading tests."""
    index_file = tmp_path / "test_index.usearch"
    yield index_file
    # Cleanup is handled by pytest's tmp_path fixture


@pytest.fixture
def large_dataset():
    # type: () -> tuple[list[str], list[str]]
    """Generate a larger dataset for performance testing."""
    ids = []
    units = []
    for i in range(100):
        # Generate ISCC-IDs (timestamp in microseconds)
        iscc_id = ic.gen_iscc_id(timestamp=10000000 + i, hub_id=i % 10, realm_id=0)["iscc"]
        ids.append(iscc_id)

        # Generate corresponding META units with varying lengths
        bit_length = [64, 128, 192, 256][i % 4]
        unit = ic.Code.rnd(ic.MT.META, bits=bit_length)
        units.append(f"ISCC:{unit}")

    return ids, units


@pytest.fixture
def edge_case_units():
    # type: () -> dict[str, str]
    """Generate edge case ISCC-UNITs for testing."""
    zeros_bytes = bytes([0] * 8)
    ones_bytes = bytes([255] * 8)
    return {
        "min_length": f"ISCC:{ic.Code.rnd(ic.MT.META, bits=64)}",  # Minimum 64 bits
        "max_length": f"ISCC:{ic.Code.rnd(ic.MT.META, bits=256)}",  # Maximum 256 bits
        "all_zeros": f"ISCC:{ic.encode_component(ic.MT.META, ic.ST.NONE, ic.VS.V0, 64, zeros_bytes)}",
        "all_ones": f"ISCC:{ic.encode_component(ic.MT.META, ic.ST.NONE, ic.VS.V0, 64, ones_bytes)}",
    }


@pytest.fixture
def sample_iscc_codes():
    # type: () -> list[str]
    """Generate valid ISCC-CODEs for testing (with 256-bit Data + Instance units)."""
    codes = []
    for i in range(10):
        # Generate 256-bit Data-Code and Instance-Code units
        data_unit = ic.Code.rnd(ic.MT.DATA, bits=256)
        instance_unit = ic.Code.rnd(ic.MT.INSTANCE, bits=256)

        # Generate ISCC-CODE from Data + Instance (wide format for 256-bit)
        iscc_code = ic.gen_iscc_code_v0([f"ISCC:{data_unit}", f"ISCC:{instance_unit}"], wide=True)["iscc"]
        codes.append(iscc_code)
    return codes


# Server testing fixtures


@pytest.fixture
def memory_index_instance():
    # type: () -> typing.Any
    """
    Create a fresh MemoryIndex instance for testing.

    Provides an isolated, in-memory index that doesn't require persistence.
    Ideal for fast, isolated server tests.

    :return: MemoryIndex instance
    """
    from iscc_vdb.indexes.memory import MemoryIndex

    return MemoryIndex()


@pytest.fixture(params=["memory", "lmdb"])
def backend_index(request, tmp_path):
    # type: (typing.Any, typing.Any) -> typing.Any
    """
    Parametrized fixture providing both index backend implementations.

    Creates fresh index instances for each test with proper cleanup.
    Tests using this fixture will automatically run against both backends.

    :param request: Pytest request object with param ("memory" or "lmdb")
    :param tmp_path: Pytest temp directory for LMDB storage
    :return: Index instance (MemoryIndex or LmdbIndexManager)
    """
    backend_type = request.param

    if backend_type == "memory":
        from iscc_vdb.indexes.memory import MemoryIndex

        index = MemoryIndex()
    elif backend_type == "lmdb":
        from iscc_vdb.indexes.lmdb import LmdbIndexManager

        # Use isolated temp directory for each test
        lmdb_path = tmp_path / "lmdb_indexes"
        index = LmdbIndexManager(lmdb_path)
    else:  # pragma: no cover
        raise ValueError(f"Unknown backend type: {backend_type}")

    yield index

    # Cleanup
    index.close()


@pytest.fixture
def test_client(backend_index):
    # type: (typing.Any) -> TestClient
    """
    Create TestClient with app configured to use parametrized backend.

    Provides a FastAPI TestClient for making HTTP requests to the API.
    Uses the backend_index fixture which parametrizes over all implementations.
    This ensures all server tests run against both MemoryIndex and LmdbIndexManager.

    :param backend_index: Parametrized index backend instance
    :return: FastAPI TestClient instance
    """
    from iscc_vdb.server import app

    # Override app.state.index with parametrized backend
    app.state.index = backend_index

    return TestClient(app)


@pytest.fixture
def sample_assets(sample_iscc_ids, sample_iscc_codes, sample_content_units):
    # type: (list[str], list[str], list[str]) -> list[typing.Any]
    """
    Generate list of IsccAsset objects with valid ISCC data for testing.

    Creates assets with iscc_id, iscc_code, units, and metadata fields populated
    from existing ISCC fixtures.

    :param sample_iscc_ids: Fixture providing ISCC-IDs
    :param sample_iscc_codes: Fixture providing ISCC-CODEs
    :param sample_content_units: Fixture providing ISCC-UNITs
    :return: List of IsccAsset objects
    """
    from iscc_vdb.schema import IsccAsset

    assets = []
    for i in range(5):
        # Provide at least 2 units (schema requires min_length=2)
        asset = IsccAsset(
            iscc_id=sample_iscc_ids[i],
            iscc_code=sample_iscc_codes[i],
            units=[
                sample_content_units[i % len(sample_content_units)],
                sample_content_units[(i + 1) % len(sample_content_units)],
            ],
            metadata={"index": i, "title": f"Test Asset {i}"},
        )
        assets.append(asset)
    return assets


@pytest.fixture
def populated_index(memory_index_instance, sample_assets):
    # type: (typing.Any, list[typing.Any]) -> typing.Any
    """
    Create a MemoryIndex pre-populated with sample assets.

    Provides an index with a default index named "testindex" containing
    sample assets. Useful for testing search and retrieval operations.

    :param memory_index_instance: Fresh MemoryIndex instance
    :param sample_assets: Sample IsccAsset objects
    :return: MemoryIndex with data
    """
    from iscc_vdb.schema import IsccIndex

    # Create test index
    memory_index_instance.create_index(IsccIndex(name="testindex"))

    # Add sample assets
    memory_index_instance.add_assets("testindex", sample_assets)

    return memory_index_instance
