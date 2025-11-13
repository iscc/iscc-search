"""
Shared utilities for ISCC-Search CLI.

Common functionality used across multiple CLI commands.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import iscc_core as ic
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    from iscc_search.protocols.index import IsccIndexProtocol  # noqa: F401

import iscc_search

__all__ = ["console", "get_default_index", "parse_simprints_from_features"]


# Shared console instance for all CLI commands
console = Console()


# Configure loguru to use rich's console for proper output coordination
logger.remove()  # Remove default handler
logger.add(
    RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        show_time=False,  # Use custom time format
        show_level=False,  # Use custom level format
        show_path=False,  # Don't show file path on right
    ),
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {module}:{function}:{line} - {message}",
    level="DEBUG",
)


def get_default_index():
    # type: () -> IsccIndexProtocol
    """
    Get or create default usearch index.

    Creates index at <data_dir>/default if it doesn't exist.

    :return: Default index instance
    """
    from iscc_search.indexes.usearch import UsearchIndexManager
    from iscc_search.schema import IsccIndex

    # Use platform-specific data directory
    data_dir = Path(iscc_search.dirs.user_data_dir)
    base_path = data_dir

    # Create directory if needed
    base_path.mkdir(parents=True, exist_ok=True)

    manager = UsearchIndexManager(str(base_path))

    # Check if index exists, create if not
    try:
        manager.get_index("default")
    except FileNotFoundError:
        # Index doesn't exist, create it
        manager.create_index(IsccIndex(name="default"))

    return manager


def parse_simprints_from_features(features, simprint_bits=None):
    # type: (list[dict], int | None) -> dict[str, list[dict]] | None
    """
    Transform features array from .iscc.json format to IsccEntry.simprints format.

    Converts from:
        [{"maintype": "semantic", "subtype": "text", "version": 0,
          "simprints": ["abc", "def"], "offsets": [0, 100], "sizes": [50, 60]}]

    To:
        {"SEMANTIC_TEXT_V0": [
            {"simprint": "abc", "offset": 0, "size": 50},
            {"simprint": "def", "offset": 100, "size": 60}
        ]}

    :param features: List of feature dicts from .iscc.json file
    :param simprint_bits: Truncate simprints to this bit length (64, 128, 192, 256)
    :return: Dict mapping simprint types to lists of simprint objects, or None if empty
    """
    if not features:
        return None

    result = {}  # type: dict[str, list[dict]]

    for feature in features:
        # Extract required fields
        maintype = feature.get("maintype", "").upper()
        subtype = feature.get("subtype", "").upper()
        version = feature.get("version", 0)
        simprints = feature.get("simprints", [])
        offsets = feature.get("offsets", [])
        sizes = feature.get("sizes", [])

        # Skip if essential data is missing
        if not maintype or not subtype or not simprints:
            logger.warning(f"Skipping feature with missing data: {feature.get('maintype')}-{feature.get('subtype')}")
            continue

        # Construct simprint type key (e.g., "SEMANTIC_TEXT_V0")
        simprint_type = f"{maintype}_{subtype}_V{version}"

        # Ensure arrays have matching lengths (use min length for safety)
        min_len = min(len(simprints), len(offsets), len(sizes))
        if min_len != len(simprints):
            logger.warning(
                f"Array length mismatch for {simprint_type}: "
                f"simprints={len(simprints)}, offsets={len(offsets)}, sizes={len(sizes)}. "
                f"Using {min_len} entries."
            )

        # Zip together simprints, offsets, sizes (with optional truncation)
        simprint_list = []
        for i in range(min_len):
            simprint_str = simprints[i]

            # Truncate simprint if requested
            if simprint_bits is not None:
                try:
                    # Decode base64 to binary
                    simprint_bytes = ic.decode_base64(simprint_str)

                    # Calculate target bytes
                    target_bytes = simprint_bits // 8

                    # Validate size
                    if len(simprint_bytes) < target_bytes:
                        raise ValueError(
                            f"Simprint too small for {simprint_type}: "
                            f"{len(simprint_bytes) * 8} bits < {simprint_bits} bits"
                        )

                    # Truncate to target size
                    simprint_bytes = simprint_bytes[:target_bytes]

                    # Re-encode to base64
                    simprint_str = ic.encode_base64(simprint_bytes)

                except Exception as e:
                    logger.error(f"Failed to truncate simprint for {simprint_type}: {e}")
                    continue  # Skip this simprint

            simprint_list.append({"simprint": simprint_str, "offset": offsets[i], "size": sizes[i]})

        # Add to result (merge if type already exists - shouldn't happen but handle gracefully)
        if simprint_type in result:
            logger.warning(f"Duplicate simprint type {simprint_type}, merging entries")
            result[simprint_type].extend(simprint_list)
        else:
            result[simprint_type] = simprint_list

    return result if result else None
