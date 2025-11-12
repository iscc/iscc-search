"""
Shared utilities for ISCC-Search CLI.

Common functionality used across multiple CLI commands.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    from iscc_search.protocols.index import IsccIndexProtocol  # noqa: F401

import iscc_search

__all__ = ["console", "get_default_index"]


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

    Creates index at <data_dir>/usearch/default if it doesn't exist.

    :return: Default index instance
    """
    from iscc_search.indexes.usearch import UsearchIndexManager
    from iscc_search.schema import IsccIndex

    # Use platform-specific data directory
    data_dir = Path(iscc_search.dirs.user_data_dir)
    base_path = data_dir / "usearch"

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
