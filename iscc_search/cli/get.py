"""
Get command for ISCC-Search CLI.

Handles retrieving full ISCC asset details by ISCC-ID.
"""

import json

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from iscc_search.cli.common import console, get_default_index

__all__ = ["get_command"]


def get_command(
    iscc_id,  # type: str
):
    # type: (...) -> None
    """
    Get an ISCC asset by ISCC-ID.

    Retrieves the full asset details for a given ISCC-ID from the default index.

    Example:
        iscc-search get ISCC:MAIGIIFJRDGEQQAA
    """
    # Get default index
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading index...", total=None)
        index = get_default_index()
        progress.remove_task(task)

    # Get asset
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Retrieving asset...", total=None)
            asset = index.get_asset("default", iscc_id)
            progress.remove_task(task)
    except FileNotFoundError:
        console.print(f"[red]Asset not found: {iscc_id}[/red]")
        index.close()
        raise typer.Exit(code=1)

    # Close index
    index.close()

    # Output as JSON
    output = {
        "iscc_id": asset.iscc_id,
        "iscc_code": asset.iscc_code,
        "units": asset.units,
        "metadata": asset.metadata,
    }

    console.print_json(json.dumps(output))
