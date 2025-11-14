"""
Get command for ISCC-Search CLI.

Handles retrieving full ISCC asset details by ISCC-ID.
"""

import json

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from iscc_search.cli.common import console

__all__ = ["get_command"]


def get_command(
    iscc_id,  # type: str
    index_name: str | None = typer.Option(None, "--index", help="Index name to use (overrides active index)"),
):
    # type: (...) -> None
    """
    Get an ISCC asset by ISCC-ID.

    Retrieves the full asset details for a given ISCC-ID from the active index.

    Example:
        iscc-search get ISCC:MAIGIIFJRDGEQQAA
        iscc-search get ISCC:MAIGIIFJRDGEQQAA --index production
    """
    from iscc_search.cli.common import get_active_index

    # Get active or specified index
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading index...", total=None)
        try:
            index, target_index_name = get_active_index(index_name)
        except ValueError as e:
            progress.remove_task(task)
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1)
        progress.remove_task(task)

    # Get asset
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Retrieving asset...", total=None)
            asset = index.get_asset(target_index_name, iscc_id)
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
