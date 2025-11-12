"""
Search command for ISCC-Search CLI.

Handles searching for similar ISCC assets in the default index.
"""

import json

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from iscc_search.cli.common import console, get_default_index
from iscc_search.schema import IsccQuery

__all__ = ["search_command"]


def search_command(
    iscc_code,  # type: str
    limit=typer.Option(3, "--limit", "-l", help="Maximum number of results"),  # type: int
):
    # type: (...) -> None
    """
    Search for similar ISCC assets.

    Returns top N most similar assets from the default index.

    Example:
        iscc-search search ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY
        iscc-search search ISCC:KEC... --limit 10
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

    # Create query using IsccQuery (not IsccEntry)
    query = IsccQuery(iscc_code=iscc_code)

    # Search
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Searching...", total=None)
        results = index.search_assets("default", query, limit=int(limit))
        progress.remove_task(task)

    # Close index
    index.close()

    # Serialize IsccSearchResult directly - faithfully reproduce index output
    output = results.model_dump(mode="json", exclude_none=True)

    console.print_json(json.dumps(output))
