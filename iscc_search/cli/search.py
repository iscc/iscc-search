"""
Search command for ISCC-Search CLI.

Handles searching for similar ISCC assets in the default index.
"""

import json

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from iscc_search.cli.common import console
from iscc_search.schema import IsccQuery

__all__ = ["search_command"]


def search_command(
    iscc_code,  # type: str
    limit=typer.Option(3, "--limit", "-l", help="Maximum number of results"),  # type: int
    index_name: str | None = typer.Option(None, "--index", help="Index name to use (overrides active index)"),
):
    # type: (...) -> None
    """
    Search for similar ISCC assets.

    Returns top N most similar assets from the active index.

    Example:
        iscc-search search ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY
        iscc-search search ISCC:KEC... --limit 10
        iscc-search search ISCC:KEC... --index production
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

    # Create query using IsccQuery (not IsccEntry)
    query = IsccQuery(iscc_code=iscc_code)

    # Search
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Searching...", total=None)
        results = index.search_assets(target_index_name, query, limit=int(limit))
        progress.remove_task(task)

    # Close index
    index.close()

    # Serialize IsccSearchResult directly - faithfully reproduce index output
    output = results.model_dump(mode="json", exclude_none=True)

    console.print_json(json.dumps(output))
