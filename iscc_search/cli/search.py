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
    meta: bool = typer.Option(False, "--meta", "-m", help="Include metadata for matched results"),
):
    # type: (...) -> None
    """
    Search for similar ISCC assets.

    Returns top N most similar assets from the default index.

    Example:
        iscc-search search ISCC:KECYCMZIOY36XXGZ7S6QJQ2AEEXPOVEHZYPK6GMSFLU3WF54UPZMTPY
        iscc-search search ISCC:KEC... --limit 10 --meta
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

    # Build matches output
    matches_output = []
    for match in results.global_matches:
        match_dict = {
            "iscc_id": match.iscc_id,
            "score": match.score,
            "types": match.types,
        }

        # If --meta flag is set, retrieve and add metadata
        if meta:
            try:
                asset = index.get_asset("default", match.iscc_id)
                if asset.metadata:
                    match_dict["metadata"] = asset.metadata
            except FileNotFoundError:
                # Asset not found, skip metadata
                pass

        matches_output.append(match_dict)

    # Close index
    index.close()

    # Output as JSON
    output = {
        "query": {"iscc_code": results.query.iscc_code, "units": results.query.units},
        "matches": matches_output,
    }

    console.print_json(json.dumps(output))
