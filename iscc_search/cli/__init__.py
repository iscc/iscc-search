"""
ISCC-Search CLI.

Command-line interface for managing ISCC indexes and searching for similar content.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

if TYPE_CHECKING:
    from iscc_search.protocol import IsccIndexProtocol  # noqa: F401

import iscc_search

__all__ = ["app", "main"]


app = typer.Typer(
    name="iscc-search",
    help="ISCC similarity search CLI",
    no_args_is_help=True,
)

console = Console()


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


@app.command()
def add(
    pattern,  # type: str
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
):
    # type: (...) -> None
    """
    Add ISCC assets from JSON files to the default index.

    Accepts file paths, directory paths, or glob patterns.
    Files must be valid JSON with 'iscc_code' or 'iscc'/'units' fields.

    Example:
        iscc-search add myfolder/*.json
        iscc-search add /path/to/assets/
        iscc-search add asset.iscc.json
    """
    from iscc_search.schema import IsccAsset

    # Get or create default index
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing index...", total=None)
        index = get_default_index()
        progress.remove_task(task)

    # Expand pattern to files
    pattern_path = Path(pattern)

    if pattern_path.is_file():
        # Single file
        files = [pattern_path]
    elif pattern_path.is_dir():
        # Directory - find all *.iscc.json files
        files = list(pattern_path.rglob("*.iscc.json"))
        if not files:
            # Fall back to *.json
            files = list(pattern_path.rglob("*.json"))
    else:
        # Glob pattern
        files = list(Path(".").glob(pattern))

    if not files:
        console.print(f"[red]No files found matching: {pattern}[/red]")
        raise typer.Exit(code=2)

    # Parse JSON files and create assets
    assets = []
    errors = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing files...", total=len(files))

        for file_path in files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)

                # Map JSON fields to IsccAsset
                asset_data = {}

                # Handle iscc_id
                if "iscc_id" in data:
                    asset_data["iscc_id"] = data["iscc_id"]

                # Handle iscc_code - try 'iscc_code' first, fall back to 'iscc'
                if "iscc_code" in data:
                    asset_data["iscc_code"] = data["iscc_code"]
                elif "iscc" in data:
                    asset_data["iscc_code"] = data["iscc"]

                # Handle units
                if "units" in data:
                    asset_data["units"] = data["units"]

                # Collect all other fields as metadata
                metadata_fields = set(data.keys()) - {"iscc_id", "iscc_code", "iscc", "units"}
                if metadata_fields:
                    asset_data["metadata"] = {k: data[k] for k in metadata_fields}

                asset = IsccAsset(**asset_data)
                assets.append(asset)

                if verbose:
                    console.print(f"[green]✓[/green] {file_path.name}")

            except Exception as e:
                error_msg = f"{file_path.name}: {str(e)}"
                errors.append(error_msg)
                if verbose:
                    console.print(f"[red]✗[/red] {error_msg}")

            progress.update(task, advance=1)

    if not assets:
        console.print("[red]No valid assets found[/red]")
        if errors:
            console.print("\nErrors:")
            for error in errors:
                console.print(f"  [red]•[/red] {error}")
        raise typer.Exit(code=4)

    # Add assets to index
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Adding {len(assets)} asset(s)...", total=None)
        results = index.add_assets("default", assets)
        progress.remove_task(task)

    # Close index to save all data
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Saving indexes...", total=None)
        index.close()
        progress.remove_task(task)

    # Count results
    created = sum(1 for r in results if r.status == "created")
    updated = sum(1 for r in results if r.status == "updated")

    # Output results as JSON
    output = {
        "files_scanned": len(files),
        "assets_added": len(assets),
        "created": created,
        "updated": updated,
        "errors": len(errors),
    }

    console.print_json(json.dumps(output))

    if errors:
        console.print("\n[yellow]Warnings:[/yellow]")
        for error in errors[:5]:  # Show first 5 errors
            console.print(f"  [yellow]•[/yellow] {error}")
        if len(errors) > 5:
            console.print(f"  ... and {len(errors) - 5} more")


@app.command()
def search(
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
    from iscc_search.schema import IsccAsset

    # Get default index
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading index...", total=None)
        index = get_default_index()
        progress.remove_task(task)

    # Create query asset
    query = IsccAsset(iscc_code=iscc_code)

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

    # Output as JSON
    output = {
        "query": {"iscc_code": results.query.iscc_code, "units": results.query.units},
        "metric": results.metric,
        "matches": [
            {
                "iscc_id": match.iscc_id,
                "score": match.score,
                "matches": match.matches,
            }
            for match in results.matches
        ],
    }

    console.print_json(json.dumps(output))


@app.command()
def version():
    # type: () -> None
    """Show version information."""
    console.print(f"iscc-search version {iscc_search.__version__}")


def main():
    # type: () -> None
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
