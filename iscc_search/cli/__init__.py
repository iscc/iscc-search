"""
ISCC-Search CLI.

Command-line interface for managing ISCC indexes and searching for similar content.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import simdjson
import typer
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
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
    seen_iscc_ids = set()  # Track ISCC-IDs to detect duplicates
    # Reuse parser instance for optimal performance (reduces allocation overhead)
    parser = simdjson.Parser()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing files...", total=len(files))

        for file_path in files:
            try:
                # Use simdjson for efficient parsing of only required fields
                with open(file_path, "rb") as f:
                    doc = parser.parse(f.read())

                # Map JSON fields to IsccAsset - extract to plain Python types immediately
                asset_data = {}

                # Handle iscc_id
                if "iscc_id" in doc:
                    iscc_id = str(doc["iscc_id"])
                    asset_data["iscc_id"] = iscc_id

                    # Check for duplicate ISCC-ID
                    if iscc_id in seen_iscc_ids:
                        logger.warning(f"Duplicate ISCC-ID encountered: {iscc_id} in {file_path.name}")
                    else:
                        seen_iscc_ids.add(iscc_id)

                # Handle iscc_code - try 'iscc_code' first, fall back to 'iscc'
                if "iscc_code" in doc:
                    asset_data["iscc_code"] = str(doc["iscc_code"])
                elif "iscc" in doc:
                    asset_data["iscc_code"] = str(doc["iscc"])

                # Handle units
                if "units" in doc:
                    # Convert to Python list of strings immediately
                    asset_data["units"] = [str(u) for u in doc["units"]]

                # Release proxy object before next parser reuse
                del doc

                # Do not collect metadata - only index ISCC fields
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
