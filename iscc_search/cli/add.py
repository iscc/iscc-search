"""
Add command for ISCC-Search CLI.

Handles adding ISCC assets from JSON files to the default index.
"""

import json
from pathlib import Path

import simdjson
import typer
from loguru import logger
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from iscc_search.cli.common import console, get_default_index
from iscc_search.schema import IsccEntry
from iscc_search.utils import timer

__all__ = ["add_command"]


def expand_pattern_to_files(pattern):
    # type: (str) -> list[Path]
    """
    Expand pattern to list of files.

    Handles single files, directories, and glob patterns.

    :param pattern: File path, directory path, or glob pattern
    :return: List of Path objects
    """
    pattern_path = Path(pattern)

    if pattern_path.is_file():
        # Single file
        return [pattern_path]
    elif pattern_path.is_dir():
        # Directory - find all *.iscc.json files
        files = list(pattern_path.rglob("*.iscc.json"))
        if not files:
            # Fall back to *.json
            files = list(pattern_path.rglob("*.json"))
        return files
    else:
        # Glob pattern
        return list(Path(".").glob(pattern))


def parse_asset_files(files, verbose=False):
    # type: (list[Path], bool) -> tuple[list[IsccEntry], list[str]]
    """
    Parse JSON files into IsccEntry objects.

    Uses simdjson for efficient parsing. Tracks duplicate ISCC-IDs.

    :param files: List of file paths to parse
    :param verbose: Show detailed progress for each file
    :return: Tuple of (assets, errors)
    """
    assets = []
    errors = []
    seen_iscc_ids = {}  # type: dict[str, str]  # Track ISCC-IDs to filename mapping
    # Reuse parser instance for optimal performance
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

                # Map JSON fields to IsccEntry - extract to plain Python types immediately
                asset_data = {}

                # Handle iscc_id
                if "iscc_id" in doc:
                    iscc_id = str(doc["iscc_id"])
                    asset_data["iscc_id"] = iscc_id

                    # Check for duplicate ISCC-ID
                    if iscc_id in seen_iscc_ids:
                        previous_file = seen_iscc_ids[iscc_id]
                        logger.warning(
                            f"Duplicate: {iscc_id} for {file_path.name.split('.')[0]} & {previous_file.split('.')[0]}"
                        )
                    else:
                        seen_iscc_ids[iscc_id] = file_path.name

                # Handle iscc_code - try 'iscc_code' first, fall back to 'iscc'
                if "iscc_code" in doc:
                    asset_data["iscc_code"] = str(doc["iscc_code"])
                elif "iscc" in doc:
                    asset_data["iscc_code"] = str(doc["iscc"])

                # Handle units
                if "units" in doc:
                    # Convert to Python list of strings immediately
                    asset_data["units"] = [str(u) for u in doc["units"]]

                # Handle metadata - collect name and filename if present
                metadata = {}
                if "name" in doc:
                    name = str(doc["name"]).strip()
                    if name:
                        metadata["name"] = name
                if "filename" in doc:
                    filename = str(doc["filename"]).strip()
                    if filename:
                        metadata["filename"] = filename

                if metadata:
                    asset_data["metadata"] = metadata

                # Release proxy object before next parser reuse
                del doc

                asset = IsccEntry(**asset_data)
                assets.append(asset)

                if verbose:
                    console.print(f"[green]✓[/green] {file_path.name}")

            except Exception as e:
                error_msg = f"{file_path.name}: {str(e)}"
                errors.append(error_msg)
                if verbose:
                    console.print(f"[red]✗[/red] {error_msg}")

            progress.update(task, advance=1)

    return assets, errors


def format_add_results(results, files_count, assets_count, errors):
    # type: (list, int, int, list[str]) -> dict
    """
    Format add command results as JSON-serializable dict.

    :param results: List of add results from index
    :param files_count: Number of files scanned
    :param assets_count: Number of assets added
    :param errors: List of error messages
    :return: Output dictionary
    """
    created = sum(1 for r in results if r.status == "created")
    updated = sum(1 for r in results if r.status == "updated")

    return {
        "files_scanned": files_count,
        "assets_added": assets_count,
        "created": created,
        "updated": updated,
        "errors": len(errors),
    }


def add_command(
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
    files = expand_pattern_to_files(pattern)

    if not files:
        console.print(f"[red]No files found matching: {pattern}[/red]")
        raise typer.Exit(code=2)

    # Parse JSON files and create assets
    assets, errors = parse_asset_files(files, verbose=verbose)

    if not assets:
        console.print("[red]No valid assets found[/red]")
        if errors:
            console.print("\nErrors:")
            for error in errors:
                console.print(f"  [red]•[/red] {error}")
        raise typer.Exit(code=4)

    # Add assets to index
    with timer(f"Indexing {len(assets)} asset(s)"):
        results = index.add_assets("default", assets)

    # Close index to save all data
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Saving indexes...", total=None)
        index.close()
        progress.remove_task(task)

    # Format and output results as JSON
    output = format_add_results(results, len(files), len(assets), errors)
    console.print_json(json.dumps(output))

    if errors:
        console.print("\n[yellow]Warnings:[/yellow]")
        for error in errors[:5]:  # Show first 5 errors
            console.print(f"  [yellow]•[/yellow] {error}")
        if len(errors) > 5:
            console.print(f"  ... and {len(errors) - 5} more")
