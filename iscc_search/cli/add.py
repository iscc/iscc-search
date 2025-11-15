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

from iscc_search.cli.common import console, parse_simprints_from_features
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
        if pattern_path.is_absolute():
            # For absolute paths, glob from parent directory
            parent = pattern_path.parent
            name = pattern_path.name
            return list(parent.glob(name))
        else:
            # For relative patterns, glob from current directory
            return list(Path(".").glob(pattern))


def parse_and_index_files(files, index, index_name, batch_size=100, verbose=False, simprint_bits=None):
    # type: (list[Path], object, str, int, bool, int | None) -> tuple[list, list[str]]
    """
    Parse JSON files and index assets in batches using optimized simdjson.

    Uses streaming approach with batched indexing to minimize memory usage.
    Leverages simdjson proxy objects and JSON pointers for efficient field extraction.

    :param files: List of file paths to parse
    :param index: Index instance for direct indexing
    :param index_name: Target index name
    :param batch_size: Number of assets per batch
    :param verbose: Show detailed progress for each file
    :param simprint_bits: Truncate simprints to this bit length (64, 128, 192, 256)
    :return: Tuple of (all_results, errors)
    """
    all_results = []  # type: list
    errors = []
    seen_iscc_ids = {}  # type: dict[str, str]  # Track ISCC-IDs to filename mapping

    # Reuse parser instance for optimal performance
    parser = simdjson.Parser()

    # Batch accumulator
    batch = []  # type: list[IsccEntry]
    batch_count = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        parse_task = progress.add_task("Parsing files...", total=len(files))

        for file_path in files:
            try:
                # Read file as bytes (optimal for simdjson)
                with open(file_path, "rb") as f:
                    file_bytes = f.read()

                # Parse with recursive=False (default) for lazy proxy objects
                doc = parser.parse(file_bytes)

                # Map JSON fields to IsccEntry using optimized extraction
                asset_data = {}

                # Handle iscc_id - use JSON pointer for top-level access
                try:
                    iscc_id = str(doc.at_pointer("/iscc_id"))
                    asset_data["iscc_id"] = iscc_id

                    # Check for duplicate ISCC-ID
                    if iscc_id in seen_iscc_ids:
                        previous_file = seen_iscc_ids[iscc_id]
                        logger.warning(
                            f"Duplicate: {iscc_id} for {file_path.name.split('.')[0]} & {previous_file.split('.')[0]}"
                        )
                    else:
                        seen_iscc_ids[iscc_id] = file_path.name
                except KeyError:
                    pass  # iscc_id not present

                # Handle iscc_code - try 'iscc_code' first, fall back to 'iscc'
                try:
                    asset_data["iscc_code"] = str(doc.at_pointer("/iscc_code"))
                except KeyError:
                    try:
                        asset_data["iscc_code"] = str(doc.at_pointer("/iscc"))
                    except KeyError:
                        pass  # Neither field present

                # Handle units - use dict-style access on proxy
                if "units" in doc:
                    units_proxy = doc["units"]  # Get proxy Array reference
                    asset_data["units"] = [str(u) for u in units_proxy]
                    del units_proxy  # Delete proxy reference

                # Handle metadata - collect name and source URI
                metadata = {}
                if "name" in doc:
                    name = str(doc["name"]).strip()
                    if name:
                        metadata["name"] = name

                # Add source URI pointing to .iscc.utf8 file
                source_path = file_path.with_suffix(".utf8")
                metadata["source"] = source_path.as_uri()

                if metadata:
                    asset_data["metadata"] = metadata

                # Handle features - iterate over proxy array without full conversion
                if "features" in doc:
                    features_proxy = doc["features"]  # Get proxy Array reference
                    features = []

                    for feature_proxy in features_proxy:
                        # Access fields via proxy dict (converts only accessed elements)
                        simprints_proxy = feature_proxy.get("simprints", [])
                        offsets_proxy = feature_proxy.get("offsets", [])
                        sizes_proxy = feature_proxy.get("sizes", [])

                        feature_dict = {
                            "maintype": str(feature_proxy.get("maintype", "")),
                            "subtype": str(feature_proxy.get("subtype", "")),
                            "version": int(feature_proxy.get("version", 0)),
                            "simprints": [str(s) for s in simprints_proxy],
                            "offsets": [int(o) for o in offsets_proxy],
                            "sizes": [int(sz) for sz in sizes_proxy],
                        }
                        features.append(feature_dict)

                    # Delete proxy references (only loop vars if they were defined)
                    del features_proxy  # Always defined at line 150
                    if features:  # Loop variables only exist if loop executed
                        del feature_proxy, simprints_proxy, offsets_proxy, sizes_proxy

                    simprints = parse_simprints_from_features(features, simprint_bits=simprint_bits)
                    if simprints:
                        asset_data["simprints"] = simprints

                asset = IsccEntry(**asset_data)
                batch.append(asset)

                # Delete doc proxy to allow parser reuse
                del doc

                if verbose:
                    console.print(f"[green]✓[/green] {file_path.name}")

                # Flush batch when full
                if len(batch) >= batch_size:
                    batch_count += 1
                    if verbose:
                        console.print(f"[cyan]Indexing batch {batch_count} ({len(batch)} assets)...[/cyan]")

                    results = index.add_assets(index_name, batch)
                    all_results.extend(results)
                    batch.clear()

            except Exception as e:
                error_msg = f"{file_path.name}: {str(e)}"
                errors.append(error_msg)
                if verbose:
                    console.print(f"[red]✗[/red] {error_msg}")

            progress.update(parse_task, advance=1)

        # Flush remaining assets
        if batch:
            batch_count += 1
            if verbose:
                console.print(f"[cyan]Indexing final batch {batch_count} ({len(batch)} assets)...[/cyan]")

            results = index.add_assets(index_name, batch)
            all_results.extend(results)
            batch.clear()

    return all_results, errors


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
    simprint_bits: int | None = typer.Option(
        None, "--simprint-bits", "-s", help="Truncate simprints to this bit length (64, 128, 192, or 256)"
    ),
    batch_size: int = typer.Option(100, "--batch-size", "-b", help="Number of assets to index per batch"),
    index_name: str | None = typer.Option(None, "--index", help="Index name to use (overrides active index)"),
):
    # type: (...) -> None
    """
    Add ISCC assets from JSON files to the active index.

    Accepts file paths, directory paths, or glob patterns.
    Files must be valid JSON with 'iscc_code' or 'iscc'/'units' fields.

    Example:
        iscc-search add myfolder/*.json
        iscc-search add -s 64 /path/to/assets/
        iscc-search add --simprint-bits 128 asset.iscc.json
        iscc-search add --batch-size 500 --index production data/*.json
    """
    from iscc_search.cli.common import get_active_index

    # Validate simprint_bits parameter
    if simprint_bits is not None:
        valid_sizes = [64, 128, 192, 256]
        if simprint_bits not in valid_sizes:
            console.print(f"[red]Invalid --simprint-bits: {simprint_bits}. Must be one of: {valid_sizes}[/red]")
            raise typer.Exit(code=3)

    # Get active or specified index
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing index...", total=None)
        try:
            index, target_index_name = get_active_index(index_name)
        except ValueError as e:
            progress.remove_task(task)
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1)
        progress.remove_task(task)

    # Expand pattern to files
    files = expand_pattern_to_files(pattern)

    if not files:
        console.print(f"[red]No files found matching: {pattern}[/red]")
        raise typer.Exit(code=2)

    # Parse JSON files and index in batches
    with timer(f"Processing {len(files)} file(s) to '{target_index_name}'"):
        results, errors = parse_and_index_files(
            files=files,
            index=index,
            index_name=target_index_name,
            batch_size=batch_size,
            verbose=verbose,
            simprint_bits=simprint_bits,
        )

    if not results:
        console.print("[red]No valid assets found[/red]")
        if errors:
            console.print("\nErrors:")
            for error in errors:
                console.print(f"  [red]•[/red] {error}")
        raise typer.Exit(code=4)

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
    output = format_add_results(results, len(files), len(results), errors)
    console.print_json(json.dumps(output))

    if errors:
        console.print("\n[yellow]Warnings:[/yellow]")
        for error in errors[:5]:  # Show first 5 errors
            console.print(f"  [yellow]•[/yellow] {error}")
        if len(errors) > 5:
            console.print(f"  ... and {len(errors) - 5} more")
