"""CLI interface for ISCC Vector Database."""

import json
import sys
from pathlib import Path
import click
import simdjson
from loguru import logger
from platformdirs import user_data_dir
from tqdm import tqdm
from iscc_vdb.lookup import IsccLookupIndex
from iscc_vdb.types import IsccItemDict


APP_NAME = "iscc-vdb"
APP_AUTHOR = "iscc"


def get_default_db_path():
    # type: () -> Path
    """
    Get default database path using platformdirs.

    :return: Path to default database file
    """
    data_dir = Path(user_data_dir(APP_NAME, APP_AUTHOR))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "lookup.mdb"


@click.group()
@click.option(
    "--db",
    type=click.Path(path_type=Path),
    default=None,
    help=f"Path to database file (default: {get_default_db_path()})",
)
@click.pass_context
def cli(ctx, db):
    # type: (click.Context, Path | None) -> None
    """ISCC Vector Database CLI."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db or get_default_db_path()
    logger.remove()
    logger.add(sys.stderr, level="INFO")


@cli.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def add(ctx, directory):
    # type: (click.Context, Path) -> None
    """
    Scan directory for *.iscc.json files and add to index.

    Searches recursively for files matching *.iscc.json pattern,
    extracts iscc_id, iscc_code, and units fields, then adds them
    to the lookup index.
    """
    db_path = ctx.obj["db_path"]
    logger.info(f"Using database: {db_path}")
    logger.info(f"Scanning directory: {directory}")

    # Find all *.iscc.json files
    json_files = list(directory.rglob("*.iscc.json"))
    logger.info(f"Found {len(json_files)} *.iscc.json files")

    if not json_files:
        logger.warning("No *.iscc.json files found")
        return

    # Initialize index
    idx = IsccLookupIndex(db_path)

    added_count = 0
    error_count = 0

    for json_file in tqdm(json_files, desc="Processing files", unit="file"):
        try:
            # Use simdjson for efficient parsing
            with open(json_file, "rb") as f:
                doc = simdjson.loads(f.read())

            # Extract only the fields we need
            item = IsccItemDict()

            # Try to get iscc_id
            if "iscc_id" in doc:
                item["iscc_id"] = doc["iscc_id"]

            # Try to get iscc_code
            if "iscc_code" in doc:
                item["iscc_code"] = doc["iscc_code"]

            # Try to get units
            if "units" in doc:
                item["units"] = doc["units"]

            # Only add if we have either iscc_code or units
            if "iscc_code" in item or "units" in item:
                idx.add(item)
                added_count += 1
                logger.debug(f"Added: {json_file.name}")
            else:
                logger.warning(f"Skipping {json_file.name}: no iscc_code or units found")

        except Exception as e:
            error_count += 1
            logger.error(f"Error processing {json_file}: {e}")

    idx.close()
    logger.info(f"Successfully added {added_count} items")
    if error_count:
        logger.warning(f"Failed to process {error_count} files")


@cli.command()
@click.argument("iscc_code", type=str)
@click.option("--limit", default=100, help="Maximum number of results (default: 100)")
@click.pass_context
def search(ctx, iscc_code, limit):
    # type: (click.Context, str, int) -> None
    """
    Search index for similar ISCC codes.

    Searches the lookup index and outputs results as pretty-printed JSON.
    """
    db_path = ctx.obj["db_path"]
    logger.info(f"Using database: {db_path}")
    logger.info(f"Searching for: {iscc_code}")

    # Initialize index
    idx = IsccLookupIndex(db_path)

    # Create search query
    query = IsccItemDict(iscc_code=iscc_code)

    # Search
    results = idx.search(query, limit=limit)

    idx.close()

    # Output as pretty JSON
    click.echo(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info(f"Found {len(results[0]['lookup_matches']) if results else 0} matches")


@cli.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--limit", default=100, help="Maximum number of results per query (default: 100)")
@click.pass_context
def match(ctx, directory, limit):
    # type: (click.Context, Path, int) -> None
    """
    Scan directory and find matches in index, excluding self-matches.

    Searches recursively for files matching *.iscc.json pattern,
    extracts iscc_id and iscc_code fields, searches the index,
    and outputs only results with matches different from the query iscc_id.
    """
    db_path = ctx.obj["db_path"]
    logger.info(f"Using database: {db_path}")
    logger.info(f"Scanning directory: {directory}")

    # Find all *.iscc.json files
    json_files = list(directory.rglob("*.iscc.json"))
    logger.info(f"Found {len(json_files)} *.iscc.json files")

    if not json_files:
        logger.warning("No *.iscc.json files found")
        return

    # Initialize index
    idx = IsccLookupIndex(db_path)

    matches_found = []
    processed_count = 0
    error_count = 0

    for json_file in tqdm(json_files, desc="Matching files", unit="file"):
        try:
            # Use simdjson for efficient parsing
            with open(json_file, "rb") as f:
                doc = simdjson.loads(f.read())

            # Extract only iscc_code and iscc_id
            query_iscc_id = doc.get("iscc_id")
            query_iscc_code = doc.get("iscc_code")

            # Skip if no iscc_code
            if not query_iscc_code:
                logger.debug(f"Skipping {json_file.name}: no iscc_code found")
                continue

            # Create search query
            query = IsccItemDict(iscc_code=query_iscc_code)

            # Search index
            results = idx.search(query, limit=limit)

            # Filter out self-matches and empty results
            if results and results[0]["lookup_matches"]:
                filtered_matches = [
                    match for match in results[0]["lookup_matches"] if match["iscc_id"] != query_iscc_id
                ]

                # Only include if there are matches after filtering
                if filtered_matches:
                    match_result = {
                        "query_file": str(json_file),
                        "query_iscc_id": query_iscc_id,
                        "query_iscc_code": query_iscc_code,
                        "matches": filtered_matches,
                    }
                    matches_found.append(match_result)

            processed_count += 1

        except Exception as e:
            error_count += 1
            logger.error(f"Error processing {json_file}: {e}")

    idx.close()

    # Output results as pretty JSON
    click.echo(json.dumps(matches_found, indent=2, ensure_ascii=False))

    logger.info(f"Processed {processed_count} files")
    logger.info(f"Found {len(matches_found)} files with matches")
    if error_count:
        logger.warning(f"Failed to process {error_count} files")


if __name__ == "__main__":
    cli()
