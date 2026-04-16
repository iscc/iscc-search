"""
Hub command for ISCC-Search CLI.

Streams published ISCC datasets from the HuggingFace Hub (parquet format)
into a local index, auto-registering one named after the dataset when needed.
"""

import json
import os
import re
import time

# Silence huggingface_hub's tqdm progress bars — they render poorly in non-TTY
# terminals (e.g. PyCharm console) where every refresh writes a new line. Our
# rich spinners + row-count progress bar already give the user feedback.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import iscc_core as ic
import typer
from loguru import logger
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from iscc_search.cli.common import console, get_active_index
from iscc_search.config import LocalIndexConfig, get_config_manager
from iscc_search.schema import IsccEntry
from iscc_search.utils import timer

__all__ = ["hub_command"]


# Composite ISCC-CODE column name and the five ISCC-UNIT columns (canonical header order)
ISCC_CODE_COLUMN = "iscc"
ISCC_UNIT_COLUMNS = ("iscc_meta", "iscc_semantic", "iscc_content", "iscc_data", "iscc_instance")
ISCC_COLUMNS = (ISCC_CODE_COLUMN, *ISCC_UNIT_COLUMNS)


# Monotonically advances past the wall-clock microsecond so tight loops assigning
# ISCC-IDs stay collision-free (ic.gen_iscc_id returns identical IDs when called
# faster than the microsecond clock).
_last_ts = 0


def next_iscc_id():
    # type: () -> str
    """Return a fresh ISCC-ID, guaranteed unique across tight call loops."""
    global _last_ts
    _last_ts = max(time.time_ns() // 1000, _last_ts + 1)
    return ic.gen_iscc_id(timestamp=_last_ts)["iscc"]


def derive_index_name(repo_id):
    # type: (str) -> str
    """
    Derive a valid IsccIndex.name from an HF repo_id.

    Strips the org prefix, drops non-alphanumerics, and removes a leading "iscc"
    (the dataset convention). Examples: ``iscc/iscc-flickr30k`` -> ``flickr30k``,
    ``iscc/iscc-book-covers`` -> ``bookcovers``.
    """
    name = re.sub(r"[^a-z0-9]", "", repo_id.rsplit("/", 1)[-1].lower())
    if name.startswith("iscc") and len(name) > 4:
        name = name[4:]
    name = name[:32]
    if not name or not name[0].isalpha():
        raise ValueError(f"Cannot derive a valid index name from '{repo_id}'")
    return name


def resolve_index(index_name, repo_id):
    # type: (str|None, str) -> tuple[object, str]
    """
    Resolve target index, auto-registering a local one if the name isn't configured yet.

    Uses ``index_name`` when given, otherwise derives it from ``repo_id``.
    """
    target = index_name or derive_index_name(repo_id)
    mgr = get_config_manager()
    mgr.load()
    if target not in {n for n, _, _ in mgr.list_indexes()}:
        mgr.add_index(LocalIndexConfig(name=target))
        console.print(f"Registered local index [cyan]{target}[/cyan]")
    return get_active_index(target)


def select_metadata_columns(schema):
    # type: (object) -> list[str]
    """
    Pick string/numeric/boolean columns from an arrow schema to preserve as metadata.

    Skips ISCC columns and binary/struct/list columns (e.g. image thumbnails).
    """
    import pyarrow as pa

    keep = []
    for field in schema:
        if field.name in ISCC_COLUMNS:
            continue
        t = field.type
        if (
            pa.types.is_string(t)
            or pa.types.is_large_string(t)
            or pa.types.is_integer(t)
            or pa.types.is_floating(t)
            or pa.types.is_boolean(t)
        ):
            keep.append(field.name)
    return keep


def row_to_iscc_entry(row, metadata_columns, repo_id):
    # type: (dict, list[str], str) -> IsccEntry | None
    """
    Map a parquet row (as dict) to an IsccEntry.

    Pulls the composite ``iscc`` and the five ``iscc_*`` unit columns; preserves
    everything in ``metadata_columns`` as opaque metadata plus ``hf_dataset``
    pointing to the repo_id. Returns ``None`` when the row has neither a
    composite ISCC-CODE nor at least two units.
    """
    iscc_code = row.get(ISCC_CODE_COLUMN) or None
    units = [row[c] for c in ISCC_UNIT_COLUMNS if row.get(c)]

    if not iscc_code and len(units) < 2:
        return None

    metadata = {"hf_dataset": repo_id}
    for col in metadata_columns:
        val = row.get(col)
        if val is None or val == "":
            continue
        metadata[col] = val

    kwargs = {"iscc_id": next_iscc_id(), "metadata": metadata}
    if iscc_code:
        kwargs["iscc_code"] = iscc_code
    if len(units) >= 2:
        kwargs["units"] = units
    return IsccEntry(**kwargs)


def list_parquet_files(repo_id, split):
    # type: (str, str|None) -> list[str]
    """List parquet files in a HF dataset repo, optionally filtered by split name."""
    from huggingface_hub import list_repo_files

    files = [f for f in list_repo_files(repo_id, repo_type="dataset") if f.endswith(".parquet")]
    if split:
        needle = split.lower()
        files = [f for f in files if needle in f.lower()]
    return sorted(files)


def flush_batch(index, index_name, batch, errors):
    # type: (object, str, list, list) -> int
    """Flush a batch to the index; collect errors; return number added."""
    if not batch:
        return 0
    try:
        results = index.add_assets(index_name, batch)
        return len(results)
    except Exception as e:
        errors.append(str(e))
        logger.error(f"Batch failed: {e}")
        return 0
    finally:
        batch.clear()


def stream_parquet_file(local_path, batch_size, repo_id, index, index_name, remaining, errors):
    # type: (str, int, str, object, str, int|None, list) -> tuple[int, int]
    """
    Stream rows from one parquet file into the index.

    ``remaining`` caps the number of rows read from this file (None = no cap).
    Returns (rows_read, assets_added).
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(local_path)
    metadata_columns = select_metadata_columns(pf.schema_arrow)
    present_iscc = [c for c in ISCC_COLUMNS if c in pf.schema_arrow.names]
    if not present_iscc:
        logger.warning(f"Skipping {local_path}: no ISCC columns found")
        return 0, 0

    read_columns = present_iscc + metadata_columns
    total = pf.metadata.num_rows if remaining is None else min(remaining, pf.metadata.num_rows)

    rows_read = 0
    added = 0
    batch = []  # type: list[IsccEntry]

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Rows from {local_path.split('/')[-1]}", total=total)
        for record_batch in pf.iter_batches(batch_size=batch_size, columns=read_columns):
            for row in record_batch.to_pylist():
                if remaining is not None and rows_read >= remaining:
                    break
                rows_read += 1
                progress.update(task, advance=1)
                entry = row_to_iscc_entry(row, metadata_columns, repo_id)
                if entry is None:
                    continue
                batch.append(entry)
                if len(batch) >= batch_size:
                    added += flush_batch(index, index_name, batch, errors)
            if remaining is not None and rows_read >= remaining:
                break
        added += flush_batch(index, index_name, batch, errors)

    return rows_read, added


def hub_command(
    repo_id,  # type: str
    split: str = typer.Option("train", "--split", help="Dataset split to index (substring match on filenames)"),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Stop after indexing N rows"),
    batch_size: int = typer.Option(500, "--batch-size", "-b", help="Rows per index batch"),
    index_name: str | None = typer.Option(
        None,
        "--index",
        help="Target index name (auto-derived from REPO_ID when omitted; auto-registered locally when missing)",
    ),
):
    # type: (...) -> None
    """
    Index an ISCC dataset published on the HuggingFace Hub.

    Auto-registers a local index named after the dataset when one doesn't exist
    yet. Expects the standard ISCC parquet schema: ``iscc``, ``iscc_meta``,
    ``iscc_semantic``, ``iscc_content``, ``iscc_data``, ``iscc_instance`` plus
    any string/numeric columns preserved as opaque metadata. Binary columns
    (e.g. image thumbnails) are skipped.

    Example:
        iscc-search hub iscc/iscc-flickr30k
        iscc-search hub iscc/iscc-book-covers --limit 10000
        iscc-search hub iscc/iscc-flickr30k --index production
    """
    # Resolve (and auto-register) target index
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        task = p.add_task("Initializing index...", total=None)
        try:
            index, target_index_name = resolve_index(index_name, repo_id)
        except ValueError as e:
            p.remove_task(task)
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1)
        p.remove_task(task)

    # Discover parquet files
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        task = p.add_task(f"Listing parquet files in {repo_id}...", total=None)
        try:
            files = list_parquet_files(repo_id, split)
        except Exception as e:
            p.remove_task(task)
            console.print(f"[red]Failed to list files for {repo_id}: {e}[/red]")
            raise typer.Exit(code=2)
        p.remove_task(task)

    if not files:
        console.print(f"[red]No parquet files found in {repo_id} for split '{split}'[/red]")
        raise typer.Exit(code=2)

    console.print(f"Found [cyan]{len(files)}[/cyan] parquet file(s) in [cyan]{repo_id}[/cyan] (split={split})")

    from huggingface_hub import hf_hub_download

    total_rows = 0
    total_added = 0
    errors = []  # type: list[str]

    with timer(f"Indexing {repo_id} → '{target_index_name}'"):
        for rel_path in files:
            remaining = None if limit is None else limit - total_rows
            if remaining is not None and remaining <= 0:
                break
            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
            ) as p:
                task = p.add_task(f"Downloading {rel_path}...", total=None)
                try:
                    local_path = hf_hub_download(repo_id, rel_path, repo_type="dataset")
                except Exception as e:
                    p.remove_task(task)
                    errors.append(f"download {rel_path}: {e}")
                    console.print(f"[yellow]Skipping {rel_path}: {e}[/yellow]")
                    continue
                p.remove_task(task)

            rows_read, added = stream_parquet_file(
                local_path=local_path,
                batch_size=batch_size,
                repo_id=repo_id,
                index=index,
                index_name=target_index_name,
                remaining=remaining,
                errors=errors,
            )
            total_rows += rows_read
            total_added += added

    # Close index so writes are flushed
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        task = p.add_task("Saving index...", total=None)
        index.close()
        p.remove_task(task)

    output = {
        "dataset": repo_id,
        "split": split,
        "index": target_index_name,
        "parquet_files": len(files),
        "rows_read": total_rows,
        "assets_added": total_added,
        "errors": len(errors),
    }
    console.print_json(json.dumps(output))

    if errors:
        console.print("\n[yellow]Warnings:[/yellow]")
        for err in errors[:5]:
            console.print(f"  [yellow]•[/yellow] {err}")
        if len(errors) > 5:
            console.print(f"  ... and {len(errors) - 5} more")
