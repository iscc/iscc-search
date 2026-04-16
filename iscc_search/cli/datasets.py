"""
Datasets command — list ISCC datasets published on the HuggingFace Hub.
"""

import json

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from iscc_search.cli.common import console

__all__ = ["datasets_command"]


def _size_from_tags(tags):
    # type: (list[str] | None) -> str
    """Extract size category (e.g. ``10K<n<100K``) from HF dataset tags."""
    for tag in tags or []:
        if tag.startswith("size_categories:"):
            return tag.split(":", 1)[1]
    return "-"


def datasets_command(
    author: str = typer.Option("iscc", "--author", "-a", help="HuggingFace org/user to browse"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max datasets to list"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of a rich table"),
):
    # type: (...) -> None
    """
    List ISCC datasets published on the HuggingFace Hub.

    Defaults to the curated ``iscc`` organization. Use ``--author`` to browse other namespaces.
    Pass the printed Repo ID to ``iscc-search hub`` to index a dataset.

    Example:
        iscc-search datasets
        iscc-search datasets --author myorg --limit 50
        iscc-search datasets --json
    """
    from huggingface_hub import list_datasets

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        task = p.add_task(f"Listing datasets for '{author}'...", total=None)
        try:
            infos = list(list_datasets(author=author, limit=limit))
        except Exception as e:
            p.remove_task(task)
            console.print(f"[red]Failed to list datasets for '{author}': {e}[/red]")
            raise typer.Exit(code=1)
        p.remove_task(task)

    if json_output:
        rows = [
            {
                "repo_id": d.id,
                "downloads": d.downloads,
                "likes": d.likes,
                "last_modified": d.last_modified.isoformat() if d.last_modified else None,
                "size_category": _size_from_tags(d.tags),
                "tags": list(d.tags or []),
            }
            for d in infos
        ]
        console.print_json(json.dumps(rows))
        return

    if not infos:
        console.print(f"[yellow]No datasets found for author '{author}'[/yellow]")
        return

    table = Table(title=f"Datasets in '{author}' ({len(infos)})")
    table.add_column("Repo ID", style="cyan", no_wrap=True)
    table.add_column("Downloads", justify="right")
    table.add_column("Likes", justify="right")
    table.add_column("Updated")
    table.add_column("Size")

    for d in infos:
        updated = d.last_modified.strftime("%Y-%m-%d") if d.last_modified else "-"
        table.add_row(d.id, str(d.downloads or 0), str(d.likes or 0), updated, _size_from_tags(d.tags))

    console.print(table)
    console.print(f"\n[dim]Hint: iscc-search hub {infos[0].id}[/dim]")
