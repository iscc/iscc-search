"""
Index management CLI commands.

Provides commands for managing index configurations (local and remote),
including registration, listing, activation, and removal.
"""

import os
import shutil
import sys
from pathlib import Path

import typer
from rich.table import Table

from iscc_search.cli.common import console
from iscc_search.config import (
    LocalIndexConfig,
    RemoteIndexConfig,
    get_config_manager,
)


def add_command(
    name: str,
    local: bool = typer.Option(False, "--local", help="Create local index"),
    remote: str | None = typer.Option(None, "--remote", help="URL of remote server"),
    path: str | None = typer.Option(None, "--path", help="Path for local index (optional)"),
    api_key: str | None = typer.Option(None, "--api-key", help="API key for remote index"),
):
    # type: (...) -> None
    """
    Register a new index configuration.

    Register either a local index or a remote index. For remote indexes,
    the API key can be provided via --api-key flag or ISCC_SEARCH_API_KEY
    environment variable.

    Examples:

        # Register local index
        iscc-search index add mylocal --local

        # Register local index with custom path
        iscc-search index add mylocal --local --path /data/iscc

        # Register remote index
        iscc-search index add production --remote https://api.example.com

        # Register remote index with API key
        iscc-search index add production --remote https://api.example.com --api-key secret
    """
    config_manager = get_config_manager()

    # Validate: must specify either --local or --remote
    if not local and remote is None:
        console.print("[red]Error: Must specify either --local or --remote[/red]")
        sys.exit(1)

    if local and remote is not None:
        console.print("[red]Error: Cannot specify both --local and --remote[/red]")
        sys.exit(1)

    # Create appropriate index config
    if local:
        index_config = LocalIndexConfig(name=name, path=path)
        console.print(f"[green]Registered local index '{name}'[/green]")
        console.print(f"Path: {index_config.path}")
    else:
        # Check for API key in env if not provided
        if api_key is None:
            api_key = os.environ.get("ISCC_SEARCH_API_KEY")

        if remote is None:
            console.print("[red]Error: Remote URL is required[/red]")
            sys.exit(1)

        index_config = RemoteIndexConfig(name=name, url=remote, api_key=api_key)
        console.print(f"[green]Registered remote index '{name}'[/green]")
        console.print(f"URL: {index_config.url}")
        if api_key:
            console.print("[dim]API key: configured[/dim]")

    # Add to config
    config_manager.add_index(index_config)

    # For remote indexes, ensure the index exists on the server
    if not local:
        from iscc_search.remote.client import RemoteIndex
        from iscc_search.schema import IsccIndex

        client = RemoteIndex(url=remote, index_name=name, api_key=api_key)
        try:
            # Check if index exists
            client.get_index(name)
        except FileNotFoundError:
            # Create index if it doesn't exist
            console.print(f"[yellow]Index '{name}' not found on server, creating...[/yellow]")
            client.create_index(IsccIndex(name=name))
            console.print(f"[green]Created index '{name}' on remote server[/green]")
        finally:
            client.close()

    # Show if this is now the active index
    if config_manager.get_active().name == name:
        console.print(f"[cyan]'{name}' is now the active index[/cyan]")


def list_command():
    # type: () -> None
    """
    List all configured indexes.

    Shows all registered indexes with their type (local/remote) and
    indicates which is currently active.

    Example:

        iscc-search index list
    """
    config_manager = get_config_manager()
    indexes = config_manager.list_indexes()

    if not indexes:
        console.print("[yellow]No indexes configured[/yellow]")
        console.print("Use 'iscc-search index add' to register an index")
        return

    # Create table
    table = Table(title="Configured Indexes")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Location/URL", style="white")
    table.add_column("Active", style="green")

    for name, cfg, is_active in indexes:
        active_marker = "✓" if is_active else ""
        if cfg.type == "local":
            location = cfg.path  # type: ignore
        else:
            location = cfg.url  # type: ignore

        table.add_row(name, cfg.type, location, active_marker)

    console.print(table)


def use_command(name: str):
    # type: (...) -> None
    """
    Set the active index.

    All commands (add, search, get) will use this index by default
    unless overridden with --index flag.

    Example:

        iscc-search index use production
    """
    config_manager = get_config_manager()

    try:
        config_manager.set_active(name)
        console.print(f"[green]Active index set to '{name}'[/green]")
    except KeyError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("Use 'iscc-search index list' to see available indexes")
        sys.exit(1)


def remove_command(
    name: str,
    delete_data: bool = typer.Option(False, "--delete-data", help="Also delete local index data"),
):
    # type: (...) -> None
    """
    Remove an index from configuration.

    Removes the index configuration. For local indexes, use --delete-data
    to also delete the actual index files.

    Examples:

        # Remove from config only
        iscc-search index remove staging

        # Remove config and delete local index data
        iscc-search index remove old-local --delete-data
    """
    config_manager = get_config_manager()

    try:
        # Get index config before removing (to check type and path)
        indexes = config_manager.list_indexes()
        index_config = None
        for idx_name, cfg, _ in indexes:
            if idx_name == name:
                index_config = cfg
                break

        if index_config is None:
            console.print(f"[red]Error: Index '{name}' not found[/red]")
            sys.exit(1)

        # Remove from config
        config_manager.remove_index(name)
        console.print(f"[green]Removed index '{name}' from configuration[/green]")

        # Delete data if requested and it's a local index
        if delete_data:
            if index_config.type == "local":
                # Construct path to the specific index subdirectory
                base_path = Path(index_config.path)  # type: ignore
                index_path = base_path / name
                if index_path.exists():
                    shutil.rmtree(index_path)
                    console.print(f"[green]Deleted index directory {index_path}[/green]")
                else:
                    console.print(f"[yellow]Warning: Index directory {index_path} not found[/yellow]")
            else:
                console.print("[yellow]Note: --delete-data only applies to local indexes[/yellow]")

    except KeyError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
