"""
ISCC-Search CLI.

Command-line interface for managing ISCC indexes and searching for similar content.
"""

import json

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

import iscc_search
from iscc_search.cli.add import add_command
from iscc_search.cli.search import search_command
from iscc_search.cli.common import console, get_default_index

__all__ = ["app", "main"]


app = typer.Typer(
    name="iscc-search",
    help="ISCC similarity search CLI",
    no_args_is_help=True,
)

# Register commands
app.command(name="add")(add_command)
app.command(name="search")(search_command)


@app.command()
def get(
    iscc_id,  # type: str
):
    """
    Get an ISCC asset by ISCC-ID.

    Retrieves the full asset details for a given ISCC-ID from the default index.

    Example:
        iscc-search get ISCC:MAIGIIFJRDGEQQAA
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

    # Get asset
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Retrieving asset...", total=None)
            asset = index.get_asset("default", iscc_id)
            progress.remove_task(task)
    except FileNotFoundError:
        console.print(f"[red]Asset not found: {iscc_id}[/red]")
        index.close()
        raise typer.Exit(code=1)

    # Close index
    index.close()

    # Output as JSON
    output = {
        "iscc_id": asset.iscc_id,
        "iscc_code": asset.iscc_code,
        "units": asset.units,
        "metadata": asset.metadata,
    }

    console.print_json(json.dumps(output))


@app.command()
def version():
    # type: () -> None
    """Show version information."""
    console.print(f"iscc-search version {iscc_search.__version__}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind server to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind server to"),
    dev: bool = typer.Option(False, "--dev", "-d", help="Run in development mode with auto-reload"),
    workers: int = typer.Option(None, "--workers", "-w", help="Number of worker processes (production only)"),
):
    """
    Start the ISCC-Search REST API server.

    By default, runs with production settings (no auto-reload).
    Use --dev flag for development mode with auto-reload enabled.

    Example:
        iscc-search serve                    # Production mode
        iscc-search serve --dev              # Development mode with auto-reload
        iscc-search serve --port 9000        # Custom port
        iscc-search serve --workers 4        # Multi-worker production
    """
    import uvicorn

    if dev and workers:
        console.print("[yellow]Warning: --workers is ignored in development mode[/yellow]")
        workers = None

    # Configure uvicorn based on mode
    uvicorn_config = {
        "app": "iscc_search.server:app",
        "host": host,
        "port": port,
        "log_level": "debug" if dev else "info",
    }

    if dev:
        # Development mode: enable reload, disable workers
        uvicorn_config["reload"] = True
        console.print(f"[green]Starting server in development mode at http://{host}:{port}[/green]")
        console.print("[yellow]Auto-reload enabled - code changes will restart server[/yellow]")
    else:
        # Production mode: no reload, optional workers
        uvicorn_config["reload"] = False
        if workers:
            uvicorn_config["workers"] = workers
            console.print(
                f"[green]Starting server in production mode at http://{host}:{port} with {workers} workers[/green]"
            )
        else:
            console.print(f"[green]Starting server in production mode at http://{host}:{port}[/green]")

    # Start server
    uvicorn.run(**uvicorn_config)


def main():
    # type: () -> None
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
