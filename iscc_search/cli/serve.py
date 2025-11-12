"""
Serve command for ISCC-Search CLI.

Handles starting the ISCC-Search REST API server.
"""

import typer

from iscc_search.cli.common import console

__all__ = ["serve_command"]


def serve_command(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind server to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind server to"),
    dev: bool = typer.Option(False, "--dev", "-d", help="Run in development mode with auto-reload"),
    workers: int = typer.Option(None, "--workers", "-w", help="Number of worker processes (production only)"),
):
    # type: (...) -> None
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
