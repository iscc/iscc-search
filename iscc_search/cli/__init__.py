"""
ISCC-Search CLI.

Command-line interface for managing ISCC indexes and searching for similar content.
"""

import typer

import iscc_search
from iscc_search.cli.add import add_command
from iscc_search.cli.get import get_command
from iscc_search.cli.search import search_command
from iscc_search.cli.serve import serve_command
from iscc_search.cli.common import console

__all__ = ["app", "main"]


app = typer.Typer(
    name="iscc-search",
    help="ISCC similarity search CLI",
    no_args_is_help=True,
)

# Register commands
app.command(name="add")(add_command)
app.command(name="get")(get_command)
app.command(name="search")(search_command)
app.command(name="serve")(serve_command)


@app.command()
def version():
    # type: () -> None
    """Show version information."""
    console.print(f"iscc-search version {iscc_search.__version__}")


def main():
    # type: () -> None
    """CLI entry point."""
    app()
