"""Entry point for running ISCC-Search server as a module."""

import uvicorn

from iscc_search.options import search_opts


def main():
    # type: () -> None
    """
    Start the ISCC-Search API server.

    Runs uvicorn server with hot reload enabled in development mode.
    """
    uvicorn.run(
        "iscc_search.server:app",
        host=search_opts.host,
        port=search_opts.port,
        reload=True,
        log_level=search_opts.log_level,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
