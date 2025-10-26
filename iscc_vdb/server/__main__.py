"""Entry point for running ISCC-VDB server as a module."""

import uvicorn


def main():
    # type: () -> None
    """
    Start the ISCC-VDB API server.

    Runs uvicorn server with hot reload enabled in development mode.
    """
    uvicorn.run(
        "iscc_vdb.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
