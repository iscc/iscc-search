"""Tests for the serve CLI command."""

import typer
import uvicorn
import pytest

from iscc_search.cli.serve import serve_command
from iscc_search.options import search_opts


def test_serve_rejects_multi_worker_with_usearch_backend(monkeypatch):
    # type: (pytest.MonkeyPatch) -> None
    """Guard raises typer.Exit when --workers > 1 is combined with usearch:// backend."""
    monkeypatch.setattr(uvicorn, "run", lambda **_kwargs: None)
    monkeypatch.setattr(search_opts, "index_uri", "usearch:///tmp/test-index")

    with pytest.raises(typer.Exit) as exc_info:
        serve_command(host="127.0.0.1", port=8000, dev=False, workers=2)

    assert exc_info.value.exit_code == 1


def test_serve_allows_multi_worker_with_lmdb_backend(monkeypatch):
    # type: (pytest.MonkeyPatch) -> None
    """Guard does not fire for non-usearch backends (LMDB supports concurrent writers)."""
    called = {}

    def fake_run(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(search_opts, "index_uri", "lmdb:///tmp/test-index")

    serve_command(host="127.0.0.1", port=8000, dev=False, workers=2)

    assert called["workers"] == 2


def test_serve_allows_single_worker_with_usearch_backend(monkeypatch):
    # type: (pytest.MonkeyPatch) -> None
    """Guard does not fire for --workers 1 with usearch:// backend."""
    called = {}

    def fake_run(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(search_opts, "index_uri", "usearch:///tmp/test-index")

    serve_command(host="127.0.0.1", port=8000, dev=False, workers=1)

    assert called["app"] == "iscc_search.server:app"


def test_serve_dev_mode_ignores_workers(monkeypatch):
    # type: (pytest.MonkeyPatch) -> None
    """Dev mode clears workers before the guard, so --workers > 1 does not raise."""
    called = {}

    def fake_run(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(search_opts, "index_uri", "usearch:///tmp/test-index")

    serve_command(host="127.0.0.1", port=8000, dev=True, workers=4)

    assert called.get("reload") is True
    assert "workers" not in called
