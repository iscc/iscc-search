#! /usr/bin/env bash
# This script is for devcontainer setup only.
# For local development on any platform, use: uv run poe install

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install --install-hooks
