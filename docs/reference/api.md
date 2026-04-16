---
icon: lucide/code
description: Python API reference for iscc-search index backends and search operations.
---

# API Reference

Auto-generated from source code docstrings.

## Index Protocol

The `IsccIndexProtocol` defines the interface that all index backends implement. CLI, REST API,
and library code all use this protocol regardless of the backend.

::: iscc_search.protocols.index.IsccIndexProtocol
    options:
      heading_level: 3

## Configuration

Server deployment settings managed through environment variables with the `ISCC_SEARCH_` prefix.
See the [Configuration Reference](configuration.md) for the full list of variables.

::: iscc_search.options.SearchOptions
    options:
      heading_level: 3

## Models

Types and convenience classes for handling ISCC codes, units, and items.

::: iscc_search.models
    options:
      heading_level: 3
