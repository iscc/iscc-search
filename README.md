# iscc-vdb

> [!WARNING]
> **This project is in early development and not ready for production use.**
>
> The API and features are subject to significant changes. Use at your own risk.

[![Release](https://img.shields.io/github/v/release/iscc/iscc-vdb)](https://img.shields.io/github/v/release/iscc/iscc-vdb)
[![Build status](https://img.shields.io/github/actions/workflow/status/iscc/iscc-vdb/main.yml?branch=main)](https://github.com/iscc/iscc-vdb/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/iscc/iscc-vdb/branch/main/graph/badge.svg)](https://codecov.io/gh/iscc/iscc-vdb)
[![Commit activity](https://img.shields.io/github/commit-activity/m/iscc/iscc-vdb)](https://img.shields.io/github/commit-activity/m/iscc/iscc-vdb)
[![License](https://img.shields.io/github/license/iscc/iscc-vdb)](https://img.shields.io/github/license/iscc/iscc-vdb)

Embedded Vector Database for ISCC

- **Github repository**: <https://github.com/iscc/iscc-vdb/>
- **Documentation** <https://vdb.iscc.codes/>

#### Prerequisites

- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/) package manager

#### Installation

Install the environment and the pre-commit hooks:

```bash
uv run poe install
```

#### Available Commands

All development tasks can be run using `poe`:

```bash
uv run poe --help  # Show all available tasks
uv run poe install  # Install dependencies and pre-commit hooks
uv run poe check    # Run all code quality checks
uv run poe test     # Run tests
uv run poe docs     # Serve documentation locally
uv run poe build    # Build distribution packages
```

The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new
release.

To finalize the set-up for publishing to PyPI, see
[here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi). For activating the
automatic documentation with MkDocs, see
[here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github). To
enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Releasing a new version

- Create an API Token on [PyPI](https://pypi.org/).
- Add the API Token to your projects secrets with the name `PYPI_TOKEN` by visiting
    [this page](https://github.com/iscc/iscc-vdb/settings/secrets/actions/new).
- Create a [new release](https://github.com/iscc/iscc-vdb/releases/new) on Github.
- Create a new tag in the form `*.*.*`.

For more details, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/cicd/#how-to-trigger-a-release).

______________________________________________________________________

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
