# AGENTS.md

## Cursor Cloud specific instructions

This is a **Jupyter Book static site** (Python-based). There are no backend services, databases, or Docker containers.

### Key commands

| Task | Command |
|------|---------|
| Install deps | `pip install -r requirements.txt` |
| Build site | `jupyter-book build . --all` |
| Preview site | `cd _build/html && python3 -m http.server 8080` |

### Non-obvious caveats

- `jupyter-book` and related scripts install to `~/.local/bin`. You must ensure this directory is on `PATH` (e.g. `export PATH="$HOME/.local/bin:$PATH"`) before running `jupyter-book` commands.
- The `requirements.txt` warns against installing `myst-cli` or `mystmd` as they conflict with the classic `jupyter-book` CLI. If the build produces `EISDIR` errors, run `pip uninstall -y myst-cli mystmd` and reinstall jupyter-book.
- The build config (`_config.yml`) uses `execute_notebooks: auto`, which means notebooks are executed during build. Some notebooks (e.g. the geospatial voronoi analysis) may fail execution if network-dependent tile servers are unavailable — the build still succeeds and produces HTML with warnings.
- There is no linter, test suite, or pre-commit hooks configured in this repository. Validation consists of a successful `jupyter-book build`.
- Built output goes to `_build/html/`. This directory is gitignored.
