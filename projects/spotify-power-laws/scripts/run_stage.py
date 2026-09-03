#!/usr/bin/env python3
"""Thin CLI for pipeline stages. Usage: python scripts/00_download.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spotify_powerlaws.pipeline import STAGES, run_all  # noqa: E402


def main(stage: str | None = None) -> None:
    if stage is None:
        stage = Path(sys.argv[0]).stem.split("_", 1)[-1]
    if stage in {"all", "pipeline"}:
        run_all()
        return
    # 00_download -> download; 01_schema -> schema
    if stage not in STAGES:
        # filenames are 00_download, 01_schema, ...
        mapped = {
            "download": "download",
            "schema": "schema",
            "forensics": "forensics",
            "estimand": "estimand",
            "validate": "validate",
            "models": "models",
            "distributional": "distributional",
            "uncertainty": "uncertainty",
            "figures": "figures",
        }.get(stage)
        if mapped is None:
            raise SystemExit(f"Unknown stage {stage!r}. Choose from {list(STAGES)}")
        stage = mapped
    STAGES[stage]()


if __name__ == "__main__":
    main()
