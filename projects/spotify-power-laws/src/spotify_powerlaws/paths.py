"""Resolve project paths from a notebook, script, or package import."""

from __future__ import annotations

from pathlib import Path


def project_dir() -> Path:
    here = Path(__file__).resolve()
    # src/spotify_powerlaws/paths.py → project root is parents[2]
    pkg_root = here.parents[2]
    if (pkg_root / "src" / "spotify_powerlaws").is_dir():
        return pkg_root
    alt = Path("projects/spotify-power-laws").resolve()
    if (alt / "src" / "spotify_powerlaws").is_dir():
        return alt
    return pkg_root


def data_dir() -> Path:
    path = project_dir() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir() -> Path:
    path = data_dir() / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_csv_path() -> Path:
    return raw_dir() / "spotify_artist_streaming_2020_2025.csv"


def artifacts_dir() -> Path:
    path = project_dir() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir() -> Path:
    path = project_dir() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path
