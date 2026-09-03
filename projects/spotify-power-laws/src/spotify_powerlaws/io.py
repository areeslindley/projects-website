"""Download and load the Kaggle CSV. Prefer the Kaggle API; fall back to the public zip."""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from . import CSV_NAME, KAGGLE_SLUG
from .paths import raw_csv_path, raw_dir

DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    f"{KAGGLE_SLUG}"
)
USER_AGENT = "spotify-power-laws/0.1 (portfolio; +https://github.com/areeslindley/projects-website)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_kaggle_cli(dest: Path) -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        return False
    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_SLUG, path=str(dest.parent), unzip=True)
        downloaded = dest.parent / CSV_NAME
        if downloaded != dest and downloaded.exists():
            downloaded.replace(dest)
        return dest.exists()
    except Exception:
        return False


def _download_public_zip(dest: Path) -> None:
    request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        members = archive.namelist()
        if CSV_NAME not in members:
            raise FileNotFoundError(f"{CSV_NAME} not in zip; members={members}")
        dest.write_bytes(archive.read(CSV_NAME))


def download_csv(force: bool = False) -> Path:
    dest = raw_csv_path()
    if dest.exists() and not force:
        return dest
    raw_dir()
    if not _download_kaggle_cli(dest):
        _download_public_zip(dest)
    if not dest.exists():
        raise FileNotFoundError(f"Failed to download {CSV_NAME}")
    return dest


def load_raw(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or raw_csv_path()
    if not csv_path.exists():
        csv_path = download_csv()
    df = pd.read_csv(csv_path)
    df["release_date"] = pd.to_datetime(df["release_date"])
    return df
