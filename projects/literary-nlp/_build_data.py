"""Download Project Gutenberg texts, strip boilerplate, and write pages.csv.

Run from the project directory or the repository root:

    python projects/literary-nlp/_build_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

PROJ = Path(__file__).resolve().parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from gutenberg_utils import BOOKS, pages_from_text, project_data_dir, strip_gutenberg_boilerplate

USER_AGENT = (
    "projects-website-literary-nlp/1.0 "
    "(https://github.com/areeslindley/projects-website; educational portfolio)"
)


def candidate_urls(gid: int) -> list[str]:
    return [
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-8.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
        f"https://www.gutenberg.org/ebooks/{gid}.txt.utf-8",
    ]


def download_book(gid: int, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size > 1000:
        return dest.read_text(encoding="utf-8", errors="replace")

    headers = {"User-Agent": USER_AGENT}
    last_error = None
    for url in candidate_urls(gid):
        try:
            resp = requests.get(url, headers=headers, timeout=60)
            if resp.status_code != 200 or len(resp.content) < 1000:
                last_error = f"{url} -> {resp.status_code}"
                continue
            text = None
            for enc in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    text = resp.content.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if not text:
                last_error = f"{url} could not decode"
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
            print(f"  downloaded {gid} from {url}")
            return text
        except requests.RequestException as exc:
            last_error = f"{url} -> {exc}"
    raise RuntimeError(f"Could not download Gutenberg {gid}: {last_error}")


def main() -> None:
    data_dir = project_data_dir()
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for book in BOOKS:
        gid = book["gutenberg_id"]
        print(f"Processing {book['title']} (PG {gid})...")
        raw_path = raw_dir / f"{book['book_id']}.txt"
        raw = download_book(gid, raw_path)
        cleaned = strip_gutenberg_boilerplate(raw)
        pages = pages_from_text(book, cleaned)
        n_chapters = pages["chapter_idx"].nunique() if not pages.empty else 0
        n_words = int(pages["n_words"].sum()) if not pages.empty else 0
        print(f"  {len(pages)} pages, {n_chapters} chapters, {n_words:,} words")
        frames.append(pages)

    out = pd.concat(frames, ignore_index=True)
    out_path = data_dir / "pages.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} pages, {out['n_words'].sum():,} words)")


if __name__ == "__main__":
    main()
