"""Download BreastMNIST 128×128 into projects/cancer-imaging/data/."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent
DATA = PROJ / "data"
DEST = DATA / "breastmnist_128.npz"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Already present: {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
        return

    try:
        from medmnist import BreastMNIST
    except ImportError as exc:
        sys.exit(
            "medmnist is required. Install with: pip install medmnist\n"
            f"Original error: {exc}"
        )

    print("Downloading BreastMNIST (128×128) via the MedMNIST API…")
    BreastMNIST(split="train", download=True, size=128, root=str(DATA))
    # The API writes breastmnist_128.npz (or similar) into root.
    candidates = sorted(DATA.glob("breastmnist*128*.npz")) + sorted(
        DATA.glob("breastmnist.npz")
    )
    if not candidates:
        sys.exit(f"Download finished but no NPZ found in {DATA}")
    src = candidates[0]
    if src.resolve() != DEST.resolve():
        shutil.copy2(src, DEST)
        print(f"Copied {src.name} → {DEST.name}")
    print(f"Wrote {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
