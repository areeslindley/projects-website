"""Shared figure style. No default matplotlib titles; axes carry units."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

INK = "#1a2f36"
TEAL = "#1b6b5a"
AMBER = "#c45c26"
SLATE = "#5c6b73"
MIST = "#e8eef0"
GOLD = "#c9a227"
NAVY = "#0d3b4c"

PALETTE = [TEAL, AMBER, NAVY, GOLD, SLATE]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "medium",
            "axes.titlecolor": INK,
            "axes.labelsize": 10,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "text.color": INK,
            "grid.color": MIST,
            "grid.linewidth": 0.8,
            "font.size": 10,
            "legend.frameon": False,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def save_fig(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
