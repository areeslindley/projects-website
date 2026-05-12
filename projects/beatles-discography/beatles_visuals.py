#!/usr/bin/env python3
"""Generate interactive Beatles discography visualizations."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import plotly.express as px

WRITER_REFERENCE_URL = (
    "https://raw.githubusercontent.com/inteligentni/Class-05-Feature-engineering/"
    "master/The%20Beatles%20songs%20dataset,%20v1,%20no%20NAs.csv"
)


def normalize_token(value: str) -> str:
    """Normalize text for robust matching."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first matching column across several naming conventions."""
    normalized_to_original = {normalize_token(col): col for col in df.columns}
    for candidate in candidates:
        key = normalize_token(candidate)
        if key in normalized_to_original:
            return normalized_to_original[key]
    return None


def load_dataset(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Dataset not found at {input_csv}. "
            "Download from Kaggle and place as beatles_spotify_dataset.csv."
        )
    return pd.read_csv(input_csv)


def enrich_writers_if_missing(
    df: pd.DataFrame, song_col: str, writer_col: str | None
) -> tuple[pd.DataFrame, str]:
    """Enrich writer data from a public Beatles songwriter reference if needed."""
    if writer_col:
        return df, writer_col

    try:
        ref = pd.read_csv(WRITER_REFERENCE_URL)
    except Exception as exc:  # pragma: no cover - network/data issues are runtime concerns
        raise ValueError(
            "Writer column not found and songwriter enrichment could not be loaded."
        ) from exc

    ref_song_col = resolve_column(ref, ["title", "song", "track", "name"])
    ref_writer_col = resolve_column(ref, ["songwriter", "writer", "writers"])
    if not ref_song_col or not ref_writer_col:
        raise ValueError(
            "Writer column not found in main dataset and enrichment schema is incompatible."
        )

    merged = df.copy()
    merged["_join_song"] = merged[song_col].astype(str).str.lower().str.strip()
    ref = ref.copy()
    ref["_join_song"] = ref[ref_song_col].astype(str).str.lower().str.strip()
    merged = merged.merge(
        ref[["_join_song", ref_writer_col]].drop_duplicates("_join_song"),
        on="_join_song",
        how="left",
    )
    merged.rename(columns={ref_writer_col: "writers_enriched"}, inplace=True)
    merged.drop(columns=["_join_song"], inplace=True)
    return merged, "writers_enriched"


def build_popularity(df: pd.DataFrame, popularity_col: str) -> pd.Series:
    """Create a numeric popularity score that works across schema variants."""
    popularity = pd.to_numeric(df[popularity_col], errors="coerce")
    normalized_name = normalize_token(popularity_col)

    # For ranking fields (e.g. Top.50.Billboard), lower rank means higher popularity.
    if "billboard" in normalized_name or "rank" in normalized_name:
        ranked = popularity.where(popularity > 0)
        return (51 - ranked).clip(lower=0).fillna(0)

    return popularity.fillna(popularity.median()).fillna(0)


def split_writers(value: str) -> list[str]:
    if pd.isna(value):
        return ["Unknown writer"]
    parts = re.split(
        r"\s*(?:,|/|&| and | feat\. | featuring )\s*", str(value), flags=re.IGNORECASE
    )
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return cleaned or ["Unknown writer"]


def create_visuals(df: pd.DataFrame, output_dir: Path) -> None:
    song_col = resolve_column(df, ["song", "title", "track_name", "name"])
    album_col = resolve_column(df, ["album", "album_name", "albumdebut"])
    writer_col = resolve_column(df, ["writers", "writer", "songwriter", "songwriters"])
    popularity_col = resolve_column(
        df,
        [
            "popularity",
            "track_popularity",
            "spotify_popularity",
            "top_50_billboard",
            "top.50.billboard",
            "rank",
        ],
    )

    if not song_col or not album_col:
        raise ValueError("Could not identify song and album columns in dataset.")
    if not popularity_col:
        raise ValueError("Could not identify a popularity column in dataset.")

    working_df, writer_col = enrich_writers_if_missing(df, song_col, writer_col)
    working_df[song_col] = working_df[song_col].fillna("Unknown song")
    working_df[album_col] = working_df[album_col].fillna("Unknown album")
    working_df["popularity_score"] = build_popularity(working_df, popularity_col)

    writer_rows = working_df[[song_col, album_col, writer_col, "popularity_score"]].copy()
    writer_rows["writer_name"] = writer_rows[writer_col].apply(split_writers)
    writer_rows = writer_rows.explode("writer_name")
    writer_rows["writer_name"] = writer_rows["writer_name"].fillna("Unknown writer")

    writer_summary = (
        writer_rows.groupby("writer_name", as_index=False)
        .agg(
            mean_popularity=("popularity_score", "mean"),
            song_count=(song_col, "nunique"),
        )
        .sort_values("mean_popularity", ascending=False)
    )

    fig_writer = px.bar(
        writer_summary.head(20),
        x="writer_name",
        y="mean_popularity",
        color="song_count",
        title="Top writers by average song popularity",
        labels={
            "writer_name": "Writer",
            "mean_popularity": "Average popularity",
            "song_count": "Songs",
        },
    )
    fig_writer.update_layout(xaxis_tickangle=-35, template="plotly_white")
    fig_writer.write_html(
        output_dir / "writer_popularity_by_writer.html",
        include_plotlyjs="cdn",
    )

    fig_album_sunburst = px.sunburst(
        working_df,
        path=[album_col, song_col],
        values="popularity_score",
        color="popularity_score",
        color_continuous_scale="Viridis",
        title="Beatles discography sunburst (Album -> Song)",
    )
    fig_album_sunburst.update_layout(template="plotly_white")
    fig_album_sunburst.write_html(
        output_dir / "sunburst_discography_by_album.html",
        include_plotlyjs="cdn",
    )

    fig_writer_sunburst = px.sunburst(
        writer_rows,
        path=["writer_name", album_col, song_col],
        values="popularity_score",
        color="popularity_score",
        color_continuous_scale="Magma",
        title="Beatles discography sunburst (Writer -> Album -> Song)",
    )
    fig_writer_sunburst.update_layout(template="plotly_white")
    fig_writer_sunburst.write_html(
        output_dir / "sunburst_discography_by_writer.html",
        include_plotlyjs="cdn",
    )


def parse_args() -> argparse.Namespace:
    default_input = Path(__file__).resolve().parent / "data" / "beatles_spotify_dataset.csv"
    default_output = Path(__file__).resolve().parent / "outputs"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to Beatles CSV dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Folder where HTML visualizations will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    df = load_dataset(args.input)
    create_visuals(df, args.output)
    print(f"Visualizations written to: {args.output}")


if __name__ == "__main__":
    main()
