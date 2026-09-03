"""Empirical schema: grain, missingness, cardinality, derived-column identities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import EXPECTED_ARTISTS, EXPECTED_COLS, EXPECTED_ROWS

UNITS = {
    "track_id": "UUID",
    "track_name": "string",
    "artist_name": "string (invented)",
    "album_name": "string",
    "release_date": "calendar date (YYYY-MM-DD)",
    "genre": "nominal (20 labels)",
    "duration_ms": "milliseconds",
    "popularity": "integer index, claimed 0–100",
    "danceability": "unit interval",
    "energy": "unit interval",
    "key": "pitch class 0–11",
    "loudness": "dB (max 0)",
    "mode": "1 = major, 0 = minor",
    "instrumentalness": "unit interval",
    "tempo": "BPM",
    "stream_count": "count; observation window unspecified",
    "country": "ISO-2",
    "explicit": "boolean",
    "label": "string",
    "release_year": "year; deterministic from release_date",
    "release_month": "month 1–12; deterministic from release_date",
    "release_day_of_week": "English weekday; deterministic from release_date",
    "duration_minutes": "minutes = round(duration_ms / 60000, 2)",
    "popularity_category": "Low/Medium/High/Very High bins of popularity",
    "loudness_category": "Quiet/Moderate/Loud bins of loudness",
    "key_name": "letter name of key",
    "mode_name": "Major/Minor",
    "is_explicit_bool": "duplicate of explicit",
    "release_quarter": "Q1–Q4 from month",
    "is_weekend_release": "Saturday/Sunday flag",
    "log_stream_count": "round(log1p(stream_count), 4)",
    "upbeat_score": "linear composite of danceability, energy, tempo",
    "artist_track_count": "full-sample group size of artist_name",
}

ROLE = {
    "track_id": "primary key",
    "track_name": "label",
    "artist_name": "clustering unit",
    "album_name": "label",
    "release_date": "release calendar (not a listening panel)",
    "genre": "covariate; not a partition of artists",
    "duration_ms": "antecedent",
    "popularity": "consequent of streams",
    "danceability": "antecedent (audio)",
    "energy": "antecedent (audio)",
    "key": "antecedent",
    "loudness": "antecedent (audio)",
    "mode": "antecedent",
    "instrumentalness": "antecedent (audio)",
    "tempo": "antecedent (audio)",
    "stream_count": "candidate target",
    "country": "covariate",
    "explicit": "antecedent",
    "label": "covariate",
    "release_year": "derived",
    "release_month": "derived",
    "release_day_of_week": "derived",
    "duration_minutes": "derived",
    "popularity_category": "derived / consequent",
    "loudness_category": "derived",
    "key_name": "derived",
    "mode_name": "derived",
    "is_explicit_bool": "derived",
    "release_quarter": "derived",
    "is_weekend_release": "derived",
    "log_stream_count": "deterministic transform of target",
    "upbeat_score": "composite of audio",
    "artist_track_count": "whole-sample statistic",
}


def schema_report(df: pd.DataFrame) -> dict:
    nunique = df.nunique(dropna=False)
    missing = df.isna().sum()
    columns = []
    for name in df.columns:
        series = df[name]
        columns.append(
            {
                "name": name,
                "dtype": str(series.dtype),
                "missing": int(missing[name]),
                "cardinality": int(nunique[name]),
                "units": UNITS.get(name, ""),
                "role": ROLE.get(name, ""),
            }
        )

    artist_n = df.groupby("artist_name").size()
    date_min = pd.Timestamp(df["release_date"].min())
    date_max = pd.Timestamp(df["release_date"].max())
    calendar_days = int((date_max - date_min).days) + 1
    grain = "track"
    if df["track_id"].nunique() != len(df):
        grain = "unknown (track_id not unique)"

    log1p = np.log1p(df["stream_count"].to_numpy(dtype=float))
    log_err = float(np.max(np.abs(df["log_stream_count"].to_numpy() - np.round(log1p, 4))))
    dur_err = float(
        np.max(
            np.abs(
                df["duration_minutes"].to_numpy()
                - np.round(df["duration_ms"].to_numpy() / 60_000, 2)
            )
        )
    )

    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_artists": int(df["artist_name"].nunique()),
        "expected_rows": EXPECTED_ROWS,
        "expected_cols": EXPECTED_COLS,
        "expected_artists": EXPECTED_ARTISTS,
        "grain": grain,
        "grain_note": (
            "One row is one track. release_date is a track-release calendar, "
            "not artist-year streaming. This is not a panel of listening."
        ),
        "date_min": str(date_min.date()),
        "date_max": str(date_max.date()),
        "n_unique_dates": int(df["release_date"].nunique()),
        "calendar_days": calendar_days,
        "complete_calendar": bool(df["release_date"].nunique() == calendar_days),
        "artist_tracks_min": int(artist_n.min()),
        "artist_tracks_median": float(artist_n.median()),
        "artist_tracks_mean": float(artist_n.mean()),
        "artist_tracks_max": int(artist_n.max()),
        "top_artist": str(artist_n.idxmax()),
        "top_artist_share": float(artist_n.max() / len(df)),
        "top10_row_share": float(artist_n.sort_values(ascending=False).head(10).sum() / len(df)),
        "log_stream_is_round_log1p": log_err < 1e-8,
        "duration_minutes_is_round_ms": dur_err < 1e-12,
        "explicit_equals_bool_flag": bool((df["explicit"] == df["is_explicit_bool"]).all()),
        "artist_track_count_is_group_size": bool(
            (
                df.groupby("artist_name")["artist_track_count"].first()
                == artist_n
            ).all()
        ),
        "missing_audio": [
            name
            for name in (
                "acousticness",
                "valence",
                "speechiness",
                "liveness",
                "time_signature",
            )
            if name not in df.columns
        ],
        "missing_success_fields": [
            name
            for name in (
                "followers",
                "monthly_listeners",
                "chart_peak",
                "chart_position",
            )
            if name not in df.columns
        ],
        "columns": columns,
        "year_counts": {int(k): int(v) for k, v in df["release_year"].value_counts().sort_index().items()},
        "genre_counts": {str(k): int(v) for k, v in df["genre"].value_counts().items()},
    }
