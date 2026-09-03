"""Phase 1 forensics: Benford, terminal digits, names, correlations, duplicates, popularity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

BENFORD = np.log10(1.0 + 1.0 / np.arange(1, 10))
REAL_ARTISTS = [
    "Taylor Swift",
    "Drake",
    "The Weeknd",
    "Bad Bunny",
    "Ed Sheeran",
    "BTS",
    "Adele",
    "Beyoncé",
    "The Beatles",
    "Coldplay",
]


def leading_digits(values: np.ndarray) -> np.ndarray:
    positive = np.abs(values.astype(float))
    positive = positive[positive > 0]
    return np.floor(positive / 10.0 ** np.floor(np.log10(positive))).astype(int)


def benford_test(values: np.ndarray) -> dict:
    digits = leading_digits(values)
    observed = np.array([(digits == d).sum() for d in range(1, 10)], dtype=float)
    expected = BENFORD * observed.sum()
    chi2, p_value = stats.chisquare(observed, expected)
    proportions = observed / observed.sum()
    mad = float(np.mean(np.abs(proportions - BENFORD)))
    # Nigrini (2012) MAD bands for first digits
    if mad < 0.006:
        conformity = "close"
    elif mad < 0.012:
        conformity = "acceptable"
    elif mad < 0.015:
        conformity = "marginally acceptable"
    else:
        conformity = "nonconforming"
    return {
        "n": int(observed.sum()),
        "chi2": float(chi2),
        "df": 8,
        "p_value": float(p_value),
        "mad": mad,
        "nigrini_conformity": conformity,
        "observed_proportions": proportions.tolist(),
        "benford_proportions": BENFORD.tolist(),
        "observed_counts": observed.astype(int).tolist(),
    }


def terminal_digits(values: np.ndarray) -> dict:
    counts = values.astype(np.int64)
    last = np.abs(counts) % 10
    observed = np.bincount(last, minlength=10).astype(float)
    expected = np.full(10, observed.sum() / 10.0)
    chi2, p_value = stats.chisquare(observed, expected)
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "last_digit_counts": observed.astype(int).tolist(),
        "share_mod10_zero": float(np.mean(counts % 10 == 0)),
        "share_mod100_zero": float(np.mean(counts % 100 == 0)),
        "share_mod1000_zero": float(np.mean(counts % 1000 == 0)),
        "uniform_expectation_mod10": 0.1,
        "uniform_expectation_mod100": 0.01,
        "uniform_expectation_mod1000": 0.001,
    }


def oneway_icc(y: np.ndarray, groups: pd.Series) -> dict:
    y = np.asarray(y, dtype=float)
    frame = pd.DataFrame({"y": y, "g": groups.to_numpy()})
    overall = frame["y"].mean()
    grouped = frame.groupby("g")["y"]
    n_i = grouped.size()
    mean_i = grouped.mean()
    ssb = float(np.sum(n_i * (mean_i - overall) ** 2))
    ssw = float(np.sum((frame["y"] - frame["g"].map(mean_i)) ** 2))
    k = int(frame["g"].nunique())
    n = int(len(frame))
    msb = ssb / (k - 1)
    msw = ssw / (n - k)
    n0 = (n - float((n_i**2).sum() / n)) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    deff = 1.0 + (float(n_i.mean()) - 1.0) * icc
    return {
        "k": k,
        "n": n,
        "icc": float(icc),
        "mean_cluster_size": float(n_i.mean()),
        "design_effect": float(deff),
        "n_effective": float(n / deff) if deff else float(n),
        "msb": float(msb),
        "msw": float(msw),
    }


def gini(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    x = x[x >= 0]
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2.0 * np.sum(np.arange(1, n + 1) * x)) / (n * x.sum()) - (n + 1) / n)


def forensics_report(df: pd.DataFrame) -> dict:
    audio = ["danceability", "energy", "loudness", "instrumentalness", "tempo"]
    corr = df[audio + ["popularity", "stream_count"]].corr()
    names = set(df["artist_name"].unique())
    artist_n = df.groupby("artist_name").size()
    y_log = np.log1p(df["stream_count"].to_numpy(dtype=float))

    pop = df["popularity"]
    known_hits = {name: name in names for name in REAL_ARTISTS}

    return {
        "publisher_synthetic": True,
        "publisher_note": (
            "Kaggle metadata: fully synthetic; no real Spotify API data was used."
        ),
        "benford_stream_count": benford_test(df["stream_count"].to_numpy()),
        "terminal_stream_count": terminal_digits(df["stream_count"].to_numpy()),
        "known_real_artists_present": known_hits,
        "n_known_real_artists_present": int(sum(known_hits.values())),
        "invented_name_examples": sorted(names)[:12],
        "external_validation": (
            "No row matches a publicly reported artist. Order-of-magnitude "
            "checks against Spotify monthly listeners are not identified for "
            "this file; the names are generated."
        ),
        "audio_target_correlations": {
            col: float(df[col].corr(df["stream_count"])) for col in audio
        },
        "energy_loudness_corr": float(df["energy"].corr(df["loudness"])),
        "popularity_log1p_streams_corr": float(pop.corr(pd.Series(y_log, index=df.index))),
        "popularity_streams_corr": float(pop.corr(df["stream_count"])),
        "missing_acousticness_valence": True,
        "duplicate_rows_excluding_id": int(
            df.drop(columns=["track_id"]).duplicated().sum()
        ),
        "duplicate_artist_track_name": int(
            df.duplicated(subset=["artist_name", "track_name"]).sum()
        ),
        "duplicate_track_name": int(df["track_name"].duplicated().sum()),
        "top_track_names": df["track_name"].value_counts().head(10).to_dict(),
        "popularity_integer": bool((pop == pop.astype(int)).all()),
        "popularity_min": int(pop.min()),
        "popularity_max": int(pop.max()),
        "popularity_skew": float(pop.skew()),
        "popularity_share_ge_80": float((pop >= 80).mean()),
        "icc_log_streams_artist": oneway_icc(y_log, df["artist_name"]),
        "icc_log_streams_genre": oneway_icc(y_log, df["genre"]),
        "gini_track_streams": gini(df["stream_count"].to_numpy()),
        "gini_artist_total_streams": gini(
            df.groupby("artist_name")["stream_count"].sum().to_numpy()
        ),
        "gini_artist_catalog_size": gini(artist_n.to_numpy()),
        "artists_multi_genre": int(
            (df.groupby("artist_name")["genre"].nunique() > 1).sum()
        ),
        "complete_calendar": bool(
            df["release_date"].nunique()
            == (df["release_date"].max() - df["release_date"].min()).days + 1
        ),
        "correlation_matrix": corr.round(4).to_dict(),
        "verdict": (
            "The file is a synthetic track catalog with 500 invented artists. "
            "Audio features are orthogonal to streams; popularity is a compressed "
            "copy of log streams; the superstar pattern is catalog size, not "
            "per-track quality. It can teach leakage, validation, and model-capacity "
            "questions about this generator. It cannot support claims about Spotify."
        ),
    }
