"""Conformal intervals, ALE, and grouped permutation importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .validate import duan_from_log1p, feature_frame, ols_pipeline, target_log


def split_conformal(df: pd.DataFrame, seed: int, levels: tuple[float, ...] = (0.5, 0.8, 0.9)) -> dict:
    """Residual split conformal on a grouped artist split, evaluated on held-out artists."""
    groups = df["artist_name"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=seed + 1)
    fit_idx, calib_idx = next(inner.split(train, groups=train["artist_name"]))
    fit = train.iloc[fit_idx]
    calib = train.iloc[calib_idx]

    y_fit = target_log(fit)
    pipe = ols_pipeline("honest")
    pipe.fit(feature_frame(fit, "honest"), y_fit)
    calib_pred = pipe.predict(feature_frame(calib, "honest"))
    scores = np.abs(target_log(calib) - calib_pred)
    test_pred = pipe.predict(feature_frame(test, "honest"))
    y_test = target_log(test)
    n_cal = scores.size
    coverage = {}
    for level in levels:
        q = float(np.quantile(scores, np.ceil((n_cal + 1) * level) / n_cal, method="higher"))
        lo = test_pred - q
        hi = test_pred + q
        inside = (y_test >= lo) & (y_test <= hi)
        coverage[str(level)] = {
            "nominal": level,
            "empirical": float(inside.mean()),
            "interval_halfwidth": q,
            "median_width": float(np.median(hi - lo)),
        }
    # coverage curve across many nominal levels
    grid = np.linspace(0.05, 0.95, 19)
    curve = []
    for level in grid:
        q = float(np.quantile(scores, min(1.0, np.ceil((n_cal + 1) * level) / n_cal), method="higher"))
        inside = np.mean(np.abs(y_test - test_pred) <= q)
        curve.append({"nominal": float(level), "empirical": float(inside)})
    resid = y_fit - pipe.predict(feature_frame(fit, "honest"))
    return {
        "n_fit_artists": int(fit["artist_name"].nunique()),
        "n_calib_artists": int(calib["artist_name"].nunique()),
        "n_test_artists": int(test["artist_name"].nunique()),
        "coverage": coverage,
        "curve": curve,
        "test_rmse_log": float(np.sqrt(np.mean((y_test - test_pred) ** 2))),
        "duan_mae_count": float(
            np.mean(np.abs(test["stream_count"].to_numpy() - duan_from_log1p(test_pred, resid)))
        ),
    }


def _ale_1d(model, X: pd.DataFrame, column: str, n_grid: int = 20) -> dict:
    """Accumulated local effects for one numeric column, centred to mean zero."""
    values = np.sort(X[column].to_numpy(dtype=float))
    bins = np.unique(np.quantile(values, np.linspace(0, 1, n_grid + 1)))
    if bins.size < 3:
        return {"x": [], "ale": []}
    effects = []
    centres = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (X[column] >= low) & (X[column] <= high if high == bins[-1] else X[column] < high)
        if mask.sum() == 0:
            continue
        left = X.loc[mask].copy()
        right = X.loc[mask].copy()
        left[column] = low
        right[column] = high
        delta = model.predict(right) - model.predict(left)
        effects.append(float(delta.mean()))
        centres.append(float(0.5 * (low + high)))
    ale = np.cumsum(effects)
    ale = ale - ale.mean()
    return {"x": centres, "ale": ale.tolist(), "column": column}


def ale_audio(df: pd.DataFrame, seed: int) -> dict:
    y = target_log(df)
    X = feature_frame(df, "honest")
    pipe = ols_pipeline("honest")
    pipe.fit(X, y)
    out = {}
    for column in ("energy", "loudness", "danceability", "tempo"):
        out[column] = _ale_1d(pipe, X, column)
        out[column]["unit"] = {
            "energy": "unit interval",
            "loudness": "dB",
            "danceability": "unit interval",
            "tempo": "BPM",
        }[column]
    return out


def grouped_permutation_importance(df: pd.DataFrame, seed: int, n_repeats: int = 8) -> dict:
    """Permute feature blocks on grouped held-out artists; ΔRMSE on log1p scale."""
    groups = df["artist_name"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    y_tr, y_te = target_log(train), target_log(test)
    X_tr = feature_frame(train, "honest")
    X_te = feature_frame(test, "honest")
    pipe = ols_pipeline("honest")
    pipe.fit(X_tr, y_tr)
    baseline = float(np.sqrt(np.mean((y_te - pipe.predict(X_te)) ** 2)))
    blocks = {
        "audio": ["danceability", "energy", "loudness", "instrumentalness", "tempo", "key", "mode"],
        "duration_explicit": ["duration_ms", "explicit"],
        "calendar": ["release_month"],
        "genre": ["genre"],
        "country": ["country"],
        "label": ["label"],
    }
    rng = np.random.default_rng(seed)
    out = {}
    for name, cols in blocks.items():
        deltas = []
        for _ in range(n_repeats):
            shuffled = X_te.copy()
            perm = rng.permutation(len(shuffled))
            block = shuffled[cols].iloc[perm]
            block.index = shuffled.index
            shuffled[cols] = block
            rmse = float(np.sqrt(np.mean((y_te - pipe.predict(shuffled)) ** 2)))
            deltas.append(rmse - baseline)
        out[name] = {
            "mean_delta_rmse": float(np.mean(deltas)),
            "p10": float(np.quantile(deltas, 0.1)),
            "p90": float(np.quantile(deltas, 0.9)),
        }
    out["baseline_rmse"] = baseline
    return out
