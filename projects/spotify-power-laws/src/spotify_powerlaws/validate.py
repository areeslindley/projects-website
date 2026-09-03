"""Grouped vs random CV, learning curves, and the Nadeau–Bengio corrected t-test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .estimand import HONEST_CATEGORICAL, HONEST_NUMERIC, LEAKAGE_CATEGORICAL, LEAKAGE_NUMERIC

N_FOLDS = 5


@dataclass
class FoldScores:
    name: str
    split: str
    rmse_log: np.ndarray
    mae_log: np.ndarray
    r2_log: np.ndarray
    rmse_count: np.ndarray
    mae_count: np.ndarray


def target_log(df: pd.DataFrame) -> np.ndarray:
    return np.log1p(df["stream_count"].to_numpy(dtype=float))


def duan_from_log1p(pred_log: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """E[Y | x] for a log1p model: exp(pred) * mean(exp(e)) - 1."""
    return np.exp(pred_log) * float(np.mean(np.exp(residuals))) - 1.0


def naive_expm1(pred_log: np.ndarray) -> np.ndarray:
    return np.expm1(pred_log)


def _encoder() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def make_preprocessor(feature_set: str) -> ColumnTransformer:
    if feature_set == "honest":
        numeric = HONEST_NUMERIC
        categorical = HONEST_CATEGORICAL
    elif feature_set == "leakage":
        numeric = HONEST_NUMERIC + LEAKAGE_NUMERIC
        categorical = HONEST_CATEGORICAL + LEAKAGE_CATEGORICAL
    else:
        raise ValueError(feature_set)
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", _encoder(), categorical),
        ]
    )


def feature_frame(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set == "honest":
        cols = HONEST_NUMERIC + HONEST_CATEGORICAL
    elif feature_set == "leakage":
        cols = HONEST_NUMERIC + LEAKAGE_NUMERIC + HONEST_CATEGORICAL + LEAKAGE_CATEGORICAL
    else:
        raise ValueError(feature_set)
    out = df[cols].copy()
    out["explicit"] = out["explicit"].astype(int)
    return out


def intercept_predict(y_train: np.ndarray, n_test: int) -> np.ndarray:
    return np.full(n_test, y_train.mean())


def ols_pipeline(feature_set: str) -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor(feature_set)),
            ("model", LinearRegression()),
        ]
    )


def ridge_pipeline(feature_set: str) -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor(feature_set)),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 16))),
        ]
    )


def _metrics(y_log: np.ndarray, pred_log: np.ndarray, y_count: np.ndarray, pred_count: np.ndarray) -> dict:
    rmse_log = float(np.sqrt(np.mean((y_log - pred_log) ** 2)))
    mae_log = float(np.mean(np.abs(y_log - pred_log)))
    ss_res = float(np.sum((y_log - pred_log) ** 2))
    ss_tot = float(np.sum((y_log - y_log.mean()) ** 2))
    r2_log = float(1.0 - ss_res / ss_tot) if ss_tot else 0.0
    rmse_count = float(np.sqrt(np.mean((y_count - pred_count) ** 2)))
    mae_count = float(np.mean(np.abs(y_count - pred_count)))
    return {
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "r2_log": r2_log,
        "rmse_count": rmse_count,
        "mae_count": mae_count,
    }


def _splitters(groups: np.ndarray, seed: int):
    return {
        "grouped": GroupKFold(n_splits=N_FOLDS),
        "random": KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed),
    }


def oof_predictions(
    df: pd.DataFrame,
    feature_set: str,
    split: str,
    seed: int,
    model: str = "ols",
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    y_log = target_log(df)
    y_count = df["stream_count"].to_numpy(dtype=float)
    groups = df["artist_name"].to_numpy()
    X = feature_frame(df, feature_set)
    splitter = _splitters(groups, seed)[split]
    pred_log = np.empty_like(y_log)
    fold_rows = []
    split_iter = (
        splitter.split(X, y_log, groups)
        if split == "grouped"
        else splitter.split(X, y_log)
    )
    for fold, (train_idx, test_idx) in enumerate(split_iter):
        y_tr, y_te = y_log[train_idx], y_log[test_idx]
        if model == "intercept":
            pred = intercept_predict(y_tr, len(test_idx))
            resid = y_tr - y_tr.mean()
        else:
            pipe = ols_pipeline(feature_set) if model == "ols" else ridge_pipeline(feature_set)
            pipe.fit(X.iloc[train_idx], y_tr)
            pred = pipe.predict(X.iloc[test_idx])
            resid = y_tr - pipe.predict(X.iloc[train_idx])
        pred_log[test_idx] = pred
        pred_count = duan_from_log1p(pred, resid)
        metrics = _metrics(y_te, pred, y_count[test_idx], pred_count)
        metrics.update({"fold": fold, "n_test": int(len(test_idx))})
        fold_rows.append(metrics)
    return pred_log, y_log, fold_rows


def nadeau_bengio_t(a: np.ndarray, b: np.ndarray, n_folds: int = N_FOLDS) -> dict:
    """Corrected resampled t-test on paired fold scores (lower is better for RMSE)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    k = len(d)
    n_test_over_train = 1.0 / (n_folds - 1)
    var = float(np.var(d, ddof=1))
    se = np.sqrt((1.0 / k + n_test_over_train) * var)
    t_stat = float(d.mean() / se) if se else 0.0
    from scipy.stats import t as student_t

    df = k - 1
    p_value = float(2 * student_t.sf(abs(t_stat), df))
    return {
        "mean_diff": float(d.mean()),
        "t": t_stat,
        "df": df,
        "p_value": p_value,
        "correction": "Nadeau-Bengio 2003; n_test/n_train = 1/(k-1)",
    }


def learning_curve_artist_subsamples(
    df: pd.DataFrame,
    seed: int,
    artist_grid: tuple[int, ...] = (25, 50, 100, 200, 350, 450),
    n_boot: int = 12,
) -> dict:
    """RMSE on a fixed held-out artist panel as the training artist count grows."""
    rng = np.random.default_rng(seed)
    artists = df["artist_name"].unique()
    held_out = rng.choice(artists, size=80, replace=False)
    train_pool = np.setdiff1d(artists, held_out)
    test = df[df["artist_name"].isin(held_out)]
    y_test = target_log(test)
    X_test = feature_frame(test, "honest")
    rows = []
    for n_art in artist_grid:
        rmses = []
        for _ in range(n_boot):
            chosen = rng.choice(train_pool, size=min(n_art, len(train_pool)), replace=False)
            train = df[df["artist_name"].isin(chosen)]
            y_tr = target_log(train)
            pipe = ols_pipeline("honest")
            pipe.fit(feature_frame(train, "honest"), y_tr)
            pred = pipe.predict(X_test)
            rmses.append(float(np.sqrt(np.mean((y_test - pred) ** 2))))
        rmses = np.asarray(rmses)
        intercept = float(np.sqrt(np.mean((y_test - y_test.mean()) ** 2)))
        rows.append(
            {
                "n_artists": int(n_art),
                "rmse_mean": float(rmses.mean()),
                "rmse_lo": float(np.quantile(rmses, 0.1)),
                "rmse_hi": float(np.quantile(rmses, 0.9)),
                "intercept_rmse": intercept,
            }
        )
    return {"held_out_artists": 80, "n_boot": n_boot, "points": rows}
