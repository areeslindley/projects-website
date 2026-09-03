"""Model ladder on the honest feature set, in order of capacity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

from .estimand import HONEST_NUMERIC
from .validate import (
    duan_from_log1p,
    feature_frame,
    make_preprocessor,
    nadeau_bengio_t,
    ols_pipeline,
    oof_predictions,
    target_log,
)


def _summarise_folds(rows: list[dict]) -> dict:
    frame = pd.DataFrame(rows)
    out = {}
    for col in ("rmse_log", "mae_log", "r2_log", "rmse_count", "mae_count"):
        out[col] = {
            "mean": float(frame[col].mean()),
            "std": float(frame[col].std(ddof=1)),
            "folds": frame[col].tolist(),
        }
    return out


def elastic_net_oof(df: pd.DataFrame, seed: int) -> dict:
    y_log = target_log(df)
    y_count = df["stream_count"].to_numpy(dtype=float)
    groups = df["artist_name"].to_numpy()
    X = feature_frame(df, "honest")
    splitter = GroupKFold(n_splits=5)
    rows = []
    path_alphas = None
    path_mse = None
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y_log, groups)):
        pipe = Pipeline(
            [
                ("prep", make_preprocessor("honest")),
                (
                    "model",
                    ElasticNetCV(
                        l1_ratio=[0.2, 0.5, 0.8],
                        alphas=np.logspace(-3, 1, 20),
                        cv=3,
                        max_iter=8000,
                        random_state=seed,
                    ),
                ),
            ]
        )
        pipe.fit(X.iloc[train_idx], y_log[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        resid = y_log[train_idx] - pipe.predict(X.iloc[train_idx])
        pred_count = duan_from_log1p(pred, resid)
        from .validate import _metrics

        metrics = _metrics(y_log[test_idx], pred, y_count[test_idx], pred_count)
        model = pipe.named_steps["model"]
        metrics.update(
            {
                "fold": fold,
                "alpha": float(model.alpha_),
                "l1_ratio": float(model.l1_ratio_),
            }
        )
        rows.append(metrics)
        if fold == 0:
            path_alphas = model.alphas_.tolist()
            mse_path = np.asarray(model.mse_path_)
            try:
                l1_index = list(model.l1_ratio).index(model.l1_ratio_)
                mse = mse_path[l1_index].mean(axis=-1)
            except (ValueError, IndexError):
                mse = mse_path.reshape(mse_path.shape[0], -1).mean(axis=1)
            path_mse = np.asarray(mse, dtype=float).tolist()
    return {
        "summary": _summarise_folds(rows),
        "folds": rows,
        "path_alphas": path_alphas,
        "path_mse": path_mse,
    }


def spline_vs_linear(df: pd.DataFrame, seed: int) -> dict:
    """Grouped OOF RMSE for linear OLS vs cubic splines on the three audio amplitudes."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from .estimand import HONEST_CATEGORICAL

    y_log = target_log(df)
    groups = df["artist_name"].to_numpy()
    X = feature_frame(df, "honest")
    spline_cols = ["danceability", "energy", "loudness"]
    other_num = [c for c in HONEST_NUMERIC if c not in spline_cols]
    linear = ols_pipeline("honest")
    spline = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "spl",
                            SplineTransformer(n_knots=6, degree=3, include_bias=False),
                            spline_cols,
                        ),
                        ("num", StandardScaler(), other_num),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            HONEST_CATEGORICAL,
                        ),
                    ]
                ),
            ),
            ("model", LinearRegression()),
        ]
    )
    splitter = GroupKFold(n_splits=5)
    lin_rmse, spl_rmse = [], []
    for train_idx, test_idx in splitter.split(X, y_log, groups):
        linear.fit(X.iloc[train_idx], y_log[train_idx])
        spline.fit(X.iloc[train_idx], y_log[train_idx])
        lin_rmse.append(
            float(np.sqrt(np.mean((y_log[test_idx] - linear.predict(X.iloc[test_idx])) ** 2)))
        )
        spl_rmse.append(
            float(np.sqrt(np.mean((y_log[test_idx] - spline.predict(X.iloc[test_idx])) ** 2)))
        )
    return {
        "linear_rmse": lin_rmse,
        "spline_rmse": spl_rmse,
        "paired_test": nadeau_bengio_t(np.array(spl_rmse), np.array(lin_rmse)),
        "note": "Positive mean_diff means splines worse than linear (RMSE).",
    }


def boosting_oof(df: pd.DataFrame, seed: int) -> dict:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import OrdinalEncoder

    from .estimand import HONEST_CATEGORICAL, HONEST_NUMERIC
    from .validate import _metrics

    y_log = target_log(df)
    y_count = df["stream_count"].to_numpy(dtype=float)
    groups = df["artist_name"].to_numpy()
    X = feature_frame(df, "honest")
    prep = ColumnTransformer(
        [
            ("num", "passthrough", HONEST_NUMERIC),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                HONEST_CATEGORICAL,
            ),
        ]
    )
    splitter = GroupKFold(n_splits=5)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y_log, groups)):
        X_tr, y_tr = X.iloc[train_idx], y_log[train_idx]
        Xt = prep.fit_transform(X_tr)
        model_es = HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=15,
            random_state=seed + fold,
        )
        model_es.fit(Xt, y_tr)
        pred = model_es.predict(prep.transform(X.iloc[test_idx]))
        resid = y_tr - model_es.predict(Xt)
        metrics = _metrics(
            y_log[test_idx],
            pred,
            y_count[test_idx],
            duan_from_log1p(pred, resid),
        )
        metrics.update(
            {
                "fold": fold,
                "n_iter": int(getattr(model_es, "n_iter_", 0) or 0),
            }
        )
        rows.append(metrics)
    return {"summary": _summarise_folds(rows), "folds": rows}


def mixedlm_artist_intercept(df: pd.DataFrame) -> dict:
    """Partial pooling of artist intercepts on the log1p scale."""
    import statsmodels.formula.api as smf

    work = df.copy()
    work["y"] = np.log1p(work["stream_count"])
    formula = (
        "y ~ danceability + energy + loudness + instrumentalness + tempo "
        "+ duration_ms + C(genre)"
    )
    try:
        model = smf.mixedlm(formula, work, groups=work["artist_name"])
        result = model.fit(method=["lbfgs"], reml=True)
        var_artist = float(result.cov_re.iloc[0, 0])
        var_resid = float(result.scale)
        icc = var_artist / (var_artist + var_resid) if (var_artist + var_resid) else 0.0
        return {
            "converged": bool(result.converged),
            "var_artist": var_artist,
            "sd_artist": float(np.sqrt(max(var_artist, 0.0))),
            "var_resid": var_resid,
            "icc": float(icc),
            "n_groups": int(work["artist_name"].nunique()),
            "llf": float(result.llf),
            "note": (
                "Artist random-intercept scale near zero means partial pooling "
                "collapses toward complete pooling for this generator."
            ),
        }
    except Exception as exc:
        return {
            "converged": False,
            "error": str(exc),
            "var_artist": 0.0,
            "sd_artist": 0.0,
            "var_resid": float(np.var(work["y"])),
            "icc": 0.0,
            "n_groups": int(work["artist_name"].nunique()),
            "note": "MixedLM failed; ANOVA ICC already showed artist effects ≈ 0.",
        }


def model_ladder(df: pd.DataFrame, seed: int) -> dict:
    intercept = oof_predictions(df, "honest", "grouped", seed, model="intercept")
    ols_grouped = oof_predictions(df, "honest", "grouped", seed, model="ols")
    ols_random = oof_predictions(df, "honest", "random", seed, model="ols")
    leak_grouped = oof_predictions(df, "leakage", "grouped", seed, model="ols")
    leak_random = oof_predictions(df, "leakage", "random", seed, model="ols")
    ridge = oof_predictions(df, "honest", "grouped", seed, model="ridge")
    enet = elastic_net_oof(df, seed)
    spline = spline_vs_linear(df, seed)
    boost = boosting_oof(df, seed)
    mixed = mixedlm_artist_intercept(df)

    def pack(rows):
        return {"summary": _summarise_folds(rows), "folds": rows}

    intercept_rmse = np.array([r["rmse_log"] for r in intercept[2]])
    ols_g = np.array([r["rmse_log"] for r in ols_grouped[2]])
    ols_r = np.array([r["rmse_log"] for r in ols_random[2]])
    leak_g = np.array([r["rmse_log"] for r in leak_grouped[2]])
    leak_r = np.array([r["rmse_log"] for r in leak_random[2]])
    boost_g = np.array([r["rmse_log"] for r in boost["folds"]])
    enet_g = np.array([r["rmse_log"] for r in enet["folds"]])

    return {
        "intercept_grouped": pack(intercept[2]),
        "ols_honest_grouped": pack(ols_grouped[2]),
        "ols_honest_random": pack(ols_random[2]),
        "ols_leakage_grouped": pack(leak_grouped[2]),
        "ols_leakage_random": pack(leak_random[2]),
        "ridge_honest_grouped": pack(ridge[2]),
        "elastic_net": enet,
        "spline_vs_linear": spline,
        "boosting": boost,
        "mixedlm": mixed,
        "tests": {
            "ols_vs_intercept_grouped": nadeau_bengio_t(ols_g, intercept_rmse),
            "boost_vs_ols_grouped": nadeau_bengio_t(boost_g, ols_g),
            "enet_vs_ols_grouped": nadeau_bengio_t(enet_g, ols_g),
            "ols_random_vs_grouped": nadeau_bengio_t(ols_r, ols_g),
            "leakage_vs_honest_grouped": nadeau_bengio_t(leak_g, ols_g),
        },
        "optimism_gap_honest_rmse": {
            "grouped_mean": float(ols_g.mean()),
            "random_mean": float(ols_r.mean()),
            "relative_drop": float((ols_g.mean() - ols_r.mean()) / ols_g.mean()),
        },
        "leakage_gap_grouped_r2": {
            "honest": float(np.mean([r["r2_log"] for r in ols_grouped[2]])),
            "leakage": float(np.mean([r["r2_log"] for r in leak_grouped[2]])),
        },
    }
