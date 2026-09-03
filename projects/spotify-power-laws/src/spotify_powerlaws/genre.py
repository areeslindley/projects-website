"""Recover planted genre from audio. A different estimand from stream prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .validate import N_FOLDS, nadeau_bengio_t

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "instrumentalness",
    "tempo",
    "key",
    "mode",
    "duration_ms",
    "explicit",
]

GENRE_ESTIMAND = {
    "target": "genre",
    "n_classes": 20,
    "working_scale": "nominal labels",
    "features": AUDIO_FEATURES,
    "question": (
        "Among tracks in this synthetic catalog, how well do antecedent audio "
        "features predict the planted genre label, with artist as the unit of independence?"
    ),
    "question_type": "explanatory of the generator (planted genre-specific audio), with a predictive rider on held-out artists",
    "unit_of_independence": "artist_name",
    "excluded": [
        "upbeat_score (composite of audio)",
        "stream_count, log_stream_count, popularity, popularity_category",
        "label, country, genre as a feature",
    ],
    "not_identified": [
        "real-world genre taxonomy",
        "recommendation quality",
        "anything about Spotify listening",
        "stream prediction (page 1 already closed that)",
    ],
    "metrics": ["balanced_accuracy", "macro_f1", "log_loss"],
}


def audio_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[AUDIO_FEATURES].copy()
    out["explicit"] = out["explicit"].astype(int)
    return out


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_proba, labels=labels)),
    }


def _align_proba(model, X, labels: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    out = np.zeros((len(X), len(labels)))
    index = {c: i for i, c in enumerate(labels)}
    for j, cls in enumerate(model.classes_):
        out[:, index[cls]] = raw[:, j]
    return out


def _make_model(name: str, seed: int, fold: int = 0):
    if name == "dummy":
        return DummyClassifier(strategy="prior", random_state=seed)
    if name == "lda":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LinearDiscriminantAnalysis(solver="svd")),
            ]
        )
    if name == "logit":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=500,
                        C=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.08,
            max_iter=200,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=15,
            random_state=seed + fold,
        )
    raise ValueError(name)


def oof_classify(
    df: pd.DataFrame,
    model_name: str,
    split: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    X = audio_frame(df)
    y = df["genre"].to_numpy()
    groups = df["artist_name"].to_numpy()
    labels = np.sort(pd.unique(y))
    if split == "grouped":
        splitter = GroupKFold(n_splits=N_FOLDS)
        splits = splitter.split(X, y, groups)
    else:
        splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        splits = splitter.split(X, y)
    pred = np.empty(len(y), dtype=object)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        model = _make_model(model_name, seed, fold)
        model.fit(X.iloc[train_idx], y[train_idx])
        y_hat = model.predict(X.iloc[test_idx])
        proba = _align_proba(model, X.iloc[test_idx], labels)
        pred[test_idx] = y_hat
        metrics = _metrics(y[test_idx], y_hat, proba, labels)
        metrics.update({"fold": fold, "n_test": int(len(test_idx))})
        fold_rows.append(metrics)
    return pred, y, fold_rows


def _summarise(rows: list[dict]) -> dict:
    frame = pd.DataFrame(rows)
    out = {}
    for col in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss"):
        out[col] = {
            "mean": float(frame[col].mean()),
            "std": float(frame[col].std(ddof=1)),
            "folds": frame[col].tolist(),
        }
    return out


def confusion_from_oof(y: np.ndarray, pred: np.ndarray, class_order: list[str]) -> dict:
    index = {c: i for i, c in enumerate(class_order)}
    k = len(class_order)
    counts = np.zeros((k, k), dtype=int)
    for true, hat in zip(y, pred):
        counts[index[true], index[hat]] += 1
    row_sum = counts.sum(axis=1, keepdims=True)
    recall = np.divide(counts, row_sum, out=np.zeros_like(counts, dtype=float), where=row_sum > 0)
    return {
        "labels": class_order,
        "counts": counts.tolist(),
        "recall": recall.tolist(),
    }


def lda_coordinates(df: pd.DataFrame, seed: int, n_scatter: int = 2500) -> dict:
    """Full-sample LD1–LD2 for a descriptive plot of planted audio clusters."""
    X = audio_frame(df)
    y = df["genre"].to_numpy()
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LinearDiscriminantAnalysis(n_components=2, solver="svd")),
        ]
    )
    coords = pipe.fit_transform(X, y)
    frame = pd.DataFrame({"ld1": coords[:, 0], "ld2": coords[:, 1], "genre": y})
    means = frame.groupby("genre")[["ld1", "ld2"]].mean()
    means["n"] = df["genre"].value_counts()
    means = means.reset_index()
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(frame), size=min(n_scatter, len(frame)), replace=False)
    energy_loud = df.groupby("genre")[["energy", "loudness"]].mean()
    energy_loud["n"] = df["genre"].value_counts()
    energy_loud = energy_loud.reset_index()
    return {
        "means": means.to_dict(orient="records"),
        "energy_loudness_means": energy_loud.to_dict(orient="records"),
        "scatter": {
            "ld1": frame.iloc[sample_idx]["ld1"].tolist(),
            "ld2": frame.iloc[sample_idx]["ld2"].tolist(),
            "genre": frame.iloc[sample_idx]["genre"].tolist(),
        },
        "explained_variance_ratio": pipe.named_steps["model"].explained_variance_ratio_.tolist(),
    }


def per_class_recall(y: np.ndarray, pred: np.ndarray, class_order: list[str]) -> dict:
    conf = confusion_from_oof(y, pred, class_order)
    recalls = {
        label: float(conf["recall"][i][i]) for i, label in enumerate(class_order)
    }
    counts = {label: int((y == label).sum()) for label in class_order}
    rarest = sorted(class_order, key=lambda g: counts[g])[:5]
    return {
        "recall": recalls,
        "n": counts,
        "rarest_five": {g: {"n": counts[g], "recall": recalls[g]} for g in rarest},
    }


def genre_ladder(df: pd.DataFrame, seed: int) -> dict:
    labels_by_freq = df["genre"].value_counts().index.tolist()
    models = ("dummy", "lda", "logit", "hgb")
    packed = {}
    oof_store = {}
    for name in models:
        for split in ("grouped", "random"):
            pred, y, rows = oof_classify(df, name, split, seed)
            key = f"{name}_{split}"
            packed[key] = {"summary": _summarise(rows), "folds": rows}
            if name == "lda" and split == "grouped":
                oof_store["y"] = y
                oof_store["pred"] = pred

    def ba(key: str) -> np.ndarray:
        return np.array(packed[key]["summary"]["balanced_accuracy"]["folds"])

    # RMSE-style tests: negate balanced accuracy so positive mean_diff means first is worse
    tests = {
        "lda_vs_dummy_grouped": nadeau_bengio_t(-ba("lda_grouped"), -ba("dummy_grouped")),
        "logit_vs_lda_grouped": nadeau_bengio_t(-ba("logit_grouped"), -ba("lda_grouped")),
        "hgb_vs_lda_grouped": nadeau_bengio_t(-ba("hgb_grouped"), -ba("lda_grouped")),
        "hgb_vs_logit_grouped": nadeau_bengio_t(-ba("hgb_grouped"), -ba("logit_grouped")),
        "lda_random_vs_grouped": nadeau_bengio_t(-ba("lda_random"), -ba("lda_grouped")),
    }
    y = np.asarray(oof_store["y"])
    pred = np.asarray(oof_store["pred"])
    return {
        "estimand": GENRE_ESTIMAND,
        "class_counts": df["genre"].value_counts().to_dict(),
        "majority_share": float(df["genre"].value_counts().iloc[0] / len(df)),
        "majority_label": str(df["genre"].value_counts().index[0]),
        "models": packed,
        "tests": tests,
        "confusion_lda_grouped": confusion_from_oof(y, pred, labels_by_freq),
        "per_class_lda_grouped": per_class_recall(y, pred, labels_by_freq),
        "lda_coordinates": lda_coordinates(df, seed),
        "optimism_gap_lda_ba": {
            "grouped_mean": float(ba("lda_grouped").mean()),
            "random_mean": float(ba("lda_random").mean()),
            "relative_lift": float(
                (ba("lda_random").mean() - ba("lda_grouped").mean()) / ba("lda_grouped").mean()
            ),
        },
        "note": (
            "Positive paired-test mean_diff follows the RMSE convention: the first "
            "model is worse. LDA vs dummy should be negative (LDA better)."
        ),
    }
