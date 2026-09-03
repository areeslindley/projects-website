"""Run stages and write artifacts/figures. Called by numbered scripts and make."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import SEED
from .distributional import distributional_report
from .forensics import forensics_report
from .io import download_csv, load_raw, sha256_file
from .models import model_ladder
from .paths import artifacts_dir
from .plots import (
    figure_conformal,
    figure_forensics,
    figure_learning_curve,
    figure_leakage,
    figure_lorenz,
    figure_shrinkage,
)
from .schema import schema_report
from .uncertainty import ale_audio, grouped_permutation_importance, split_conformal
from .validate import learning_curve_artist_subsamples
from .estimand import ESTIMAND


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def write_json(name: str, payload: dict) -> Path:
    path = artifacts_dir() / name
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def load_json(name: str) -> dict:
    return json.loads((artifacts_dir() / name).read_text())


def run_download() -> dict:
    path = download_csv()
    payload = {
        "path": "data/raw/spotify_artist_streaming_2020_2025.csv",
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    write_json("download.json", payload)
    return payload


def run_schema() -> dict:
    df = load_raw()
    report = schema_report(df)
    write_json("schema.json", report)
    return report


def run_forensics() -> dict:
    df = load_raw()
    report = forensics_report(df)
    write_json("forensics.json", report)
    return report


def run_estimand() -> dict:
    write_json("estimand.json", ESTIMAND)
    return ESTIMAND


def run_validate() -> dict:
    df = load_raw()
    curve = learning_curve_artist_subsamples(df, SEED)
    write_json("learning_curve.json", curve)
    return curve


def run_models() -> dict:
    df = load_raw()
    report = model_ladder(df, SEED)
    write_json("models.json", report)
    return report


def run_distributional() -> dict:
    df = load_raw()
    report = distributional_report(df, SEED)
    write_json("distributional.json", report)
    return report


def run_uncertainty() -> dict:
    df = load_raw()
    conf = split_conformal(df, SEED)
    ale = ale_audio(df, SEED)
    perm = grouped_permutation_importance(df, SEED)
    payload = {"conformal": conf, "ale": ale, "permutation": perm}
    write_json("uncertainty.json", payload)
    return payload


def run_figures() -> dict:
    schema = load_json("schema.json")
    forensics = load_json("forensics.json")
    models = load_json("models.json")
    curve = load_json("learning_curve.json")
    dist = load_json("distributional.json")
    uncertainty = load_json("uncertainty.json")
    figure_forensics(forensics, schema)
    figure_leakage(models)
    figure_learning_curve(curve)
    figure_lorenz(dist, schema)
    figure_conformal(uncertainty["conformal"])
    figure_shrinkage(models, uncertainty["permutation"])
    paths = {
        "forensics": "figures/01_forensics.png",
        "leakage": "figures/02_leakage_gap.png",
        "learning_curve": "figures/03_learning_curve.png",
        "lorenz": "figures/04_lorenz.png",
        "conformal": "figures/05_conformal.png",
        "shrinkage": "figures/06_shrinkage_importance.png",
    }
    write_json("figures.json", paths)
    return paths


STAGES = {
    "download": run_download,
    "schema": run_schema,
    "forensics": run_forensics,
    "estimand": run_estimand,
    "validate": run_validate,
    "models": run_models,
    "distributional": run_distributional,
    "uncertainty": run_uncertainty,
    "figures": run_figures,
}


def run_all() -> dict:
    out = {}
    for name, fn in STAGES.items():
        out[name] = fn()
    write_json("metrics.json", {
        "seed": SEED,
        "csv": "data/raw/spotify_artist_streaming_2020_2025.csv",
        "schema_n": out["schema"]["n_rows"],
        "verdict": out["forensics"]["verdict"],
        "models_tests": out["models"]["tests"],
        "optimism_gap": out["models"]["optimism_gap_honest_rmse"],
        "leakage_gap": out["models"]["leakage_gap_grouped_r2"],
        "mixedlm": out["models"]["mixedlm"],
        "conformal_80": out["uncertainty"]["conformal"]["coverage"]["0.8"],
        "powerlaw_tracks": out["distributional"]["track_streams"]["powerlaw"],
    })
    return out
