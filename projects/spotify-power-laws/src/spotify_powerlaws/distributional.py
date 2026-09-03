"""Clauset–Shalizi–Newman power-law tests and Lorenz curves."""

from __future__ import annotations

import numpy as np
from scipy import stats

from .forensics import gini


def _pl_alpha(x: np.ndarray, xmin: float) -> float:
    tail = x[x >= xmin]
    n = tail.size
    return 1.0 + n / float(np.sum(np.log(tail / xmin)))


def _pl_cdf(x: np.ndarray, xmin: float, alpha: float) -> np.ndarray:
    return 1.0 - (xmin / x) ** (alpha - 1.0)


def _ks_distance(x: np.ndarray, xmin: float, alpha: float) -> float:
    tail = np.sort(x[x >= xmin])
    n = tail.size
    if n < 50:
        return np.inf
    empirical = np.arange(1, n + 1) / n
    theoretical = _pl_cdf(tail, xmin, alpha)
    return float(np.max(np.abs(empirical - theoretical)))


def choose_xmin(x: np.ndarray, n_grid: int = 28) -> dict:
    x = np.sort(x[x > 0])
    lo = np.quantile(x, 0.05)
    hi = np.quantile(x, 0.8)
    grid = np.unique(np.geomspace(max(lo, x.min()), hi, n_grid))
    best = None
    for xmin in grid:
        alpha = _pl_alpha(x, xmin)
        ks = _ks_distance(x, xmin, alpha)
        n_tail = int((x >= xmin).sum())
        candidate = {"xmin": float(xmin), "alpha": float(alpha), "ks": ks, "n_tail": n_tail}
        if best is None or ks < best["ks"]:
            best = candidate
    return best or {"xmin": float(x.min()), "alpha": np.nan, "ks": np.inf, "n_tail": 0}


def _bootstrap_gof(x: np.ndarray, xmin: float, alpha: float, n_boot: int, seed: int) -> dict:
    """Semi-parametric Clauset GOF: p = share of synthetic KS >= observed KS."""
    rng = np.random.default_rng(seed)
    tail = x[x >= xmin]
    n = tail.size
    below = x[x < xmin]
    observed_ks = _ks_distance(x, xmin, alpha)
    hits = 0
    for _ in range(n_boot):
        u = rng.random(n)
        synthetic_tail = xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))
        if below.size:
            synthetic_body = rng.choice(below, size=below.size, replace=True)
            synthetic = np.concatenate([synthetic_body, synthetic_tail])
        else:
            synthetic = synthetic_tail
        ks = _ks_distance(synthetic, xmin, alpha)
        if ks >= observed_ks:
            hits += 1
    p_value = (hits + 1) / (n_boot + 1)
    return {"observed_ks": float(observed_ks), "p_value": float(p_value), "n_boot": n_boot}


def _lognormal_ll(x: np.ndarray) -> tuple[float, tuple[float, float]]:
    shape, loc, scale = stats.lognorm.fit(x, floc=0)
    ll = float(np.sum(stats.lognorm.logpdf(x, shape, loc=loc, scale=scale)))
    return ll, (float(shape), float(scale))


def _stretched_exp_ll(x: np.ndarray) -> tuple[float, tuple[float, float]]:
    # Weibull with loc=0 as a stretched-exponential family
    c, loc, scale = stats.weibull_min.fit(x, floc=0)
    ll = float(np.sum(stats.weibull_min.logpdf(x, c, loc=loc, scale=scale)))
    return ll, (float(c), float(scale))


def _pl_ll(x: np.ndarray, xmin: float, alpha: float) -> float:
    n = x.size
    return float(n * np.log(alpha - 1) - n * np.log(xmin) - alpha * np.sum(np.log(x / xmin)))


def vuong(ll_a: np.ndarray, ll_b: np.ndarray) -> dict:
    """Vuong (1989) non-nested likelihood-ratio test on per-observation loglik."""
    d = ll_a - ll_b
    n = d.size
    r = float(d.sum())
    sigma = float(d.std(ddof=1))
    z = r / (sigma * np.sqrt(n)) if sigma else 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    return {"z": float(z), "p_value": p, "mean_log_ratio": float(d.mean())}


def powerlaw_suite(values: np.ndarray, seed: int, n_boot: int = 80) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    fit = choose_xmin(x)
    xmin, alpha = fit["xmin"], fit["alpha"]
    gof = _bootstrap_gof(x, xmin, alpha, n_boot=n_boot, seed=seed)
    tail = x[x >= xmin]
    pl_obs = _pl_ll(tail, xmin, alpha) / tail.size  # mean; vuong uses vectors
    # per-observation loglik for power law (continuous Pareto)
    ll_pl = np.log(alpha - 1) - np.log(xmin) - alpha * np.log(tail / xmin)
    shape, loc, scale = stats.lognorm.fit(tail, floc=0)
    ll_ln = stats.lognorm.logpdf(tail, shape, loc=0, scale=scale)
    c, loc_w, scale_w = stats.weibull_min.fit(tail, floc=0)
    ll_we = stats.weibull_min.logpdf(tail, c, loc=0, scale=scale_w)
    return {
        "xmin": xmin,
        "alpha": alpha,
        "n_tail": int(tail.size),
        "n": int(x.size),
        "ks": fit["ks"],
        "gof": gof,
        "vuong_pl_vs_lognormal": vuong(ll_pl, ll_ln),
        "vuong_pl_vs_stretched_exp": vuong(ll_pl, ll_we),
        "lognormal_params": {"sigma": float(shape), "scale": float(scale)},
        "stretched_exp_params": {"c": float(c), "scale": float(scale_w)},
        "interpretation": (
            "A non-significant Vuong test means lognormal is not distinguishable "
            "from a power law on the fitted tail. That is a result, not a failure."
        ),
    }


def lorenz_curve(values: np.ndarray, n_points: int = 200) -> dict:
    x = np.sort(np.asarray(values, dtype=float))
    x = x[x >= 0]
    shares = np.cumsum(x) / x.sum()
    pop = np.arange(1, x.size + 1) / x.size
    idx = np.linspace(0, x.size - 1, n_points).astype(int)
    return {
        "population": pop[idx].tolist(),
        "share": shares[idx].tolist(),
        "gini": gini(x),
    }


def bootstrap_gini(values: np.ndarray, seed: int, n_boot: int = 400) -> dict:
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    draws = np.array([gini(rng.choice(x, size=x.size, replace=True)) for _ in range(n_boot)])
    point = gini(x)
    return {
        "gini": float(point),
        "ci80": [float(np.quantile(draws, 0.1)), float(np.quantile(draws, 0.9))],
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "n_boot": n_boot,
    }


def distributional_report(df, seed: int) -> dict:
    streams = df["stream_count"].to_numpy(dtype=float)
    artist_totals = df.groupby("artist_name")["stream_count"].sum().to_numpy(dtype=float)
    catalog = df.groupby("artist_name").size().to_numpy(dtype=float)
    year_gini = {}
    for year, sub in df.groupby("release_year"):
        year_gini[int(year)] = bootstrap_gini(sub["stream_count"].to_numpy(), seed + int(year), n_boot=200)
    return {
        "track_streams": {
            "powerlaw": powerlaw_suite(streams, seed),
            "lorenz": lorenz_curve(streams),
            "gini": bootstrap_gini(streams, seed),
        },
        "artist_totals": {
            "powerlaw": powerlaw_suite(artist_totals, seed + 1),
            "lorenz": lorenz_curve(artist_totals),
            "gini": bootstrap_gini(artist_totals, seed + 1),
        },
        "artist_catalog_size": {
            "lorenz": lorenz_curve(catalog),
            "gini": bootstrap_gini(catalog, seed + 2),
        },
        "gini_by_year": year_gini,
        "survivorship": (
            "500 named artists is a truncated sample whose mass sits in a "
            "planted tail: one invented artist is 24% of rows. Every tail "
            "index and Gini here describes that generator, not Spotify."
        ),
    }
