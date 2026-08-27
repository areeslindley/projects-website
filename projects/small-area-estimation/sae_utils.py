"""Fay–Herriot small-area estimators used by the SAE notebooks.

Implements the area-level model of Fay & Herriot (1979) in the same form as
R's ``sae::eblupFH`` / ``emdi::fh``: known sampling variances, linear linking
model, and EBLUP shrinkage. Bayesian inference is a Gibbs sampler for the
conjugate hierarchical specification, with an optional PyMC NUTS fit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize


def project_dir() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "sae_utils.py").exists():
        return here
    alt = Path("projects/small-area-estimation").resolve()
    return alt if alt.exists() else here


def project_data_dir() -> Path:
    return project_dir() / "data"


def load_panel() -> pd.DataFrame:
    path = project_data_dir() / "la_sae_panel.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run projects/small-area-estimation/_build_data.py"
        )
    df = pd.read_csv(path)
    df["in_model"] = df["in_model"].astype(bool)
    return df


def modelling_frame(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Areas with direct rate, sampling variance, claimant rate, and ONS benchmark."""
    df = load_panel() if panel is None else panel.copy()
    m = df.loc[df["in_model"]].copy().reset_index(drop=True)
    if m.empty:
        raise ValueError("Modelling subset is empty — check la_sae_panel.csv")
    return m


def design_matrix(df: pd.DataFrame, covariates: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    covariates = covariates or ["claimant_rate"]
    X = np.column_stack([np.ones(len(df)), df[covariates].to_numpy(dtype=float)])
    names = ["intercept"] + list(covariates)
    return X, names


# ---------------------------------------------------------------------------
# Classical EBLUP (Fay–Herriot)
# ---------------------------------------------------------------------------

def _gls(X: np.ndarray, y: np.ndarray, V_diag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = 1.0 / V_diag
    XtW = X.T * w
    XtWX = XtW @ X
    XtWy = XtW @ y
    beta = np.linalg.solve(XtWX, XtWy)
    return beta, XtWX


def fh_reml_sigma_u(X: np.ndarray, y: np.ndarray, psi: np.ndarray) -> dict[str, Any]:
    """Profile residual maximum likelihood for σ²_u (Rao & Molina, 2015, §5.2)."""
    n, p = X.shape

    def neg_reml(log_s2: np.ndarray) -> float:
        s2 = float(np.exp(log_s2[0]))
        V = s2 + psi
        beta, XtWX = _gls(X, y, V)
        resid = y - X @ beta
        # log|V| + log|X'V^{-1}X| + residual quadratic form
        logdet_V = np.sum(np.log(V))
        sign, logdet_XtWX = np.linalg.slogdet(XtWX)
        if sign <= 0:
            return 1e12
        quad = np.sum(resid**2 / V)
        return 0.5 * (logdet_V + logdet_XtWX + quad)

    # Method-of-moments start (Prasad–Rao 1990)
    beta_ols, _ = _gls(X, y, np.ones_like(psi))
    pmat = X @ np.linalg.pinv(X.T @ X) @ X.T
    mom = ((y - X @ beta_ols) ** 2 - psi).sum() / max(n - p, 1)
    s2_0 = float(max(mom, 1e-6))

    opt = optimize.minimize(neg_reml, x0=np.array([np.log(s2_0)]), method="Nelder-Mead")
    sigma_u2 = float(np.exp(opt.x[0]))
    V = sigma_u2 + psi
    beta, XtWX = _gls(X, y, V)
    return {
        "sigma_u2": sigma_u2,
        "sigma_u": float(np.sqrt(sigma_u2)),
        "beta": beta,
        "XtWX": XtWX,
        "method": "REML",
        "converged": bool(opt.success),
        "mom_start": s2_0,
    }


def _pv_trace(X: np.ndarray, V: np.ndarray) -> float:
    """tr(P_V^2) where P_V = V^{-1} - V^{-1}X(X'V^{-1}X)^{-1}X'V^{-1} and ∂V/∂σ² = I."""
    invV = 1.0 / V
    XtW = X.T * invV
    XtWX = XtW @ X
    # P_V is n×n; form tr(P_V^2) via the diagonal + Woodbury terms.
    # P_V = D - uu-type: D = diag(invV), B = X @ XtWX^{-1} @ X' * outer invV
    XtWX_inv = np.linalg.inv(XtWX)
    # tr(P^2) = sum_i P_ii^2 + off-diagonals. Compute P explicitly (n ~ 250).
    WX = X * invV[:, None]
    P = np.diag(invV) - WX @ XtWX_inv @ WX.T
    return float(np.trace(P @ P))


def eblup_fh(
    df: pd.DataFrame,
    covariates: list[str] | None = None,
    y_col: str = "direct_rate",
    psi_col: str = "psi",
) -> pd.DataFrame:
    """EBLUP point estimates, shrinkage factors, and Prasad–Rao / Datta–Lahiri MSE."""
    covariates = covariates or ["claimant_rate"]
    y = df[y_col].to_numpy(dtype=float)
    psi = df[psi_col].to_numpy(dtype=float)
    X, names = design_matrix(df, covariates)
    fit = fh_reml_sigma_u(X, y, psi)
    s2 = fit["sigma_u2"]
    beta = fit["beta"]
    V = s2 + psi
    gamma = s2 / V
    theta_syn = X @ beta
    theta_eblup = gamma * y + (1.0 - gamma) * theta_syn

    # Datta–Lahiri (2000) / Rao–Molina: MSE ≈ g1 + g2 + 2 g3 for REML
    g1 = gamma * psi
    XtWX_inv = np.linalg.inv(fit["XtWX"])
    g2 = np.einsum("ij,jk,ik->i", X, XtWX_inv, X) * (1.0 - gamma) ** 2
    var_s2 = 2.0 / _pv_trace(X, V)
    g3 = (psi**2) * var_s2 / (V**3)
    mse = g1 + g2 + 2.0 * g3
    se = np.sqrt(np.maximum(mse, 0.0))

    out = df.copy()
    out["gamma"] = gamma
    out["theta_syn"] = theta_syn
    out["eblup"] = theta_eblup
    out["eblup_mse"] = mse
    out["eblup_se"] = se
    out["eblup_ci_lo"] = theta_eblup - 1.96 * se
    out["eblup_ci_hi"] = theta_eblup + 1.96 * se
    out.attrs["fit"] = {
        **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fit.items()},
        "coef_names": names,
        "var_sigma_u2": var_s2,
        "covariates": covariates,
    }
    return out


def jackknife_mse(
    df: pd.DataFrame,
    covariates: list[str] | None = None,
    y_col: str = "direct_rate",
    psi_col: str = "psi",
) -> np.ndarray:
    """Delete-one jackknife MSE for the EBLUP (Jiang, Lahiri & Wan 2002, simplified).

    MSE_i ≈ g1i(σ̂²) - ((n-1)/n) Σ_j (θ̂_i(-j) - θ̄_i)^2 is unstable for g1;
    we use the standard jackknife variance of the EBLUP itself:
        ((n-1)/n) Σ_j (θ̂_i(-j) - θ̄_i)^2
    and report it alongside Prasad–Rao, not instead of it.
    """
    covariates = covariates or ["claimant_rate"]
    n = len(df)
    y = df[y_col].to_numpy(dtype=float)
    psi = df[psi_col].to_numpy(dtype=float)
    X, _ = design_matrix(df, covariates)
    leave = np.zeros((n, n))
    for j in range(n):
        mask = np.ones(n, dtype=bool)
        mask[j] = False
        fit = fh_reml_sigma_u(X[mask], y[mask], psi[mask])
        s2 = fit["sigma_u2"]
        beta = fit["beta"]
        V = s2 + psi
        gamma = s2 / V
        leave[j] = gamma * y + (1.0 - gamma) * (X @ beta)
    theta_bar = leave.mean(axis=0)
    jack_var = ((n - 1) / n) * np.sum((leave - theta_bar) ** 2, axis=0)
    return jack_var


# ---------------------------------------------------------------------------
# Bayesian Fay–Herriot (Gibbs)
# ---------------------------------------------------------------------------

def gibbs_fh(
    df: pd.DataFrame,
    covariates: list[str] | None = None,
    y_col: str = "direct_rate",
    psi_col: str = "psi",
    draws: int = 4000,
    burn: int = 1000,
    thin: int = 2,
    seed: int = 42,
    prior_beta_var: float = 1e4,
    ig_a: float = 0.5,
    ig_b: float = 0.5,
) -> dict[str, Any]:
    """Conjugate Gibbs sampler for the area-level hierarchical model.

    y_i | θ_i ~ N(θ_i, ψ_i),
    θ_i | β, σ²_u ~ N(x_i'β, σ²_u),
    β ~ N(0, τ² I),  σ²_u ~ InverseGamma(a, b)  (rate parameterisation via scale b).
    """
    covariates = covariates or ["claimant_rate"]
    rng = np.random.default_rng(seed)
    y = df[y_col].to_numpy(dtype=float)
    psi = df[psi_col].to_numpy(dtype=float)
    X, names = design_matrix(df, covariates)
    n, p = X.shape

    # Initialise at the EBLUP
    eblup = eblup_fh(df, covariates=covariates, y_col=y_col, psi_col=psi_col)
    theta = eblup["eblup"].to_numpy(dtype=float)
    beta = np.array(eblup.attrs["fit"]["beta"], dtype=float)
    s2 = float(eblup.attrs["fit"]["sigma_u2"])

    n_keep = (draws - burn) // thin
    theta_s = np.empty((n_keep, n))
    beta_s = np.empty((n_keep, p))
    s2_s = np.empty(n_keep)
    k = 0
    prec_beta = np.eye(p) / prior_beta_var

    for t in range(draws):
        # θ | y, β, σ²_u
        prec = 1.0 / psi + 1.0 / s2
        mean = (y / psi + (X @ beta) / s2) / prec
        theta = rng.normal(mean, np.sqrt(1.0 / prec))
        # β | θ, σ²_u
        XtX = X.T @ X
        Vβ = np.linalg.inv(XtX / s2 + prec_beta)
        mβ = Vβ @ (X.T @ theta) / s2
        beta = rng.multivariate_normal(mβ, Vβ)
        # σ²_u | θ, β  ~ IG(a + n/2, b + 0.5 ||θ - Xβ||²)
        resid = theta - X @ beta
        a_post = ig_a + n / 2.0
        b_post = ig_b + 0.5 * np.dot(resid, resid)
        s2 = 1.0 / rng.gamma(a_post, 1.0 / b_post)
        s2 = max(s2, 1e-8)
        if t >= burn and (t - burn) % thin == 0:
            theta_s[k] = theta
            beta_s[k] = beta
            s2_s[k] = s2
            k += 1

    return {
        "theta": theta_s[:k],
        "beta": beta_s[:k],
        "sigma_u2": s2_s[:k],
        "coef_names": names,
        "covariates": covariates,
        "draws": k,
        "prior": {"beta_var": prior_beta_var, "ig_a": ig_a, "ig_b": ig_b},
    }


def summarise_theta(samples: np.ndarray, index: pd.Index | None = None) -> pd.DataFrame:
    q = np.quantile(samples, [0.025, 0.5, 0.975], axis=0)
    out = pd.DataFrame(
        {
            "bayes_mean": samples.mean(axis=0),
            "bayes_median": q[1],
            "bayes_ci_lo": q[0],
            "bayes_ci_hi": q[2],
            "bayes_sd": samples.std(axis=0, ddof=1),
        }
    )
    if index is not None:
        out.index = index
    return out


def posterior_predictive_y(theta_draws: np.ndarray, psi: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(theta_draws, np.sqrt(psi))


def fit_pymc_fh(
    df: pd.DataFrame,
    covariates: list[str] | None = None,
    y_col: str = "direct_rate",
    psi_col: str = "psi",
    draws: int = 800,
    tune: int = 800,
    chains: int = 2,
    seed: int = 42,
    target_accept: float = 0.9,
) -> dict[str, Any]:
    """NUTS fit of the same hierarchical model in PyMC. Raises if PyMC is missing."""
    import pymc as pm

    covariates = covariates or ["claimant_rate"]
    y = df[y_col].to_numpy(dtype=float)
    psi = df[psi_col].to_numpy(dtype=float)
    X, names = design_matrix(df, covariates)
    n, p = X.shape
    sigma_e = np.sqrt(psi)

    with pm.Model() as model:
        sigma_u = pm.HalfNormal("sigma_u", sigma=5.0)
        beta = pm.Normal("beta", mu=0.0, sigma=10.0, shape=p)
        mu_theta = pm.math.dot(X, beta)
        theta = pm.Normal("theta", mu=mu_theta, sigma=sigma_u, shape=n)
        pm.Normal("y_obs", mu=theta, sigma=sigma_e, observed=y)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=True,
        )

    theta_draws = np.asarray(idata.posterior["theta"]).reshape(-1, n)
    beta_draws = np.asarray(idata.posterior["beta"]).reshape(-1, p)
    sigma_u_draws = np.asarray(idata.posterior["sigma_u"]).reshape(-1)
    return {
        "idata": idata,
        "model": model,
        "theta": theta_draws,
        "beta": beta_draws,
        "sigma_u2": sigma_u_draws**2,
        "coef_names": names,
        "covariates": covariates,
        "draws": theta_draws.shape[0],
    }


def rhat_split(draws: np.ndarray, chains: int = 2) -> float:
    """Crude split-R̂ for a 1-d parameter stored as concatenated chain draws."""
    n = len(draws)
    if n < 4 or chains < 2:
        return np.nan
    m = n // chains
    draws = draws[: m * chains].reshape(chains, m)
    chain_means = draws.mean(axis=1)
    chain_vars = draws.var(axis=1, ddof=1)
    W = chain_vars.mean()
    B = m * chain_means.var(ddof=1)
    var_hat = (1 - 1 / m) * W + B / m
    if W <= 0:
        return np.nan
    return float(np.sqrt(var_hat / W))


def coverage_rate(truth: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((truth >= lo) & (truth <= hi)))
