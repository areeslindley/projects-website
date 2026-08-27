"""Generate small-area estimation notebooks. Run from the project directory or repo root."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

PROJ = Path(__file__).parent
KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"},
}

IMPORTS = """
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

PROJ_DIR = Path('.').resolve()
if not (PROJ_DIR / 'sae_utils.py').exists():
    PROJ_DIR = Path('projects/small-area-estimation').resolve()
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from sae_utils import (
    load_panel, modelling_frame, design_matrix, eblup_fh, fh_reml_sigma_u,
    jackknife_mse, gibbs_fh, summarise_theta, posterior_predictive_y,
    rhat_split, coverage_rate, project_data_dir,
)

def display_plotly(fig):
    \"\"\"Embed Plotly with CDN JS — fig.show() is blank in Jupyter Book HTML.\"\"\"
    display(HTML(fig.to_html(include_plotlyjs='cdn', full_html=False)))

FIG_DIR = PROJ_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR = project_data_dir()
PANEL = load_panel()
MODEL = modelling_frame(PANEL)
print(f'Panel: {len(PANEL)} local authorities; modelling subset: {len(MODEL)}')
print(f'Period (APS / ONS model-based): {PANEL[\"direct_period\"].dropna().iloc[0]}')
"""


def md(text: str):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str):
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def nav(title, prev_l, prev_t, next_l, next_t, desc):
    prev_p = f"[← Previous: {prev_t}]({prev_l})" if prev_l else ""
    next_p = f"[Next: {next_t} →]({next_l})" if next_l else ""
    sep = " | " if prev_p and next_p else ""
    return md(f"# {title}\n\n**Navigation**: {prev_p}{sep}{next_p}\n\n{desc}\n")


def footer(prev_l, prev_t, next_l, next_t):
    prev_p = f"[← Previous: {prev_t}]({prev_l})" if prev_l else ""
    next_p = f"[Next: {next_t} →]({next_l})" if next_l else ""
    sep = " | " if prev_p and next_p else ""
    return md(f"---\n\n**Navigation**: {prev_p}{sep}{next_p}\n")


def save(name, cells):
    nb = {"cells": cells, "metadata": KERNEL, "nbformat": 4, "nbformat_minor": 5}
    (PROJ / name).write_text(json.dumps(nb, indent=1))
    print(f"Wrote {name}")


def nb01():
    cells = [
        nav(
            "Introduction: the small-area problem",
            "index.md", "Project Overview",
            "02_data_acquisition.ipynb", "Data Acquisition",
            "Direct survey estimates become unpublishably noisy once you slice the "
            "Annual Population Survey (APS) to local authorities. This notebook sets "
            "out why, and writes down the Fay–Herriot model that official statistics "
            "has used for two decades to borrow strength.",
        ),
        code(IMPORTS),
        md(
            "## Why direct estimates fail\n\n"
            "The APS is a household survey designed to support **national and regional** "
            "labour-market estimates. At local-authority (LA) level the achieved sample "
            "in a 12-month window is often a few hundred economically active adults, "
            "sometimes fewer. Unemployment is a rare outcome (~4–6% of the economically "
            "active), so the **numerator sample** is a handful of people.\n\n"
            "Write $y_i$ for the direct Horvitz–Thompson estimator of the unemployment "
            "rate $\\theta_i$ in area $i$, and $\\psi_i = \\mathrm{Var}(y_i \\mid \\theta_i)$ "
            "for its design-based sampling variance (treated as known). Then\n\n"
            "$$\n"
            "y_i = \\theta_i + e_i, \\qquad e_i \\sim N(0, \\psi_i).\n"
            "$$\n\n"
            "If $\\psi_i$ is large relative to the between-area variation in $\\theta_i$, "
            "the ranking of LAs by $y_i$ is mostly ranking of sampling error. ONS "
            "historically responded by **suppressing** estimates with large CVs and, "
            "for unemployment specifically, by publishing a **model-based** series "
            "instead (Nomis dataset NM_127)."
        ),
        code(
            "print(PANEL[['direct_rate','direct_rate_ci','psi','cv','ons_mb_rate']].describe().round(3).to_string())\n"
            "print('\\nShare of LAs with a published APS rate: '\n"
            "      f\"{PANEL['direct_rate'].notna().mean():.0%}\")\n"
            "print(f\"Share with a published 95% CI (modelling inputs): {PANEL['psi'].notna().mean():.0%}\")\n"
        ),
        md(
            "## Fay–Herriot intuition\n\n"
            "The linking model assumes the true rates vary smoothly with auxiliary "
            "information $x_i$ that is known without (or with negligible) sampling error. "
            "ONS uses the **claimant count** — an administrative register — as the "
            "workhorse covariate, because it is strongly correlated with ILO unemployment "
            "and has no survey variance:\n\n"
            "$$\n"
            "\\theta_i = x_i'\\beta + u_i, \\qquad u_i \\sim N(0, \\sigma_u^2).\n"
            "$$\n\n"
            "The EBLUP is a convex combination of the direct estimate and the regression "
            "synthetic estimator $\\hat{\\theta}_i^{\\mathrm{syn}} = x_i'\\hat\\beta$:\n\n"
            "$$\n"
            "\\hat{\\theta}_i^{\\mathrm{EBLUP}} = \\gamma_i y_i + (1-\\gamma_i) x_i'\\hat\\beta, "
            "\\qquad \\gamma_i = \\frac{\\sigma_u^2}{\\sigma_u^2 + \\psi_i}.\n"
            "$$\n\n"
            "Areas with small $\\psi_i$ (large samples) keep $\\gamma_i \\approx 1$ and "
            "barely shrink; noisy areas are pulled toward the regression surface. "
            "That is the whole idea — **shrinkage in proportion to sampling variance**, "
            "not a uniform smooth."
        ),
        code(
            "# Schematic: shrinkage as a function of sampling variance, holding σ²_u fixed\n"
            "psi_grid = np.linspace(0.05, 8, 200)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "for s2, lab in [(0.5, r'$\\sigma_u^2 = 0.5$'), (1.5, r'$\\sigma_u^2 = 1.5$'), (4.0, r'$\\sigma_u^2 = 4$')]:\n"
            "    ax.plot(psi_grid, s2 / (s2 + psi_grid), label=lab, lw=2)\n"
            "ax.set_xlabel(r'Sampling variance $\\psi_i$')\n"
            "ax.set_ylabel(r'Shrinkage weight $\\gamma_i$')\n"
            "ax.set_title('Fay–Herriot: how much the direct estimate is trusted')\n"
            "ax.legend()\n"
            "fig.tight_layout()\n"
            "fig.savefig(FIG_DIR / '01_shrinkage_weight.png', dpi=120)\n"
            "fig\n"
        ),
        md(
            "## Two inferential routes\n\n"
            "| | Classical EBLUP | Bayesian hierarchical |\n"
            "|---|---|---|\n"
            "| $\\sigma_u^2$ | Point estimate (REML / Prasad–Rao MOM) | Full posterior |\n"
            "| $\\theta_i$ | Plug-in EBLUP + analytic or jackknife MSE | Posterior mean / median and credible interval |\n"
            "| Uncertainty in $\\hat\\beta, \\hat\\sigma_u^2$ | Corrected for via $g_1+g_2+2g_3$ (Datta–Lahiri) | Propagated automatically |\n"
            "| Assumptions | Frequentist MSE under the model; $\\psi_i$ known | Prior on $\\beta, \\sigma_u^2$; same sampling model |\n\n"
            "Neither is a free lunch. The frequentist MSE approximations are **asymptotic "
            "in the number of areas**, not in the sample size per area, and they treat "
            "$\\psi_i$ as known (it is estimated from the survey). The Bayesian interval "
            "is well-calibrated only if the prior and the linking model are adequate. "
            "We will stress-test both against ONS's published model-based figures — "
            "which are themselves a production Fay–Herriot, not ground truth."
        ),
        md(
            "## What this project is not\n\n"
            "- It is **not** the ONS production pipeline. ONS has used variants with "
            "benchmarking to regional APS totals, time-series linking, and (in some "
            "years) more elaborate covariance structures. We implement the textbook "
            "area-level model on public aggregates.\n"
            "- It does **not** use APS microdata. All inputs are LA-level Nomis tables "
            "under the Open Government Licence.\n"
            "- $\\psi_i$ is recovered from published 95% CI half-widths as "
            "$(c_i / 1.96)^2$. That ignores uncertainty in the variance estimate itself."
        ),
        md(
            "## Key takeaways\n\n"
            "- Direct APS unemployment rates at LA level are often suppressed or have "
            "CVs too large to support rankings.\n"
            "- Fay–Herriot shrinks $y_i$ toward $x_i'\\beta$ with weight $\\gamma_i$ "
            "determined by the **ratio** of process variance to sampling variance.\n"
            "- ONS model-based estimates are the natural external benchmark for an "
            "open-source reimplementation — agreement is a check, not a proof."
        ),
        footer("index.md", "Project Overview", "02_data_acquisition.ipynb", "Data Acquisition"),
    ]
    save("01_introduction.ipynb", cells)


def nb02():
    cells = [
        nav(
            "Data acquisition",
            "01_introduction.ipynb", "Introduction",
            "03_direct_estimates.ipynb", "Direct Estimates",
            "All inputs are open Nomis / ONS tables. This notebook documents the API "
            "calls and loads the bundled panel used by later chapters.",
        ),
        code(IMPORTS),
        md(
            "## Sources and licence\n\n"
            "| Source | Nomis id | Role |\n"
            "|---|---|---|\n"
            "| Annual Population Survey, unemployment rate aged 16–64 | `NM_17_5` variable 84 | Direct estimator $y_i$, 95% CI → $\\psi_i$ |\n"
            "| APS economic inactivity rate aged 16–64 | `NM_17_5` variable 111 | Optional auxiliary covariate |\n"
            "| Model-based estimates of unemployment | `NM_127_1` item 2 | ONS production benchmark |\n"
            "| Claimant count as % of residents aged 16–64 | `NM_162_1` measure 2 | Administrative covariate $x_i$ |\n"
            "| LAD December 2023 BGC boundaries | ONS Open Geography | Choropleths |\n\n"
            "Geography: `TYPE424` — local authority districts / unitaries as of April 2023. "
            "The APS and model-based series used here are the 12 months **Apr 2025–Mar 2026**; "
            "the claimant snapshot is **March 2026** (end of that window).\n\n"
            "> **Attribution.** Contains public sector information licensed under the "
            "[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). "
            "Source: Office for National Statistics and [Nomis](https://www.nomisweb.co.uk/)."
        ),
        md(
            "## Nomis REST shape\n\n"
            "Nomis is a SDMX-style API. A typical CSV pull is\n\n"
            "`https://www.nomisweb.co.uk/api/v01/dataset/{id}.data.csv?geography=TYPE424&...&time=latest`\n\n"
            "APS percentages expose four `MEASURES`: the rate (`20599`), numerator, "
            "denominator, and CI half-width (`21003`). Model-based estimates expose "
            "value (`20100`) and confidence (`20701`). Notebooks **do not** hit Nomis "
            "at build time; they read `data/la_sae_panel.csv`. Refresh with:\n\n"
            "```bash\n"
            "python projects/small-area-estimation/_build_data.py\n"
            "```"
        ),
        code(
            "print('Bundled files:')\n"
            "for p in sorted(DATA_DIR.glob('*')):\n"
            "    if p.is_file():\n"
            "        print(f'  {p.name:28s} {p.stat().st_size/1024:7.1f} KB')\n"
            "\n"
            "print('\\nPanel columns:', list(PANEL.columns))\n"
            "print(PANEL.head(8).to_string(index=False))\n"
        ),
        md(
            "## Coverage and suppression\n\n"
            "Nomis flags unpublished APS cells (`OBS_STATUS` other than `A`). After the "
            "pivot, suppression appears as missing `direct_rate` or missing CI. The "
            "Fay–Herriot fit below uses only LAs with a published rate **and** CI, "
            "plus a claimant rate and an ONS model-based figure — the `in_model` flag."
        ),
        code(
            "summary = (\n"
            "    PANEL.groupby('country')\n"
            "    .agg(\n"
            "        n=('GEOGRAPHY_CODE', 'size'),\n"
            "        with_direct=('direct_rate', 'count'),\n"
            "        with_ci=('psi', 'count'),\n"
            "        in_model=('in_model', 'sum'),\n"
            "        mean_direct=('direct_rate', 'mean'),\n"
            "        mean_ons=('ons_mb_rate', 'mean'),\n"
            "        mean_claimant=('claimant_rate', 'mean'),\n"
            "    )\n"
            "    .round(2)\n"
            ")\n"
            "summary\n"
        ),
        code(
            "fig = px.scatter(\n"
            "    PANEL.dropna(subset=['direct_rate', 'ons_mb_rate']),\n"
            "    x='direct_rate', y='ons_mb_rate', color='country',\n"
            "    hover_name='GEOGRAPHY_NAME',\n"
            "    trendline='ols',\n"
            "    labels={'direct_rate': 'APS direct rate (%)', 'ons_mb_rate': 'ONS model-based rate (%)'},\n"
            "    title='Direct APS vs ONS model-based unemployment rate',\n"
            ")\n"
            "lims = [0, max(PANEL['direct_rate'].max(), PANEL['ons_mb_rate'].max())]\n"
            "fig.add_trace(go.Scatter(x=lims, y=lims, mode='lines', name='y = x',\n"
            "                         line=dict(dash='dash', color='grey')))\n"
            "display_plotly(fig)\n"
        ),
        md(
            "## Constructing $\\psi_i$\n\n"
            "If Nomis publishes a 95% CI half-width $c_i$ on the percentage scale,\n\n"
            "$$\n"
            "\\hat\\psi_i = (c_i / 1.96)^2.\n"
            "$$\n\n"
            "This assumes a symmetric Wald interval. Design-based intervals for "
            "proportions can be asymmetric at the boundary; we drop areas without a "
            "published $c_i$ rather than inventing a generalised variance function, "
            "so the likelihood in later chapters is faithful to the published errors."
        ),
        code(
            "m = MODEL.copy()\n"
            "print(f'Modelling subset n={len(m)}')\n"
            "print(m[['GEOGRAPHY_NAME','country','direct_rate','direct_rate_ci','psi','cv',\n"
            "         'claimant_rate','ons_mb_rate']].head(10).to_string(index=False))\n"
            "print('\\nCV quintiles:', m['cv'].quantile([0.2, 0.4, 0.6, 0.8]).round(3).to_dict())\n"
        ),
        md(
            "## Key takeaways\n\n"
            "- Four open Nomis tables plus ONS boundaries are sufficient to fit and "
            "validate an area-level SAE model — no Accredited Researcher access.\n"
            "- Suppression is a feature of the problem: many LAs have $y_i$ but no CI, "
            "and some have neither.\n"
            "- $\\psi_i$ comes from published CIs, not from a binomial approximation "
            "to weighted counts (those counts are **population estimates**, not sample sizes)."
        ),
        footer("01_introduction.ipynb", "Introduction", "03_direct_estimates.ipynb", "Direct Estimates"),
    ]
    save("02_data_acquisition.ipynb", cells)


def nb03():
    cells = [
        nav(
            "Direct estimates and sampling variance",
            "02_data_acquisition.ipynb", "Data Acquisition",
            "04_eblup.ipynb", "Classical EBLUP",
            "We inspect $y_i$ and $\\psi_i$ directly: where the APS is stable, where it "
            "is noise, and how that tracks the economically active population used as a "
            "size proxy.",
        ),
        code(IMPORTS),
        md(
            "## The design-based object\n\n"
            "$y_i$ is an estimated **percentage** of economically active 16–64 year olds "
            "who are ILO unemployed. The Nomis denominator is a **weighted population "
            "estimate**, not the APS sample size. CVs therefore need the published CI; "
            "you cannot reconstruct $\\psi_i$ from $y_i(1-y_i)/N_i^{\\mathrm{pop}}$.\n\n"
            "A rule of thumb in ONS quality reports is that CVs above ~20% are too "
            "volatile for unaided publication. That is exactly the regime Fay–Herriot "
            "is designed for."
        ),
        code(
            "m = MODEL.copy()\n"
            "m['size_proxy'] = m['direct_rate_denominator']\n"
            "print(m[['direct_rate','psi','cv','size_proxy']].describe().round(3).to_string())\n"
            "print(f\"\\nShare with CV > 0.20: {(m['cv'] > 0.20).mean():.1%}\")\n"
            "print(f\"Share with CV > 0.30: {(m['cv'] > 0.30).mean():.1%}\")\n"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))\n"
            "axes[0].scatter(m['size_proxy']/1e3, m['direct_rate'], c=m['cv'], cmap='magma_r', s=28, alpha=0.85)\n"
            "axes[0].set_xlabel('Economically active population (thousands, APS weighted)')\n"
            "axes[0].set_ylabel('APS unemployment rate (%)')\n"
            "axes[0].set_title('Direct rates vs size')\n"
            "sc = axes[1].scatter(m['size_proxy']/1e3, m['cv'], c=m['direct_rate'], cmap='viridis', s=28, alpha=0.85)\n"
            "axes[1].axhline(0.20, color='crimson', ls='--', lw=1, label='CV = 20%')\n"
            "axes[1].set_xlabel('Economically active population (thousands)')\n"
            "axes[1].set_ylabel('Coefficient of variation')\n"
            "axes[1].set_title('Instability vs size')\n"
            "axes[1].legend()\n"
            "fig.colorbar(sc, ax=axes[1], label='Direct rate (%)')\n"
            "fig.tight_layout()\n"
            "fig.savefig(FIG_DIR / '03_cv_vs_size.png', dpi=120)\n"
            "fig\n"
        ),
        md(
            "The left panel is the usual heteroskedastic scatter: smaller LAs fan out. "
            "The right panel is the diagnostic that matters for SAE — CV declines with "
            "size, but even some large authorities sit above 20% because unemployment "
            "is rare. Those points will receive $\\gamma_i$ well below 1."
        ),
        code(
            "fig = px.scatter(\n"
            "    m, x='size_proxy', y='direct_rate',\n"
            "    error_y='direct_rate_ci', color='country',\n"
            "    hover_name='GEOGRAPHY_NAME',\n"
            "    log_x=True,\n"
            "    labels={'size_proxy': 'APS economically active (weighted)', 'direct_rate': 'Unemployment rate (%)'},\n"
            "    title='Direct estimates with published 95% CI half-widths',\n"
            ")\n"
            "display_plotly(fig)\n"
        ),
        md(
            "## Direct vs claimant count\n\n"
            "If the linking model is going to help, $x_i$ must be correlated with "
            "$\\theta_i$ **after** accounting for the fact that $y_i$ is noisy. A raw "
            "scatter of $y_i$ against claimant rate understates the true correlation "
            "(attenuation). ONS's production model is betting that the administrative "
            "series tracks the ILO concept closely enough for $u_i$ to be small."
        ),
        code(
            "r = np.corrcoef(m['claimant_rate'], m['direct_rate'])[0, 1]\n"
            "print(f'corr(claimant, direct) = {r:.3f}')\n"
            "if m['inactivity_rate'].notna().all():\n"
            "    r2 = np.corrcoef(m['inactivity_rate'], m['direct_rate'])[0, 1]\n"
            "    print(f'corr(inactivity, direct) = {r2:.3f}')\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8, 5))\n"
            "for c, g in m.groupby('country'):\n"
            "    ax.scatter(g['claimant_rate'], g['direct_rate'], s=28, alpha=0.8, label=c)\n"
            "ax.set_xlabel('Claimant count rate (% of 16–64 residents)')\n"
            "ax.set_ylabel('APS unemployment rate (%)')\n"
            "ax.set_title('Linking covariate vs noisy direct estimator')\n"
            "ax.legend()\n"
            "fig.tight_layout()\n"
            "fig.savefig(FIG_DIR / '03_claimant_vs_direct.png', dpi=120)\n"
            "fig\n"
        ),
        md(
            "## Key takeaways\n\n"
            "- Published CIs, not population denominators, identify $\\psi_i$.\n"
            "- A large share of even the *publishable* LAs have CV > 20%.\n"
            "- Claimant count co-moves with APS unemployment, which is necessary "
            "(not sufficient) for a useful linking model."
        ),
        footer("02_data_acquisition.ipynb", "Data Acquisition", "04_eblup.ipynb", "Classical EBLUP"),
    ]
    save("03_direct_estimates.ipynb", cells)


def nb04():
    cells = [
        nav(
            "Fay–Herriot: classical EBLUP",
            "03_direct_estimates.ipynb", "Direct Estimates",
            "05_bayesian.ipynb", "Bayesian Hierarchical Model",
            "REML fit of $\\sigma_u^2$, EBLUP shrinkage, Prasad–Rao / Datta–Lahiri MSE, "
            "and a delete-one jackknife — the same objects as `sae::eblupFH` / `emdi::fh`.",
        ),
        code(IMPORTS),
        md(
            "## The frequentist target\n\n"
            "Stacking the sampling and linking models gives the marginal\n\n"
            "$$\n"
            "y \\mid \\beta, \\sigma_u^2 \\sim N\\bigl(X\\beta,\\; \\sigma_u^2 I + \\Psi\\bigr), "
            "\\qquad \\Psi = \\mathrm{diag}(\\psi_i).\n"
            "$$\n\n"
            "REML maximises the residual likelihood after projecting out $\\beta$ "
            "(Rao & Molina, 2015, §5.2). Given $\\hat{\\sigma}_u^2$, GLS supplies "
            "$\\hat{\\beta}$ and the EBLUP $\\hat{\\theta}_i = \\hat{\\gamma}_i y_i + "
            "(1-\\hat{\\gamma}_i) x_i'\\hat{\\beta}$.\n\n"
            "We implement this in `sae_utils.eblup_fh` rather than wrapping R. The "
            "algebra is that of `sae::eblupFH(..., method = \"REML\")` and "
            "`emdi::fh(..., method = \"reml\")`: known $\\psi_i$, linear mean, "
            "independent random effects. What we **do not** include is emdi's "
            "spatial/temporal extensions or automatic transformation of $y$."
        ),
        code(
            "fit = eblup_fh(MODEL, covariates=['claimant_rate'])\n"
            "info = fit.attrs['fit']\n"
            "print('REML σ²_u = {:.3f}  (σ_u = {:.3f})'.format(info['sigma_u2'], info['sigma_u']))\n"
            "print('MOM start  = {:.3f}'.format(info['mom_start']))\n"
            "print('converged  =', info['converged'])\n"
            "print('Var(σ²_u)  = {:.3f}'.format(info['var_sigma_u2']))\n"
            "beta = np.array(info['beta'])\n"
            "for name, b in zip(info['coef_names'], beta):\n"
            "    print(f'  β[{name}] = {b:.3f}')\n"
            "print(f\"\\nmedian γ = {fit['gamma'].median():.3f}  \"\n"
            "      f\"min={fit['gamma'].min():.3f}  max={fit['gamma'].max():.3f}\")\n"
            "fit[['GEOGRAPHY_NAME','direct_rate','eblup','gamma','eblup_se','ons_mb_rate']].head(8)\n"
        ),
        md(
            "## MSE: $g_1 + g_2 + 2g_3$\n\n"
            "Prasad & Rao (1990) decompose the EBLUP MSE. Datta & Lahiri (2000) show "
            "that for REML the leading correction is **twice** $g_3$:\n\n"
            "$$\n"
            "\\mathrm{mse}(\\hat\\theta_i) \\approx g_{1i}(\\hat\\sigma_u^2) + g_{2i}(\\hat\\sigma_u^2) "
            "+ 2 g_{3i}(\\hat\\sigma_u^2),\n"
            "$$\n\n"
            "- $g_1 = \\gamma_i \\psi_i$ — leading term, the posterior variance if "
            "$\\beta,\\sigma_u^2$ were known;\n"
            "- $g_2$ — uncertainty in $\\hat\\beta$;\n"
            "- $g_3$ — uncertainty in $\\hat\\sigma_u^2$, involving "
            "$\\widehat{\\mathrm{Var}}(\\hat\\sigma_u^2) = 2 / \\mathrm{tr}(P_V^2)$.\n\n"
            "These are **model-based** MSEs. They understate error if the linking model "
            "is misspecified, and they ignore uncertainty in $\\psi_i$."
        ),
        code(
            "print(fit[['eblup_mse','eblup_se']].describe().round(3).to_string())\n"
            "print('\\nMedian SE: direct CI/1.96 vs EBLUP')\n"
            "print(pd.DataFrame({\n"
            "    'direct_se': MODEL['direct_rate_ci']/1.96,\n"
            "    'eblup_se': fit['eblup_se'],\n"
            "}).median().round(3))\n"
        ),
        md(
            "## Jackknife MSE\n\n"
            "Jiang, Lahiri & Wan (2002) jackknife the EBLUP by deleting one area at a "
            "time and refitting $\\sigma_u^2$. We report the jackknife **variance of "
            "the EBLUP** as a check on $g_1+g_2+2g_3$, not as a replacement: with "
            "$m \\approx 100$ areas the delete-one perturbation of $\\hat\\sigma_u^2$ "
            "is noisy, and the two estimators can diverge in the tails."
        ),
        code(
            "jack_var = jackknife_mse(MODEL, covariates=['claimant_rate'])\n"
            "fit = fit.copy()\n"
            "fit['eblup_se_jack'] = np.sqrt(np.maximum(jack_var, 0))\n"
            "print(fit[['eblup_se','eblup_se_jack']].describe().round(3).to_string())\n"
            "fig, ax = plt.subplots(figsize=(5.5, 5.5))\n"
            "ax.scatter(fit['eblup_se'], fit['eblup_se_jack'], s=22, alpha=0.8)\n"
            "mmax = max(fit['eblup_se'].max(), fit['eblup_se_jack'].max())\n"
            "ax.plot([0, mmax], [0, mmax], '--', color='grey')\n"
            "ax.set_xlabel('Prasad–Rao / Datta–Lahiri SE')\n"
            "ax.set_ylabel('Jackknife SE')\n"
            "ax.set_title('Two estimates of EBLUP uncertainty')\n"
            "fig.tight_layout()\n"
            "fig.savefig(FIG_DIR / '04_mse_jackknife.png', dpi=120)\n"
            "fig\n"
        ),
        md(
            "## Shrinkage in the data\n\n"
            "Plot $y_i$ against the EBLUP. Points far from $y=x$ are the areas the "
            "model is willing to move; they should be the high-$\\psi_i$ ones."
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
            "sc = axes[0].scatter(fit['direct_rate'], fit['eblup'], c=fit['gamma'], cmap='coolwarm', s=32)\n"
            "lims = [fit[['direct_rate','eblup']].min().min(), fit[['direct_rate','eblup']].max().max()]\n"
            "axes[0].plot(lims, lims, '--', color='grey')\n"
            "axes[0].set_xlabel('Direct APS rate (%)')\n"
            "axes[0].set_ylabel('EBLUP (%)')\n"
            "axes[0].set_title('Shrinkage: colour = γ')\n"
            "fig.colorbar(sc, ax=axes[0], label=r'$\\gamma_i$')\n"
            "axes[1].scatter(fit['psi'], fit['gamma'], s=28, alpha=0.8, c='#0d3b4c')\n"
            "axes[1].set_xlabel(r'$\\psi_i$')\n"
            "axes[1].set_ylabel(r'$\\gamma_i$')\n"
            "axes[1].set_title('Weight on the direct estimate vs sampling variance')\n"
            "fig.tight_layout()\n"
            "fig.savefig(FIG_DIR / '04_shrinkage.png', dpi=120)\n"
            "fig\n"
        ),
        md(
            "## Adding inactivity\n\n"
            "A second covariate (APS inactivity) is itself a survey estimate, so it "
            "violates the 'known $x$' assumption. We fit it as a sensitivity check, "
            "not as the preferred specification. If $\\hat\\sigma_u^2$ barely moves, "
            "claimant count already absorbs most of the between-area signal."
        ),
        code(
            "m2 = MODEL.dropna(subset=['inactivity_rate']).copy()\n"
            "fit2 = eblup_fh(m2, covariates=['claimant_rate', 'inactivity_rate'])\n"
            "i2 = fit2.attrs['fit']\n"
            "print('claimant-only σ²_u = {:.3f}'.format(info['sigma_u2']))\n"
            "print('claimant+inactivity σ²_u = {:.3f}'.format(i2['sigma_u2']))\n"
            "print('coefficients:', dict(zip(i2['coef_names'], np.round(i2['beta'], 3))))\n"
        ),
        md(
            "## Key takeaways\n\n"
            "- REML $\\hat\\sigma_u^2$ determines the **global** amount of borrowing; "
            "$\\psi_i$ then allocates it locally via $\\gamma_i$.\n"
            "- Datta–Lahiri MSE is the analytic default; the jackknife is a useful "
            "disagreement diagnostic, not an automatic upgrade.\n"
            "- Treating a survey covariate as $x_i$ is a specification error — keep "
            "the production-style model to claimant count."
        ),
        footer("03_direct_estimates.ipynb", "Direct Estimates", "05_bayesian.ipynb", "Bayesian Hierarchical Model"),
    ]
    save("04_eblup.ipynb", cells)


def nb05():
    cells = [
        nav(
            "Fay–Herriot: Bayesian hierarchical model",
            "04_eblup.ipynb", "Classical EBLUP",
            "06_model_comparison.ipynb", "Model Comparison",
            "The same area-level model, now with a joint posterior over "
            "$(\\theta, \\beta, \\sigma_u^2)$. A conjugate Gibbs sampler is the "
            "workhorse; a PyMC NUTS fit is the independent check.",
        ),
        code(IMPORTS),
        md(
            "## Why Bayes here\n\n"
            "The EBLUP is a **posterior mean** for $\\theta_i$ under a flat prior on "
            "$\\beta$ and a plug-in $\\hat\\sigma_u^2$. That plug-in step is the "
            "awkward bit: $g_3$ is a Taylor correction for not knowing $\\sigma_u^2$. "
            "A hierarchical Bayesian model just samples it.\n\n"
            "$$\n"
            "\\begin{aligned}\n"
            "y_i \\mid \\theta_i &\\sim N(\\theta_i, \\psi_i), \\\\\n"
            "\\theta_i \\mid \\beta, \\sigma_u^2 &\\sim N(x_i'\\beta, \\sigma_u^2), \\\\\n"
            "\\beta &\\sim N(0, \\tau^2 I), \\\\\n"
            "\\sigma_u^2 &\\sim \\mathrm{InverseGamma}(a, b).\n"
            "\\end{aligned}\n"
            "$$\n\n"
            "This is conjugate, so Gibbs is exact (up to Monte Carlo) and cheap. "
            "NUTS in PyMC does not need conjugacy; we use it to confirm that the "
            "posterior is not an artefact of the Inverse-Gamma prior. A half-normal "
            "on $\\sigma_u$ (PyMC) and IG$(0.5,0.5)$ on $\\sigma_u^2$ (Gibbs) are "
            "**not** the same prior — if $\\theta_i$ posteriors still agree, the "
            "likelihood is doing the work.\n\n"
            "That last point is the frequentist/Bayes distinction that actually "
            "matters here. It is **not** that one produces point estimates and the "
            "other intervals. Both produce both. The difference is whether "
            "uncertainty in $\\sigma_u^2$ is a correction term or a sampled parameter, "
            "and whether you are willing to insert a prior that remains informative "
            "when $m$ is small."
        ),
        md(
            "## Gibbs sampler\n\n"
            "Full conditionals (Gelman et al. notation; scale parameterisation of IG):\n\n"
            "- $\\theta_i \\mid \\cdot \\sim N\\big(v_i(y_i/\\psi_i + x_i'\\beta/\\sigma_u^2),\\, v_i\\big)$, "
            "$v_i^{-1} = \\psi_i^{-1} + \\sigma_u^{-2}$\n"
            "- $\\beta \\mid \\cdot \\sim N(m_\\beta, V_\\beta)$ with precision "
            "$X'X/\\sigma_u^2 + \\tau^{-2}I$\n"
            "- $\\sigma_u^2 \\mid \\cdot \\sim \\mathrm{IG}(a+m/2,\\, b + \\tfrac12\\|\\theta-X\\beta\\|^2)$\n\n"
            "$\\tau^2 = 10^4$ is weakly informative on the percentage-point scale."
        ),
        code(
            "gibbs = gibbs_fh(MODEL, covariates=['claimant_rate'], draws=4000, burn=1000, thin=2, seed=42)\n"
            "theta_g = gibbs['theta']\n"
            "sum_g = summarise_theta(theta_g, index=MODEL.index)\n"
            "bayes = MODEL.join(sum_g)\n"
            "print(f\"Kept draws: {gibbs['draws']}\")\n"
            "print(f\"Posterior mean σ²_u = {gibbs['sigma_u2'].mean():.3f}  \"\n"
            "      f\"median={np.median(gibbs['sigma_u2']):.3f}\")\n"
            "print('β posterior means:', dict(zip(gibbs['coef_names'], np.round(gibbs['beta'].mean(0), 3))))\n"
            "for i, name in enumerate(gibbs['coef_names']):\n"
            "    d = gibbs['beta'][:, i]\n"
            "    print(f\"  {name}: mean={d.mean():.3f}  95% CI=({np.quantile(d,0.025):.3f}, {np.quantile(d,0.975):.3f})\")\n"
            "print(f\"split-R̂(σ²_u) ≈ {rhat_split(gibbs['sigma_u2']):.3f}\")\n"
            "bayes[['GEOGRAPHY_NAME','direct_rate','bayes_mean','bayes_ci_lo','bayes_ci_hi']].head()\n"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "axes[0].plot(gibbs['sigma_u2'], lw=0.6, color='#0d3b4c')\n"
            "axes[0].set_title(r'Gibbs trace: $\\sigma_u^2$')\n"
            "axes[0].set_xlabel('draw')\n"
            "axes[1].hist(gibbs['sigma_u2'], bins=40, color='#2a9d8f', edgecolor='white')\n"
            "axes[1].axvline(gibbs['sigma_u2'].mean(), color='crimson', ls='--', label='posterior mean')\n"
            "e = eblup_fh(MODEL, covariates=['claimant_rate'])\n"
            "axes[1].axvline(e.attrs['fit']['sigma_u2'], color='black', ls=':', label='REML')\n"
            "axes[1].set_title(r'Posterior of $\\sigma_u^2$ vs REML')\n"
            "axes[1].legend()\n"
            "fig.tight_layout()\n"
            "fig.savefig(FIG_DIR / '05_sigma_u_posterior.png', dpi=120)\n"
            "fig\n"
        ),
        md(
            "## PyMC / NUTS\n\n"
            "The non-conjugate specification puts a half-normal prior on $\\sigma_u$ "
            "and independent $N(0,10^2)$ priors on $\\beta$. If PyMC is unavailable "
            "in this runtime, the cell records the failure and the Gibbs posterior "
            "remains the Bayesian result."
        ),
        code(
            "pymc_ok = False\n"
            "pymc_sum = None\n"
            "try:\n"
            "    from sae_utils import fit_pymc_fh\n"
            "    pymc_fit = fit_pymc_fh(\n"
            "        MODEL, covariates=['claimant_rate'],\n"
            "        draws=600, tune=600, chains=2, seed=42,\n"
            "    )\n"
            "    pymc_sum = summarise_theta(pymc_fit['theta'], index=MODEL.index).add_prefix('pymc_')\n"
            "    pymc_ok = True\n"
            "    print(f\"NUTS draws: {pymc_fit['draws']}\")\n"
            "    print(f\"Posterior mean σ²_u (NUTS) = {pymc_fit['sigma_u2'].mean():.3f}\")\n"
            "    print(f\"split-R̂(σ_u) ≈ {rhat_split(np.sqrt(pymc_fit['sigma_u2'])):.3f}\")\n"
            "except Exception as exc:\n"
            "    print('PyMC fit skipped:', type(exc).__name__, '-', exc)\n"
        ),
        code(
            "if pymc_ok:\n"
            "    cmp = pd.DataFrame({\n"
            "        'gibbs_mean': bayes['bayes_mean'],\n"
            "        'pymc_mean': pymc_sum['pymc_bayes_mean'],\n"
            "    })\n"
            "    print('corr(Gibbs, NUTS) θ means:', np.corrcoef(cmp['gibbs_mean'], cmp['pymc_mean'])[0,1].round(4))\n"
            "    print((cmp['gibbs_mean'] - cmp['pymc_mean']).describe().round(3))\n"
            "    fig, ax = plt.subplots(figsize=(5.5, 5.5))\n"
            "    ax.scatter(cmp['gibbs_mean'], cmp['pymc_mean'], s=22, alpha=0.85)\n"
            "    lims = [cmp.min().min(), cmp.max().max()]\n"
            "    ax.plot(lims, lims, '--', color='grey')\n"
            "    ax.set_xlabel('Gibbs posterior mean')\n"
            "    ax.set_ylabel('PyMC NUTS posterior mean')\n"
            "    ax.set_title(r'$\\theta_i$ under two priors / two samplers')\n"
            "    fig.tight_layout()\n"
            "    fig.savefig(FIG_DIR / '05_gibbs_vs_pymc.png', dpi=120)\n"
            "    fig\n"
            "else:\n"
            "    print('No NUTS comparison in this runtime.')\n"
        ),
        md(
            "## Posterior predictive check\n\n"
            "Draw $y_i^{\\mathrm{rep}} \\sim N(\\theta_i^{(s)}, \\psi_i)$ and compare "
            "the replicated distribution of, say, the cross-area variance of $y$ to "
            "the observed one. A linking model that is too rigid (tiny $\\sigma_u^2$) "
            "under-replicates the spread of the direct estimates; one that ignores "
            "$\\psi_i$ over-replicates it."
        ),
        code(
            "y = MODEL['direct_rate'].to_numpy()\n"
            "psi = MODEL['psi'].to_numpy()\n"
            "y_rep = posterior_predictive_y(theta_g, psi, seed=1)\n"
            "obs_var = y.var(ddof=1)\n"
            "rep_var = y_rep.var(axis=1, ddof=1)\n"
            "pval = (rep_var >= obs_var).mean()\n"
            "print(f'Observed Var(y) = {obs_var:.3f}')\n"
            "print(f'Replicated mean Var(y) = {rep_var.mean():.3f}  (PPP = {pval:.3f})')\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "ax.hist(rep_var, bins=40, color='#90caf9', edgecolor='white')\n"
            "ax.axvline(obs_var, color='crimson', lw=2, label='observed')\n"
            "ax.set_xlabel(r'Var$(y^{rep})$ across areas')\n"
            "ax.set_title('Posterior predictive check: cross-area variance of the direct estimator')\n"
            "ax.legend()\n"
            "fig.tight_layout()\n"
            "fig.savefig(FIG_DIR / '05_ppc_variance.png', dpi=120)\n"
            "fig\n"
        ),
        md(
            "## Key takeaways\n\n"
            "- The Bayesian $\\theta_i$ posterior mean **is** an EBLUP with $\\sigma_u^2$ "
            "integrated rather than plugged in; it should sit close to the REML EBLUP "
            "when $m$ is large.\n"
            "- Credible intervals automatically include uncertainty in $\\beta$ and "
            "$\\sigma_u^2$. They still assume $\\psi_i$ known and $u_i$ iid normal.\n"
            "- Prior disagreement (IG vs half-normal) is a feature: if Gibbs and NUTS "
            "diverge, the variance component is weakly identified and the EBLUP MSE "
            "formula is in the same fragile regime."
        ),
        footer("04_eblup.ipynb", "Classical EBLUP", "06_model_comparison.ipynb", "Model Comparison"),
    ]
    save("05_bayesian.ipynb", cells)


def nb06():
    cells = [
        nav(
            "Model comparison and validation",
            "05_bayesian.ipynb", "Bayesian Hierarchical Model",
            None, None,
            "Direct vs EBLUP vs Bayesian vs ONS published model-based estimates, "
            "with shrinkage diagnostics and local-authority choropleths.",
        ),
        code(IMPORTS),
        md(
            "## Putting four estimators on the same map\n\n"
            "For each LA in the modelling subset we now have:\n\n"
            "1. **Direct** APS rate $y_i$ with Wald interval from the published CI;\n"
            "2. **EBLUP** $\\hat\\theta}_i$ with Datta–Lahiri MSE;\n"
            "3. **Bayesian** posterior mean and 95% central interval (Gibbs);\n"
            "4. **ONS model-based** rate, the production figure for the same period.\n\n"
            "ONS is not truth. It is a competing (and more elaborate) estimate of "
            "the same $\\theta_i$. Close agreement means we have recovered the "
            "public-data skeleton of their model; systematic disagreement points to "
            "benchmarking, extra covariates, temporal smoothing, or a different "
            "variance function."
        ),
        code(
            "eblup = eblup_fh(MODEL, covariates=['claimant_rate'])\n"
            "gibbs = gibbs_fh(MODEL, covariates=['claimant_rate'], draws=4000, burn=1000, thin=2, seed=42)\n"
            "bayes = summarise_theta(gibbs['theta'], index=MODEL.index)\n"
            "cmp = eblup.copy()\n"
            "cmp = cmp.join(bayes)\n"
            "cmp['direct_se'] = cmp['direct_rate_ci'] / 1.96\n"
            "cmp['direct_lo'] = cmp['direct_rate'] - 1.96 * cmp['direct_se']\n"
            "cmp['direct_hi'] = cmp['direct_rate'] + 1.96 * cmp['direct_se']\n"
            "\n"
            "def metrics(a, b):\n"
            "    d = a - b\n"
            "    return pd.Series({\n"
            "        'bias': d.mean(),\n"
            "        'MAE': d.abs().mean(),\n"
            "        'RMSE': np.sqrt((d**2).mean()),\n"
            "        'corr': np.corrcoef(a, b)[0, 1],\n"
            "    })\n"
            "\n"
            "rows = {\n"
            "    'Direct vs ONS': metrics(cmp['direct_rate'], cmp['ons_mb_rate']),\n"
            "    'EBLUP vs ONS': metrics(cmp['eblup'], cmp['ons_mb_rate']),\n"
            "    'Bayes vs ONS': metrics(cmp['bayes_mean'], cmp['ons_mb_rate']),\n"
            "    'EBLUP vs Bayes': metrics(cmp['eblup'], cmp['bayes_mean']),\n"
            "}\n"
            "tab = pd.DataFrame(rows).T.round(3)\n"
            "print(tab.to_string())\n"
            "print('\\nONS interval coverage of our EBLUP (where ONS CI exists):')\n"
            "has = cmp['ons_mb_rate_ci'].notna()\n"
            "print('share of EBLUPs inside ONS 95% CI:',\n"
            "      coverage_rate(cmp.loc[has, 'eblup'],\n"
            "                    cmp.loc[has, 'ons_mb_rate'] - cmp.loc[has, 'ons_mb_rate_ci'],\n"
            "                    cmp.loc[has, 'ons_mb_rate'] + cmp.loc[has, 'ons_mb_rate_ci']))\n"
            "tab\n"
        ),
        code(
            "fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharex=True, sharey=True)\n"
            "pairs = [\n"
            "    ('direct_rate', 'Direct APS'),\n"
            "    ('eblup', 'EBLUP'),\n"
            "    ('bayes_mean', 'Bayesian mean'),\n"
            "]\n"
            "lims = [cmp[['direct_rate','eblup','bayes_mean','ons_mb_rate']].min().min(),\n"
            "        cmp[['direct_rate','eblup','bayes_mean','ons_mb_rate']].max().max()]\n"
            "for ax, (col, lab) in zip(axes, pairs):\n"
            "    ax.scatter(cmp['ons_mb_rate'], cmp[col], s=22, alpha=0.8, c=cmp['gamma'], cmap='coolwarm')\n"
            "    ax.plot(lims, lims, '--', color='grey', lw=1)\n"
            "    ax.set_xlabel('ONS model-based (%)')\n"
            "    ax.set_title(lab)\n"
            "    ax.set_ylabel('This project (%)' if ax is axes[0] else '')\n"
            "fig.tight_layout()\n"
            "fig.savefig(FIG_DIR / '06_vs_ons_scatter.png', dpi=120)\n"
            "fig\n"
        ),
        md(
            "## Where they diverge\n\n"
            "Largest $|\\hat{\\theta}^{\\mathrm{EBLUP}} - \\hat{\\theta}^{\\mathrm{ONS}}|$ "
            "areas are the interesting residuals: possible extra ONS covariates, "
            "regional benchmarking, or a different $\\psi_i$."
        ),
        code(
            "cmp['abs_diff_ons'] = (cmp['eblup'] - cmp['ons_mb_rate']).abs()\n"
            "cmp['signed_diff'] = cmp['eblup'] - cmp['ons_mb_rate']\n"
            "print(cmp.nlargest(8, 'abs_diff_ons')[\n"
            "    ['GEOGRAPHY_NAME','country','direct_rate','eblup','bayes_mean','ons_mb_rate','gamma','psi']\n"
            "].round(2).to_string(index=False))\n"
        ),
        md(
            "## Interval comparison\n\n"
            "If Bayes is doing its job, credible intervals should be **narrower** than "
            "direct Wald intervals, especially at small $\\gamma_i$, and roughly "
            "comparable to EBLUP MSE intervals. They will not match ONS intervals "
            "unless the production variance function is the same."
        ),
        code(
            "cmp['width_direct'] = cmp['direct_hi'] - cmp['direct_lo']\n"
            "cmp['width_eblup'] = cmp['eblup_ci_hi'] - cmp['eblup_ci_lo']\n"
            "cmp['width_bayes'] = cmp['bayes_ci_hi'] - cmp['bayes_ci_lo']\n"
            "print(cmp[['width_direct','width_eblup','width_bayes']].median().round(2))\n"
            "fig, ax = plt.subplots(figsize=(8, 5))\n"
            "ax.scatter(cmp['gamma'], cmp['width_direct'], s=16, alpha=0.5, label='Direct 95%')\n"
            "ax.scatter(cmp['gamma'], cmp['width_eblup'], s=16, alpha=0.7, label='EBLUP 95%')\n"
            "ax.scatter(cmp['gamma'], cmp['width_bayes'], s=16, alpha=0.7, label='Bayes 95%')\n"
            "ax.set_xlabel(r'$\\gamma_i$ (weight on direct)')\n"
            "ax.set_ylabel('Interval width (percentage points)')\n"
            "ax.set_title('Precision gain is concentrated in the noisy (low-γ) areas')\n"
            "ax.legend()\n"
            "fig.tight_layout()\n"
            "fig.savefig(FIG_DIR / '06_interval_width.png', dpi=120)\n"
            "fig\n"
        ),
        md("## Choropleths"),
        code(
            "import geopandas as gpd\n"
            "\n"
            "geo_path = DATA_DIR / 'la_boundaries.geojson'\n"
            "gdf = gpd.read_file(geo_path)\n"
            "gdf = gdf.merge(\n"
            "    cmp[['GEOGRAPHY_CODE','GEOGRAPHY_NAME','direct_rate','eblup','bayes_mean','ons_mb_rate','gamma','signed_diff']],\n"
            "    on='GEOGRAPHY_CODE', how='left',\n"
            ")\n"
            "plot_df = gdf.dropna(subset=['eblup'])\n"
            "\n"
            "fig, axes = plt.subplots(2, 2, figsize=(12, 14))\n"
            "specs = [\n"
            "    (axes[0,0], 'direct_rate', 'Direct APS'),\n"
            "    (axes[0,1], 'eblup', 'EBLUP'),\n"
            "    (axes[1,0], 'bayes_mean', 'Bayesian mean'),\n"
            "    (axes[1,1], 'ons_mb_rate', 'ONS model-based'),\n"
            "]\n"
            "vmin = plot_df[['direct_rate','eblup','bayes_mean','ons_mb_rate']].min().min()\n"
            "vmax = plot_df[['direct_rate','eblup','bayes_mean','ons_mb_rate']].max().max()\n"
            "for ax, col, title in specs:\n"
            "    plot_df.plot(column=col, ax=ax, cmap='YlOrRd', vmin=vmin, vmax=vmax, linewidth=0.1, edgecolor='grey', legend=True)\n"
            "    ax.set_axis_off()\n"
            "    ax.set_title(title)\n"
            "fig.suptitle('Unemployment rate (%) — modelling subset', y=0.99)\n"
            "fig.tight_layout()\n"
            "fig.savefig(FIG_DIR / '06_choropleth_rates.png', dpi=120)\n"
            "fig\n"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(8, 10))\n"
            "plot_df.plot(column='signed_diff', ax=ax, cmap='RdBu_r', legend=True,\n"
            "             linewidth=0.1, edgecolor='grey',\n"
            "             legend_kwds={'label': 'EBLUP − ONS (pp)'})\n"
            "ax.set_axis_off()\n"
            "ax.set_title('Where this EBLUP diverges from ONS model-based estimates')\n"
            "fig.tight_layout()\n"
            "fig.savefig(FIG_DIR / '06_choropleth_diff.png', dpi=120)\n"
            "fig\n"
        ),
        md(
            "## Interactive residual view"
        ),
        code(
            "fig = px.scatter(\n"
            "    cmp, x='ons_mb_rate', y='eblup',\n"
            "    color='gamma', size='psi',\n"
            "    hover_name='GEOGRAPHY_NAME',\n"
            "    hover_data={'direct_rate':':.1f', 'bayes_mean':':.1f', 'country': True},\n"
            "    labels={'ons_mb_rate': 'ONS model-based (%)', 'eblup': 'EBLUP (%)', 'gamma': 'γ'},\n"
            "    title='EBLUP vs ONS (size = ψ, colour = γ)',\n"
            "    color_continuous_scale='RdBu',\n"
            ")\n"
            "lims = [cmp[['ons_mb_rate','eblup']].min().min(), cmp[['ons_mb_rate','eblup']].max().max()]\n"
            "fig.add_trace(go.Scatter(x=lims, y=lims, mode='lines', name='y = x',\n"
            "                         line=dict(dash='dash', color='grey')))\n"
            "display_plotly(fig)\n"
        ),
        md(
            "## When model-based estimation helps\n\n"
            "The payoff is concentrated where $\\gamma_i$ is small: the EBLUP and Bayes "
            "intervals are shorter than the direct interval, and both sit closer to ONS "
            "than $y_i$ does. Where $\\gamma_i \\approx 1$, all four series essentially "
            "reproduce the APS — as they should. Publishing a model-based figure for "
            "Birmingham is a different proposition from publishing one for a small "
            "district, and the shrinkage factor makes that distinction **visible**.\n\n"
            "Disagreement with ONS, where it occurs, is more likely\n\n"
            "- **benchmarking**: ONS may calibrate LA estimates to regional APS totals, "
            "which our independent FH does not;\n"
            "- **the variance function**: we take published CIs as $\\psi_i$, ONS "
            "estimates $\\psi_i$ from survey design internally;\n"
            "- **covariates and dynamics**: a single claimant snapshot vs a production "
            "model that can include lagged rates or additional admin sources;\n"
            "- **geography**: mid-year boundary changes and GB vs UK coverage.\n\n"
            "Those gaps are expected. The useful result is that a transparent, "
            "reproducible Fay–Herriot on open aggregates recovers the same **map** "
            "and the same **shrinkage pattern** as the official series."
        ),
        md(
            "## Key takeaways\n\n"
            "- EBLUP and Bayes posterior means are nearly interchangeable at this $m$; "
            "the Bayesian contribution is the posterior of $\\sigma_u^2$ and honest "
            "propagation into $\\theta_i$ intervals.\n"
            "- Both improve on the direct estimator as a predictor of the ONS series, "
            "which is the right check for an open reimplementation.\n"
            "- Model-based estimation earns its keep in the high-$\\psi_i$ tail — "
            "exactly the areas the APS cannot publish with a straight face."
        ),
        footer("05_bayesian.ipynb", "Bayesian Hierarchical Model", None, None),
    ]
    save("06_model_comparison.ipynb", cells)


if __name__ == "__main__":
    nb01()
    nb02()
    nb03()
    nb04()
    nb05()
    nb06()
