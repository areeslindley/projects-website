# Small Area Estimation for UK Local Authorities

<div style="background: linear-gradient(135deg, #0d3b4c 0%, #2a9d8f 100%); color: white; padding: 2em; border-radius: 10px; margin: 2em 0; text-align: center;">
  <h2 style="color: white; margin-top: 0;">Borrowing strength when the sample runs out</h2>
  <p style="font-size: 1.1em; margin-bottom: 0;">A Fay–Herriot area-level model for APS unemployment rates, compared with ONS’s published model-based estimates</p>
</div>

## Project Overview

Direct survey estimates of unemployment become too volatile to publish once the Annual Population Survey is broken down to local authorities: the numerator sample is often a handful of unemployed people. ONS has long addressed this with **model-based estimates** that borrow strength from the claimant count. This project is an open-source reimplementation of that class of model — classical EBLUP and a Bayesian hierarchical version — validated against the Nomis-published official series.

The analysis uses only openly licensed aggregates (no Accredited Researcher / SDS access). Sampling variances $\psi_i$ are recovered from published 95% confidence intervals. The linking model is the textbook Fay–Herriot specification: a linear mean in the claimant-count rate and iid area random effects.

## Project Structure

Six notebooks build the analysis step by step:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5em; margin: 2em 0;">

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #0d3b4c;">
  <h3 style="margin-top: 0; color: #0d3b4c;">1. Introduction</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="01_introduction.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Why direct estimates fail; Fay–Herriot shrinkage; EBLUP vs Bayes</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #145c69;">
  <h3 style="margin-top: 0; color: #145c69;">2. Data acquisition</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="02_data_acquisition.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Nomis API: APS rates &amp; CIs, claimant count, ONS model-based series</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #1a7a7a;">
  <h3 style="margin-top: 0; color: #1a7a7a;">3. Direct estimates</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="03_direct_estimates.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Construct $y_i$ and $\psi_i$; instability versus area size</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #2a9d8f;">
  <h3 style="margin-top: 0; color: #2a9d8f;">4. Classical EBLUP</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="04_eblup.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">REML $\sigma_u^2$, Prasad–Rao / Datta–Lahiri MSE, jackknife</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #1b6b8a;">
  <h3 style="margin-top: 0; color: #1b6b8a;">5. Bayesian FH</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="05_bayesian.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Gibbs sampler and PyMC NUTS; posterior of $\theta_i$ and $\sigma_u^2$</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #c45c26;">
  <h3 style="margin-top: 0; color: #c45c26;">6. Validation</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="06_model_comparison.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Direct vs EBLUP vs Bayes vs ONS; choropleths; shrinkage</p>
</div>

</div>

## Key Questions

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; margin: 1.5em 0;">

- When does the APS local-authority unemployment rate stop being a publication-grade statistic?
- How much should a noisy area shrink toward the claimant-count regression, and does that match ONS?
- Do REML EBLUPs and Bayesian posterior means disagree once $\sigma_u^2$ is uncertain?
- Where do an open Fay–Herriot and the official model-based series diverge — and why might that be expected?

</div>

## Preview Figures

![Shrinkage weight vs sampling variance](figures/01_shrinkage_weight.png)

![Direct-estimate CV vs area size](figures/03_cv_vs_size.png)

![EBLUP vs ONS model-based estimates](figures/06_vs_ons_scatter.png)

## Dataset

<div style="background: #e8f4f8; padding: 1.5em; border-radius: 8px; margin: 1.5em 0;">

- **Direct estimates:** Nomis APS (`NM_17_5`, unemployment rate aged 16–64) with published confidence intervals
- **Benchmark:** Nomis model-based estimates of unemployment (`NM_127_1`)
- **Auxiliary:** Claimant count as % of residents aged 16–64 (`NM_162_1`); APS inactivity as a sensitivity covariate
- **Boundaries:** ONS Open Geography LAD December 2023 BGC
- **Licence:** [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) — see `data/ATTRIBUTION.txt`

Refresh the bundled panel with `python projects/small-area-estimation/_build_data.py`.

</div>

## Technical Stack

**Python** • Fay–Herriot • REML EBLUP • PyMC • Nomis API • geopandas • Plotly • Official Statistics

---

<div style="text-align: center; margin: 2em 0; padding: 1.5em; background: #f0f0f0; border-radius: 8px;">
  <p style="font-size: 1.2em; margin: 0;"><strong>Ready to explore?</strong></p>
  <p style="margin: 0.5em 0 0 0;"><a href="01_introduction.html" style="background: #0d3b4c; color: white; padding: 0.7em 2em; text-decoration: none; border-radius: 5px; display: inline-block;">Start with Introduction →</a></p>
</div>
