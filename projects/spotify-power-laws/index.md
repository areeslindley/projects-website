# Power Laws and Popularity

<div style="background: linear-gradient(135deg, #0d3b4c 0%, #c45c26 100%); color: white; padding: 2em; border-radius: 10px; margin: 2em 0; text-align: center;">
  <h2 style="color: white; margin-top: 0;">What 500 artists can and cannot tell you about streaming success</h2>
  <p style="font-size: 1.1em; margin-bottom: 0;">A synthetic Spotify catalog, treated as a small-<em>n</em> inference problem wearing an ML costume</p>
</div>

<div style="background: #fff3e0; padding: 1.2em 1.5em; border-radius: 8px; border-left: 4px solid #e65100; margin: 1.5em 0;">
<strong>Synthetic data.</strong> The Kaggle file is a 50,000-row track table with 500 invented artist names. The publisher states that no Spotify API data were used. Nothing below is a claim about the streaming industry.
</div>

## Project Overview

Kaggle will sell you 50,000 rows and 33 columns of "Spotify artist streaming analytics, 2020–2025". That looks like a machine-learning problem. It is not. After a schema check the grain is **track**, not artist-year; the names are generated; audio features are orthogonal to streams; and popularity is almost `log1p` of the target. The interesting work is estimand definition, a leakage audit, validation that matches the clustering, and stopping when extra model capacity stops paying.

The honest conclusion is the one worth publishing: **flexible models buy nothing at this generating process, leaked features buy \(R^2 = 1\), and the planted superstar tail is catalog size rather than quality.**

## The page

One executable chapter. Code lives in `src/spotify_powerlaws/` and numbered scripts; the notebook loads artifacts.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5em; margin: 2em 0;">

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #0d3b4c;">
  <h3 style="margin-top: 0; color: #0d3b4c;">Analysis</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="01_analysis.html">Read the page →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Forensics, estimand, grouped CV, model ladder, power-law tests, conformal coverage</p>
</div>

</div>

## Key questions

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; margin: 1.5em 0;">

- Is this an artist panel, a track catalog, or a synthetic script with a Spotify costume?
- Which columns are antecedent to streams, and which are the target in disguise?
- Does grouped-by-artist CV change the honest model — or only fail to save a tautological one?
- When does added capacity stop paying, and does partial pooling find artist effects the ANOVA already said were zero?

</div>

## Preview figures

![Forensics dashboard](figures/01_forensics.png)

![Leakage gap](figures/02_leakage_gap.png)

![Lorenz curves](figures/04_lorenz.png)

## Dataset

<div style="background: #e8f4f8; padding: 1.5em; border-radius: 8px; margin: 1.5em 0;">

- **Source:** [Kaggle: Spotify Artist Streaming Analytics 2020–2025](https://www.kaggle.com/datasets/beamhonor0911/spotify-artist-streaming-analytics-20202025) (CC BY 4.0)
- **File:** 50,000 tracks × 33 columns; 500 invented artists
- **Grain:** track (`track_id` unique). Not artist-year, not a listening panel
- **Rebuild:** `make download` writes `data/raw/` (gitignored). SHA-256 in `artifacts/download.json`
- **This site:** notebooks read committed `artifacts/` and `figures/` only

</div>

## Technical stack

**Python** • pandas • scikit-learn • statsmodels MixedLM • matplotlib • scipy

Pipeline: `projects/spotify-power-laws/Makefile` (`00_download.py` … `08_figures.py`)

---

<div style="text-align: center; margin: 2em 0; padding: 1.5em; background: #f0f0f0; border-radius: 8px;">
  <p style="font-size: 1.2em; margin: 0;"><strong>Start with the forensics.</strong></p>
  <p style="margin: 0.5em 0 0 0;"><a href="01_analysis.html" style="background: #0d3b4c; color: white; padding: 0.7em 2em; text-decoration: none; border-radius: 5px; display: inline-block;">Open the analysis →</a></p>
</div>
