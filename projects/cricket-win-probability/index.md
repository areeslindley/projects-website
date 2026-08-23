# ODI Win Probability: From DLS to Data

<div style="background: linear-gradient(135deg, #1b5e20 0%, #f9a825 100%); color: white; padding: 2em; border-radius: 10px; margin: 2em 0; text-align: center;">
  <h2 style="color: white; margin-top: 0;">Ball-by-ball win probability for one-day cricket</h2>
  <p style="font-size: 1.1em; margin-bottom: 0;">Resource-based baselines and calibrated machine-learning models applied to ODI chases</p>
</div>

## Project Overview

Live win-probability gauges have become a staple of cricket broadcasts — but how are those numbers computed? This project walks through the statistical machinery behind **P(batting team wins | current state)** in ODIs, from the resource logic underlying Duckworth–Lewis–Stern (DLS) rain rules to data-driven models trained on ball-by-ball history.

The analysis uses [Cricsheet](https://cricsheet.org/) ODI data parsed into game-state features at every ball of the second innings. A simplified DLS-style resource table provides an interpretable baseline; logistic regression and LightGBM models are trained on historical chases and compared on calibration metrics. Interactive Plotly charts showcase famous World Cup matches.

> **Disclaimer:** The DLS implementation here is an educational approximation — not the ICC-official calculator.

## Project Structure

Six notebooks build the analysis step by step:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5em; margin: 2em 0;">

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #1b5e20;">
  <h3 style="margin-top: 0; color: #1b5e20;">1. Introduction & Data</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="01_introduction.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">ODI scoring, win probability definition, Cricsheet JSON parsing</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #2e7d32;">
  <h3 style="margin-top: 0; color: #2e7d32;">2. Game State Features</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="02_game_state.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">RRR, wickets remaining, phase indicators, feature correlations</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #558b2f;">
  <h3 style="margin-top: 0; color: #558b2f;">3. DLS & Resources</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="03_dls_resources.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Resource tables, par scores, DLS-style win probability baseline</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #f9a825;">
  <h3 style="margin-top: 0; color: #f57f17;">4. Modelling</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="04_modelling.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Logistic regression, LightGBM, Brier score vs DLS</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #1565c0;">
  <h3 style="margin-top: 0; color: #1565c0;">5. Live Charts</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="05_live_charts.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Interactive Plotly win-probability lines for famous chases</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #6a1b9a;">
  <h3 style="margin-top: 0; color: #6a1b9a;">6. Evaluation</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="06_evaluation_case_studies.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Calibration plots, ML–DLS disagreement, case-study summary</p>
</div>

</div>

## Key Questions

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; margin: 1.5em 0;">

- When does a chase become statistically unlikely?
- How well does a resource table approximate historical outcomes?
- Does a machine-learning model improve on DLS-style probabilities?
- Which famous chases had the steepest probability swings?

</div>

## Preview Figures

![Cumulative runs — 2019 World Cup Final](figures/01_innings_runs.png)

![Win probability — 2019 World Cup Final chase](figures/05_wc2019_winprob.png)

![Calibration diagram](figures/06_calibration.png)

## Dataset

<div style="background: #e8f4f8; padding: 1.5em; border-radius: 8px; margin: 1.5em 0;">

- **Primary source:** [Cricsheet ODI ball-by-ball JSON](https://cricsheet.org/)
- **Bundled case studies:** 2019 World Cup Final, 2011 WC QF (India vs Australia), 2011 WC upset (Ireland vs England)
- **Training sample:** 400 ODIs, ~120,000 chase balls (see `data/odi_training_sample.csv`)

</div>

## Technical Stack

**Python** • Cricsheet • scikit-learn • LightGBM • Plotly • pandas • Sports Analytics • Calibration

---

<div style="text-align: center; margin: 2em 0; padding: 1.5em; background: #f0f0f0; border-radius: 8px;">
  <p style="font-size: 1.2em; margin: 0;"><strong>Ready to explore?</strong></p>
  <p style="margin: 0.5em 0 0 0;"><a href="01_introduction.html" style="background: #1b5e20; color: white; padding: 0.7em 2em; text-decoration: none; border-radius: 5px; display: inline-block;">Start with Introduction →</a></p>
</div>
