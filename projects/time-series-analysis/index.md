# Time Series Analysis: Classical Methods

<div style="background: linear-gradient(135deg, #1565c0 0%, #00838f 100%); color: white; padding: 2em; border-radius: 10px; margin: 2em 0; text-align: center;">
  <h2 style="color: white; margin-top: 0;">📈 Decomposing Time, Forecasting the Future</h2>
  <p style="font-size: 1.1em; margin-bottom: 0;">A comprehensive tour of classical time series methods applied to benchmark datasets</p>
</div>

## Project Overview

Time series data — sequences of observations indexed by time — appear throughout science, economics, and industry. This project provides a rigorous yet accessible walkthrough of the classical toolkit: from the humble exponential smoother through to full ARIMA-family models and a brief introduction to state-space representations. Each method is motivated theoretically, implemented in Python, and applied to canonical benchmark datasets so that behaviour is immediately interpretable.

Datasets used include:

- **Air Passengers** (Box & Jenkins, 1976): monthly international airline passenger counts 1949–1960 — the *drosophila* of time series, exhibiting clear trend, multiplicative seasonality, and increasing variance
- **Monthly Milk Production** (Cryer, 1986): monthly pounds of milk per cow 1962–1975 — additive seasonality with linear trend
- **Sunspot Numbers** (WDC-SILSO): annual sunspot counts from 1700 — stationary-ish with quasi-periodic ~11-year solar cycle; classic AR/ARMA showcase
- **Mauna Loa CO₂** (Keeling, NOAA): monthly atmospheric CO₂ 1958–present — strong trend plus annual seasonality; good for Fourier decomposition
- **UK Retail Sales** (ONS): monthly index values — real-world irregularity, trading-day effects, and the challenges of official statistics

## Project Structure

This analysis is organised into eight interconnected notebooks:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5em; margin: 2em 0;">

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #1565c0;">
  <h3 style="margin-top: 0; color: #1565c0;">📐 1. Introduction & Data Loading</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="01_introduction.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Loading datasets, visualising raw series, ACF/PACF, classical and STL decomposition</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #00838f;">
  <h3 style="margin-top: 0; color: #00838f;">📉 2. Simple Exponential Smoothing</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="02_ses.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Level smoothing, α optimisation, ARIMA(0,1,1) equivalence</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #2e7d32;">
  <h3 style="margin-top: 0; color: #2e7d32;">🌡️ 3. Holt-Winters Smoothing</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="03_holt_winters.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Trend and seasonal extensions, ETS framework, AIC model selection</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #6a1b9a;">
  <h3 style="margin-top: 0; color: #6a1b9a;">🔊 4. Fourier Analysis</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="04_fourier.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">DFT, periodograms, spectral density, Fourier regressors</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #e65100;">
  <h3 style="margin-top: 0; color: #e65100;">🔁 5. ARMA & ARIMA</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="05_arma_arima.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Stationarity testing, Box-Jenkins methodology, AR/MA identification</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #0277bd;">
  <h3 style="margin-top: 0; color: #0277bd;">❄️ 6. SARIMA</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="06_sarima.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Seasonal differencing, airline model, auto-ARIMA, production seasonal adjustment</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #4527a0;">
  <h3 style="margin-top: 0; color: #4527a0;">🔮 7. State-Space Models</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="07_state_space.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Kalman filter, structural time series, Harvey's framework</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #c62828;">
  <h3 style="margin-top: 0; color: #c62828;">🏆 8. Model Comparison</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="08_model_comparison.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Cross-validation, forecast metrics, Diebold-Mariano test</p>
</div>

</div>

## Key Questions

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; margin: 1.5em 0;">

- 🔍 How does the choice of smoothing parameter α affect forecast responsiveness vs stability?
- 🔍 What does the autocorrelation structure of a residual series tell us about model adequacy?
- 🔍 When does seasonal differencing outperform explicit Fourier seasonality?
- 🔍 How do ARIMA and structural state-space models relate mathematically, and do they diverge in practice?
- 🔍 Which classical method produces the best out-of-sample forecasts on benchmark data?

</div>

## Datasets

| Dataset | Frequency | Length | Key Features |
|---|---|---|---|
| Air Passengers | Monthly | 144 obs (1949–1960) | Multiplicative trend+season, growing variance |
| Milk Production | Monthly | 168 obs (1962–1975) | Additive trend+season |
| Sunspots | Annual | 300+ obs (1700–present) | Quasi-periodic, near-stationarity |
| Mauna Loa CO₂ | Monthly | 700+ obs (1958–present) | Strong nonlinear trend, annual seasonality |
| UK Retail Sales | Monthly | ~300 obs | Real-world noise, calendar effects |

## Technical Approach

<div style="background: #fff5e6; padding: 1.5em; border-radius: 8px; border-left: 4px solid #ff7f0e; margin: 1.5em 0;">

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1em; margin-top: 1em;">

<div>
  <strong>📊 Exploratory Decomposition</strong><br>
  <span style="font-size: 0.9em; color: #666;">Classical additive/multiplicative decomposition, STL</span>
</div>

<div>
  <strong>📐 Stationarity Testing</strong><br>
  <span style="font-size: 0.9em; color: #666;">ADF, KPSS, variance-ratio tests</span>
</div>

<div>
  <strong>🔧 Model Fitting</strong><br>
  <span style="font-size: 0.9em; color: #666;">statsmodels, pmdarima auto-ARIMA</span>
</div>

<div>
  <strong>🔊 Spectral Analysis</strong><br>
  <span style="font-size: 0.9em; color: #666;">scipy.signal, manual DFT, Fourier regressors</span>
</div>

<div>
  <strong>✅ Validation</strong><br>
  <span style="font-size: 0.9em; color: #666;">Time-series cross-validation, rolling windows</span>
</div>

</div>

</div>

## Expected Outcomes

<div style="background: #e8f5e9; padding: 1.5em; border-radius: 8px; margin: 1.5em 0;">

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1em; margin-top: 1em;">

<div>
  ✅ <strong>Understood</strong> the theoretical basis of each classical method
</div>

<div>
  ✅ <strong>Identified</strong> which method suits which data generating process
</div>

<div>
  ✅ <strong>Produced</strong> interpretable, well-diagnosed models
</div>

<div>
  ✅ <strong>Benchmarked</strong> all methods head-to-head on common datasets
</div>

</div>

</div>

---

<div style="text-align: center; margin: 2em 0; padding: 1.5em; background: #f0f0f0; border-radius: 8px;">
  <p style="font-size: 1.2em; margin: 0;"><strong>Ready to explore?</strong></p>
  <p style="margin: 0.5em 0 0 0;"><a href="01_introduction.html" style="background: #1565c0; color: white; padding: 0.7em 2em; text-decoration: none; border-radius: 5px; display: inline-block;">Start with Introduction & Data Loading →</a></p>
</div>
