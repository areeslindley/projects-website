"""Generate time series analysis notebooks. Run once from repo root."""
import json
from pathlib import Path

PROJ = Path(__file__).parent
KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"},
}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def nav_header(title, prev_link, prev_title, next_link, next_title, desc):
    prev_part = f"[← Previous: {prev_title}]({prev_link})" if prev_link else ""
    next_part = f"[Next: {next_title} →]({next_link})" if next_link else ""
    sep = " | " if prev_part and next_part else ""
    return md(
        f"# {title}\n\n"
        f"**Navigation**: {prev_part}{sep}{next_part}\n\n"
        f"{desc}\n"
    )


def nav_footer(prev_link, prev_title, next_link, next_title):
    prev_part = f"[← Previous: {prev_title}]({prev_link})" if prev_link else ""
    next_part = f"[Next: {next_title} →]({next_link})" if next_link else ""
    sep = " | " if prev_part and next_part else ""
    return md(f"---\n\n**Navigation**: {prev_part}{sep}{next_part}\n")


def save(name, cells):
    nb = {"cells": cells, "metadata": KERNEL, "nbformat": 4, "nbformat_minor": 5}
    path = PROJ / name
    path.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {path.name}")


LOADERS = '''
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

DATA_DIR = Path('data')
if not DATA_DIR.exists():
    DATA_DIR = Path('projects/time-series-analysis/data')

def load_series(name):
    """Load a bundled CSV as a DatetimeIndex Series."""
    df = pd.read_csv(DATA_DIR / name, parse_dates=['date'])
    return df.set_index('date')['value'].sort_index()

def load_air_passengers():
    path = DATA_DIR / 'air_passengers.csv'
    if path.exists():
        return load_series('air_passengers.csv')
    from statsmodels.datasets import get_rdataset
    air = get_rdataset('AirPassengers', 'datasets').data
    s = air['value']
    s.index = pd.date_range('1949-01', periods=len(s), freq='MS')
    return s

def load_milk():
    path = DATA_DIR / 'monthly_milk.csv'
    if path.exists():
        return load_series('monthly_milk.csv')
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-milk-production-pounds.csv'
    return pd.read_csv(url, index_col=0, parse_dates=True).squeeze()

def load_co2():
    from statsmodels.datasets import co2
    return co2.load().data.resample('MS').mean().ffill().squeeze()

def load_sunspots():
    from statsmodels.datasets import sunspots
    d = sunspots.load_pandas().data
    return d.set_index('YEAR')['SUNACTIVITY']

def load_retail():
    path = DATA_DIR / 'uk_retail_sales.csv'
    if path.exists():
        return load_series('uk_retail_sales.csv')
    raise FileNotFoundError('uk_retail_sales.csv not found in data/')
'''

# --- 01 Introduction ---
cells01 = [
    nav_header(
        "Introduction & Data Loading",
        "index.md", "Project Overview",
        "02_ses.ipynb", "Simple Exponential Smoothing",
        "This notebook loads five benchmark time series, visualises their structure, "
        "and introduces decomposition and autocorrelation as exploratory tools. "
        "Classical methods — as opposed to modern ML forecasters — assume explicit "
        "structure (trend, seasonality, autocorrelation) that we can diagnose visually and statistically.",
    ),
    code(LOADERS.strip()),
    md("## Datasets\n\nWe load five canonical series covering different frequencies and features."),
    code("""
# Load all five benchmark datasets
air = load_air_passengers()
milk = load_milk()
co2 = load_co2()
spots = load_sunspots()
retail = load_retail()

datasets = {
    'Air Passengers': air,
    'Milk Production': milk,
    'Mauna Loa CO2': co2,
    'Sunspots': spots,
    'UK Retail Sales': retail,
}
for name, s in datasets.items():
    freq = pd.infer_freq(s.index[:3]) if isinstance(s.index, pd.DatetimeIndex) else 'annual'
    print(f"{name:20s} n={len(s):4d}  freq={freq or 'irregular'}")
"""),
    md("## Visualisation\n\nTime plots reveal trend, seasonality, and changing variance at a glance."),
    code("""
fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 2.5 * len(datasets)), sharex=False)
for ax, (name, s) in zip(axes, datasets.items()):
    s.plot(ax=ax, color='steelblue', lw=1.2)
    ax.set_title(name)
    ax.set_ylabel('Value')
fig.tight_layout()
fig
"""),
    md(
        "## Time Series Features\n\n"
        "- **Trend**: long-run increase or decrease (Air Passengers, CO₂)\n"
        "- **Seasonality**: fixed-period cycles (monthly period 12, annual period ~11 for sunspots)\n"
        "- **Cyclicality**: non-fixed period oscillations\n"
        "- **Irregularity**: noise and one-off events (UK Retail Sales)\n\n"
        "Weak stationarity requires constant mean, constant variance, and autocovariance "
        "depending only on lag — rarely satisfied in raw economic series."
    ),
    md("## Autocorrelation\n\nACF and PACF summarise linear dependence at each lag."),
    code("""
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
series_list = list(datasets.items())[:3]
for i, (name, s) in enumerate(series_list):
    plot_acf(s.dropna(), ax=axes[0, i], lags=min(40, len(s)//2 - 1), title=f'ACF: {name}')
    plot_pacf(s.dropna(), ax=axes[1, i], lags=min(40, len(s)//2 - 1), title=f'PACF: {name}', method='ywm')
fig.tight_layout()
fig
"""),
    md("## Classical Decomposition\n\nAdditive: $y_t = T_t + S_t + R_t$. Multiplicative when seasonal amplitude grows with level."),
    code("""
from statsmodels.tsa.seasonal import seasonal_decompose

# Air Passengers — multiplicative seasonality
decomp_mul = seasonal_decompose(air, model='multiplicative', period=12)
fig = decomp_mul.plot()
fig.set_size_inches(12, 8)
fig.suptitle('Air Passengers — Multiplicative Decomposition', y=1.02)
fig
"""),
    md("## STL Decomposition\n\nSTL (Seasonal-Trend decomposition using Loess) is robust to outliers and handles evolving seasonality."),
    code("""
from statsmodels.tsa.seasonal import STL

stl = STL(milk, period=12, robust=True).fit()
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
milk.plot(ax=axes[0], title='Observed', color='steelblue')
stl.trend.plot(ax=axes[1], title='Trend', color='darkorange')
stl.seasonal.plot(ax=axes[2], title='Seasonal', color='seagreen')
stl.resid.plot(ax=axes[3], title='Remainder', color='gray')
fig.tight_layout()
fig
"""),
    md(
        "## Key Takeaways\n\n"
        "- **Benchmark diversity**: Each dataset highlights different features — multiplicative season (Air Passengers), "
        "additive season (Milk), quasi-periodicity (Sunspots), trend+season (CO₂), real-world noise (Retail).\n"
        "- **ACF/PACF patterns**: Slow decay in ACF indicates non-stationarity; spikes at lag 12 signal monthly seasonality.\n"
        "- **Decomposition choice**: Multiplicative when seasonal swings grow with level; STL preferred when outliers or evolving season matter.\n"
        "- **Next steps**: With features identified, we proceed to modelling — starting with the simplest forecaster, SES."
    ),
    nav_footer("index.md", "Project Overview", "02_ses.ipynb", "Simple Exponential Smoothing"),
]

# --- 02 SES ---
cells02 = [
    nav_header(
        "Simple Exponential Smoothing (SES)",
        "01_introduction.ipynb", "Introduction",
        "03_holt_winters.ipynb", "Holt-Winters",
        "Simple Exponential Smoothing updates a level estimate with geometrically decaying weights. "
        "It is appropriate for series without trend or seasonality after preprocessing.",
    ),
    code(LOADERS.strip() + "\nfrom scipy.optimize import minimize_scalar\nfrom statsmodels.tsa.holtwinters import SimpleExpSmoothing\nfrom statsmodels.tsa.arima.model import ARIMA"),
    md(
        "## The SES Recurrence\n\n"
        "$$\\hat{y}_{t+1|t} = \\alpha y_t + (1-\\alpha)\\hat{y}_{t|t-1}$$\n\n"
        "Expanded: $\\ell_t = \\sum_{j=0}^{t-1} \\alpha(1-\\alpha)^j y_{t-j} + (1-\\alpha)^t \\ell_0$. "
        "Forecasts are flat: $\\hat{y}_{T+h|T} = \\ell_T$."
    ),
    code("""
# Weight profiles for different alpha values
alphas = [0.1, 0.3, 0.7]
lags = np.arange(30)
fig, ax = plt.subplots(figsize=(10, 6))
for a in alphas:
    weights = a * (1 - a) ** lags
    ax.plot(lags, weights, label=f'alpha={a}, half-life={np.log(0.5)/np.log(1-a):.1f} lags')
ax.set_xlabel('Lag j')
ax.set_ylabel('Weight on y_{t-j}')
ax.set_title('Geometric Decay of SES Weights')
ax.legend()
fig
"""),
    md("## Manual SSE Optimisation\n\nWe minimise one-step-ahead sum of squared errors to estimate $\\alpha$."),
    code("""
def ses_forecast(y, alpha, l0=None):
    l0 = l0 if l0 is not None else y[0]
    level = l0
    preds = []
    for obs in y:
        preds.append(level)
        level = alpha * obs + (1 - alpha) * level
    return np.array(preds), level

def ses_sse(alpha, y):
    preds, _ = ses_forecast(y, alpha)
    return np.sum((y[1:] - preds[:-1]) ** 2)

spots = load_sunspots().astype(float)
# Use a stationary window (recent sunspot cycle)
y = spots.loc[1900:].values
result = minimize_scalar(ses_sse, bounds=(0.01, 0.99), method='bounded', args=(y,))
alpha_hat = result.x
print(f'Optimal alpha (manual): {alpha_hat:.4f}')
"""),
    code("""
# Compare manual vs statsmodels SES
model = SimpleExpSmoothing(y, initialization_method='estimated').fit(optimized=True)
print(f'Optimized alpha (statsmodels): {model.params["smoothing_level"]:.4f}')

manual_preds, _ = ses_forecast(y, alpha_hat)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y, label='Observed', alpha=0.7)
ax.plot(manual_preds, label='Manual SES', ls='--')
ax.plot(model.fittedvalues, label='statsmodels SES', ls=':')
ax.set_title('SES on Sunspots (1900+)')
ax.legend()
fig
"""),
    md("## ARIMA(0,1,1) Equivalence\n\nSES on differenced data is equivalent to ARIMA(0,1,1) with $\\theta = 1 - \\alpha$."),
    code("""
milk = load_milk().values
alpha_m = minimize_scalar(ses_sse, bounds=(0.01, 0.99), method='bounded', args=(milk,)).x
arima = ARIMA(milk, order=(0, 1, 1)).fit()
params = arima.params
theta = params['ma.L1'] if hasattr(params, 'index') and 'ma.L1' in params.index else float(params[-1])
print(f'SES alpha: {alpha_m:.4f}  <=>  ARIMA theta: {theta:.4f}  (1-alpha = {1-alpha_m:.4f})')
"""),
    md(
        "## Key Takeaways\n\n"
        "- **Alpha controls memory**: Higher $\\alpha$ reacts faster to shocks; lower $\\alpha$ gives smoother level estimates.\n"
        "- **Flat forecasts**: SES cannot extrapolate trend — inappropriate for Air Passengers without differencing.\n"
        "- **ARIMA link**: SES $\\equiv$ ARIMA(0,1,1) — connecting exponential smoothing to the Box-Jenkins framework.\n"
        "- **Limitation**: Assumes no trend or seasonality in the target series."
    ),
    nav_footer("01_introduction.ipynb", "Introduction", "03_holt_winters.ipynb", "Holt-Winters"),
]

# --- 03 Holt-Winters ---
cells03 = [
    nav_header(
        "Holt-Winters Exponential Smoothing",
        "02_ses.ipynb", "Simple Exponential Smoothing",
        "04_fourier.ipynb", "Fourier Analysis",
        "Holt's method adds trend; Holt-Winters adds seasonality. The ETS framework unifies error, trend, and seasonal components.",
    ),
    code(LOADERS.strip() + "\nfrom statsmodels.tsa.holtwinters import ExponentialSmoothing"),
    md(
        "## Holt-Winters Recurrences\n\n"
        "Level: $\\ell_t = \\alpha(y_t - s_{t-m}) + (1-\\alpha)(\\ell_{t-1} + b_{t-1})$\n\n"
        "Trend: $b_t = \\beta^*(\\ell_t - \\ell_{t-1}) + (1-\\beta^*)b_{t-1}$\n\n"
        "Seasonal: $s_t = \\gamma(y_t - \\ell_{t-1} - b_{t-1}) + (1-\\gamma)s_{t-m}$\n\n"
        "Forecast: $\\hat{y}_{t+h|t} = \\ell_t + hb_t + s_{t+h-m(k+1)}$"
    ),
    code("""
air = load_air_passengers()
milk = load_milk()

# Multiplicative HW for Air Passengers
hw_mul = ExponentialSmoothing(
    air, trend='add', seasonal='mul', seasonal_periods=12,
    initialization_method='estimated'
).fit(optimized=True)
print(f'Air Passengers HW (mul): AIC={hw_mul.aic:.1f}')

# Additive HW for Milk
hw_add = ExponentialSmoothing(
    milk, trend='add', seasonal='add', seasonal_periods=12,
    initialization_method='estimated'
).fit(optimized=True)
print(f'Milk HW (add): AIC={hw_add.aic:.1f}')
"""),
    code("""
# ETS model selection via AIC
configs = [
    ('SES', dict()),
    ('Holt', dict(trend='add')),
    ('HW add', dict(trend='add', seasonal='add', seasonal_periods=12)),
    ('HW mul', dict(trend='add', seasonal='mul', seasonal_periods=12)),
]
rows = []
for label, kw in configs:
    m = ExponentialSmoothing(air, initialization_method='estimated', **kw).fit(optimized=True)
    rows.append({'Model': label, 'AIC': m.aic, 'BIC': m.bic})
pd.DataFrame(rows).sort_values('AIC')
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
air.plot(ax=axes[0], label='Observed', alpha=0.6)
hw_mul.fittedvalues.plot(ax=axes[0], label='HW fitted', color='crimson')
axes[0].set_title('Air Passengers — Multiplicative HW')

milk.plot(ax=axes[1], label='Observed', alpha=0.6)
hw_add.fittedvalues.plot(ax=axes[1], label='HW fitted', color='darkgreen')
axes[1].set_title('Milk — Additive HW')
for ax in axes:
    ax.legend()
fig.tight_layout()
fig
"""),
    md(
        "## Key Takeaways\n\n"
        "- **Multiplicative vs additive**: Use multiplicative when seasonal amplitude scales with level (Air Passengers).\n"
        "- **ETS framework**: All HW variants are special cases of Error-Trend-Seasonal state-space models.\n"
        "- **Damped trend**: Undamped linear trends produce unrealistic long-horizon forecasts — damped variants preferred in production.\n"
        "- **AIC selection**: Compare ETS variants systematically rather than guessing seasonal form."
    ),
    nav_footer("02_ses.ipynb", "Simple Exponential Smoothing", "04_fourier.ipynb", "Fourier Analysis"),
]

# --- 04 Fourier ---
cells04 = [
    nav_header(
        "Fourier Analysis & Spectral Methods",
        "03_holt_winters.ipynb", "Holt-Winters",
        "05_arma_arima.ipynb", "ARMA & ARIMA",
        "Fourier analysis decomposes a series into sinusoids at different frequencies, "
        "revealing periodic structure and enabling flexible seasonal regression.",
    ),
    code(LOADERS.strip() + "\nimport statsmodels.formula.api as smf\nfrom scipy.signal import welch, periodogram"),
    md("## Discrete Fourier Transform\n\nThe DFT maps a series to frequency domain: $X_k = \\sum_{t=0}^{N-1} x_t e^{-2\\pi i k t / N}$."),
    code("""
def manual_dft(x):
    N = len(x)
    k = np.arange(N // 2 + 1)
    t = np.arange(N)
    return np.array([np.sum(x * np.exp(-2j * np.pi * k_i * t / N)) for k_i in k])

co2 = load_co2().dropna().values[-120:]  # last 10 years
dft_manual = manual_dft(co2 - co2.mean())
dft_numpy = np.fft.rfft(co2 - co2.mean())
np.allclose(np.abs(dft_manual), np.abs(dft_numpy), rtol=1e-10)
"""),
    code("""
# Periodogram of CO2 and Air Passengers
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (s, title) in zip(axes, [(load_co2().dropna(), 'Mauna Loa CO2'), (load_air_passengers(), 'Air Passengers')]):
    y = s.values - s.values.mean()
    freqs, psd = periodogram(y, fs=12 if 'CO2' in title or 'Passengers' in title else 1)
    ax.semilogy(freqs, psd)
    ax.set_title(f'Periodogram: {title}')
    ax.set_xlabel('Frequency (cycles per year)')
    ax.set_ylabel('Power')
fig.tight_layout()
fig
"""),
    md("## Welch's Method\n\nWindowing (Hann) reduces spectral leakage compared to the raw periodogram."),
    code("""
co2_vals = load_co2().dropna().values
f_raw, p_raw = periodogram(co2_vals - co2_vals.mean(), fs=12)
f_welch, p_welch = welch(co2_vals - co2_vals.mean(), fs=12, window='hann', nperseg=256)
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(f_raw, p_raw, alpha=0.5, label='Raw periodogram')
ax.semilogy(f_welch, p_welch, label='Welch (Hann window)')
ax.set_xlabel('Frequency (cycles/month)')
ax.legend()
ax.set_title('Spectral Leakage and Windowing')
fig
"""),
    md("## Fourier Regressors\n\nFlexible seasonality via $K$ pairs of $\\sin, \\cos$ terms at harmonics of period $m$."),
    code("""
def fourier_features(n, period=12, K=3):
    t = np.arange(n)
    cols = {}
    for k in range(1, K + 1):
        cols[f'sin_{k}'] = np.sin(2 * np.pi * k * t / period)
        cols[f'cos_{k}'] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols)

co2_s = load_co2().dropna()
n = len(co2_s)
X = fourier_features(n, period=12, K=3)
X['trend'] = np.arange(n)
X['y'] = co2_s.values
model = smf.ols('y ~ trend + sin_1 + cos_1 + sin_2 + cos_2 + sin_3 + cos_3', data=X).fit()
print(model.summary().tables[1])
"""),
    md(
        "## Key Takeaways\n\n"
        "- **Dominant frequencies**: CO₂ shows strong annual cycle (freq ≈ 1/12); Air Passengers shows trend + season.\n"
        "- **Windowing**: Raw periodograms are noisy; Welch smoothing with Hann window improves estimation.\n"
        "- **Fourier regression**: Flexible alternative to fixed seasonal dummies; trade-off controlled by $K$.\n"
        "- **Limitation**: Assumes fixed seasonal period — inappropriate when seasonality evolves."
    ),
    nav_footer("03_holt_winters.ipynb", "Holt-Winters", "05_arma_arima.ipynb", "ARMA & ARIMA"),
]

# --- 05 ARMA ARIMA ---
cells05 = [
    nav_header(
        "ARMA & ARIMA Models",
        "04_fourier.ipynb", "Fourier Analysis",
        "06_sarima.ipynb", "SARIMA",
        "ARIMA models capture autocorrelation structure via autoregressive and moving-average components, "
        "with differencing to achieve stationarity.",
    ),
    code(LOADERS.strip() + "\nfrom statsmodels.tsa.stattools import adfuller, kpss\nfrom statsmodels.tsa.arima.model import ARIMA\nfrom statsmodels.stats.diagnostic import acorr_ljungbox\nfrom statsmodels.graphics.tsaplots import plot_acf, plot_pacf"),
    md("## Unit Root Tests\n\nADF: $H_0$ = unit root. KPSS: $H_0$ = stationarity. Use both together."),
    code("""
air = load_air_passengers()
spots = load_sunspots().astype(float)

for name, s in [('Air Passengers', air), ('Sunspots', spots)]:
    adf = adfuller(s.dropna())
    kpss_stat, kpss_p, _, _ = kpss(s.dropna(), regression='c', nlags='auto')
    print(f'{name}: ADF p={adf[1]:.4f}  KPSS p={kpss_p:.4f}')
"""),
    md("## AR(2) on Sunspots\n\nPACF cuts off at lag $p$ for AR($p$); sunspots are classically modelled as AR(2)."),
    code("""
spots = load_sunspots().astype(float).loc[1900:2008]
model_ar2 = ARIMA(spots, order=(2, 0, 0)).fit()
print(model_ar2.summary().tables[1])
lb = acorr_ljungbox(model_ar2.resid, lags=[10, 20], return_df=True)
print('Ljung-Box on residuals:\\n', lb)
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
spots.plot(ax=axes[0], label='Observed', alpha=0.7)
model_ar2.fittedvalues.plot(ax=axes[0], label='AR(2) fitted', color='crimson')
axes[0].legend()
axes[0].set_title('Sunspots AR(2)')

plot_acf(model_ar2.resid.dropna(), ax=axes[1], lags=30, title='ACF of AR(2) residuals')
fig.tight_layout()
fig
"""),
    md("## Differenced Air Passengers\n\nNon-seasonal ARIMA on $\\Delta y_t$ as a stepping stone to SARIMA."),
    code("""
d_air = air.diff().dropna()
d_model = ARIMA(air, order=(0, 1, 1)).fit()
print(f'ARIMA(0,1,1) AIC={d_model.aic:.1f}')
print(d_model.summary().tables[1])
"""),
    md(
        "## Key Takeaways\n\n"
        "- **Stationarity first**: ADF and KPSS together reduce misclassification of integration order.\n"
        "- **Identification**: PACF cutoff → AR order; ACF cutoff → MA order; both tail off → mixed ARMA.\n"
        "- **Diagnostics**: Ljung-Box on residuals tests remaining autocorrelation; $Q \\sim \\chi^2_h$.\n"
        "- **Box-Jenkins**: Identify → Estimate → Diagnose → Forecast — iterate until residuals are white noise."
    ),
    nav_footer("04_fourier.ipynb", "Fourier Analysis", "06_sarima.ipynb", "SARIMA"),
]

# --- 06 SARIMA ---
cells06 = [
    nav_header(
        "SARIMA & Seasonal Extensions",
        "05_arma_arima.ipynb", "ARMA & ARIMA",
        "07_state_space.ipynb", "State-Space Models",
        "SARIMA extends ARIMA with seasonal differencing and seasonal AR/MA terms. "
        "The Box-Jenkins airline model ARIMA(0,1,1)(0,1,1)₁₂ remains the canonical benchmark.",
    ),
    code(LOADERS.strip() + "\nfrom statsmodels.tsa.statespace.sarimax import SARIMAX\nfrom statsmodels.stats.diagnostic import acorr_ljungbox\nfrom pmdarima import auto_arima"),
    md("## The Airline Model\n\nARIMA$(0,1,1)(0,1,1)_{12}$: one regular and one seasonal difference, one MA and one seasonal MA."),
    code("""
air = load_air_passengers()
airline = SARIMAX(air, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
print(airline.summary().tables[1])
print(f'AIC={airline.aic:.1f}')
"""),
    code("""
# Residual diagnostics
fig = airline.plot_diagnostics(figsize=(12, 8))
fig.suptitle('Airline Model Diagnostics', y=1.02)
fig
"""),
    code("""
# auto-ARIMA with constrained search for CI speed
auto = auto_arima(
    air, seasonal=True, m=12, stepwise=True, suppress_warnings=True,
    max_p=2, max_q=2, max_P=1, max_Q=1, information_criterion='aic'
)
print(auto.summary())
"""),
    code("""
# 24-month forecast with intervals
forecast = airline.get_forecast(steps=24)
fc_mean = forecast.predicted_mean
fc_ci = forecast.conf_int()
fig, ax = plt.subplots(figsize=(12, 5))
air.plot(ax=ax, label='Observed')
fc_mean.plot(ax=ax, label='Forecast', color='crimson')
ax.fill_between(fc_ci.index, fc_ci.iloc[:, 0], fc_ci.iloc[:, 1], alpha=0.3, color='crimson')
ax.legend()
ax.set_title('Air Passengers — 24-month SARIMA Forecast')
fig
"""),
    md(
        "Production seasonal adjustment systems (X-13ARIMA-SEATS, TRAMO-SEATS, JDemetra+) "
        "implement these models at scale for official statistics — the airline model is their intellectual ancestor."
    ),
    code("""
milk = load_milk()
milk_sarima = SARIMAX(milk, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
print(f'Milk SARIMA AIC={milk_sarima.aic:.1f}')
"""),
    md(
        "## Key Takeaways\n\n"
        "- **Seasonal differencing**: $(1-B^{12})y_t$ removes fixed monthly season; combine with regular differencing.\n"
        "- **Airline model**: Still competitive after 50 years — parsimony beats complexity on short series.\n"
        "- **auto-ARIMA**: Useful for exploration; always verify with residual diagnostics.\n"
        "- **Production link**: ONS seasonal adjustment pipelines extend these ideas with regARIMA and calendar effects."
    ),
    nav_footer("05_arma_arima.ipynb", "ARMA & ARIMA", "07_state_space.ipynb", "State-Space Models"),
]

# --- 07 State Space ---
cells07 = [
    nav_header(
        "Introduction to State-Space & Structural Models",
        "06_sarima.ipynb", "SARIMA",
        "08_model_comparison.ipynb", "Model Comparison",
        "State-space models separate observed data from unobserved components (level, trend, season), "
        "estimated via the Kalman filter.",
    ),
    code(LOADERS.strip() + "\nfrom statsmodels.tsa.statespace.structural import UnobservedComponents"),
    md(
        "## State-Space Form\n\n"
        "Observation: $y_t = Z_t \\alpha_t + \\varepsilon_t$. "
        "State: $\\alpha_{t+1} = T_t \\alpha_t + R_t \\eta_t$.\n\n"
        "Local level $\\equiv$ ARIMA(0,1,1). Local linear trend $\\equiv$ ARIMA(0,2,2)."
    ),
    code("""
air = load_air_passengers()

# Local level model
uc_level = UnobservedComponents(air, level='local level').fit(disp=False)
print('Local level AIC:', uc_level.aic)

# Local linear trend + seasonal
uc_struct = UnobservedComponents(
    air, level='local linear trend', seasonal=12
).fit(disp=False)
print('Structural (trend+season) AIC:', uc_struct.aic)
"""),
    code("""
# Extract components
fig = uc_struct.plot_components(figsize=(12, 9))
fig.suptitle('Harvey Structural Model — Air Passengers', y=1.02)
fig
"""),
    md(
        "## ARIMA Equivalences\n\n"
        "| Structural model | Equivalent ARIMA |\n"
        "|---|---|\n"
        "| Local level | ARIMA(0,1,1) |\n"
        "| Local linear trend | ARIMA(0,2,2) |\n"
        "| Local level + seasonal(12) | ARIMA(0,1,1)(0,1,1)₁₂ |\n\n"
        "Production tools: **KFAS** (R), **statsmodels** (Python), used internally by TRAMO-SEATS and JDemetra+."
    ),
    md(
        "## Key Takeaways\n\n"
        "- **Interpretability**: Structural models decompose into level, trend, season — unlike black-box ARIMA.\n"
        "- **Kalman filter**: Optimal recursive estimation under Gaussian linear assumptions.\n"
        "- **Equivalence**: Many ARIMA models have structural representations — choose based on purpose.\n"
        "- **Missing data**: State-space handles gaps naturally; ARIMA requires complete series."
    ),
    nav_footer("06_sarima.ipynb", "SARIMA", "08_model_comparison.ipynb", "Model Comparison"),
]

# --- 08 Model Comparison ---
cells08 = [
    nav_header(
        "Model Comparison & Selection",
        "07_state_space.ipynb", "State-Space Models",
        None, None,
        "We compare all classical methods head-to-head using time-series cross-validation, "
        "proper forecast metrics, and the Diebold-Mariano test for statistical significance.",
    ),
    code(LOADERS.strip() + """
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents
from scipy import stats

def mae(y, yhat): return np.mean(np.abs(y - yhat))
def rmse(y, yhat): return np.sqrt(np.mean((y - yhat) ** 2))
def mape(y, yhat): return np.mean(np.abs((y - yhat) / y)) * 100
def smape(y, yhat): return np.mean(2 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat))) * 100

def crps_normal(y, mu, sigma):
    \"\"\"CRPS for Gaussian predictive distribution.\"\"\"
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))

def diebold_mariano(e1, e2, h=1):
    \"\"\"DM test: H0 equal forecast accuracy.\"\"\"
    d = e1**2 - e2**2
    d_mean = d.mean()
    n = len(d)
    gamma0 = np.var(d, ddof=1)
    var_d = gamma0 / n
    if var_d <= 0:
        return 0.0, 1.0
    dm_stat = d_mean / np.sqrt(var_d)
    p = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p
"""),
    md("## Time Series Cross-Validation\n\nExpanding-window CV respects temporal ordering — no random shuffling."),
    code("""
def evaluate_models(series, test_size=12):
    train, test = series.iloc[:-test_size], series.iloc[-test_size:]
    results = {}

    # Holt-Winters multiplicative
    hw = ExponentialSmoothing(
        train, trend='add', seasonal='mul', seasonal_periods=12,
        initialization_method='estimated'
    ).fit(optimized=True)
    fc = hw.forecast(len(test))
    results['HW Multiplicative'] = {'MAE': mae(test, fc), 'RMSE': rmse(test, fc), 'MAPE': mape(test, fc)}

    # SARIMA airline
    sar = SARIMAX(train, order=(0,1,1), seasonal_order=(0,1,1,12)).fit(disp=False)
    fc = sar.forecast(len(test))
    results['SARIMA Airline'] = {'MAE': mae(test, fc), 'RMSE': rmse(test, fc), 'MAPE': mape(test, fc)}

    # Structural
    uc = UnobservedComponents(train, level='local linear trend', seasonal=12).fit(disp=False)
    fc = uc.forecast(len(test))
    results['Structural'] = {'MAE': mae(test, fc), 'RMSE': rmse(test, fc), 'MAPE': mape(test, fc)}

    return pd.DataFrame(results).T

air = load_air_passengers()
milk = load_milk()
air_results = evaluate_models(air)
milk_results = evaluate_models(milk)
print('Air Passengers (12-month holdout):')
display(air_results.style.format('{:.2f}'))
print('Milk Production (12-month holdout):')
display(milk_results.style.format('{:.2f}'))
"""),
    code("""
# Diebold-Mariano: HW vs SARIMA on Air Passengers
train = air.iloc[:-12]
test = air.iloc[-12:]
hw = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=12,
                        initialization_method='estimated').fit(optimized=True)
sar = SARIMAX(train, order=(0,1,1), seasonal_order=(0,1,1,12)).fit(disp=False)
e_hw = test.values - hw.forecast(12).values
e_sar = test.values - sar.forecast(12).values
dm, p = diebold_mariano(e_hw, e_sar)
print(f'Diebold-Mariano (HW vs SARIMA): stat={dm:.3f}, p={p:.4f}')
"""),
    md(
        "## When to Use Which\n\n"
        "- **SES**: No trend/season; maximum simplicity\n"
        "- **Holt-Winters**: Clear trend+season; transparent parameters\n"
        "- **ARIMA/SARIMA**: Complex autocorrelation; formal inference\n"
        "- **Structural**: Interpretable components; missing data; calendar effects\n\n"
        "M-competitions (M1–M5) showed exponential smoothing often matches ARIMA; combinations frequently win."
    ),
    md(
        "## Key Takeaways\n\n"
        "- **Out-of-sample matters**: In-sample AIC can mislead; always hold out recent data.\n"
        "- **No universal winner**: Best model depends on DGP, horizon, and series length.\n"
        "- **DM test**: Formal comparison of forecast accuracy; use with caution at small sample sizes.\n"
        "- **Looking ahead**: Neural forecasters (N-BEATS, TFT), Prophet, and gradient boosting extend this toolkit in future projects."
    ),
    nav_footer("07_state_space.ipynb", "State-Space Models", None, None),
]

if __name__ == '__main__':
    save('01_introduction.ipynb', cells01)
    save('02_ses.ipynb', cells02)
    save('03_holt_winters.ipynb', cells03)
    save('04_fourier.ipynb', cells04)
    save('05_arma_arima.ipynb', cells05)
    save('06_sarima.ipynb', cells06)
    save('07_state_space.ipynb', cells07)
    save('08_model_comparison.ipynb', cells08)
