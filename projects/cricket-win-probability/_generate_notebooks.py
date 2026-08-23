"""Generate cricket win-probability notebooks. Run from project directory."""

from __future__ import annotations

import json
from pathlib import Path

PROJ = Path(__file__).parent
KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"},
}

IMPORTS = '''
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

PROJ_DIR = Path('.').resolve()
if not (PROJ_DIR / 'cricsheet_utils.py').exists():
    PROJ_DIR = Path('projects/cricket-win-probability').resolve()
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from cricsheet_utils import (
    project_data_dir, load_cricsheet_json, match_metadata, balls_to_dataframe,
    build_game_states, cumulative_runs_by_innings, resource_remaining,
    dls_win_probability,
)
DATA_DIR = project_data_dir()
SAMPLE_DIR = DATA_DIR / 'sample_matches'
'''


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str):
    return {
        "cell_type": "code",
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
            "Introduction & Data Loading",
            "index.md", "Project Overview",
            "02_game_state.ipynb", "Game State Features",
            "This notebook introduces ODI win probability, the Cricsheet data format, and loads a famous match for exploration.",
        ),
        md(
            "## ODI scoring recap\n\n"
            "A standard ODI allots **50 overs** (300 balls) and **10 wickets** per innings. "
            "The team batting second must surpass the first-innings total to win.\n\n"
            "| Symbol | Meaning |\n|--------|--------|\n"
            "| R | Runs scored so far |\n| W | Wickets lost |\n| B | Balls bowled |\n"
            "| RRR | Runs required per remaining ball |"
        ),
        md(
            "## What is win probability?\n\n"
            "Win probability is **P(batting team wins | current game state)**. "
            "Broadcasters show it as a live gauge; analysts use it to quantify how dramatically a chase shifted. "
            "Required run rate (RRR) is a related heuristic but does not map cleanly to probability."
        ),
        code(IMPORTS + "\nprint('Ready:', DATA_DIR)\n"),
        md("## Load a case-study match\n\nWe use bundled Cricsheet-format JSON from the 2019 World Cup Final."),
        code(
            "match_path = SAMPLE_DIR / '2019_wc_final_eng_nz.json'\n"
            "doc = load_cricsheet_json(match_path)\n"
            "meta = match_metadata(doc)\n"
            "balls = balls_to_dataframe(doc)\n"
            "meta\n"
        ),
        md("## Ball-by-ball table"),
        code("balls.head(10)\n"),
        md("## Cumulative runs by innings"),
        code(
            "cum = cumulative_runs_by_innings(balls)\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "for label, grp in cum.groupby('innings_label'):\n"
            "    ax.plot(grp['x'], grp['cumulative_runs'], label=label, linewidth=2)\n"
            "ax.set_xlabel('Overs')\nax.set_ylabel('Runs')\n"
            "ax.set_title('2019 World Cup Final — cumulative runs')\n"
            "ax.legend()\nplt.tight_layout()\n"
            "fig_path = PROJ_DIR / 'figures' / '01_innings_runs.png'\n"
            "fig_path.parent.mkdir(exist_ok=True)\n"
            "fig.savefig(fig_path, dpi=120)\nplt.show()\n"
            "print(f'Saved {fig_path}')\n"
        ),
        md(
            "## Preview\n\n"
            "Each ball in the chase carries a binary outcome label: did the batting team eventually win? "
            "Later chapters model that probability from the live state."
        ),
        code(
            "states = build_game_states(doc)\n"
            "print(f'Chase balls: {len(states)}')\n"
            "print(f\"Chase team won: {states['batting_team_won'].iloc[-1]}\")\n"
            "states[['ball_number', 'runs_scored', 'wickets_lost', 'runs_required', 'runs_required_per_ball']].tail()\n"
        ),
        footer("index.md", "Project Overview", "02_game_state.ipynb", "Game State Features"),
    ]
    save("01_introduction.ipynb", cells)


def nb02():
    cells = [
        nav(
            "Game State Features",
            "01_introduction.ipynb", "Introduction",
            "03_dls_resources.ipynb", "DLS & Resource Methods",
            "We encode each ball in the chase as a feature vector for modelling.",
        ),
        code(IMPORTS),
        code(
            "training = pd.read_csv(DATA_DIR / 'odi_training_sample.csv')\n"
            "feature_cols = [\n"
            "    'runs_scored', 'wickets_lost', 'balls_bowled', 'balls_remaining',\n"
            "    'wickets_remaining', 'runs_required', 'runs_required_per_ball',\n"
            "    'current_run_rate', 'target', 'overs_completed',\n"
            "    'phase_powerplay', 'phase_middle', 'phase_death',\n"
            "]\n"
            "training[feature_cols].describe().T\n"
        ),
        md("## Correlation structure"),
        code(
            "corr = training[feature_cols + ['batting_team_won']].corr()\n"
            "fig, ax = plt.subplots(figsize=(9, 7))\n"
            "im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)\n"
            "ax.set_xticks(range(len(corr)))\n"
            "ax.set_yticks(range(len(corr)))\n"
            "ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)\n"
            "ax.set_yticklabels(corr.columns, fontsize=8)\n"
            "plt.colorbar(im, ax=ax)\n"
            "ax.set_title('Feature correlation matrix')\n"
            "plt.tight_layout()\nplt.show()\n"
        ),
        md("## RRR vs outcome at the final ball"),
        code(
            "finals = training.groupby('match_idx').tail(1)\n"
            "fig, ax = plt.subplots(figsize=(8, 4))\n"
            "for won, grp in finals.groupby('batting_team_won'):\n"
            "    label = 'Chase won' if won else 'Chase lost'\n"
            "    ax.hist(grp['runs_required_per_ball'].dropna(), bins=30, alpha=0.6, label=label)\n"
            "ax.set_xlabel('RRR at last ball')\nax.set_ylabel('Matches')\n"
            "ax.legend()\nax.set_title('Final-ball required rate by outcome')\n"
            "plt.tight_layout()\nplt.show()\n"
        ),
        md("## Example: high-pressure state"),
        code(
            "pressure = training[(training['over'] >= 43) & (training['wickets_lost'] >= 5)]\n"
            "pressure.nlargest(1, 'runs_required_per_ball')[feature_cols + ['batting_team_won']]\n"
        ),
        footer("01_introduction.ipynb", "Introduction", "03_dls_resources.ipynb", "DLS & Resource Methods"),
    ]
    save("02_game_state.ipynb", cells)


def nb03():
    cells = [
        nav(
            "DLS & Resource Methods",
            "02_game_state.ipynb", "Game State",
            "04_modelling.ipynb", "Modelling",
            "A simplified Duckworth–Lewis–Stern resource model provides an interpretable baseline.",
        ),
        md(
            "> **Note:** ICC uses proprietary DLS tables with protected overs. "
            "This chapter implements an **educational approximation** to teach the resource logic."
        ),
        code(IMPORTS),
        md("## Resource remaining surface"),
        code(
            "overs_grid = np.linspace(0, 50, 51)\n"
            "wickets_grid = range(10)\n"
            "surface = np.array([[resource_remaining(o, w) for o in overs_grid] for w in wickets_grid])\n"
            "fig, ax = plt.subplots(figsize=(9, 5))\n"
            "im = ax.imshow(surface, aspect='auto', origin='lower', cmap='YlGnBu')\n"
            "ax.set_xlabel('Overs completed')\nax.set_ylabel('Wickets lost')\n"
            "ax.set_title('Simplified ODI resource remaining (%)')\n"
            "plt.colorbar(im, ax=ax, label='Resource %')\nplt.tight_layout()\nplt.show()\n"
        ),
        md("## Ball-by-ball DLS probability for WC Final"),
        code(
            "doc = load_cricsheet_json(SAMPLE_DIR / '2019_wc_final_eng_nz.json')\n"
            "states = build_game_states(doc)\n"
            "first_runs = int(states['target'].iloc[0] - 1)\n"
            "states = states.copy()\n"
            "states['dls_prob'] = dls_win_probability(states, first_runs)\n"
            "fig, ax = plt.subplots(figsize=(10, 4))\n"
            "ax.plot(states['overs_completed'], states['dls_prob'], color='#2e7d32', linewidth=2)\n"
            "ax.set_xlabel('Overs (chase)')\nax.set_ylabel('P(win)')\n"
            "ax.set_ylim(0, 1)\n"
            "ax.set_title('DLS-style win probability — 2019 World Cup Final chase')\n"
            "plt.tight_layout()\nplt.show()\n"
            "states[['over', 'runs_scored', 'wickets_lost', 'dls_prob']].tail(8)\n"
        ),
        footer("02_game_state.ipynb", "Game State", "04_modelling.ipynb", "Modelling"),
    ]
    save("03_dls_resources.ipynb", cells)


def nb04():
    cells = [
        nav(
            "Historical Modelling",
            "03_dls_resources.ipynb", "DLS Resources",
            "05_live_charts.ipynb", "Live Charts",
            "We train logistic regression and LightGBM on historical chase balls and compare against DLS.",
        ),
        code(
            IMPORTS
            + "\nfrom sklearn.linear_model import LogisticRegression\n"
            "from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.pipeline import Pipeline\n"
            "import lightgbm as lgb\n"
        ),
        code(
            "training = pd.read_csv(DATA_DIR / 'odi_training_sample.csv')\n"
            "feature_cols = [\n"
            "    'runs_scored', 'wickets_lost', 'balls_remaining', 'wickets_remaining',\n"
            "    'runs_required', 'runs_required_per_ball', 'current_run_rate',\n"
            "    'overs_completed', 'phase_powerplay', 'phase_death',\n"
            "]\n"
            "train_matches = training['match_idx'].unique()\n"
            "rng = np.random.default_rng(42)\n"
            "rng.shuffle(train_matches)\n"
            "split = int(0.8 * len(train_matches))\n"
            "train_ids, val_ids = set(train_matches[:split]), set(train_matches[split:])\n"
            "train_df = training[training['match_idx'].isin(train_ids)]\n"
            "val_df = training[training['match_idx'].isin(val_ids)]\n"
            "X_train, y_train = train_df[feature_cols], train_df['batting_team_won']\n"
            "X_val, y_val = val_df[feature_cols], val_df['batting_team_won']\n"
            "print(f'Train balls: {len(X_train):,} | Val balls: {len(X_val):,}')\n"
        ),
        md("## Logistic regression"),
        code(
            "logit = Pipeline([\n"
            "    ('scale', StandardScaler()),\n"
            "    ('clf', LogisticRegression(max_iter=1000, random_state=42)),\n"
            "])\n"
            "logit.fit(X_train, y_train)\n"
            "p_logit = logit.predict_proba(X_val)[:, 1]\n"
        ),
        md("## LightGBM"),
        code(
            "lgbm = lgb.LGBMClassifier(\n"
            "    n_estimators=200, learning_rate=0.05, max_depth=5,\n"
            "    random_state=42, verbose=-1,\n"
            ")\n"
            "lgbm.fit(X_train, y_train)\n"
            "p_lgbm = lgbm.predict_proba(X_val)[:, 1]\n"
        ),
        md("## Metrics comparison"),
        code(
            "p_dls = val_df['dls_prob'].values\n"
            "results = []\n"
            "for name, preds in [('Logistic regression', p_logit), ('LightGBM', p_lgbm), ('DLS baseline', p_dls)]:\n"
            "    results.append({\n"
            "        'Model': name,\n"
            "        'Brier score': round(brier_score_loss(y_val, preds), 4),\n"
            "        'Log loss': round(log_loss(y_val, preds), 4),\n"
            "        'AUC': round(roc_auc_score(y_val, preds), 4),\n"
            "    })\n"
            "metrics = pd.DataFrame(results)\n"
            "metrics\n"
        ),
        md("## Logistic coefficients"),
        code(
            "coefs = pd.Series(logit.named_steps['clf'].coef_[0], index=feature_cols)\n"
            "fig, ax = plt.subplots(figsize=(8, 4))\n"
            "coefs.sort_values().plot(kind='barh', ax=ax, color='#1565c0')\n"
            "ax.set_title('Logistic regression coefficients (scaled features)')\n"
            "ax.set_xlabel('Coefficient')\nplt.tight_layout()\nplt.show()\n"
        ),
        code(
            "import joblib\n"
            "model_dir = PROJ_DIR / 'data'\n"
            "joblib.dump(logit, model_dir / 'logit_model.joblib')\n"
            "joblib.dump(lgbm, model_dir / 'lgbm_model.joblib')\n"
            "print('Models saved.')\n"
        ),
        footer("03_dls_resources.ipynb", "DLS Resources", "05_live_charts.ipynb", "Live Charts"),
    ]
    save("04_modelling.ipynb", cells)


def nb05():
    cells = [
        nav(
            "Live Win Probability Charts",
            "04_modelling.ipynb", "Modelling",
            "06_evaluation_case_studies.ipynb", "Evaluation & Case Studies",
            "Interactive Plotly charts compare ML and DLS probabilities ball-by-ball.",
        ),
        code(
            IMPORTS
            + "\nimport plotly.graph_objects as go\n"
            "from IPython.display import HTML, display\n"
            "import joblib\n"
            "logit = joblib.load(DATA_DIR / 'logit_model.joblib')\n"
            "lgbm = joblib.load(DATA_DIR / 'lgbm_model.joblib')\n"
            "feature_cols = [\n"
            "    'runs_scored', 'wickets_lost', 'balls_remaining', 'wickets_remaining',\n"
            "    'runs_required', 'runs_required_per_ball', 'current_run_rate',\n"
            "    'overs_completed', 'phase_powerplay', 'phase_death',\n"
            "]\n"
        ),
        code(
            "def win_prob_states(doc):\n"
            "    states = build_game_states(doc).copy()\n"
            "    first_runs = int(states['target'].iloc[0] - 1)\n"
            "    states['dls_prob'] = dls_win_probability(states, first_runs)\n"
            "    states['ml_prob'] = lgbm.predict_proba(states[feature_cols])[:, 1]\n"
            "    return states\n"
            "\n"
            "def win_prob_chart(doc, title):\n"
            "    \"\"\"Plotly figure; use display_plotly() so Jupyter Book embeds HTML.\"\"\"\n"
            "    states = win_prob_states(doc)\n"
            "    fig = go.Figure()\n"
            "    fig.add_trace(go.Scatter(x=states['overs_completed'], y=states['ml_prob'],\n"
            "                             mode='lines', name='LightGBM', line=dict(color='#1565c0', width=2)))\n"
            "    fig.add_trace(go.Scatter(x=states['overs_completed'], y=states['dls_prob'],\n"
            "                             mode='lines', name='DLS baseline', line=dict(color='#2e7d32', width=2, dash='dash')))\n"
            "    wicket_rows = states[states['ball_number'].isin(\n"
            "        balls_to_dataframe(doc).query('innings == 2 and is_wicket == 1')['ball_number']\n"
            "    )]\n"
            "    for _, w in wicket_rows.iterrows():\n"
            "        fig.add_vline(x=w['overs_completed'], line_width=1, line_dash='dot', line_color='gray', opacity=0.4)\n"
            "    fig.update_layout(title=title, xaxis_title='Overs (chase)', yaxis_title='P(win)',\n"
            "                      yaxis=dict(range=[0, 1]), template='plotly_white', height=450)\n"
            "    return fig\n"
            "\n"
            "def display_plotly(fig):\n"
            "    \"\"\"Embed Plotly with CDN JS — fig.show() is blank in Jupyter Book HTML.\"\"\"\n"
            "    display(HTML(fig.to_html(include_plotlyjs='cdn', full_html=False)))\n"
            "\n"
            "def plot_win_prob_mpl(states, title, save_path=None):\n"
            "    fig, ax = plt.subplots(figsize=(10, 4))\n"
            "    ax.plot(states['overs_completed'], states['ml_prob'], label='LightGBM', color='#1565c0', linewidth=2)\n"
            "    ax.plot(states['overs_completed'], states['dls_prob'], label='DLS baseline', color='#2e7d32', linewidth=2, linestyle='--')\n"
            "    ax.set_xlabel('Overs (chase)')\n"
            "    ax.set_ylabel('P(win)')\n"
            "    ax.set_ylim(0, 1)\n"
            "    ax.set_title(title)\n"
            "    ax.legend()\n"
            "    plt.tight_layout()\n"
            "    if save_path is not None:\n"
            "        fig.savefig(save_path, dpi=120)\n"
            "    plt.show()\n"
            "    return fig\n"
        ),
        md("## 2019 World Cup Final"),
        code(
            "doc = load_cricsheet_json(SAMPLE_DIR / '2019_wc_final_eng_nz.json')\n"
            "states = win_prob_states(doc)\n"
            "display_plotly(win_prob_chart(doc, '2019 World Cup Final — England chase'))\n"
            "plot_win_prob_mpl(\n"
            "    states,\n"
            "    '2019 World Cup Final — England chase',\n"
            "    save_path=PROJ_DIR / 'figures' / '05_wc2019_winprob.png',\n"
            ")\n"
        ),
        md("## 2011 World Cup Quarter-Final — India chase"),
        code(
            "doc2 = load_cricsheet_json(SAMPLE_DIR / '2011_wc_qf_ind_aus.json')\n"
            "states2 = win_prob_states(doc2)\n"
            "display_plotly(win_prob_chart(doc2, '2011 WC QF — India chase vs Australia'))\n"
            "plot_win_prob_mpl(states2, '2011 WC QF — India chase vs Australia')\n"
        ),
        footer("04_modelling.ipynb", "Modelling", "06_evaluation_case_studies.ipynb", "Evaluation & Case Studies"),
    ]
    save("05_live_charts.ipynb", cells)


def nb06():
    cells = [
        nav(
            "Evaluation & Case Studies",
            "05_live_charts.ipynb", "Live Charts",
            "index.md", "Project Overview",
            "Calibration analysis, model–DLS disagreement, and limitations.",
        ),
        code(
            IMPORTS
            + "\nfrom sklearn.metrics import brier_score_loss\n"
            "import joblib\n"
            "lgbm = joblib.load(DATA_DIR / 'lgbm_model.joblib')\n"
            "feature_cols = [\n"
            "    'runs_scored', 'wickets_lost', 'balls_remaining', 'wickets_remaining',\n"
            "    'runs_required', 'runs_required_per_ball', 'current_run_rate',\n"
            "    'overs_completed', 'phase_powerplay', 'phase_death',\n"
            "]\n"
            "val_df = pd.read_csv(DATA_DIR / 'odi_training_sample.csv')\n"
            "val_matches = val_df['match_idx'].unique()[-80:]\n"
            "val_df = val_df[val_df['match_idx'].isin(val_matches)]\n"
            "val_df = val_df.copy()\n"
            "val_df['ml_prob'] = lgbm.predict_proba(val_df[feature_cols])[:, 1]\n"
        ),
        md("## Calibration (reliability diagram)"),
        code(
            "def reliability_curve(y_true, y_prob, n_bins=10):\n"
            "    bins = np.linspace(0, 1, n_bins + 1)\n"
            "    mids, obs = [], []\n"
            "    for lo, hi in zip(bins[:-1], bins[1:]):\n"
            "        mask = (y_prob >= lo) & (y_prob < hi)\n"
            "        if mask.sum() == 0:\n"
            "            continue\n"
            "        mids.append((lo + hi) / 2)\n"
            "        obs.append(y_true[mask].mean())\n"
            "    return np.array(mids), np.array(obs)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(6, 6))\n"
            "ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')\n"
            "for name, col in [('LightGBM', 'ml_prob'), ('DLS baseline', 'dls_prob')]:\n"
            "    mids, obs = reliability_curve(val_df['batting_team_won'].values, val_df[col].values)\n"
            "    ax.plot(mids, obs, 'o-', label=name)\n"
            "ax.set_xlabel('Predicted P(win)')\nax.set_ylabel('Observed win rate')\n"
            "ax.set_title('Calibration on validation chase balls')\n"
            "ax.legend()\nplt.tight_layout()\n"
            "fig.savefig(PROJ_DIR / 'figures' / '06_calibration.png', dpi=120)\nplt.show()\n"
        ),
        md("## Where ML and DLS disagree most"),
        code(
            "val_df['prob_diff'] = val_df['ml_prob'] - val_df['dls_prob']\n"
            "fig, ax = plt.subplots(figsize=(8, 4))\n"
            "ax.scatter(val_df['overs_completed'], val_df['prob_diff'], alpha=0.05, s=8)\n"
            "ax.axhline(0, color='k', linewidth=0.8)\n"
            "ax.set_xlabel('Overs completed (chase)')\nax.set_ylabel('ML prob − DLS prob')\n"
            "ax.set_title('Probability disagreement by over')\nplt.tight_layout()\nplt.show()\n"
        ),
        md("## Case-study summary"),
        code(
            "cases = []\n"
            "for fname in sorted(SAMPLE_DIR.glob('*.json')):\n"
            "    doc = load_cricsheet_json(fname)\n"
            "    meta = match_metadata(doc)\n"
            "    states = build_game_states(doc)\n"
            "    states['ml_prob'] = lgbm.predict_proba(states[feature_cols])[:, 1]\n"
            "    cases.append({\n"
            "        'Match': fname.stem,\n"
            "        'Event': meta['event'],\n"
            "        'Winner': meta['winner'],\n"
            "        'Max ML P(win)': round(states['ml_prob'].max(), 3),\n"
            "        'Min ML P(win)': round(states['ml_prob'].min(), 3),\n"
            "        'Final ML P(win)': round(states['ml_prob'].iloc[-1], 3),\n"
            "    })\n"
            "pd.DataFrame(cases)\n"
        ),
        md(
            "## Limitations & further reading\n\n"
            "- Simplified DLS tables; no rain-adjusted par scores mid-match\n"
            "- No player quality, venue, or pitch effects\n"
            "- ODI only; T20 and Test need separate resource structures\n"
            "- Further reading: Duckworth & Lewis (1998), Jayadevan (VJD method), ESPNcricinfo win-probability models"
        ),
        footer("05_live_charts.ipynb", "Live Charts", "index.md", "Project Overview"),
    ]
    save("06_evaluation_case_studies.ipynb", cells)


def main():
    nb01()
    nb02()
    nb03()
    nb04()
    nb05()
    nb06()
    print("All notebooks generated.")


if __name__ == "__main__":
    main()
