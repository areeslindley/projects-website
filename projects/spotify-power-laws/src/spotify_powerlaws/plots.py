"""Argument-carrying figures. Axes labelled with units; no leftover default titles."""

from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter

from .paths import figures_dir
from .style import AMBER, INK, MIST, NAVY, SLATE, TEAL, apply_style, save_fig


def figure_forensics(forensics: dict, schema: dict) -> str:
    apply_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
    digits = np.arange(1, 10)
    ben = forensics["benford_stream_count"]
    ax = axes[0]
    ax.bar(digits, ben["observed_proportions"], color=TEAL, width=0.7, label="Observed")
    ax.plot(digits, ben["benford_proportions"], color=AMBER, marker="o", linewidth=1.8, label="Benford")
    ax.set_xlabel("Leading digit of stream count")
    ax.set_ylabel("Share of tracks")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xticks(digits)
    ax.legend(loc="upper right")
    ax.set_title(
        f"Benford MAD = {ben['mad']:.3f} ({ben['nigrini_conformity']})",
        loc="left",
        pad=8,
    )

    ax = axes[1]
    last = np.arange(10)
    counts = np.array(forensics["terminal_stream_count"]["last_digit_counts"], dtype=float)
    ax.bar(last, counts / counts.sum(), color=NAVY, width=0.7)
    ax.axhline(0.1, color=AMBER, linewidth=1.6, linestyle="--", label="Uniform 0.10")
    ax.set_xlabel("Terminal digit of stream count")
    ax.set_ylabel("Share of tracks")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xticks(last)
    ax.legend(loc="lower right")
    p_term = forensics["terminal_stream_count"]["p_value"]
    ax.set_title(f"No rounding heap (χ² p = {p_term:.2f})", loc="left", pad=8)

    ax = axes[2]
    ax.set_axis_off()
    lines = [
        f"{schema['n_rows']:,} tracks · {schema['n_artists']} invented artists",
        f"Grain: {schema['grain']} (not artist-year)",
        f"Largest catalog: {schema['top_artist']} ({schema['top_artist_share']:.0%} of rows)",
        "Real names present: none of 10 checks",
        "Acousticness / valence: absent",
        f"energy–loudness r = {forensics['energy_loudness_corr']:.2f}",
        f"audio vs streams |r| ≤ {max(abs(v) for v in forensics['audio_target_correlations'].values()):.3f}",
        f"popularity vs log1p(streams) r = {forensics['popularity_log1p_streams_corr']:.2f}",
    ]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        FancyBboxPatch((0.02, 0.04), 0.96, 0.92, boxstyle="round,pad=0.02,rounding_size=0.04",
                       facecolor=MIST, edgecolor="none")
    )
    ax.set_title("What the file actually is", loc="left", pad=8)
    for i, line in enumerate(lines):
        ax.text(0.08, 0.88 - i * 0.105, line, fontsize=9, color=INK, va="top")
    fig.tight_layout()
    path = figures_dir() / "01_forensics.png"
    save_fig(fig, path)
    return str(path)


def figure_leakage(models: dict) -> str:
    apply_style()
    import matplotlib.pyplot as plt

    labels = [
        "Honest\ngrouped",
        "Honest\nrandom k-fold",
        "Leakage\ngrouped",
        "Leakage\nrandom k-fold",
    ]
    keys = [
        "ols_honest_grouped",
        "ols_honest_random",
        "ols_leakage_grouped",
        "ols_leakage_random",
    ]
    r2 = [models[k]["summary"]["r2_log"]["mean"] for k in keys]
    r2_std = [models[k]["summary"]["r2_log"]["std"] for k in keys]
    colors = [TEAL, TEAL, AMBER, AMBER]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, r2, yerr=r2_std, color=colors, width=0.72, capsize=4, ecolor=SLATE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Outer-fold $R^2$ on log1p stream count")
    ax.set_ylim(0, 1.05)
    ax.axhline(0, color=INK, linewidth=0.6)
    intercept_r2 = models["intercept_grouped"]["summary"]["r2_log"]["mean"]
    ax.axhline(intercept_r2, color=SLATE, linestyle=":", linewidth=1.2)
    ax.set_title(
        "Leakage, not model class, is what inflates $R^2$",
        loc="left",
        pad=10,
    )
    fig.tight_layout()
    path = figures_dir() / "02_leakage_gap.png"
    save_fig(fig, path)
    return str(path)


def figure_learning_curve(curve: dict) -> str:
    apply_style()
    import matplotlib.pyplot as plt

    pts = curve["points"]
    n = [p["n_artists"] for p in pts]
    mean = [p["rmse_mean"] for p in pts]
    lo = [p["rmse_lo"] for p in pts]
    hi = [p["rmse_hi"] for p in pts]
    intercept = pts[0]["intercept_rmse"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.fill_between(n, lo, hi, color=TEAL, alpha=0.18, label="10–90% bootstrap band")
    ax.plot(n, mean, color=TEAL, marker="o", linewidth=2, label="Honest OLS")
    ax.axhline(intercept, color=AMBER, linestyle="--", linewidth=1.6, label="Constant mean on held-out artists")
    ax.set_xlabel("Number of training artists")
    ax.set_ylabel("RMSE on log1p stream count (80 held-out artists)")
    ax.legend(loc="upper right")
    ax.set_title("Added artists do not buy a coupling the generator did not plant", loc="left", pad=10)
    fig.tight_layout()
    path = figures_dir() / "03_learning_curve.png"
    save_fig(fig, path)
    return str(path)


def figure_lorenz(dist: dict, schema: dict) -> str:
    apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.plot([0, 1], [0, 1], color=SLATE, linestyle=":", linewidth=1.2, label="Equality")
    track = dist["track_streams"]["lorenz"]
    artist = dist["artist_totals"]["lorenz"]
    catalog = dist["artist_catalog_size"]["lorenz"]
    ax.plot(track["population"], track["share"], color=TEAL, linewidth=2.2, label=f"Track streams (Gini {track['gini']:.2f})")
    ax.plot(artist["population"], artist["share"], color=AMBER, linewidth=2.2, label=f"Artist total streams (Gini {artist['gini']:.2f})")
    ax.plot(catalog["population"], catalog["share"], color=NAVY, linewidth=2.0, linestyle="--", label=f"Artist catalog size (Gini {catalog['gini']:.2f})")
    ax.set_xlabel("Share of tracks or artists, ranked by size")
    ax.set_ylabel("Share of streams (or tracks)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.set_title(
        f"The planted tail is catalog size — {schema['top_artist']} is {schema['top_artist_share']:.0%} of rows",
        loc="left",
        pad=10,
    )
    fig.tight_layout()
    path = figures_dir() / "04_lorenz.png"
    save_fig(fig, path)
    return str(path)


def figure_conformal(conf: dict) -> str:
    apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    nom = [p["nominal"] for p in conf["curve"]]
    emp = [p["empirical"] for p in conf["curve"]]
    ax.plot([0, 1], [0, 1], color=SLATE, linestyle=":", linewidth=1.2, label="Nominal = empirical")
    ax.plot(nom, emp, color=TEAL, marker="o", linewidth=2, label="Split conformal (grouped)")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage on held-out artists")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    c80 = conf["coverage"]["0.8"]["empirical"]
    ax.set_title(f"80% intervals covered {c80:.0%} of held-out log-streams", loc="left", pad=10)
    fig.tight_layout()
    path = figures_dir() / "05_conformal.png"
    save_fig(fig, path)
    return str(path)


def figure_shrinkage(models: dict, perm: dict) -> str:
    apply_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    mixed = models["mixedlm"]
    names = ["Artist intercept\n(SD, log1p streams)", "Residual SD"]
    vals = [mixed["sd_artist"], float(np.sqrt(mixed["var_resid"]))]
    ax.barh(names, vals, color=[AMBER, TEAL], height=0.55)
    ax.set_xlabel("Standard deviation on log1p stream count")
    ax.set_title(f"Partial pooling collapses (ICC = {mixed['icc']:.4f})", loc="left", pad=8)

    ax = axes[1]
    blocks = ["audio", "duration_explicit", "calendar", "genre", "country", "label"]
    means = [perm[b]["mean_delta_rmse"] for b in blocks]
    lo = [perm[b]["p10"] for b in blocks]
    hi = [perm[b]["p90"] for b in blocks]
    y = np.arange(len(blocks))
    ax.barh(y, means, color=NAVY, height=0.6)
    ax.errorbar(means, y, xerr=[np.array(means) - np.array(lo), np.array(hi) - np.array(means)], fmt="none", ecolor=SLATE, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(blocks)
    ax.axvline(0, color=INK, linewidth=0.7)
    ax.set_xlabel("ΔRMSE on log1p streams after grouped permutation")
    ax.set_title("Honest blocks barely move out-of-artist RMSE", loc="left", pad=8)
    fig.tight_layout()
    path = figures_dir() / "06_shrinkage_importance.png"
    save_fig(fig, path)
    return str(path)


def figure_genre_lda(genre: dict) -> str:
    apply_style()
    import matplotlib.pyplot as plt
    import pandas as pd

    scatter = genre["lda_coordinates"]["scatter"]
    means = pd.DataFrame(genre["lda_coordinates"]["means"])
    genres = sorted(set(scatter["genre"]))
    cmap = plt.get_cmap("tab20")
    color = {g: cmap(i / max(len(genres) - 1, 1)) for i, g in enumerate(genres)}
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    ax.scatter(
        scatter["ld1"],
        scatter["ld2"],
        c=[color[g] for g in scatter["genre"]],
        s=8,
        alpha=0.25,
        linewidths=0,
    )
    for row in means.itertuples():
        ax.scatter(row.ld1, row.ld2, s=40 + 0.004 * row.n, c=[color[row.genre]], edgecolors=INK, linewidths=0.6, zorder=3)
        ax.annotate(row.genre, (row.ld1, row.ld2), textcoords="offset points", xytext=(5, 4), fontsize=8, color=INK)
    ev = genre["lda_coordinates"]["explained_variance_ratio"]
    ax.set_xlabel(f"LD1 ({ev[0]:.0%} of between-genre scatter)")
    ax.set_ylabel(f"LD2 ({ev[1]:.0%} of between-genre scatter)")
    ax.set_title("Audio separates planted genres even though it does not predict streams", loc="left", pad=10)
    fig.tight_layout()
    path = figures_dir() / "07_genre_lda.png"
    save_fig(fig, path)
    return str(path)


def figure_genre_confusion(genre: dict) -> str:
    apply_style()
    import matplotlib.pyplot as plt

    labels = genre["confusion_lda_grouped"]["labels"]
    recall = np.array(genre["confusion_lda_grouped"]["recall"])
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    im = ax.imshow(recall, cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted genre (LDA, grouped by artist)")
    ax.set_ylabel("True genre")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (row-normalised)")
    diag = float(np.mean([recall[i, i] for i in range(len(labels))]))
    ax.set_title(f"Mean diagonal recall = {diag:.2f} — collisions are among neighbouring planted styles", loc="left", pad=10)
    fig.tight_layout()
    path = figures_dir() / "08_genre_confusion.png"
    save_fig(fig, path)
    return str(path)


def figure_genre_ladder(genre: dict) -> str:
    apply_style()
    import matplotlib.pyplot as plt

    names = ["dummy", "lda", "logit", "hgb"]
    labels = ["Dummy\n(prior)", "LDA", "Multinomial\nlogit", "HGB"]
    grouped = [genre["models"][f"{n}_grouped"]["summary"]["balanced_accuracy"]["mean"] for n in names]
    grouped_std = [genre["models"][f"{n}_grouped"]["summary"]["balanced_accuracy"]["std"] for n in names]
    random = [genre["models"][f"{n}_random"]["summary"]["balanced_accuracy"]["mean"] for n in names]
    random_std = [genre["models"][f"{n}_random"]["summary"]["balanced_accuracy"]["std"] for n in names]
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(x - width / 2, grouped, width, yerr=grouped_std, color=TEAL, capsize=3, label="Grouped by artist", ecolor=SLATE)
    ax.bar(x + width / 2, random, width, yerr=random_std, color=AMBER, capsize=3, label="Random k-fold", ecolor=SLATE)
    ax.axhline(0.05, color=SLATE, linestyle=":", linewidth=1.2, label="Chance (1/20)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left")
    lda_g = grouped[1]
    logit_g = grouped[2]
    ax.set_title(
        f"Grouped balanced accuracy: LDA {lda_g:.2f}, logit {logit_g:.2f}; HGB does not beat logit",
        loc="left",
        pad=10,
    )
    fig.tight_layout()
    path = figures_dir() / "09_genre_ladder.png"
    save_fig(fig, path)
    return str(path)
