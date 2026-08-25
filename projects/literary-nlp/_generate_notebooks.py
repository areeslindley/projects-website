"""Generate literary NLP notebooks. Run from the project directory or repo root."""

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
if not (PROJ_DIR / 'gutenberg_utils.py').exists():
    PROJ_DIR = Path('projects/literary-nlp').resolve()
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from gutenberg_utils import (
    load_pages, book_catalog, title_of, BOOK_COLORS, BOOKS,
    THEMATIC_KEYWORDS, all_stopwords, ensure_nltk_data,
    add_vader_sentiment, add_nrc_emotions, NRC_EMOTIONS,
    keyword_counts, character_mentions, third_label,
)

def display_plotly(fig):
    \"\"\"Embed Plotly with CDN JS — fig.show() is blank in Jupyter Book HTML.\"\"\"
    display(HTML(fig.to_html(include_plotlyjs='cdn', full_html=False)))

PAGES = load_pages()
CATALOG = book_catalog()
print(f\"Loaded {len(PAGES):,} pages across {PAGES['book_id'].nunique()} books\")
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
            "Corpus & Cleaning",
            "index.md", "Project Overview",
            "02_sentiment.ipynb", "Sentiment Trajectories",
            "Eight public-domain novels, stripped of Gutenberg boilerplate and sliced into page-sized windows.",
        ),
        md(
            "## Why pages?\n\n"
            "Project Gutenberg files are continuous plain text. They have chapter headings, "
            "but no printer pagination. To compare books of very different lengths we define a "
            "**page** as a window of about **250 words** — roughly a printed octavo page — "
            "and a **progress** coordinate from 0% (opening) to 100% (close).\n\n"
            "Chapter headings (`CHAPTER`, `STAVE`, `LETTER`, and Wells-style `I.` / `II.`) "
            "are retained as labels so later charts can be read against the plot structure."
        ),
        md(
            "## Why lexicons, not transformers?\n\n"
            "This site rebuilds every notebook in GitHub Actions (`execute_notebooks: auto`, "
            "600-second timeout). VADER and the NRC emotion lexicon score a few thousand pages "
            "in seconds, need no GPU, and pin to a word list rather than a moving model checkpoint. "
            "The cost is coarseness: they miss irony and period idiom. Treat every score as a "
            "**relative trajectory**, not a claim about how a human reader feels."
        ),
        code(IMPORTS),
        md("## The eight titles"),
        code(
            "summary = (\n"
            "    PAGES.groupby(['book_id', 'title', 'author', 'year'], as_index=False)\n"
            "    .agg(pages=('page_idx', 'count'), chapters=('chapter_idx', 'nunique'), words=('n_words', 'sum'))\n"
            "    .sort_values('year')\n"
            ")\n"
            "summary\n"
        ),
        md("## How long is each novel?"),
        code(
            "order = summary.sort_values('words')\n"
            "colors = [BOOK_COLORS[b] for b in order['book_id']]\n"
            "fig, ax = plt.subplots(figsize=(9, 5))\n"
            "ax.barh(order['title'], order['words'] / 1000, color=colors)\n"
            "ax.set_xlabel('Thousands of words')\n"
            "ax.set_title('Corpus size after Gutenberg cleanup')\n"
            "plt.tight_layout()\n"
            "fig_dir = PROJ_DIR / 'figures'\n"
            "fig_dir.mkdir(exist_ok=True)\n"
            "fig.savefig(fig_dir / '01_corpus_lengths.png', dpi=120)\n"
            "plt.show()\n"
        ),
        md("## A sample page\n\nThe first page of *Dracula* after boilerplate has been stripped:"),
        code(
            "sample = PAGES.loc[PAGES['book_id'] == 'dracula'].iloc[0]\n"
            "print(f\"Chapter: {sample['chapter_title']}\")\n"
            "print(f\"Words: {sample['n_words']}\")\n"
            "print()\n"
            "print(sample['text'][:900] + '…')\n"
        ),
        md(
            "## Page-window diagnostics\n\n"
            "Most pages sit near the 250-word target. Short remainder pages are merged into the "
            "previous window so the series is not spiked by a 20-word tail."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(8, 4))\n"
            "ax.hist(PAGES['n_words'], bins=25, color='#4a0e0e', alpha=0.85)\n"
            "ax.axvline(250, color='#c9a227', linestyle='--', label='Target (250 words)')\n"
            "ax.set_xlabel('Words per page window')\n"
            "ax.set_ylabel('Pages')\n"
            "ax.set_title('Page-window length distribution')\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "PAGES['n_words'].describe().to_frame('words_per_page')\n"
        ),
        md(
            "## Attribution\n\n"
            "Plain-text ebooks from [Project Gutenberg](https://www.gutenberg.org/). "
            "See `data/README.md` for Gutenberg IDs and the rebuild command."
        ),
        footer("index.md", "Project Overview", "02_sentiment.ipynb", "Sentiment Trajectories"),
    ]
    save("01_corpus.ipynb", cells)


def nb02():
    cells = [
        nav(
            "Sentiment Trajectories",
            "01_corpus.ipynb", "Corpus & Cleaning",
            "03_emotions.ipynb", "Emotion Lexicon",
            "VADER compound scores along each novel — a rolling mood line from first page to last.",
        ),
        md(
            "## Method\n\n"
            "[VADER](https://github.com/cjhutto/vaderSentiment) (Valence Aware Dictionary and sEntiment Reasoner) "
            "assigns each page a **compound** score in $[-1, 1]$: below 0 leans negative, above 0 leans positive. "
            "We then take a centred rolling mean (8 pages) so a single melodramatic paragraph does not dominate.\n\n"
            "The x-axis is **progress through the book** (0–100%), not raw page number, so *Alice* and *Dracula* "
            "can be drawn on the same chart."
        ),
        code(IMPORTS),
        code(
            "ensure_nltk_data()\n"
            "scored = add_vader_sentiment(PAGES)\n"
            "print(scored.groupby('title')['compound'].agg(['mean', 'std', 'min', 'max']).round(3))\n"
        ),
        md("## Overlay: eight novels, one axis"),
        code(
            "fig = go.Figure()\n"
            "for book_id, grp in scored.groupby('book_id', sort=False):\n"
            "    fig.add_trace(go.Scatter(\n"
            "        x=grp['progress'] * 100,\n"
            "        y=grp['compound_smooth'],\n"
            "        mode='lines',\n"
            "        name=title_of(book_id),\n"
            "        line=dict(color=BOOK_COLORS[book_id], width=2),\n"
            "        hovertemplate='%{x:.0f}% progress<br>sentiment %{y:.2f}<extra>' + title_of(book_id) + '</extra>',\n"
            "    ))\n"
            "fig.update_layout(\n"
            "    title='Smoothed VADER sentiment vs progress',\n"
            "    xaxis_title='Progress through the book (%)',\n"
            "    yaxis_title='Compound sentiment (rolling mean)',\n"
            "    template='plotly_white',\n"
            "    height=520,\n"
            "    legend=dict(orientation='h', y=-0.2),\n"
            "    shapes=[dict(type='line', x0=0, x1=100, y0=0, y1=0,\n"
            "                 line=dict(color='gray', width=1, dash='dot'))],\n"
            ")\n"
            "display_plotly(fig)\n"
        ),
        md("Static copy for the project landing page:"),
        code(
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "for book_id, grp in scored.groupby('book_id', sort=False):\n"
            "    ax.plot(grp['progress'] * 100, grp['compound_smooth'],\n"
            "            color=BOOK_COLORS[book_id], label=title_of(book_id), linewidth=1.8)\n"
            "ax.axhline(0, color='gray', linestyle=':', linewidth=1)\n"
            "ax.set_xlabel('Progress through the book (%)')\n"
            "ax.set_ylabel('Compound sentiment (rolling mean)')\n"
            "ax.set_title('Smoothed VADER sentiment vs progress')\n"
            "ax.legend(loc='lower left', fontsize=8, ncol=2)\n"
            "plt.tight_layout()\n"
            "fig.savefig(PROJ_DIR / 'figures' / '02_sentiment_overlay.png', dpi=120)\n"
            "plt.show()\n"
        ),
        md("## One panel per book"),
        code(
            "facet = scored.copy()\n"
            "facet['Progress (%)'] = facet['progress'] * 100\n"
            "fig = px.line(\n"
            "    facet, x='Progress (%)', y='compound_smooth', color='title', facet_col='title',\n"
            "    facet_col_wrap=2, color_discrete_map={title_of(k): v for k, v in BOOK_COLORS.items()},\n"
            "    labels={'compound_smooth': 'Sentiment'},\n"
            ")\n"
            "fig.update_layout(template='plotly_white', height=900, showlegend=False,\n"
            "                  title='Sentiment trajectory by novel')\n"
            "fig.update_yaxes(matches=None)\n"
            "fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))\n"
            "display_plotly(fig)\n"
        ),
        md(
            "## Does the ending darken?\n\n"
            "Mean compound score in the first third versus the last third. "
            "A negative delta means the close is darker than the opening."
        ),
        code(
            "ends = scored.copy()\n"
            "ends['third'] = third_label(ends['progress'])\n"
            "pivot = (\n"
            "    ends.groupby(['title', 'third'], observed=True)['compound'].mean()\n"
            "    .unstack('third')[['beginning', 'end']]\n"
            ")\n"
            "pivot['delta (end − start)'] = pivot['end'] - pivot['beginning']\n"
            "pivot['arc'] = np.where(pivot['delta (end − start)'] < -0.02, 'darkens',\n"
            "                        np.where(pivot['delta (end − start)'] > 0.02, 'brightens', 'flat'))\n"
            "pivot.round(3).sort_values('delta (end − start)')\n"
        ),
        md(
            "## How to read the arcs\n\n"
            "- **A Christmas Carol** is the control case: five staves from miserliness to generosity, "
            "so the line should rise.\n"
            "- **Pride and Prejudice** stays relatively warm — social friction, not gothic dread.\n"
            "- **Alice** is playful; dips are more likely nonsense-court tension than tragedy.\n"
            "- **Dracula**, **Frankenstein**, and **Dorian Gray** are the titles we expect to finish darker "
            "than they begin. VADER may still score a triumphant vampire-hunt ending as “positive” "
            "because of words like *win*, *light*, and *god*. That is a feature of the lexicon, not a plot spoiler."
        ),
        footer("01_corpus.ipynb", "Corpus & Cleaning", "03_emotions.ipynb", "Emotion Lexicon"),
    ]
    save("02_sentiment.ipynb", cells)


def nb03():
    cells = [
        nav(
            "Emotion Lexicon",
            "02_sentiment.ipynb", "Sentiment Trajectories",
            "04_term_frequency.ipynb", "Term Frequency",
            "NRC emotions — joy, fear, sadness, anger, trust, anticipation — as a richer map than a single polarity score.",
        ),
        md(
            "## Method\n\n"
            "The [NRC Emotion Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) "
            "(Mohammad & Turney) tags words with eight basic emotions. We use the `nrclex` package "
            "and keep six: **joy, fear, sadness, anger, trust, anticipation**. "
            "Each page is summarised as affect *frequencies* (share of emotion-bearing tokens), "
            "then smoothed with the same 8-page rolling mean as the sentiment chapter.\n\n"
            "Binary polarity in the previous notebook can hide a book that is both joyful and fearful. "
            "Gothic fiction often is."
        ),
        code(IMPORTS),
        code(
            "emo = add_nrc_emotions(PAGES)\n"
            "mean_emo = emo.groupby('title')[list(NRC_EMOTIONS)].mean().round(3)\n"
            "mean_emo\n"
        ),
        md("## Mean emotional palette"),
        code(
            "heat = mean_emo.copy()\n"
            "fig = px.imshow(\n"
            "    heat, color_continuous_scale='YlOrRd', aspect='auto',\n"
            "    labels=dict(color='Mean frequency'),\n"
            "    title='Average NRC emotion share by novel',\n"
            ")\n"
            "fig.update_layout(template='plotly_white', height=480)\n"
            "display_plotly(fig)\n"
        ),
        md(
            "## Joy versus fear along the plot\n\n"
            "Two emotions that ought to trade places in a gothic novel, and stay inverted in a comedy of manners."
        ),
        code(
            "long = emo.melt(\n"
            "    id_vars=['title', 'book_id', 'progress'],\n"
            "    value_vars=['joy_smooth', 'fear_smooth'],\n"
            "    var_name='emotion', value_name='frequency',\n"
            ")\n"
            "long['emotion'] = long['emotion'].str.replace('_smooth', '')\n"
            "long['Progress (%)'] = long['progress'] * 100\n"
            "fig = px.line(\n"
            "    long, x='Progress (%)', y='frequency', color='emotion', facet_col='title',\n"
            "    facet_col_wrap=2, color_discrete_map={'joy': '#c9a227', 'fear': '#4a0e0e'},\n"
            ")\n"
            "fig.update_layout(template='plotly_white', height=900, title='Joy vs fear vs progress')\n"
            "fig.update_yaxes(matches=None)\n"
            "fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))\n"
            "display_plotly(fig)\n"
        ),
        md("## *A Christmas Carol* — five staves, six emotions"),
        code(
            "carol = emo.loc[emo['book_id'] == 'christmas_carol']\n"
            "fig = go.Figure()\n"
            "palette = {\n"
            "    'joy': '#c9a227', 'trust': '#2e7d32', 'anticipation': '#1565c0',\n"
            "    'fear': '#4a0e0e', 'sadness': '#546e7a', 'anger': '#c62828',\n"
            "}\n"
            "for emo_name in NRC_EMOTIONS:\n"
            "    fig.add_trace(go.Scatter(\n"
            "        x=carol['progress'] * 100, y=carol[f'{emo_name}_smooth'],\n"
            "        mode='lines', name=emo_name, line=dict(color=palette[emo_name], width=2),\n"
            "        stackgroup='one',\n"
            "    ))\n"
            "fig.update_layout(\n"
            "    title='A Christmas Carol — stacked NRC emotions',\n"
            "    xaxis_title='Progress (%)', yaxis_title='Smoothed frequency',\n"
            "    template='plotly_white', height=480,\n"
            ")\n"
            "display_plotly(fig)\n"
            "print('Stave headings:')\n"
            "print(carol.groupby('chapter_title')['page_idx'].min().sort_values())\n"
        ),
        md(
            "## Reading the lexicon\n\n"
            "*Dracula* and *Frankenstein* should load on **fear** and **sadness**; "
            "*Pride and Prejudice* on **trust** and **joy**; "
            "*A Christmas Carol* should shift toward joy and trust after the last ghost. "
            "NRC is still a word list — *heart* is tagged positively even when it is breaking — "
            "so spikes want a glance at the surrounding page, not a causal story."
        ),
        footer("02_sentiment.ipynb", "Sentiment Trajectories", "04_term_frequency.ipynb", "Term Frequency"),
    ]
    save("03_emotions.ipynb", cells)


def nb04():
    cells = [
        nav(
            "Term Frequency vs Page",
            "03_emotions.ipynb", "Emotion Lexicon",
            "05_wordclouds.ipynb", "Word Clouds",
            "Thematic keywords as time series, then TF–IDF to show which words actually distinguish each novel.",
        ),
        md(
            "## Keyword trajectories\n\n"
            "For each book we track a handful of plot-bearing words (counts per page, then an 8-page "
            "rolling mean). This is closer to close reading than a sentiment score: you can see *blood* "
            "arrive in *Dracula*, *morlock* in *The Time Machine*, and *christmas* take over the Carol."
        ),
        code(IMPORTS),
        code(
            "def keyword_figure(book_id):\n"
            "    df = keyword_counts(PAGES, book_id)\n"
            "    words = THEMATIC_KEYWORDS[book_id]\n"
            "    fig = go.Figure()\n"
            "    for word in words:\n"
            "        smooth = df[word].rolling(8, min_periods=1, center=True).mean()\n"
            "        fig.add_trace(go.Scatter(\n"
            "            x=df['progress'] * 100, y=smooth, mode='lines', name=word,\n"
            "        ))\n"
            "    fig.update_layout(\n"
            "        title=f'{title_of(book_id)} — keyword frequency vs progress',\n"
            "        xaxis_title='Progress (%)', yaxis_title='Mentions per page (rolling mean)',\n"
            "        template='plotly_white', height=420, legend=dict(orientation='h', y=-0.2),\n"
            "    )\n"
            "    return fig\n"
            "\n"
            "for book_id in ['dracula', 'pride_and_prejudice', 'time_machine', 'christmas_carol']:\n"
            "    display_plotly(keyword_figure(book_id))\n"
        ),
        md(
            "## Distinctive vocabulary (TF–IDF)\n\n"
            "Raw counts favour function words. TF–IDF down-weights terms that appear in every novel "
            "and surfaces the words that are *characteristic* of one title relative to the other seven. "
            "Each book is treated as a single document (all pages concatenated)."
        ),
        code(
            "from sklearn.feature_extraction.text import TfidfVectorizer\n"
            "\n"
            "docs, labels = [], []\n"
            "for book_id, grp in PAGES.groupby('book_id', sort=False):\n"
            "    docs.append(' '.join(grp['text'].astype(str)))\n"
            "    labels.append(title_of(book_id))\n"
            "\n"
            "stops = list(all_stopwords())\n"
            "vec = TfidfVectorizer(stop_words=stops, max_features=4000, min_df=1, max_df=0.9)\n"
            "X = vec.fit_transform(docs)\n"
            "terms = np.array(vec.get_feature_names_out())\n"
            "\n"
            "top_n = 12\n"
            "chosen = []\n"
            "for row in X.toarray():\n"
            "    chosen.extend(terms[row.argsort()[::-1][:top_n]])\n"
            "chosen = list(dict.fromkeys(chosen))[:40]\n"
            "idx = [np.where(terms == t)[0][0] for t in chosen]\n"
            "heat = pd.DataFrame(X.toarray()[:, idx], index=labels, columns=chosen)\n"
            "\n"
            "fig = px.imshow(\n"
            "    heat, color_continuous_scale='YlOrRd', aspect='auto',\n"
            "    labels=dict(color='TF–IDF'),\n"
            "    title='Distinctive terms (TF–IDF) across the eight novels',\n"
            ")\n"
            "fig.update_layout(template='plotly_white', height=560)\n"
            "fig.update_xaxes(tickangle=45)\n"
            "display_plotly(fig)\n"
        ),
        md("Top five TF–IDF terms per novel:"),
        code(
            "rows = []\n"
            "matrix = X.toarray()\n"
            "for i, label in enumerate(labels):\n"
            "    order = matrix[i].argsort()[::-1][:5]\n"
            "    rows.append({'title': label, 'top terms': ', '.join(terms[order])})\n"
            "pd.DataFrame(rows)\n"
        ),
        footer("03_emotions.ipynb", "Emotion Lexicon", "05_wordclouds.ipynb", "Word Clouds"),
    ]
    save("04_term_frequency.ipynb", cells)


def nb05():
    cells = [
        nav(
            "Word Clouds",
            "04_term_frequency.ipynb", "Term Frequency",
            "06_characters.ipynb", "Character Presence",
            "Vocabulary at a glance: one cloud per novel, then beginning / middle / end thirds for three contrasting plots.",
        ),
        md(
            "## Method\n\n"
            "Word clouds are a blunt instrument, but they are a good *first* look at what remains "
            "after English stopwords and Gutenberg artefacts (`chapter`, `gutenberg`, `said`, `mr`…) "
            "are removed. Later thirds of a book should not look like the opening if the plot has moved."
        ),
        code(IMPORTS),
        code(
            "from wordcloud import WordCloud\n"
            "import matplotlib.font_manager as fm\n"
            "\n"
            "STOPS = all_stopwords()\n"
            "FONT = fm.findfont(fm.FontProperties(family='DejaVu Sans'))\n"
            "\n"
            "def cloud_image(text, width=700, height=420):\n"
            "    wc = WordCloud(\n"
            "        width=width, height=height, background_color='white',\n"
            "        stopwords=STOPS, colormap='copper', max_words=80, collocations=False,\n"
            "        random_state=1, font_path=FONT,\n"
            "    )\n"
            "    return wc.generate(text).to_array()\n"
            "\n"
            "book_ids = [b['book_id'] for b in BOOKS]\n"
            "fig, axes = plt.subplots(4, 2, figsize=(12, 16))\n"
            "for ax, book_id in zip(axes.ravel(), book_ids):\n"
            "    text = ' '.join(PAGES.loc[PAGES['book_id'] == book_id, 'text'].astype(str))\n"
            "    ax.imshow(cloud_image(text), interpolation='bilinear')\n"
            "    ax.set_title(title_of(book_id), color=BOOK_COLORS[book_id])\n"
            "    ax.axis('off')\n"
            "plt.tight_layout()\n"
            "out = PROJ_DIR / 'figures' / '05_wordclouds.png'\n"
            "fig.savefig(out, dpi=110)\n"
            "plt.close()\n"
            "print('Saved', out)\n"
        ),
        md("![Whole-book word clouds](figures/05_wordclouds.png)"),
        md(
            "## Beginning, middle, end\n\n"
            "Three books with different promised arcs: gothic tightening (*Dracula*), "
            "redemption (*A Christmas Carol*), and social resolution (*Pride and Prejudice*)."
        ),
        code(
            "focus = ['dracula', 'christmas_carol', 'pride_and_prejudice']\n"
            "thirds = PAGES.copy()\n"
            "thirds['third'] = third_label(thirds['progress'])\n"
            "fig, axes = plt.subplots(len(focus), 3, figsize=(12, 10))\n"
            "for i, book_id in enumerate(focus):\n"
            "    for j, label in enumerate(['beginning', 'middle', 'end']):\n"
            "        chunk = thirds.loc[(thirds['book_id'] == book_id) & (thirds['third'] == label), 'text']\n"
            "        axes[i, j].imshow(cloud_image(' '.join(chunk.astype(str)), width=500, height=320),\n"
            "                          interpolation='bilinear')\n"
            "        axes[i, j].axis('off')\n"
            "        bits = []\n"
            "        if j == 0:\n"
            "            bits.append(title_of(book_id))\n"
            "        if i == 0:\n"
            "            bits.append(label.capitalize())\n"
            "        if bits:\n"
            "            axes[i, j].set_title(' — '.join(bits) if len(bits) == 2 else bits[0])\n"
            "plt.tight_layout()\n"
            "thirds_path = PROJ_DIR / 'figures' / '05_wordclouds_thirds.png'\n"
            "fig.savefig(thirds_path, dpi=110)\n"
            "plt.close()\n"
            "print('Saved', thirds_path)\n"
        ),
        md("![Beginning, middle, and end word clouds](figures/05_wordclouds_thirds.png)"),
        md(
            "If the method is working, Carol’s closing third should look more like *christmas / blessing / good* "
            "than *counting-house / clerk*; Dracula should pick up hunt-and-night vocabulary; "
            "Elizabeth and Darcy should still dominate Austen, with *marriage* more visible late than early."
        ),
        footer("04_term_frequency.ipynb", "Term Frequency", "06_characters.ipynb", "Character Presence"),
    ]
    save("05_wordclouds.ipynb", cells)


def nb06():
    cells = [
        nav(
            "Character Presence",
            "05_wordclouds.ipynb", "Word Clouds",
            "index.md", "Project Overview",
            "Who is on the page? Mention rates reconstruct plot occupancy without a parser or a spoiler-heavy summary.",
        ),
        md(
            "## Method\n\n"
            "Named-entity models need a downloaded spaCy pipeline and still confuse *Miss Bennet* with "
            "the wrong sister. For a small, famous corpus, **curated alias lists** are more reliable: "
            "Elizabeth / Lizzy / Eliza, Mina / Mrs. Harker, Scrooge, Heathcliff.\n\n"
            "Each alias is counted with a word-boundary regex on the page window. "
            "The chart is an 8-page rolling mean of mentions per page — a stage-direction track."
        ),
        code(IMPORTS),
        code(
            "def character_figure(book_id):\n"
            "    df = character_mentions(PAGES, book_id)\n"
            "    names = [c for c in df.columns if c not in PAGES.columns]\n"
            "    fig = go.Figure()\n"
            "    for name in names:\n"
            "        smooth = df[name].rolling(8, min_periods=1, center=True).mean()\n"
            "        fig.add_trace(go.Scatter(\n"
            "            x=df['progress'] * 100, y=smooth, mode='lines', name=name,\n"
            "        ))\n"
            "    fig.update_layout(\n"
            "        title=f'{title_of(book_id)} — character mentions vs progress',\n"
            "        xaxis_title='Progress (%)', yaxis_title='Mentions per page (rolling mean)',\n"
            "        template='plotly_white', height=420, legend=dict(orientation='h', y=-0.25),\n"
            "    )\n"
            "    return fig, names, df\n"
        ),
        md("## Four plots as occupancy charts"),
        code(
            "for book_id in ['dracula', 'wuthering_heights', 'pride_and_prejudice', 'christmas_carol']:\n"
            "    fig, names, _ = character_figure(book_id)\n"
            "    display_plotly(fig)\n"
        ),
        md("## Who dominates the last third?"),
        code(
            "rows = []\n"
            "for book in BOOKS:\n"
            "    df = character_mentions(PAGES, book['book_id'])\n"
            "    names = [c for c in df.columns if c not in PAGES.columns]\n"
            "    last = df.loc[df['progress'] >= 2 / 3, names].sum()\n"
            "    first = df.loc[df['progress'] < 1 / 3, names].sum()\n"
            "    if last.empty:\n"
            "        continue\n"
            "    rows.append({\n"
            "        'title': book['title'],\n"
            "        'opening lead': first.idxmax() if first.sum() else '—',\n"
            "        'closing lead': last.idxmax() if last.sum() else '—',\n"
            "        'closing mentions': int(last.max()) if last.sum() else 0,\n"
            "    })\n"
            "pd.DataFrame(rows)\n"
        ),
        md(
            "## What the occupancy charts should show\n\n"
            "- **Dracula:** Jonathan’s journal opens the book; Mina, Van Helsing, and the Count "
            "share the hunt in the close.\n"
            "- **Wuthering Heights:** Heathcliff is a constant; the second-generation names "
            "(Hareton, the younger Cathy) rise later.\n"
            "- **Pride and Prejudice:** Elizabeth is the baseline; Darcy’s mentions should climb "
            "after the first proposal / letter stretch.\n"
            "- **A Christmas Carol:** Scrooge is everywhere; the Ghosts arrive in sequence, "
            "Tiny Tim clusters in the Cratchit scenes and the end.\n\n"
            "Alias lists are still a compromise. *Bob* will catch Bob Cratchit and the occasional "
            "unrelated *bob*; *creature* in *Frankenstein* is the point of the book and also a generic noun. "
            "Read the lines as stage lighting, not as a census."
        ),
        footer("05_wordclouds.ipynb", "Word Clouds", "index.md", "Project Overview"),
    ]
    save("06_characters.ipynb", cells)


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
