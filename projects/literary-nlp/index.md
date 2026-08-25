# Literary NLP: Sentiment Across Classic Novels

<div style="background: linear-gradient(135deg, #4a0e0e 0%, #c9a227 100%); color: white; padding: 2em; border-radius: 10px; margin: 2em 0; text-align: center;">
  <h2 style="color: white; margin-top: 0;">Mapping mood through eight public-domain novels</h2>
  <p style="font-size: 1.1em; margin-bottom: 0;">Sentiment, emotion, vocabulary, and character presence as the plot unfolds</p>
</div>

## Project Overview

Do novels darken as they head for the ending, or do they brighten? This project treats classic fiction as a time series: each book is split into page-sized windows of text, then a different NLP technique is applied on each page to watch language and mood change from opening to close.

The corpus is eight well-known public-domain titles from [Project Gutenberg](https://www.gutenberg.org/) — gothic horror, a redemption fable, a marriage plot, and a trip down a rabbit-hole — chosen so the trajectories have something to disagree about.

Lexicon methods (VADER and the NRC emotion lexicon) are used throughout. They are fast, fully reproducible in a Jupyter Book build, and good enough to compare *relative* arcs. They are not a substitute for a literary reading: irony, archaic diction, and free indirect style can fool a word list.

## Project Structure

Six notebooks, each applying one technique to the same eight books:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5em; margin: 2em 0;">

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #4a0e0e;">
  <h3 style="margin-top: 0; color: #4a0e0e;">1. Corpus &amp; Cleaning</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="01_corpus.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Gutenberg texts, boilerplate stripping, page windows, chapter splits</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #6d4c41;">
  <h3 style="margin-top: 0; color: #6d4c41;">2. Sentiment Trajectories</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="02_sentiment.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">VADER compound scores vs progress — does the ending darken or brighten?</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #ad1457;">
  <h3 style="margin-top: 0; color: #ad1457;">3. Emotion Lexicon</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="03_emotions.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">NRC joy, fear, sadness, anger, trust, and anticipation through the plot</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #1565c0;">
  <h3 style="margin-top: 0; color: #1565c0;">4. Term Frequency</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="04_term_frequency.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Keyword trajectories vs page, and TF–IDF distinctive vocabulary</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #6a1b9a;">
  <h3 style="margin-top: 0; color: #6a1b9a;">5. Word Clouds</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="05_wordclouds.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Whole-book clouds and beginning / middle / end vocabulary shift</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #c9a227;">
  <h3 style="margin-top: 0; color: #8a7010;">6. Character Presence</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="06_characters.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Who is on the page as the story moves — mention rates vs progress</p>
</div>

</div>

## Key Questions

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; margin: 1.5em 0;">

- Which titles become darker in the final third, and which recover?
- Do emotion lexicons pick up gothic fear versus Austenite trust?
- Which words are distinctive to each novel once stopwords are stripped?
- Can character mention-rates reconstruct a plot without reading it?

</div>

## Preview Figures

![Corpus size by novel](figures/01_corpus_lengths.png)

![Smoothed sentiment vs progress](figures/02_sentiment_overlay.png)

![Word clouds for the eight novels](figures/05_wordclouds.png)

## Dataset

<div style="background: #e8f4f8; padding: 1.5em; border-radius: 8px; margin: 1.5em 0;">

- **Source:** [Project Gutenberg](https://www.gutenberg.org/) plain-text ebooks
- **Titles:** *Dracula*, *Wuthering Heights*, *The Time Machine*, *Pride and Prejudice*, *Frankenstein*, *Alice’s Adventures in Wonderland*, *A Christmas Carol*, *The Picture of Dorian Gray*
- **Unit of analysis:** ~250-word page windows (a Gutenberg file has no printer pagination)
- **Bundled file:** `data/pages.csv` — committed so CI does not fetch Gutenberg at build time

</div>

## Technical Stack

**Python** • NLTK (VADER) • NRCLex • scikit-learn (TF–IDF) • wordcloud • Plotly • pandas

---

<div style="text-align: center; margin: 2em 0; padding: 1.5em; background: #f0f0f0; border-radius: 8px;">
  <p style="font-size: 1.2em; margin: 0;"><strong>Ready to explore?</strong></p>
  <p style="margin: 0.5em 0 0 0;"><a href="01_corpus.html" style="background: #4a0e0e; color: white; padding: 0.7em 2em; text-decoration: none; border-radius: 5px; display: inline-block;">Start with the corpus →</a></p>
</div>
