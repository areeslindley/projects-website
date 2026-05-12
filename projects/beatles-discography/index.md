# The Beatles Discography Visual Story

<div style="background: linear-gradient(135deg, #4b2e83 0%, #1f4e79 100%); color: white; padding: 2em; border-radius: 10px; margin: 2em 0; text-align: center;">
  <h2 style="color: white; margin-top: 0;">🎸 Beatles on Spotify: Writers, Albums, and Popularity</h2>
  <p style="font-size: 1.1em; margin-bottom: 0;">Interactive visual analytics across The Beatles catalogue</p>
</div>

## Project Overview

This project explores The Beatles Spotify dataset from Kaggle:

- **Dataset:** [The Beatles Spotify Dataset (Kaggle)](https://www.kaggle.com/datasets/jarredpriester/the-beatles-spotify-dataset)
- **Main goals:**
  1. Categorise song popularity by writer / writers
  2. Visualise the full discography with interactive sunburst charts

The analysis uses Python, pandas, and Plotly to create interactive visuals that can be exported as HTML.

## Visuals included

- **Writer popularity chart**: Average popularity and song count by writer (including collaborations)
- **Sunburst (Album view)**: Album → Song hierarchy for the whole discography
- **Sunburst (Writer view)**: Writer → Album → Song hierarchy to highlight writing patterns

## Data note

The code is designed to work with common Beatles dataset schemas. If writer fields are missing in your local extract, it can enrich writer information from a public Beatles songwriter reference file.

## Next step

Open one of the pages below:

- [Build the visuals](01_interactive_visuals.md)
- [See the live interactive visuals](02_live_interactive_visuals.ipynb)
