# Build interactive Beatles visuals

This page shows how to generate interactive charts from the Beatles dataset:

- [The Beatles Spotify Dataset (Kaggle)](https://www.kaggle.com/datasets/jarredpriester/the-beatles-spotify-dataset)

## 1) Download the data

Download the Kaggle CSV and place it at:

`projects/beatles-discography/data/beatles_spotify_dataset.csv`

If Kaggle CLI is configured:

```bash
kaggle datasets download -d jarredpriester/the-beatles-spotify-dataset -p projects/beatles-discography/data --unzip
```

## 2) Generate the visual outputs

Run from repository root:

```bash
python3 projects/beatles-discography/beatles_visuals.py
```

This script will create:

- `projects/beatles-discography/outputs/writer_popularity_by_writer.html`
- `projects/beatles-discography/outputs/sunburst_discography_by_album.html`
- `projects/beatles-discography/outputs/sunburst_discography_by_writer.html`

## 3) What each visual shows

### Writer popularity by writer / writers

- Aggregates songs by writer (splits collaborations like "Lennon and McCartney")
- Computes average popularity score for each writer
- Colors bars by number of unique songs

This answers: **Which writers are associated with higher-popularity songs?**

### Sunburst: full discography by album

- Hierarchy: **Album -> Song**
- Segment size and color reflect popularity score
- Useful for album-level comparisons and hit concentration

### Sunburst: full discography by writer

- Hierarchy: **Writer -> Album -> Song**
- Highlights how writing contributions are spread across albums
- Makes collaborations and writer dominance visually obvious

## Implementation notes

The script supports common schema variants:

- Song columns: `song`, `title`, `track_name`, `name`
- Writer columns: `writer`, `writers`, `songwriter`, `songwriters`
- Popularity columns: `popularity`, `track_popularity`, `top.50.billboard`, etc.

If writer fields are missing, it attempts an enrichment join using a public Beatles songwriter reference dataset keyed by song title.
