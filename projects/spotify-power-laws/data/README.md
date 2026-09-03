# Data folder

The Kaggle file is **not committed**. Notebooks load `artifacts/` and `figures/` so GitHub Actions does not depend on Kaggle.

## Source

- **Dataset:** [beamhonor0911/spotify-artist-streaming-analytics-20202025](https://www.kaggle.com/datasets/beamhonor0911/spotify-artist-streaming-analytics-20202025)
- **File:** `spotify_artist_streaming_2020_2025.csv` (~12 MB, 50,000 × 33)
- **Licence:** CC BY 4.0
- **Publisher statement:** fully synthetic; no real Spotify API data was used

## Rebuild

From this project directory (network required):

```bash
make download
```

or from the repository root:

```bash
python projects/spotify-power-laws/scripts/00_download.py
```

The script prefers the Kaggle API if credentials exist, and otherwise uses the public dataset zip endpoint. The SHA-256 of the downloaded file is written to `artifacts/download.json`.

## Grain

One row is one **track**, not an artist-year. There are 500 invented artist names. `release_date` is a release calendar, not a listening panel.
