"""Power Laws and Popularity — analysis package for the synthetic Spotify catalog."""

from .paths import artifacts_dir, data_dir, figures_dir, project_dir, raw_csv_path

SEED = 20260903
KAGGLE_SLUG = "beamhonor0911/spotify-artist-streaming-analytics-20202025"
CSV_NAME = "spotify_artist_streaming_2020_2025.csv"
EXPECTED_ROWS = 50_000
EXPECTED_COLS = 33
EXPECTED_ARTISTS = 500

__all__ = [
    "SEED",
    "KAGGLE_SLUG",
    "CSV_NAME",
    "EXPECTED_ROWS",
    "EXPECTED_COLS",
    "EXPECTED_ARTISTS",
    "project_dir",
    "data_dir",
    "artifacts_dir",
    "figures_dir",
    "raw_csv_path",
]
