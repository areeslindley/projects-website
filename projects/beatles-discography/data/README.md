# Data folder

Place your Kaggle download in this folder as:

- `beatles_spotify_dataset.csv`

The project uses the dataset from:

<https://www.kaggle.com/datasets/jarredpriester/the-beatles-spotify-dataset>

If you have Kaggle CLI configured, run from repository root:

```bash
kaggle datasets download -d jarredpriester/the-beatles-spotify-dataset -p projects/beatles-discography/data --unzip
```

Then rename the extracted CSV (if needed) to:

`projects/beatles-discography/data/beatles_spotify_dataset.csv`
