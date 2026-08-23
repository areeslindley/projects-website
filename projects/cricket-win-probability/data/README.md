# Data folder

This project uses ball-by-ball ODI data from [Cricsheet](https://cricsheet.org/) (ODC-BY 1.0 — attribution required).

## Bundled files (for reproducible builds)

- `sample_matches/` — three case-study matches in Cricsheet JSON format
- `odi_training_sample.csv` — pre-built ball-state training rows (~120k rows from 400 simulated ODIs)

## Full Cricsheet download (optional)

To rebuild the training sample from real matches, download the ODI archive:

```bash
curl -L -o /tmp/odis_json.zip https://cricsheet.org/downloads/odis_json.zip
unzip /tmp/odis_json.zip -d projects/cricket-win-probability/data/raw/
```

Then adapt `_build_data.py` to parse files from `data/raw/`.

## Regenerate bundled data

From the project directory:

```bash
python _build_data.py
python _generate_notebooks.py
```

## Attribution

Match data © Cricsheet contributors, used under the [Open Data Commons Attribution License](https://opendatacommons.org/licenses/by/1-0/).
