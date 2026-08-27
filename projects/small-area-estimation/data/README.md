# Data folder

Local-authority inputs for the small-area estimation project. Notebooks read **only** these bundled files so GitHub Actions does not depend on Nomis uptime.

## Bundled files

- `la_sae_panel.csv` — one row per LAD/UA (April 2023 geography): APS direct unemployment rate and CI, sampling variance $\psi_i$, claimant-count rate, APS inactivity, ONS model-based rate
- `la_boundaries.geojson` — simplified ONS Open Geography LAD December 2023 BGC polygons
- `ATTRIBUTION.txt` — Open Government Licence notice

## Refresh from Nomis / ONS

From the repository root (requires network):

```bash
python projects/small-area-estimation/_build_data.py
```

Raw Nomis CSV pulls are written to `data/raw/` (not committed).

## Attribution

Contains public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

**Source:** Office for National Statistics and [Nomis](https://www.nomisweb.co.uk/) (official census and labour market statistics).

| Series | Nomis dataset | Period used in the bundled panel |
| --- | --- | --- |
| APS unemployment rate, 16–64 | `NM_17_5` variable 84 | Apr 2025–Mar 2026 |
| APS economic inactivity rate, 16–64 | `NM_17_5` variable 111 | Apr 2025–Mar 2026 |
| Model-based unemployment rate | `NM_127_1` item 2 | Apr 2025–Mar 2026 |
| Claimant count, % of 16–64 residents | `NM_162_1` measure 2 | March 2026 |
| LAD boundaries | ONS Open Geography (ArcGIS FeatureServer) | December 2023 BGC |
