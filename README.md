# Data science portfolio (Jupyter Book)

Personal portfolio of data science and statistics projects, built as a [Jupyter Book](https://jupyterbook.org/) and published to GitHub Pages.

## Live site

**https://areeslindley.github.io/projects-website/**

The book opens on the portfolio home page; use the left sidebar to move between sections.

## Projects

Brief summaries of what is in the book today. This list will grow as new chapters are added to `_toc.yml`.

| Project | Description |
| -------- | ----------- |
| **Cancer imaging** | BreastMNIST ultrasound: dummy → PCA-logistic → HOG/LBP forest → CNN → ResNet-18, with Grad-CAM and a full classification-metric dashboard. |
| **Small area estimation** | Fay–Herriot EBLUP and Bayesian hierarchical model for APS unemployment rates at UK local-authority level, validated against ONS model-based estimates. |
| **Literary NLP** | Sentiment, emotion, term frequency, word clouds, and character presence mapped through eight public-domain novels from Project Gutenberg. |
| **Titanic survival analysis** | End-to-end classification workflow: exploratory analysis, data cleaning, modelling, and evaluation with clear narrative and figures. |
| **French rugby Voronoi analysis** | Geospatial view of French professional rugby using Voronoi-style territory ideas, computational geometry, and interactive mapping for sports analytics. |
| **Beatles discography visual story** | Interactive visuals from a Beatles Spotify-style dataset: songwriter-level popularity, sunburst views of the catalogue by album and by writer, and a static chart gallery. |
| **UK terrain rugby** | Raster and vector spatial analysis of UK elevation with rugby club locations. |
| **UK urban systems network** | TTWA nodes, census commuting flows, and Nystuen–Dacey partitions. |
| **Time series analysis** | Classical forecasting: exponential smoothing, Fourier methods, ARIMA/SARIMA, and state-space models. |
| **Cricket win probability** | Ball-by-ball ODI win probability from DLS-style resources and calibrated ML models. |

## Source and local build

- **Repository:** https://github.com/areeslindley/projects-website  
- **Local HTML:** after `pip install -r requirements.txt`, run `jupyter-book build .` and open `_build/html/index.html` in a browser.

Deployment to the live URL above runs via GitHub Actions on pushes to `main` (see `.github/workflows/deploy.yml`).
