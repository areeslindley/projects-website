# UK Terrain & Rugby Geography

<div style="background: linear-gradient(135deg, #1f4e79 0%, #2d6a4f 100%); color: white; padding: 2em; border-radius: 10px; margin: 2em 0; text-align: center;">
  <h2 style="color: white; margin-top: 0;">🗺️ UK Terrain & Rugby Geography</h2>
  <p style="font-size: 1.1em; margin-bottom: 0;">Raster and vector spatial analysis with <code style="background: rgba(255,255,255,0.2); color: white; padding: 0.1em 0.4em; border-radius: 4px;">tidyterra</code></p>
</div>

## Project Overview

This project uses the **tidyterra** R package to work with raster and vector spatial data in a familiar **tidyverse**-style workflow. We combine freely available global grids—national SRTM-based elevation from **geodata** and (in the first notebook) WorldClim temperature—with country boundaries from **rnaturalearth**, then overlay **English Premiership**, **United Rugby Championship**, **Super Rugby**, and **French Top 14** stadium locations to produce publication-quality maps.

The emphasis is on reproducible geospatial pipelines: harmonising coordinate reference systems, clipping rasters to national boundaries or regional extents, deriving categorical elevation bands, and linking point observations to gridded terrain.

## Project Structure

The analysis is organised into five R Markdown notebooks:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5em; margin: 2em 0;">

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #1f4e79;">
  <h3 style="margin-top: 0; color: #1f4e79;">⛰️ 1. Terrain &amp; climate</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="01_terrain_analysis.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">UK elevation (hypsometric mapping) and faceted seasonal mean temperature from WorldClim</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #d62728;">
  <h3 style="margin-top: 0; color: #d62728;">🏉 2. Premiership overlay</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="02_rugby_overlay.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">English Premiership grounds on UK elevation, table, bar chart, and map</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #9467bd;">
  <h3 style="margin-top: 0; color: #9467bd;">🏆 3. URC overlay</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="03_urc_overlay.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">United Rugby Championship stadiums across Ireland, the UK, Italy, and South Africa</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #e377c2;">
  <h3 style="margin-top: 0; color: #c71585;">🌏 4. Super Rugby overlay</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="04_super_rugby_overlay.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Super Rugby franchises in New Zealand, Australia, and South Africa — elevation table, bar charts, static maps, and interactive Leaflet maps</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #2ca02c;">
  <h3 style="margin-top: 0; color: #2ca02c;">🇫🇷 5. Top 14 overlay</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="05_top14_overlay.html">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">French Top 14 grounds on national elevation — table, bar chart, static map, and interactive Leaflet map</p>
</div>

</div>

## Key Questions

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; margin: 1.5em 0;">

- How does **tidyterra** let us treat **SpatRaster** objects with **dplyr**-style verbs while keeping raster semantics?
- What do **multi-layer** rasters represent for monthly climate, and how do we pick representative **seasonal** slices?
- Where do **Premiership**, **URC**, **Super Rugby**, and **Top 14** stadiums sit in regional relief, and what does that imply about geography (e.g. coastal vs Highveld vs Massif Central)?

</div>

## Data Sources

<div style="background: #e8f4f8; padding: 1.5em; border-radius: 8px; margin: 1.5em 0;">

| Source | Role |
|--------|------|
| **geodata** | National elevation (`elevation_30s` for GBR, IRL, ITA, ZAF, NZL, AUS, FRA) and WorldClim country extracts (`worldclim_country`) |
| **rnaturalearth** | Country boundaries (medium scale) as **sf** / **SpatVector** |
| **Manual coordinates** | Premiership (10), URC (16), Super Rugby (14), and Top 14 (14) stadium centre points (WGS84); edit the notebooks to refine |

All downloads require an **internet** connection at build time; files are cached under `tempdir()` by default in the notebooks.

</div>

## Technical Approach

<div style="background: #fff5e6; padding: 1.5em; border-radius: 8px; border-left: 4px solid #ff7f0e; margin: 1.5em 0;">

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1em; margin-top: 1em;">

<div>
  <strong>Raster–vector alignment</strong><br>
  <span style="font-size: 0.9em; color: #666;">CRS checks before <code>crop</code> / <code>mask</code></span>
</div>

<div>
  <strong>tidyterra verbs</strong><br>
  <span style="font-size: 0.9em; color: #666;"><code>mutate</code>, <code>select</code>, <code>rename</code> on <code>SpatRaster</code></span>
</div>

<div>
  <strong>ggplot2 integration</strong><br>
  <span style="font-size: 0.9em; color: #666;"><code>geom_spatraster</code>, <code>geom_spatvector</code>, thematic palettes</span>
</div>

<div>
  <strong>Point extraction</strong><br>
  <span style="font-size: 0.9em; color: #666;"><code>terra::extract()</code> at stadium locations</span>
</div>

</div>

</div>

## Expected Outcomes

<div style="background: #e8f5e9; padding: 1.5em; border-radius: 8px; margin: 1.5em 0;">

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1em; margin-top: 1em;">

<div>
  ✅ <strong>Publication-style</strong> UK elevation and temperature maps
</div>

<div>
  ✅ <strong>Derived layers</strong> (elevation bands, seasonal subsets) using tidyterra
</div>

<div>
  ✅ <strong>Club-level</strong> terrain context for Premiership, URC, Super Rugby, and Top 14 geography
</div>

<div>
  ✅ <strong>Reproducible</strong> R Markdown notebooks for Jupyter Book
</div>

</div>

</div>

---

<div style="text-align: center; margin: 2em 0; padding: 1.5em; background: #f0f0f0; border-radius: 8px;">
  <p style="font-size: 1.2em; margin: 0;"><strong>Ready to explore?</strong></p>
  <p style="margin: 0.5em 0 0 0;"><a href="01_terrain_analysis.html" style="background: #1f4e79; color: white; padding: 0.7em 2em; text-decoration: none; border-radius: 5px; display: inline-block;">Start with terrain analysis →</a></p>
</div>
