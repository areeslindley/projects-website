"""Fetch and cache openly licensed ONS/NOMIS inputs for the SAE project.

Notebooks read the bundled CSVs so GitHub Actions does not depend on NOMIS
uptime. Re-run this script (network required) to refresh:

    python projects/small-area-estimation/_build_data.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import requests

PROJ = Path(__file__).parent
DATA = PROJ / "data"
RAW = DATA / "raw"
UA = {"User-Agent": "areeslindley-projects-website/sae (personal portfolio)"}
NOMIS = "https://www.nomisweb.co.uk/api/v01/dataset"
# District / unitary local authorities as of April 2023 (current GB+NI LADs).
GEO = "TYPE424"
# Align claimant snapshot with the end of the APS/model-based 12-month window
# when possible; fall back to latest if that date is missing.
CLAIMANT_TIME_PREF = "2026-03"


def _get(url: str, timeout: int = 120) -> str:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.text


def nomis_csv(dataset: str, **params) -> pd.DataFrame:
    q = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{NOMIS}/{dataset}.data.csv?{q}"
    print(f"GET {url}")
    df = pd.read_csv(io.StringIO(_get(url)))
    print(f"  -> {len(df):,} rows, period={df['DATE_NAME'].iloc[0] if len(df) else 'empty'}")
    return df


def pivot_measures(df: pd.DataFrame, value_name: str, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Spread MEASURES_NAME into columns, one row per local authority."""
    keep = ["GEOGRAPHY_NAME", "GEOGRAPHY_CODE", "DATE_NAME", "MEASURES_NAME", "OBS_VALUE", "OBS_STATUS"]
    extra_cols = extra_cols or []
    wide = (
        df[keep]
        .pivot_table(
            index=["GEOGRAPHY_CODE", "GEOGRAPHY_NAME", "DATE_NAME"],
            columns="MEASURES_NAME",
            values="OBS_VALUE",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    rename = {}
    if "Variable" in wide.columns:
        rename["Variable"] = value_name
    if "Value" in wide.columns:
        rename["Value"] = value_name
    if "Confidence" in wide.columns:
        rename["Confidence"] = f"{value_name}_ci"
    if "Numerator" in wide.columns:
        rename["Numerator"] = f"{value_name}_numerator"
    if "Denominator" in wide.columns:
        rename["Denominator"] = f"{value_name}_denominator"
    wide = wide.rename(columns=rename)
    status = (
        df.groupby(["GEOGRAPHY_CODE"])["OBS_STATUS"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
        .rename(f"{value_name}_status")
    )
    wide = wide.merge(status, on="GEOGRAPHY_CODE", how="left")
    return wide


def country_from_code(code: str) -> str:
    if not isinstance(code, str) or not code:
        return "Unknown"
    prefix = code[0]
    return {"E": "England", "W": "Wales", "S": "Scotland", "N": "Northern Ireland"}.get(prefix, "Unknown")


def fetch_aps_unemployment() -> pd.DataFrame:
    # Variable 84: unemployment rate, aged 16–64. Measures include the rate,
    # estimated numerator/denominator, and 95% CI half-width.
    df = nomis_csv(
        "NM_17_5",
        geography=GEO,
        variable=84,
        measures="20599,21001,21002,21003",
        time="latest",
    )
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW / "aps_unemployment_long.csv", index=False)
    wide = pivot_measures(df, "direct_rate")
    wide["direct_period"] = wide["DATE_NAME"]
    return wide.drop(columns=["DATE_NAME"])


def fetch_aps_inactivity() -> pd.DataFrame:
    # Variable 111: % economically inactive, aged 16–64 — auxiliary covariate.
    df = nomis_csv(
        "NM_17_5",
        geography=GEO,
        variable=111,
        measures="20599",
        time="latest",
    )
    df.to_csv(RAW / "aps_inactivity_long.csv", index=False)
    out = df.rename(columns={"OBS_VALUE": "inactivity_rate"})[
        ["GEOGRAPHY_CODE", "inactivity_rate"]
    ].drop_duplicates("GEOGRAPHY_CODE")
    return out


def fetch_model_based() -> pd.DataFrame:
    df = nomis_csv(
        "NM_127_1",
        geography=GEO,
        item=2,
        measures="20100,20701",
        time="latest",
    )
    df.to_csv(RAW / "ons_model_based_long.csv", index=False)
    wide = pivot_measures(df, "ons_mb_rate")
    wide["ons_mb_period"] = wide["DATE_NAME"]
    return wide.drop(columns=["DATE_NAME"])


def fetch_claimant() -> pd.DataFrame:
    params = dict(
        geography=GEO,
        gender=0,
        age=0,
        measure=2,
        measures=20100,
        time=CLAIMANT_TIME_PREF,
    )
    try:
        df = nomis_csv("NM_162_1", **params)
        if df["OBS_VALUE"].notna().sum() < 50:
            raise ValueError("preferred claimant date too sparse")
    except Exception as exc:
        print(f"Claimant time={CLAIMANT_TIME_PREF} failed ({exc}); using latest")
        params["time"] = "latest"
        df = nomis_csv("NM_162_1", **params)
    df.to_csv(RAW / "claimant_count_long.csv", index=False)
    out = df.rename(columns={"OBS_VALUE": "claimant_rate", "DATE_NAME": "claimant_period"})[
        ["GEOGRAPHY_CODE", "claimant_rate", "claimant_period"]
    ].drop_duplicates("GEOGRAPHY_CODE")
    return out


def fetch_boundaries() -> Path | None:
    """ONS Open Geography LAD 2023 BGC, simplified for the repo."""
    url = (
        "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
        "Local_Authority_Districts_December_2023_Boundaries_UK_BGC/FeatureServer/0/"
        "query?where=1%3D1&outFields=LAD23CD,LAD23NM&outSR=4326&f=geojson"
    )
    print(f"GET {url}")
    try:
        payload = json.loads(_get(url, timeout=180))
    except Exception as exc:
        print(f"Boundary download failed: {exc}")
        return None
    n = len(payload.get("features", []))
    print(f"  -> {n} features")
    if n == 0:
        return None

    try:
        import geopandas as gpd

        gdf = gpd.GeoDataFrame.from_features(payload, crs="EPSG:4326")
        gdf = gdf.rename(columns={"LAD23CD": "GEOGRAPHY_CODE", "LAD23NM": "GEOGRAPHY_NAME"})
        gdf["geometry"] = gdf["geometry"].simplify(0.01, preserve_topology=True)
        out = DATA / "la_boundaries.geojson"
        gdf.to_file(out, driver="GeoJSON")
        print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
        return out
    except Exception as exc:
        print(f"geopandas simplify failed ({exc}); writing raw GeoJSON")
        out = DATA / "la_boundaries.geojson"
        out.write_text(json.dumps(payload))
        return out


def assemble_panel() -> pd.DataFrame:
    direct = fetch_aps_unemployment()
    inactive = fetch_aps_inactivity()
    ons = fetch_model_based()
    claimant = fetch_claimant()

    panel = direct.merge(ons, on=["GEOGRAPHY_CODE", "GEOGRAPHY_NAME"], how="outer")
    panel = panel.merge(inactive, on="GEOGRAPHY_CODE", how="left")
    panel = panel.merge(claimant, on="GEOGRAPHY_CODE", how="left")
    panel["country"] = panel["GEOGRAPHY_CODE"].map(country_from_code)

    # Sampling variance of the direct rate (percentage points).
    # NOMIS "Confidence" on APS percentages is the 95% CI half-width.
    panel["psi"] = (panel["direct_rate_ci"] / 1.96) ** 2
    panel["cv"] = panel["direct_rate_ci"] / (1.96 * panel["direct_rate"])
    panel["in_model"] = (
        panel["direct_rate"].notna()
        & panel["psi"].notna()
        & (panel["psi"] > 0)
        & panel["claimant_rate"].notna()
        & panel["ons_mb_rate"].notna()
        & panel["country"].isin(["England", "Wales", "Scotland"])
    )

    cols = [
        "GEOGRAPHY_CODE",
        "GEOGRAPHY_NAME",
        "country",
        "direct_period",
        "direct_rate",
        "direct_rate_ci",
        "direct_rate_numerator",
        "direct_rate_denominator",
        "psi",
        "cv",
        "claimant_rate",
        "claimant_period",
        "inactivity_rate",
        "ons_mb_rate",
        "ons_mb_rate_ci",
        "ons_mb_period",
        "in_model",
    ]
    panel = panel[cols].sort_values(["country", "GEOGRAPHY_NAME"]).reset_index(drop=True)
    return panel


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    panel = assemble_panel()
    out = DATA / "la_sae_panel.csv"
    panel.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(panel)} areas, {int(panel['in_model'].sum())} in modelling subset)")
    print(panel.groupby("country")["in_model"].agg(["size", "sum"]).to_string())
    fetch_boundaries()
    (DATA / "ATTRIBUTION.txt").write_text(
        "Contains public sector information licensed under the Open Government "
        "Licence v3.0.\n\n"
        "Source: Office for National Statistics and Nomis (official census and "
        "labour market statistics).\n"
        "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/\n"
        "https://www.nomisweb.co.uk/\n"
    )
    print("Done.")


if __name__ == "__main__":
    main()
