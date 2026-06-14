# Fetch live ONS / NOMIS inputs and write RDS caches for uk-urban-systems-network.
# Run from repo root: Rscript projects/uk-urban-systems-network/scripts/fetch_live_data.R

suppressPackageStartupMessages({
  library(tidyverse)
  library(sf)
  library(here)
  library(data.table)
  library(jsonlite)
  library(httr2)
})

proj_dir <- here::here("projects", "uk-urban-systems-network")
data_dir <- file.path(proj_dir, "data")
raw_dir <- file.path(data_dir, "raw")
scripts_dir <- file.path(proj_dir, "scripts")
dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)

source(file.path(scripts_dir, "fetch_scotland_od.R"))

ONS_TTWA_GEOJSON <- paste0(
  "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/",
  "Travel_to_Work_Areas_Dec_2011_GCB_in_United_Kingdom_2022/FeatureServer/0/",
  "query?where=1%3D1&outFields=*&f=geojson"
)

LSOA_TTWA_BASE <- paste0(
  "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/",
  "LSOA11_TTWA11_UK_LU_04a063122da94a2c9bc94bcef63866aa/FeatureServer/0/query"
)

# England and Wales MSOA → LSOA (2011), not England-only
MSOA_LSOA_EW_BASE <- paste0(
  "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/",
  "OA11_LSOA11_MSOA11_LAD11_EW_LUv2_b3fe7c68f4b2420185eaff6284d4c125/FeatureServer/0/query"
)

fetch_arcgis_attributes <- function(base_url, out_fields, page_size = 2000L) {
  offset <- 0L
  chunks <- list()
  repeat {
    qs <- list(
      where = "1=1",
      outFields = paste(out_fields, collapse = ","),
      resultOffset = offset,
      resultRecordCount = page_size,
      f = "json"
    )
    resp <- jsonlite::fromJSON(
      httr2::request(base_url) |>
        httr2::req_url_query(!!!qs) |>
        httr2::req_perform() |>
        httr2::resp_body_string(),
      flatten = TRUE
    )
    feats <- resp$features
    if (is.null(feats) || nrow(feats) == 0) break
    if ("attributes" %in% names(feats) && is.data.frame(feats$attributes)) {
      feats <- feats$attributes
    } else if (any(grepl("^attributes\\.", names(feats)))) {
      attr_cols <- grep("^attributes\\.", names(feats), value = TRUE)
      feats <- feats[, attr_cols, drop = FALSE]
      names(feats) <- sub("^attributes\\.", "", names(feats))
    }
    chunks[[length(chunks) + 1]] <- as_tibble(feats)
    if (nrow(feats) < page_size) break
    offset <- offset + page_size
  }
  bind_rows(chunks)
}

OD_ZIP_URL <- "https://www.nomisweb.co.uk/output/census/2021/odwp01ew.zip"
OD_ZIP_PATH <- file.path(raw_dir, "odwp01ew.zip")
OD_MSOA_CSV <- file.path(raw_dir, "ODWP01EW_MSOA.csv")

message("=== TTWA boundaries (ONS) ===")
ttwa_boundaries <- sf::st_read(ONS_TTWA_GEOJSON, quiet = TRUE) |>
  rename(ttwa11cd = TTWA11CD, ttwa11nm = TTWA11NM) |>
  mutate(
    country = case_when(
      str_starts(ttwa11cd, "W") ~ "W",
      str_starts(ttwa11cd, "S") ~ "S",
      str_starts(ttwa11cd, "N") ~ "N",
      TRUE ~ "E"
    )
  )

ttwa_codes <- ttwa_boundaries$ttwa11cd

message("=== MSOA → TTWA lookup (England & Wales via LSOA, UK LSOA→TTWA) ===")
lsoa_ttwa <- fetch_arcgis_attributes(LSOA_TTWA_BASE, c("LSOA11CD", "TTWA11CD"))
msoa_lsoa_ew <- fetch_arcgis_attributes(MSOA_LSOA_EW_BASE, c("MSOA11CD", "LSOA11CD"))

msoa_ttwa <- msoa_lsoa_ew |>
  distinct(MSOA11CD, LSOA11CD) |>
  inner_join(lsoa_ttwa, by = "LSOA11CD") |>
  distinct(msoa11cd = MSOA11CD, ttwa11cd = TTWA11CD)

message("MSOA→TTWA pairs: ", nrow(msoa_ttwa))

message("=== NOMIS ODWP01EW (England & Wales 2021) ===")
if (!file.exists(OD_ZIP_PATH)) {
  utils::download.file(OD_ZIP_URL, OD_ZIP_PATH, mode = "wb", quiet = TRUE)
}
if (!file.exists(OD_MSOA_CSV)) {
  utils::unzip(OD_ZIP_PATH, files = "ODWP01EW_MSOA.csv", exdir = raw_dir)
}

message("Reading MSOA OD table (large file) …")
od_raw <- data.table::fread(
  OD_MSOA_CSV,
  showProgress = TRUE,
  col.names = c(
    "residence_msoa", "residence_label", "workplace_msoa", "workplace_label",
    "powi_code", "powi_label", "flow"
  )
)

od_ew <- od_raw |>
  filter(
    powi_code == 3,
    str_detect(workplace_msoa, "^E"),
    str_detect(residence_msoa, "^E")
  ) |>
  transmute(
    residence_msoa,
    workplace_msoa,
    flow = as.integer(flow),
    source = "EW2021",
    geo_level = "MSOA"
  )

message("EW commuting MSOA pairs: ", nrow(od_ew))

message("=== Scotland 2022 IZ → TTWA ===")
od_scot <- tryCatch(
  fetch_scotland_workplace_od(raw_dir, ttwa_codes),
  error = function(e) {
    message("Scotland fetch warning: ", conditionMessage(e))
    tibble(
      residence_msoa = character(),
      workplace_msoa = character(),
      flow = integer(),
      source = character(),
      geo_level = character()
    )
  }
)
message("Scotland TTWA pairs: ", nrow(od_scot))

od_flows_raw <- bind_rows(od_ew, od_scot)
message("Combined OD rows: ", nrow(od_flows_raw))

# Working-age population proxy per TTWA (EW MSOA totals + Scotland from OD)
res_pop_ew <- od_raw |>
  filter(str_detect(residence_msoa, "^E")) |>
  group_by(residence_msoa) |>
  summarise(working_age_pop = sum(flow, na.rm = TRUE), .groups = "drop") |>
  left_join(msoa_ttwa, by = c("residence_msoa" = "msoa11cd")) |>
  filter(!is.na(ttwa11cd)) |>
  group_by(ttwa11cd) |>
  summarise(working_age_pop = sum(working_age_pop), .groups = "drop")

res_pop_sc <- od_scot |>
  group_by(ttwa11cd = residence_msoa) |>
  summarise(working_age_pop = sum(flow, na.rm = TRUE), .groups = "drop")

res_pop <- bind_rows(res_pop_ew, res_pop_sc) |>
  group_by(ttwa11cd) |>
  summarise(working_age_pop = sum(working_age_pop), .groups = "drop")

ttwa_boundaries <- ttwa_boundaries |>
  left_join(res_pop, by = "ttwa11cd") |>
  mutate(working_age_pop = replace_na(working_age_pop, 0L))

saveRDS(ttwa_boundaries, file.path(data_dir, "ttwa_boundaries.rds"))
saveRDS(od_flows_raw, file.path(data_dir, "od_flows_raw.rds"))
saveRDS(msoa_ttwa, file.path(data_dir, "msoa_ttwa_lookup.rds"))

message("Saved: ttwa_boundaries.rds, od_flows_raw.rds, msoa_ttwa_lookup.rds")
