# Scotland Census 2022 workplace OD at IZ → TTWA (called from fetch_live_data.R)

fetch_scotland_workplace_od <- function(raw_dir, ttwa_codes) {
  suppressPackageStartupMessages({
    library(tidyverse)
    library(readxl)
  })

  iz_ttwa_path <- file.path(raw_dir, "scotland_dz2011_lookup.csv")
  xlsx_path <- file.path(raw_dir, "scotland_od_part6.xlsx")
  xlsx_url <- "https://www.scotlandscensus.gov.uk/media/4mfjtqk5/origin-destination-lower-geographies-1-part-6.xlsx"

  if (!file.exists(iz_ttwa_path)) {
    message("Downloading Scotland DZ2011 lookup (IZ → TTWA) …")
    utils::download.file(
      "https://statistics.gov.scot/downloads/file?id=50d30936-6de2-4ae0-a131-c2105aa74647%2FDataZone2011lookup_2024-12-16.csv",
      iz_ttwa_path,
      mode = "wb",
      quiet = TRUE
    )
  }

  if (!file.exists(xlsx_path)) {
    message("Downloading Scotland OD Part 6 (IZ workplace × travel method) …")
    utils::download.file(xlsx_url, xlsx_path, mode = "wb", quiet = TRUE)
  }

  lookup <- readr::read_csv(iz_ttwa_path, show_col_types = FALSE) |>
    transmute(
      iz2011cd = IZ2011_Code,
      ttwa11cd = TTWA2011_Code
    ) |>
    distinct() |>
    filter(
      !is.na(iz2011cd), !is.na(ttwa11cd),
      str_detect(iz2011cd, "^S"),
      ttwa11cd %in% ttwa_codes
    ) |>
    group_by(iz2011cd) |>
    slice_head(n = 1) |>
    ungroup()

  message("Reading Scotland Table 6b (IZ workplace flows) …")
  od_iz <- readxl::read_excel(xlsx_path, sheet = "Table 6b", skip = 4) |>
    set_names(c("workplace_iz", "workplace_code", "residence_iz", "residence_code", "method", "flow")) |>
    mutate(flow = as.integer(flow)) |>
    filter(
      method != "Work from home",
      method != "All",
      !method %in% c("No code required", NA),
      str_detect(residence_code, "^S"),
      str_detect(workplace_code, "^S"),
      residence_code != workplace_code
    ) |>
    group_by(residence_code, workplace_code) |>
    summarise(flow = sum(flow, na.rm = TRUE), .groups = "drop")

  message("Scotland IZ pairs (commuting): ", nrow(od_iz))

  od_ttwa <- od_iz |>
    left_join(lookup |> rename(residence_code = iz2011cd, origin_ttwa = ttwa11cd), by = "residence_code") |>
    left_join(lookup |> rename(workplace_code = iz2011cd, dest_ttwa = ttwa11cd), by = "workplace_code") |>
    filter(!is.na(origin_ttwa), !is.na(dest_ttwa)) |>
    group_by(origin_ttwa, dest_ttwa) |>
    summarise(flow = sum(flow, na.rm = TRUE), .groups = "drop")

  od_ttwa |>
    transmute(
      residence_msoa = origin_ttwa,
      workplace_msoa = dest_ttwa,
      flow,
      source = "SC2022",
      geo_level = "TTWA"
    )
}
