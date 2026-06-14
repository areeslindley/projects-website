options(repos = c(CRAN = "https://cloud.r-project.org"))

install.packages(c(
  "tidyverse", "sf", "sfnetworks", "tidygraph", "igraph",
  "ggraph", "tmap", "patchwork", "here", "scales",
  "ggspatial", "viridis", "IRkernel", "data.table", "remotes",
  "httr2", "jsonlite", "readxl",
  "leaflet", "htmlwidgets", "htmltools", "IRdisplay"
))

# nomisr is not yet on CRAN for R 4.5+; 2021 OD data uses bulk CSV (see scripts/fetch_live_data.R)
if (!requireNamespace("nomisr", quietly = TRUE)) {
  remotes::install_github("ropensci/nomisr")
}

IRkernel::installspec(user = TRUE)
