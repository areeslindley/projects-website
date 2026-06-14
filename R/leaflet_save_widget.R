# Save a self-contained Leaflet htmlwidget without leaving *_files/ next to the output.
# Prevents Sphinx/Jupyter Book from injecting leaflet CSS site-wide.
save_leaflet_widget_selfcontained <- function(map, out_path) {
  staging_dir <- file.path(tempdir(), "leaflet-widget-staging")
  dir.create(staging_dir, recursive = TRUE, showWarnings = FALSE)
  staging_path <- file.path(staging_dir, basename(out_path))
  suppressWarnings(htmlwidgets::saveWidget(map, staging_path, selfcontained = TRUE))
  staging_files <- paste0(tools::file_path_sans_ext(staging_path), "_files")
  if (dir.exists(staging_files)) {
    unlink(staging_files, recursive = TRUE)
  }
  dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
  file.copy(staging_path, out_path, overwrite = TRUE)
  out_files <- paste0(tools::file_path_sans_ext(out_path), "_files")
  if (dir.exists(out_files)) {
    unlink(out_files, recursive = TRUE)
  }
  invisible(out_path)
}
