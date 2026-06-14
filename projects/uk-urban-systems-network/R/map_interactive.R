# Interactive Leaflet maps for uk-urban-systems-network (sourced from notebooks 06/07).

`%||%` <- function(x, y) if (is.null(x)) y else x

find_book_root <- function() {
  path <- getwd()
  for (i in seq_len(8)) {
    if (file.exists(file.path(path, "_config.yml"))) {
      return(normalizePath(path, winslash = "/"))
    }
    parent <- dirname(path)
    if (identical(parent, path)) break
    path <- parent
  }
  stop("Could not find Jupyter Book root (_config.yml).")
}

debug_log <- function(message, data = list(), hypothesisId = "H1") {
  # #region agent log
  tryCatch({
    payload <- list(
      sessionId = "0556ee",
      runId = "fix-styling",
      hypothesisId = hypothesisId,
      location = "map_interactive.R:include_leaflet_map",
      message = message,
      data = data,
      timestamp = round(as.numeric(Sys.time()) * 1000)
    )
    line <- if (requireNamespace("jsonlite", quietly = TRUE)) {
      jsonlite::toJSON(payload, auto_unbox = TRUE)
    } else {
      sprintf(
        '{"sessionId":"0556ee","hypothesisId":"%s","message":"%s"}',
        hypothesisId,
        gsub('"', '\\\\"', message)
      )
    }
    cat(line, "\n", file = "/Users/areeslindley/Documents/Git_repositories/projects-website/.cursor/debug-0556ee.log", append = TRUE)
  }, error = function(e) invisible(NULL))
  # #endregion
}

include_leaflet_map <- function(map, title, file_stem) {
  maps_dir <- file.path(find_book_root(), "_static", "uk-urban-systems-network", "maps")
  dir.create(maps_dir, recursive = TRUE, showWarnings = FALSE)
  out_path <- file.path(maps_dir, paste0(file_stem, ".html"))
  book_root <- find_book_root()
  save_helper <- file.path(book_root, "R", "leaflet_save_widget.R")
  if (file.exists(save_helper)) {
    source(save_helper, local = TRUE)
    save_leaflet_widget_selfcontained(map, out_path)
  } else {
    suppressWarnings(htmlwidgets::saveWidget(map, out_path, selfcontained = TRUE))
    out_files_dir <- paste0(tools::file_path_sans_ext(out_path), "_files")
    if (dir.exists(out_files_dir)) unlink(out_files_dir, recursive = TRUE)
  }
  out_files_dir <- paste0(tools::file_path_sans_ext(out_path), "_files")
  debug_log(
    "widget published to _static",
    list(
      out_path = out_path,
      out_files_exists = dir.exists(out_files_dir),
      html_bytes = file.info(out_path)$size
    ),
    "H1"
  )
  iframe_src <- paste0("../../_static/uk-urban-systems-network/maps/", file_stem, ".html")
  title_esc <- htmltools::htmlEscape(title)
  iframe_html <- paste0(
    "<div class=\"leaflet-map-embed\" style=\"width:100%;max-width:100%;margin:1em 0;\">",
    "<h4 style=\"text-align:center;font-weight:bold;margin:0.6em 0 0.2em 0;\">", title_esc, "</h4>",
    "<iframe src=\"", iframe_src, "\" title=\"", title_esc, "\" width=\"100%\" height=\"560\" ",
    "style=\"border:none;display:block;max-width:100%;\"></iframe>",
    "</div>"
  )
  IRdisplay::display_html(iframe_html)
  invisible(out_path)
}

region_palette <- function(n_regions) {
  base <- c(
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5",
    "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
    "#e5c494", "#b3b3b3", "#1b9e77", "#d95f02", "#7570b3",
    "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666",
    "#8dd3c7", "#fb8072", "#80b1d3", "#fdb462", "#b3de69",
    "#fccde5", "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f"
  )
  if (n_regions <= length(base)) {
    stats::setNames(base[seq_len(n_regions)], as.character(seq_len(n_regions)))
  } else {
    grDevices::colorRampPalette(base)(n_regions) |>
      stats::setNames(as.character(seq_len(n_regions)))
  }
}

prepare_edge_segments <- function(od_ttwa_pairs, ttwa_centroids, min_flow = 100L) {
  node_xy <- ttwa_centroids |>
    mutate(
      x = sf::st_coordinates(geometry)[, 1],
      y = sf::st_coordinates(geometry)[, 2]
    ) |>
    sf::st_drop_geometry() |>
    select(ttwa11cd, x, y, country)

  od_ttwa_pairs |>
    filter(flow >= min_flow) |>
    left_join(node_xy |> rename(origin_ttwa = ttwa11cd, x = x, y = y), by = "origin_ttwa") |>
    left_join(
      node_xy |> rename(dest_ttwa = ttwa11cd, xend = x, yend = y),
      by = "dest_ttwa"
    ) |>
    filter(!is.na(x), !is.na(xend))
}

leaflet_ttwa_network <- function(
    ttwa_sf,
    partition_df,
    edge_segments,
    mode = c("partition", "final"),
    min_flow = 100L
) {
  mode <- match.arg(mode)
  n_regions <- n_distinct(partition_df$region_id)
  pal <- region_palette(n_regions)

  ttwa_sf <- ttwa_sf |>
    left_join(partition_df, by = "ttwa11cd") |>
    left_join(
      partition_df |>
        select(anchor_ttwa, region_id) |>
        distinct(anchor_ttwa, .keep_all = TRUE) |>
        rename(anchor_region = region_id),
      by = c("anchor_ttwa" = "anchor_ttwa")
    )

  fill_col <- pal[as.character(ttwa_sf$region_id)]

  country_labels <- c(E = "England", W = "Wales", S = "Scotland", N = "Northern Ireland")

  make_pop <- function(nm, cty, pop, anchor, rid) {
    paste0(
      "<strong>", nm, "</strong><br>",
      "Country: ", country_labels[[cty]] %||% cty, "<br>",
      "Working-age pop. (proxy): ", format(pop, big.mark = ","), "<br>",
      "Regional system: ", rid, "<br>",
      "Anchor: ", anchor
    )
  }

  map <- leaflet::leaflet(options = leaflet::leafletOptions(minZoom = 4)) |>
    leaflet::addProviderTiles(leaflet::providers$CartoDB.Positron)

  for (cty in c("E", "W", "S")) {
    sub <- ttwa_sf |> filter(country == cty)
    if (nrow(sub) == 0) next
    map <- map |>
      leaflet::addPolygons(
        data = sub,
        fillColor = pal[as.character(sub$region_id)],
        fillOpacity = if (mode == "final") 0.75 else 0.55,
        color = "#333333",
        weight = 0.4,
        opacity = 0.6,
        popup = mapply(
          make_pop,
          sub$ttwa11nm,
          sub$country,
          sub$working_age_pop,
          sub$anchor_ttwa,
          sub$region_id,
          USE.NAMES = FALSE
        ),
        group = country_labels[[cty]]
      )
  }

  if (nrow(edge_segments) > 0) {
    edge_segments <- edge_segments |>
      left_join(partition_df |> rename(origin_ttwa = ttwa11cd), by = "origin_ttwa")

    edge_lines <- edge_segments |>
      mutate(
        geometry = purrr::pmap(
          list(x, y, xend, yend),
          function(x, y, xend, yend) {
            sf::st_linestring(matrix(c(x, y, xend, yend), ncol = 2, byrow = TRUE))
          }
        )
      ) |>
      sf::st_as_sf(crs = sf::st_crs(ttwa_sf))

    map <- map |>
      leaflet::addPolylines(
        data = edge_lines,
        color = pal[as.character(edge_lines$region_id)],
        opacity = 0.35,
        weight = ~scales::rescale(flow, to = c(0.3, 2.5)),
        group = "Commuting flows"
      )
  }

  overlay_present <- vapply(c("E", "W", "S"), function(cty) {
    if (any(ttwa_sf$country == cty, na.rm = TRUE)) country_labels[[cty]] else NA_character_
  }, character(1))
  overlay_present <- overlay_present[!is.na(overlay_present)]
  if (nrow(edge_segments) > 0) {
    overlay_present <- c(overlay_present, "Commuting flows")
  }

  map |>
    leaflet::addLayersControl(
      overlayGroups = overlay_present,
      options = leaflet::layersControlOptions(collapsed = FALSE)
    )
}
