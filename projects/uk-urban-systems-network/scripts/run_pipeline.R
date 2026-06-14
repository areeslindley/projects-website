# Rebuild RDS + figures after fetch_live_data.R (chapters 03–07 logic).
# Run from repo root: Rscript projects/uk-urban-systems-network/scripts/run_pipeline.R

suppressPackageStartupMessages({
  library(tidyverse)
  library(sf)
  library(ggplot2)
  library(scales)
  library(ggraph)
  library(tidygraph)
  library(patchwork)
  library(here)
})

proj_dir <- here::here("projects", "uk-urban-systems-network")
data_dir <- file.path(proj_dir, "data")
fig_dir <- file.path(proj_dir, "figures")
MIN_FLOW <- 50L

ttwa_boundaries <- readRDS(file.path(data_dir, "ttwa_boundaries.rds"))
od_flows_raw <- readRDS(file.path(data_dir, "od_flows_raw.rds"))
msoa_ttwa <- readRDS(file.path(data_dir, "msoa_ttwa_lookup.rds"))

aggregate_msoa_to_ttwa <- function(od, lookup) {
  lookup_unique <- lookup |>
    group_by(msoa11cd) |>
    slice_head(n = 1) |>
    ungroup()
  od |>
    filter(geo_level == "MSOA") |>
    left_join(lookup_unique |> rename(origin_ttwa = ttwa11cd, residence_msoa = msoa11cd), by = "residence_msoa") |>
    left_join(lookup_unique |> rename(dest_ttwa = ttwa11cd, workplace_msoa = msoa11cd), by = "workplace_msoa") |>
    filter(!is.na(origin_ttwa), !is.na(dest_ttwa)) |>
    group_by(origin_ttwa, dest_ttwa) |>
    summarise(flow = sum(flow, na.rm = TRUE), .groups = "drop")
}

od_ttwa_ew <- aggregate_msoa_to_ttwa(od_flows_raw, msoa_ttwa)
od_ttwa_sc <- od_flows_raw |>
  filter(geo_level == "TTWA") |>
  transmute(origin_ttwa = residence_msoa, dest_ttwa = workplace_msoa, flow)

od_ttwa_pairs_raw <- bind_rows(od_ttwa_ew, od_ttwa_sc)
od_ttwa_pairs <- od_ttwa_pairs_raw |>
  filter(origin_ttwa != dest_ttwa, flow >= MIN_FLOW)

ttwa_centroids <- ttwa_boundaries |>
  mutate(geometry = sf::st_point_on_surface(geometry)) |>
  select(ttwa11cd, ttwa11nm, country, working_age_pop, geometry)

ttwa_population <- ttwa_centroids |> st_drop_geometry() |> select(ttwa11cd, working_age_pop)

saveRDS(od_ttwa_pairs, file.path(data_dir, "od_ttwa_pairs.rds"))
saveRDS(ttwa_centroids, file.path(data_dir, "ttwa_centroids.rds"))
saveRDS(ttwa_population, file.path(data_dir, "ttwa_population.rds"))

message("03: edges=", nrow(od_ttwa_pairs), " nodes=", n_distinct(c(od_ttwa_pairs$origin_ttwa, od_ttwa_pairs$dest_ttwa)))

# 04 nodes map
p_nodes <- ggplot(ttwa_centroids) +
  geom_sf(aes(size = working_age_pop), colour = "#2c3e50", alpha = 0.75) +
  scale_size_area(labels = comma, max_size = 10) +
  coord_sf(crs = sf::st_crs(ttwa_centroids)) +
  labs(title = "UK Travel to Work Areas (2011)", subtitle = "Point size ∝ working-age population") +
  theme_minimal()
ggsave(file.path(fig_dir, "04_nodes_map.png"), p_nodes, width = 9, height = 10, dpi = 300)

# 05 flows
nodes_tbl <- ttwa_centroids |>
  mutate(name = ttwa11cd, x = sf::st_coordinates(geometry)[, 1], y = sf::st_coordinates(geometry)[, 2]) |>
  st_drop_geometry()
edge_tbl <- od_ttwa_pairs |> transmute(from = origin_ttwa, to = dest_ttwa, flow)
g_net <- tidygraph::tbl_graph(nodes = nodes_tbl, edges = edge_tbl, directed = TRUE)
layout_manual <- ggraph::create_layout(g_net, layout = "manual", x = nodes_tbl$x, y = nodes_tbl$y)
p_flows <- ggraph::ggraph(layout_manual) +
  ggraph::geom_edge_link(aes(alpha = flow), colour = "grey20", show.legend = FALSE) +
  ggraph::geom_node_point(size = 1.2, colour = "grey10") +
  ggraph::scale_edge_alpha_continuous(range = c(0.05, 0.9)) +
  coord_sf(crs = sf::st_crs(ttwa_centroids)) +
  theme_void()
ggsave(file.path(fig_dir, "05_flows_network.png"), p_flows, width = 9, height = 10, dpi = 300)
saveRDS(list(graph = g_net, nodes_sf = ttwa_centroids, edges = edge_tbl), file.path(data_dir, "sfnetwork_graph.rds"))

# 06 partition
nystuen_dacey_dominant_flow <- function(edges, node_ids) {
  all_edges <- edges |> filter(origin_ttwa != dest_ttwa)
  dominant <- all_edges |>
    group_by(origin_ttwa) |>
    slice_max(flow, n = 1, with_ties = FALSE) |>
    ungroup() |>
    transmute(ttwa = origin_ttwa, partner = dest_ttwa, dominant_flow = flow)
  inflow <- all_edges |>
    group_by(dest_ttwa) |>
    summarise(inflow = sum(flow), .groups = "drop") |>
    rename(ttwa = dest_ttwa)
  node_set <- unique(node_ids)
  total_in <- inflow |>
    right_join(tibble(ttwa = node_set), by = "ttwa") |>
    mutate(inflow = replace_na(inflow, 0))
  is_terminal <- function(ttwa) {
    row <- dominant |> filter(ttwa == !!ttwa)
    if (nrow(row) == 0) return(TRUE)
    partner <- row$partner[1]
    partner_in <- total_in |> filter(ttwa == partner) |> pull(inflow)
    if (length(partner_in) == 0) partner_in <- 0
    self_in <- total_in |> filter(ttwa == !!ttwa) |> pull(inflow)
    partner_in < self_in
  }
  anchor_of <- function(ttwa) {
    visited <- character()
    current <- ttwa
    repeat {
      if (is_terminal(current)) return(current)
      nxt <- dominant |> filter(ttwa == current) |> pull(partner)
      if (length(nxt) == 0 || nxt[1] %in% visited) return(current)
      visited <- c(visited, current)
      current <- nxt[1]
    }
  }
  tibble(ttwa11cd = node_set, anchor_ttwa = vapply(node_set, anchor_of, character(1))) |>
    mutate(region_id = as.integer(factor(anchor_ttwa)))
}

node_ids <- unique(c(od_ttwa_pairs$origin_ttwa, od_ttwa_pairs$dest_ttwa))
partition_labels <- nystuen_dacey_dominant_flow(od_ttwa_pairs, node_ids)
saveRDS(partition_labels, file.path(data_dir, "partition_labels.rds"))
message("06: regions=", n_distinct(partition_labels$anchor_ttwa))

nodes_plot <- ttwa_centroids |>
  left_join(partition_labels, by = "ttwa11cd") |>
  mutate(name = ttwa11cd, x = sf::st_coordinates(geometry)[, 1], y = sf::st_coordinates(geometry)[, 2]) |>
  st_drop_geometry()
edges_plot <- od_ttwa_pairs |>
  left_join(partition_labels |> rename(from = ttwa11cd), by = c("origin_ttwa" = "from"))
g_part <- tidygraph::tbl_graph(nodes = nodes_plot, edges = edges_plot |> transmute(from = origin_ttwa, to = dest_ttwa, flow, region_id), directed = TRUE)
layout_part <- ggraph::create_layout(g_part, layout = "manual", x = nodes_plot$x, y = nodes_plot$y)
n_part <- n_distinct(partition_labels$region_id)
pal <- setNames(scales::hue_pal()(n_part), as.character(seq_len(n_part)))
p_part <- ggraph::ggraph(layout_part) +
  ggraph::geom_edge_link(aes(colour = factor(region_id)), alpha = 0.4, show.legend = FALSE) +
  ggraph::geom_node_point(aes(colour = factor(region_id)), size = 2) +
  scale_edge_colour_manual(values = pal) +
  scale_colour_manual(values = pal) +
  coord_sf(crs = sf::st_crs(ttwa_centroids)) +
  theme_void()
ggsave(file.path(fig_dir, "06_partition_map.png"), p_part, width = 9, height = 10, dpi = 300)

# 07 final
region_palette <- function(n) {
  base <- c("#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3", "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666", "#8dd3c7", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f")
  if (n <= length(base)) stats::setNames(base[seq_len(n)], as.character(seq_len(n))) else grDevices::colorRampPalette(base)(n) |> stats::setNames(as.character(seq_len(n)))
}
n_reg <- n_distinct(partition_labels$region_id)
region_pal <- region_palette(n_reg)
ttwa_regions <- ttwa_boundaries |> left_join(partition_labels, by = "ttwa11cd")
node_xy <- ttwa_centroids |> mutate(x = sf::st_coordinates(geometry)[, 1], y = sf::st_coordinates(geometry)[, 2]) |> st_drop_geometry()
edge_segments <- od_ttwa_pairs |>
  left_join(node_xy |> rename(origin_ttwa = ttwa11cd, x = x, y = y), by = "origin_ttwa") |>
  left_join(node_xy |> rename(dest_ttwa = ttwa11cd, xend = x, yend = y), by = "dest_ttwa") |>
  left_join(partition_labels |> rename(origin_ttwa = ttwa11cd), by = "origin_ttwa")

p_main <- ggplot() +
  geom_sf(data = ttwa_regions, aes(fill = factor(region_id)), colour = NA, alpha = 0.85) +
  geom_curve(data = edge_segments, aes(x = x, y = y, xend = xend, yend = yend, linewidth = flow, colour = factor(region_id)), alpha = 0.35, curvature = 0.15, inherit.aes = FALSE, show.legend = FALSE) +
  geom_point(data = node_xy, aes(x = x, y = y), size = 0.3, colour = "grey10", inherit.aes = FALSE) +
  scale_fill_manual(values = region_pal, guide = "none") +
  scale_colour_manual(values = region_pal, guide = "none") +
  scale_linewidth_continuous(range = c(0.1, 1.5), guide = "none") +
  coord_sf(crs = sf::st_crs(ttwa_boundaries), default_crs = sf::st_crs(ttwa_boundaries)) +
  labs(title = "UK regional urban systems", subtitle = "TTWA choropleth + dominant-flow network") +
  theme_void()

LONDON_ANCHOR <- ttwa_boundaries |>
  st_drop_geometry() |>
  filter(str_detect(ttwa11nm, regex("London", ignore_case = TRUE))) |>
  pull(ttwa11cd) |>
  first()

london_nodes <- partition_labels |> filter(anchor_ttwa == LONDON_ANCHOR) |> pull(ttwa11cd)
london_sf <- ttwa_regions |> filter(ttwa11cd %in% london_nodes)
london_edges <- edge_segments |> filter(origin_ttwa %in% london_nodes, dest_ttwa %in% london_nodes)
london_xy <- node_xy |> filter(ttwa11cd %in% london_nodes)
bb <- sf::st_bbox(london_sf) + c(-0.15, -0.15, 0.15, 0.15)
p_london <- ggplot() +
  geom_sf(data = london_sf, aes(fill = factor(region_id)), colour = NA, alpha = 0.9) +
  geom_curve(data = london_edges, aes(x = x, y = y, xend = xend, yend = yend, linewidth = flow), colour = "grey30", alpha = 0.5, curvature = 0.2, inherit.aes = FALSE) +
  geom_point(data = london_xy, aes(x = x, y = y), size = 1, colour = "grey10", inherit.aes = FALSE) +
  scale_fill_manual(values = region_pal, guide = "none") +
  scale_linewidth_continuous(range = c(0.2, 2), guide = "none") +
  coord_sf(xlim = c(bb["xmin"], bb["xmax"]), ylim = c(bb["ymin"], bb["ymax"]), crs = sf::st_crs(ttwa_boundaries), expand = FALSE) +
  labs(title = "London system (inset)") +
  theme_void()

p_final <- p_main + p_london + patchwork::plot_layout(widths = c(3, 1))
ggsave(file.path(fig_dir, "07_final_map.png"), p_final, width = 14, height = 10, dpi = 300)
message("07: London anchor=", LONDON_ANCHOR)
message("Done.")
