#!/usr/bin/env bash
# Check internal links in built Jupyter Book HTML (run after jupyter-book build).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/_build/html"
PORT="${LINK_CHECK_PORT:-8765}"

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "error: ${BUILD_DIR} not found — run jupyter-book build first" >&2
  exit 1
fi

echo "Checking for unconverted user-facing .md / .ipynb hrefs..."
BAD_HREFS="$(
  grep -rE 'href="[^"]*\.(md|ipynb)"' "${BUILD_DIR}" --include='*.html' \
    | grep -v '_sources/' \
    | grep -v '/_build/' \
    || true
)"
if [[ -n "${BAD_HREFS}" ]]; then
  echo "error: found unconverted internal links:" >&2
  echo "${BAD_HREFS}" >&2
  exit 1
fi

echo "Running linkinator via local server on port ${PORT}..."
cd "${BUILD_DIR}"
python3 -m http.server "${PORT}" >/dev/null 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" >/dev/null 2>&1 || true' EXIT
sleep 1

npx --yes linkinator "http://127.0.0.1:${PORT}/intro.html" \
  --recurse \
  --skip '^(mailto:|https?://|#)' \
  --skip '_sources/' \
  --skip '/_build/' \
  --verbosity error

echo "Link check passed."

echo "Checking UK urban systems figure embeds..."
URBAN_DIR="${BUILD_DIR}/projects/uk-urban-systems-network"
FIGURE_PAGES=(
  "01_introduction.html"
  "03_data_cleaning.html"
  "04_nodes.html"
  "05_flows.html"
  "06_partition.html"
  "07_final_map.html"
  "index.html"
)
for page in "${FIGURE_PAGES[@]}"; do
  if ! grep -q '<img' "${URBAN_DIR}/${page}"; then
    echo "error: no embedded images in ${URBAN_DIR}/${page}" >&2
    exit 1
  fi
done

for map in 06_partition_interactive.html 07_final_map_interactive.html; do
  if [[ ! -f "${BUILD_DIR}/_static/uk-urban-systems-network/maps/${map}" ]]; then
    echo "error: missing interactive map ${map} in build output" >&2
    exit 1
  fi
done

echo "UK urban systems figure check passed."
