#!/usr/bin/env bash
# Remove htmlwidgets *_files/ dirs so Sphinx does not inject Leaflet CSS site-wide.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
count=0
while IFS= read -r -d '' dir; do
  rm -rf "$dir"
  count=$((count + 1))
done < <(find "$ROOT/_static" "$ROOT/_build" -type d -name '*_files' -print0 2>/dev/null || true)
echo "prune_htmlwidget_deps: removed ${count} *_files directories"
