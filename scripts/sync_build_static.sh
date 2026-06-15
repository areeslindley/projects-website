#!/usr/bin/env bash
# Copy uk-urban-systems-network static assets into the built site after notebook execution.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_STATIC="${ROOT}/_build/html/_static/uk-urban-systems-network"
SRC_STATIC="${ROOT}/_static/uk-urban-systems-network"

mkdir -p "${BUILD_STATIC}/maps"

if [[ -f "${SRC_STATIC}/french_datar_2011.png" ]]; then
  cp -f "${SRC_STATIC}/french_datar_2011.png" "${BUILD_STATIC}/"
fi

if compgen -G "${SRC_STATIC}/maps/*.html" > /dev/null; then
  cp -f "${SRC_STATIC}/maps/"*.html "${BUILD_STATIC}/maps/"
fi

echo "sync_build_static: copied uk-urban-systems-network assets to _build/html/_static"
