#!/usr/bin/env bash
# Clone a pinned commit of an upstream method's repo into external/<dirname>.
# Usage: scripts/fetch_method.sh trust_region_soc
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "${1:-}" in
  trust_region_soc)
    URL="https://github.com/DenisBless/TrustRegionSOC"
    COMMIT="a93471f8698ab4754edc810184390769828ad4fa"
    DIRNAME="TrustRegionSOC"
    ;;
  "")
    echo "usage: scripts/fetch_method.sh <method_name>" >&2
    echo "known methods: trust_region_soc" >&2
    exit 1
    ;;
  *)
    echo "unknown method '$1' -- add it to scripts/fetch_method.sh and soc_bench/vendor_paths.py" >&2
    exit 1
    ;;
esac

DEST="$ROOT/external/$DIRNAME"

if [ -d "$DEST" ]; then
  echo "$DEST already exists, skipping clone."
  exit 0
fi

mkdir -p "$ROOT/external"
git clone "$URL" "$DEST"
git -C "$DEST" checkout "$COMMIT"
echo "Vendored $1 @ $COMMIT -> $DEST"
