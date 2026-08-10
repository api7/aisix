#!/usr/bin/env bash
# One command: build -> deploy to the rig -> pin -> run every load point ->
# collect results. Run from any checkout; the tree you run it from is the tree
# that gets measured.
#
#   bench/onthebench/bench.sh [user@rig-host] [local-results-dir]
#
# The rig only needs to be a fresh Ubuntu 24.04 arm64 box with passwordless
# ssh + sudo; rig-setup.sh provisions everything else idempotently.
set -euo pipefail

RIG="${1:-ubuntu@54.179.77.47}"
DEST="${2:-bench-results}"

SRC_LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMIT=$(git -C "$SRC_LOCAL" rev-parse HEAD)
DIRTY=$(git -C "$SRC_LOCAL" status --porcelain | wc -l)
RUNID="$(date -u +%Y%m%d-%H%M%S)-${COMMIT:0:7}"
[ "$DIRTY" = 0 ] || echo "WARNING: measuring a dirty tree ($DIRTY changed files); recorded in metadata" >&2

echo "== rsync source -> $RIG (commit ${COMMIT:0:12}) =="
rsync -az --delete --exclude=target --exclude=.git --exclude=docs/superpowers \
    "$SRC_LOCAL"/ "$RIG":aisix-src/

echo "== provision rig =="
ssh "$RIG" 'bash aisix-src/bench/onthebench/rig-setup.sh'

echo "== build (native release on the rig) =="
ssh "$RIG" 'source ~/.cargo/env && cd aisix-src && cargo build --release --bin aisix'

echo "== run baseline ($RUNID) =="
ssh "$RIG" "AISIX_COMMIT=$COMMIT AISIX_DIRTY=$DIRTY \
    bash aisix-src/bench/onthebench/run-baseline.sh \"\$HOME/aisix-src\" \"\$HOME/bench-results/$RUNID\""

echo "== collect results =="
mkdir -p "$DEST/$RUNID"
rsync -az "$RIG":"bench-results/$RUNID/" "$DEST/$RUNID/"
echo "results: $DEST/$RUNID"
