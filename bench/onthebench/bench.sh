#!/usr/bin/env bash
# One command: build -> deploy to the rig -> pin -> run every load point ->
# collect results. Run from any checkout; the tree you run it from is the tree
# that gets measured.
#
#   bench/onthebench/bench.sh <user@rig-host> [local-results-dir]
#   BENCH_RIG=user@rig-host bench/onthebench/bench.sh
#
# The rig only needs to be a fresh Ubuntu 24.04 arm64 box with passwordless
# ssh + sudo; rig-setup.sh provisions everything else idempotently. The rig
# address is never committed here: this is a public repository, and a default
# target would publish a live passwordless-sudo box.
set -euo pipefail

RIG="${1:-${BENCH_RIG:-}}"
[ -n "$RIG" ] || { echo "usage: bench.sh <user@rig-host> [results-dir] (or export BENCH_RIG)"; exit 2; }
DEST="${2:-bench-results}"

SRC_LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMIT=$(git -C "$SRC_LOCAL" rev-parse HEAD)
DIRTY=$(git -C "$SRC_LOCAL" status --porcelain | wc -l)
RUNID="$(date -u +%Y%m%d-%H%M%S)-${COMMIT:0:7}"
[ "$DIRTY" = 0 ] || echo "WARNING: measuring a dirty tree ($DIRTY changed files); recorded in metadata" >&2

echo "== rsync source -> $RIG (commit ${COMMIT:0:12}) =="
rsync -az --delete --exclude=target --exclude=.git --exclude=docs/superpowers \
    --exclude=bench-results "$SRC_LOCAL"/ "$RIG":aisix-src/

echo "== provision rig =="
ssh "$RIG" 'bash aisix-src/bench/onthebench/rig-setup.sh'

echo "== build (native release on the rig) =="
ssh "$RIG" 'source ~/.cargo/env && cd aisix-src && cargo build --release --bin aisix'

echo "== run baseline ($RUNID) =="
# BENCH_-prefixed, never AISIX_-prefixed: aisix reads AISIX_* environment
# variables as config overrides, so an AISIX_COMMIT in the gateway's
# environment becomes an unknown top-level config field and refuses boot.
RUN_RC=0
ssh "$RIG" "BENCH_SRC_COMMIT=$COMMIT BENCH_SRC_DIRTY=$DIRTY \
    bash aisix-src/bench/onthebench/run-baseline.sh \"\$HOME/aisix-src\" \"\$HOME/bench-results/$RUNID\"" ||
    RUN_RC=$?

# Collect whatever the run produced even when it aborted mid-way: partial
# results with their metadata beat stranded results on a disposable rig.
echo "== collect results =="
mkdir -p "$DEST/$RUNID"
rsync -az "$RIG":"bench-results/$RUNID/" "$DEST/$RUNID/" || true
echo "results: $DEST/$RUNID"
exit "$RUN_RC"
