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
# The pid suffix keeps two same-commit runs started in the same second from
# sharing a remote output directory and merging their results.
RUNID="$(date -u +%Y%m%d-%H%M%S)-${COMMIT:0:7}-$$"
[ "$DIRTY" = 0 ] || echo "WARNING: measuring a dirty tree ($DIRTY changed files); recorded in metadata" >&2

echo "== rsync source -> $RIG (commit ${COMMIT:0:12}) =="
# The gitignore filter keeps every ignored file off the rig — .env files and
# other gitignored local material may hold credentials, and none of it belongs
# in the measured tree. Untracked-but-not-ignored files still sync, preserving
# "the tree you run it from is the tree that gets measured".
rsync -az --delete --exclude=.git --filter=':- .gitignore' \
    "$SRC_LOCAL"/ "$RIG":aisix-src/

echo "== provision rig =="
ssh "$RIG" 'bash aisix-src/bench/onthebench/rig-setup.sh'

echo "== build (native release on the rig) =="
ssh "$RIG" 'source ~/.cargo/env && cd aisix-src && cargo build --locked --release --bin aisix'

echo "== run baseline ($RUNID) =="
# BENCH_-prefixed, never AISIX_-prefixed: aisix reads AISIX_* environment
# variables as config overrides, so an AISIX_COMMIT in the gateway's
# environment becomes an unknown top-level config field and refuses boot.
#
# Method overrides (BENCH_GRID, BENCH_REPS, ...) forward to the rig so a
# spot-check or an extra delay tier is a local env var, not a script edit;
# printf %q keeps values with spaces intact through the remote shell.
ENVPASS="BENCH_SRC_COMMIT=$COMMIT BENCH_SRC_DIRTY=$DIRTY"
for v in BENCH_GRID BENCH_REPS BENCH_WINDOW BENCH_WARMUP BENCH_MAX_TRIES \
         BENCH_FLOOR_REPS BENCH_FLAMEGRAPH BENCH_PERF_STACK; do
    # if, not `[ ] &&`: harmless here, but the && form as a function's last
    # statement is exactly how the grid_concs regression happened once.
    if [ -n "${!v:-}" ]; then ENVPASS="$ENVPASS $v=$(printf %q "${!v}")"; fi
done
RUN_RC=0
ssh "$RIG" "$ENVPASS \
    bash aisix-src/bench/onthebench/run-baseline.sh \"\$HOME/aisix-src\" \"\$HOME/bench-results/$RUNID\"" ||
    RUN_RC=$?

# Collect whatever the run produced even when it aborted mid-way: partial
# results with their metadata beat stranded results on a disposable rig.
echo "== collect results =="
mkdir -p "$DEST/$RUNID"
rsync -az "$RIG":"bench-results/$RUNID/" "$DEST/$RUNID/" || true
echo "results: $DEST/$RUNID"
exit "$RUN_RC"
