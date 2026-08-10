#!/usr/bin/env bash
# On-rig runner for ANY gateway-shaped process ("entrant"): same instruments,
# same core split, same windows, floors and validity policy as run-baseline.sh
# measures aisix — both source lib.sh, so a cross-target comparison is
# apples-to-apples by construction.
#
# Usage: run-entrant.sh <entrant-dir> <out-dir>
#
# <entrant-dir>/entrant.sh is sourced (before lib.sh, so it can override the
# request shape) and must provide:
#
#   ENTRANT_NAME             short identifier stamped into results and meta
#   entrant_start <out-dir>  boot the target listening on 127.0.0.1:$GW_PORT,
#                            upstream pointed at 127.0.0.1:$MOCK_PORT, pinned
#                            to $GW_CORES; must set GW_PID to the pid whose
#                            /proc CPU and RSS are sampled
#
# and may provide:
#
#   entrant_prepare <out-dir>  one-time fetch/build, before any measurement
#   entrant_stop               teardown beyond `kill $GW_PID` (containers);
#                            must tolerate being called when entrant_start
#                            never ran (cleanup on a prepare failure)
#   entrant_meta_json          extra identity JSON object (version, digest, …)
#   REQ_PATH / BODY / AUTH_HEADER  request shape, when the entrant's ingress
#                            speaks a dialect other than OpenAI chat
#
# entrant.sh is sourced before lib.sh: harness variables (GW_PORT, GW_CORES,
# MOCK_PORT, ...) are available inside the hook functions when they are
# called, but NOT at the entrant.sh top level.
#
# The entrant dir itself is deliberately not part of this repository: the
# harness carries the method; what it points at is the operator's business.
#
# Flamegraphs default OFF here (BENCH_FLAMEGRAPH=1 opts in): most shipped
# release binaries are stripped, and a symbol-less flamegraph is noise. A
# stripped binary skips the flamegraph with a warning instead of failing the
# run — the throughput numbers stand on their own.
set -euo pipefail

ENTRANT_DIR="${1:?usage: run-entrant.sh <entrant-dir> <out-dir>}"
OUT="${2:?usage: run-entrant.sh <entrant-dir> <out-dir>}"

[ -f "$ENTRANT_DIR/entrant.sh" ] || { echo "FATAL: $ENTRANT_DIR/entrant.sh missing"; exit 1; }
ENTRANT_DIR="$(cd "$ENTRANT_DIR" && pwd)"

# The entrant script first (it may set REQ_PATH/BODY/AUTH_HEADER), then the
# lib, which resolves defaults from whatever the entrant left unset.
BENCH_FLAMEGRAPH="${BENCH_FLAMEGRAPH:-0}"
# shellcheck source=/dev/null
source "$ENTRANT_DIR/entrant.sh"
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# The name is spliced verbatim into meta.json and every JSONL record; a
# constrained identifier set keeps that safe without escaping at every site.
[[ "${ENTRANT_NAME:-}" =~ ^[A-Za-z0-9._-]+$ ]] ||
    { echo "FATAL: entrant.sh must set ENTRANT_NAME matching [A-Za-z0-9._-]+ (got '${ENTRANT_NAME:-}')"; exit 1; }

rig_sanity
bench_init

if type entrant_prepare >/dev/null 2>&1; then
    echo "== prepare ($ENTRANT_NAME) ==" >&2
    # The pipe puts entrant_prepare in a subshell: state it wants to hand to
    # entrant_start or entrant_meta_json must go through files, not variables.
    # The explicit subshell re-arms errexit — a `prepare || die` form would
    # disable set -e inside the hook and let a mid-prepare failure (a failed
    # fetch before a succeeding final step) pass as success, measuring stale
    # artifacts; the rc capture keeps pipefail out of the decision.
    set +e
    ( set -e; entrant_prepare "$OUT" ) 2>&1 | tee "$OUT/prepare.log" >&2
    PREP_RC=${PIPESTATUS[0]}
    set -e
    [ "$PREP_RC" -eq 0 ] ||
        { echo "FATAL: entrant_prepare failed (rc=$PREP_RC, see $OUT/prepare.log)"; exit 1; }
fi

start_target() {
    echo "== target ($ENTRANT_NAME) ==" >&2
    # aisix reads AISIX_* environment variables as config overrides; no
    # measured process gets them, whatever it is — same hygiene for every
    # entrant, and one less variable between two entrants' environments.
    while read -r v; do unset "$v"; done < <(compgen -v | grep '^AISIX_' || true)
    GW_PID=""
    entrant_start "$OUT"
    # /proc existence, not kill -0: a containerized target's pid belongs to
    # another user, where kill -0 reports EPERM and would read as death.
    [ -n "$GW_PID" ] && [ -d "/proc/$GW_PID" ] ||
        { echo "FATAL: entrant_start did not leave a live pid in GW_PID"; exit 1; }
    wait_http_200 "http://127.0.0.1:$GW_PORT$REQ_PATH" "$ENTRANT_NAME"
    sleep 3
    RSS_IDLE=$(rss_kb "$GW_PID")
    THREADS=$(ps -T -p "$GW_PID" 2>/dev/null | tail -n +2 | wc -l) || THREADS=0
    # CPU% and RSS cover GW_PID alone. A multi-process target (master +
    # workers) would be silently understated; surface it loudly and record
    # the count so the numbers can never pass unannotated.
    CHILDREN=$(pgrep -cP "$GW_PID" 2>/dev/null) || CHILDREN=0
    [ "$CHILDREN" -eq 0 ] ||
        echo "WARNING: target pid $GW_PID has $CHILDREN child processes; CPU%/RSS cover this pid only" >&2
    echo "  pid=$GW_PID idle_rss=${RSS_IDLE}kB threads=$THREADS children=$CHILDREN" >&2
}

write_meta() { # write_meta <rss_hwm_kb-or-null>
    # /proc/<pid>/exe may belong to another user (a container's init); sudo -n
    # is how the rest of the rig already escalates, and "unknown" is recorded
    # rather than failing the run — entrant_meta_json carries identity anyway.
    local bin_sha identity=null
    bin_sha=$(sudo -n sha256sum "/proc/$GW_PID/exe" 2>/dev/null | cut -d' ' -f1 || true)
    # The identity hook is evaluated and validated before the heredoc: a hook
    # that fails or emits partial output must fail the run loudly, not ship a
    # meta.json that is silently null or unparseable.
    if type entrant_meta_json >/dev/null 2>&1; then
        identity=$(entrant_meta_json) ||
            { echo "FATAL: entrant_meta_json failed"; exit 1; }
        printf '%s' "$identity" | python3 -c 'import json,sys; json.load(sys.stdin)' ||
            { echo "FATAL: entrant_meta_json returned invalid JSON: $identity"; exit 1; }
    fi
    cat > "$OUT/meta.json" <<EOF
{
  "entrant": "$ENTRANT_NAME",
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "rig": $(meta_rig_json),
  "cores": $(meta_cores_json),
  "instruments": $(meta_instruments_json),
  "method": $(meta_method_json),
  "target": {
    "binary_sha256": "${bin_sha:-unknown}",
    "rss_idle_kb": ${RSS_IDLE:-null}, "rss_hwm_kb": $1,
    "threads": ${THREADS:-null},
    "children": ${CHILDREN:-null},
    "identity": $identity
  }
}
EOF
}

# ---- tiers: per tier, mock -> floor -> target points ------------------------

FIRST_TIER=1
for TTFT in $(grid_ttfts); do
    echo "== mock (${TTFT}ms TTFT) + rig floor ==" >&2
    start_mock "$TTFT"
    floor_tier "$TTFT"
    if [ "$FIRST_TIER" = 1 ]; then
        start_target
        write_meta null
        FIRST_TIER=0
    fi
    for CONC in $(grid_concs "$TTFT"); do
        run_point "$TTFT" "$CONC"
    done
    if [ "$TTFT" = 0 ] && [ "$FLAMEGRAPH" = 1 ] && grid_has 0 128; then
        # Same >100 symbol threshold as the baseline's require_symbols: a
        # nearly-stripped binary would render a noise flamegraph, not a
        # useful one. Soft skip, unlike the baseline - the throughput
        # numbers stand on their own.
        SYMS=$(nm "/proc/$GW_PID/exe" 2>/dev/null | grep -c ' [tT] ') || SYMS=0
        if [ "$SYMS" -gt 100 ]; then
            flamegraph_point 128 "$ENTRANT_NAME c=128 0-delay (4 pinned cores)"
        else
            echo "WARNING: target binary is stripped or unreadable ($SYMS text symbols); skipping flamegraph" >&2
        fi
    fi
done

# ---- final metadata (adds the memory high-water mark) -----------------------

RSS_HWM=$(awk '/VmHWM/{print $2}' "/proc/$GW_PID/status" 2>/dev/null || true)
write_meta "${RSS_HWM:-null}"

[ "$HARNESS_RC" = 0 ] ||
    echo "FATAL: one or more points are incomplete - do not use this run as a reference" >&2
echo "== done: $OUT ==" >&2
exit "$HARNESS_RC"
