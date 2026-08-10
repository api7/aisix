#!/usr/bin/env bash
# On-rig baseline runner, replicating the onthebench methodology on our own box:
# 16-core m7g.4xlarge split into three disjoint pinned groups — gateway 4 cores,
# load generator 6, mock upstream 6 — default shipped config, local mock, fixed
# concurrency grid, 25s windows, >=4 valid (fail=0) repetitions per point.
#
# Usage: run-baseline.sh <aisix-src-dir> <out-dir>
#
# The default grid (ttft_ms:concurrency): 0:16 0:32 0:128 10:768 — the same
# grid as the api7/aisix#891 A/B tables. BENCH_GRID / BENCH_REPS / the other
# BENCH_* knobs in lib.sh override it, so a two-window spot-check or an extra
# delay tier is an invocation, not a script edit. The rig floor (loadgen
# driving the mock directly) is recorded per tier so gateway numbers can
# always be checked against the instrument's own ceiling. One on-CPU
# flamegraph is taken at the 0-delay c=128 saturation point when the grid has
# one (perf -> inferno, the #847 workflow).
set -euo pipefail

SRC="${1:?usage: run-baseline.sh <aisix-src-dir> <out-dir>}"
OUT="${2:?usage: run-baseline.sh <aisix-src-dir> <out-dir>}"

ENTRANT_NAME=aisix
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

BIN="$SRC/target/release/aisix"

# ---- sanity -----------------------------------------------------------------

[ -x "$BIN" ] || { echo "FATAL: $BIN missing - build first"; exit 1; }
rig_sanity
[ "$FLAMEGRAPH" = 1 ] && require_symbols "$BIN"

bench_init

# ---- config: the default-shipped-config claim set ---------------------------
# Every line exists because the process cannot run the benchmark without it:
#   resources_file  boot (standalone source; without it aisix demands etcd)
#   proxy.addr      bind (the port the harness drives)
#   admin.admin_keys boot (config refuses to load with no admin key)
# The resources file wires the mock as the only upstream and mints the one
# client key; aisix has no anonymous mode, so a key is boot-required.
cat > "$OUT/config.yaml" <<EOF
resources_file: "$OUT/resources.yaml"
proxy:
  addr: "0.0.0.0:$GW_PORT"
admin:
  admin_keys:
    - "aisix-admin-dummy"
EOF
cat > "$OUT/resources.yaml" <<EOF
_format_version: "1"
provider_keys:
  - display_name: mock-openai
    provider: openai
    adapter: openai
    api_base: "http://127.0.0.1:$MOCK_PORT/v1"
    api_key: "sk-mock"
models:
  - display_name: gpt-4o-mini
    provider: openai
    model_name: gpt-4o-mini
    provider_key: mock-openai
api_keys:
  - display_name: bench
    key_env: BENCH_AISIX_KEY
    allowed_models: ["*"]
EOF

start_gateway() {
    echo "== gateway ==" >&2
    # aisix reads AISIX_* environment variables as config overrides; nothing
    # from the harness environment may leak into the measured process.
    while read -r v; do unset "$v"; done < <(compgen -v | grep '^AISIX_' || true)
    BENCH_AISIX_KEY=bench-token taskset -c "$GW_CORES" "$BIN" --config "$OUT/config.yaml" \
        > "$OUT/gateway.log" 2>&1 &
    GW_PID=$!
    wait_http_200 "http://127.0.0.1:$GW_PORT$REQ_PATH" "gateway"
    sleep 3
    RSS_IDLE=$(rss_kb "$GW_PID")
    TPC_WORKERS=$(ps -T -p "$GW_PID" | grep -c 'tpc-' || true)
    echo "  pid=$GW_PID idle_rss=${RSS_IDLE}kB tpc_workers=$TPC_WORKERS" >&2
    # Serving-mode self-check, asserted rather than merely recorded: measuring
    # the shared-runtime fallback (or a wrong worker count) while calling it
    # the default would be a silently wrong baseline. The expected count is
    # derived from the gateway's own affinity mask, not hardcoded.
    local gw_nproc
    gw_nproc=$(taskset -c "$GW_CORES" nproc)
    [ "$TPC_WORKERS" -eq "$gw_nproc" ] ||
        { echo "FATAL: expected $gw_nproc tpc- workers under the $GW_CORES affinity, got $TPC_WORKERS"; exit 1; }
}

# Metadata is written right after the gateway starts, before the first
# measured window, so an aborted run still leaves results.jsonl attributable
# to a commit and binary; the end of the run rewrites it with the final
# memory high-water mark.
write_meta() { # write_meta <rss_hwm_kb-or-null>
    cat > "$OUT/meta.json" <<EOF
{
  "commit": "${BENCH_SRC_COMMIT:-unknown}",
  "dirty_files": ${BENCH_SRC_DIRTY:-0},
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "rig": $(meta_rig_json),
  "cores": $(meta_cores_json),
  "instruments": $(meta_instruments_json),
  "method": $(meta_method_json),
  "gateway": {
    "binary_sha256": "$(sha256sum "$BIN" | cut -d' ' -f1)",
    "rss_idle_kb": $RSS_IDLE, "rss_hwm_kb": $1,
    "tpc_workers": $TPC_WORKERS
  }
}
EOF
}

# ---- tiers: per tier, mock -> floor -> gateway points -----------------------
# The gateway starts once, after the first tier's floor (so the 0-delay floor
# never competes with an idle gateway for anything), and stays up across mock
# restarts exactly as it would across upstream churn.

FIRST_TIER=1
for TTFT in $(grid_ttfts); do
    echo "== mock (${TTFT}ms TTFT) + rig floor ==" >&2
    start_mock "$TTFT"
    floor_tier "$TTFT"
    if [ "$FIRST_TIER" = 1 ]; then
        start_gateway
        write_meta null
        FIRST_TIER=0
    fi
    for CONC in $(grid_concs "$TTFT"); do
        run_point "$TTFT" "$CONC"
    done
    if [ "$TTFT" = 0 ] && [ "$FLAMEGRAPH" = 1 ] && grid_has 0 128; then
        flamegraph_point 128 "aisix c=128 0-delay (4 pinned cores)"
    fi
done

# ---- final metadata (adds the memory high-water mark) -----------------------

RSS_HWM=$(awk '/VmHWM/{print $2}' "/proc/$GW_PID/status" 2>/dev/null || true)
write_meta "${RSS_HWM:-null}"

[ "$HARNESS_RC" = 0 ] ||
    echo "FATAL: one or more points are incomplete - do not use this run as a baseline" >&2
echo "== done: $OUT ==" >&2
exit "$HARNESS_RC"
