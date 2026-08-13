#!/usr/bin/env bash
# Post-load RSS decay runner (api7/aisix#968): after a burst of large-payload
# traffic stops, does the gateway hand freed pages back to the OS, or does
# RSS ratchet at the burst peak? One saturating burst of BENCH_DECAY_BODY_KB
# bodies, then BENCH_DECAY_S seconds of idle sampling (VmRSS/VmHWM at ~2 Hz,
# smaps_rollup Pss/LazyFree at ~1 Hz), all appended to results.jsonl by
# decay_leg in lib.sh.
#
# Deliberately a separate runner rather than a run-baseline.sh tier: the large
# bodies drive VmHWM far above anything the baseline grid produces, and a
# shared process lifetime would poison meta.json's rss_hwm_kb against every
# historical baseline. This runner gets a fresh gateway, its own idle anchor,
# and its own meta.json.
#
# Usage: run-decay.sh <aisix-src-dir> <out-dir>
set -euo pipefail

SRC="${1:?usage: run-decay.sh <aisix-src-dir> <out-dir>}"
OUT="${2:?usage: run-decay.sh <aisix-src-dir> <out-dir>}"

DECAY_CONC="${BENCH_DECAY_CONC:-64}"
DECAY_BURST_S="${BENCH_DECAY_BURST_S:-60}"
DECAY_S="${BENCH_DECAY_S:-120}"
BODY_KB="${BENCH_DECAY_BODY_KB:-120}"

# Same refuse-don't-collect policy as the lib.sh knobs: nonsense must fail
# here, not after a gateway is up. The body cap is a transport limit, not a
# taste choice: the body travels to otb as one argv string and Linux
# MAX_ARG_STRLEN is 128 KiB, so MB-scale bodies need an @file mode in otb
# first (out of scope for #968; 126 leaves room for the JSON envelope).
for _k in DECAY_CONC:"$DECAY_CONC" DECAY_BURST_S:"$DECAY_BURST_S" DECAY_S:"$DECAY_S" DECAY_BODY_KB:"$BODY_KB"; do
    [[ "${_k#*:}" =~ ^[1-9][0-9]*$ ]] ||
        { echo "FATAL: BENCH_${_k%%:*} must be a positive integer, got '${_k#*:}'"; exit 1; }
done
[ "$BODY_KB" -le 126 ] ||
    { echo "FATAL: BENCH_DECAY_BODY_KB must be <= 126 (argv transport limit), got '$BODY_KB'"; exit 1; }

# A legal chat-completions request padded to ~BODY_KB, standing in for an
# inline-base64 multimodal payload — the traffic shape the issue names as the
# ratchet driver. Set before sourcing lib.sh so readiness probes, the burst,
# and meta all see the same body.
BODY=$(python3 -c 'import json, sys
pad = "x" * (int(sys.argv[1]) * 1024)
print(json.dumps({"model": "gpt-4o-mini",
                  "messages": [{"role": "user", "content": pad}],
                  "max_tokens": 16}))' "$BODY_KB")

ENTRANT_NAME=aisix
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

BIN="$SRC/target/release/aisix"

# ---- sanity -----------------------------------------------------------------

[ -x "$BIN" ] || { echo "FATAL: $BIN missing - build first"; exit 1; }
rig_sanity

bench_init

# ---- config: same default-shipped-config claim set as run-baseline.sh -------

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
    # Readiness posts $BODY, so a gateway that cannot carry the large payload
    # end to end fails here, before anything is measured.
    wait_http_200 "http://127.0.0.1:$GW_PORT$REQ_PATH" "gateway"
    assert_listener "$GW_PORT" "$GW_PID" "gateway"
    sleep 3
    RSS_IDLE=$(rss_kb "$GW_PID")
    TPC_WORKERS=$(ps -T -p "$GW_PID" | grep -c 'tpc-' || true)
    echo "  pid=$GW_PID idle_rss=${RSS_IDLE}kB tpc_workers=$TPC_WORKERS" >&2
    local gw_nproc
    gw_nproc=$(taskset -c "$GW_CORES" nproc)
    [ "$TPC_WORKERS" -eq "$gw_nproc" ] ||
        { echo "FATAL: expected $gw_nproc tpc- workers under the $GW_CORES affinity, got $TPC_WORKERS"; exit 1; }
}

write_meta() {
    cat > "$OUT/meta.json" <<EOF
{
  "kind": "decay",
  "commit": "${BENCH_SRC_COMMIT:-unknown}",
  "dirty_files": ${BENCH_SRC_DIRTY:-0},
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "rig": $(meta_rig_json),
  "cores": $(meta_cores_json),
  "instruments": $(meta_instruments_json),
  "method": {
    "burst_conc": $DECAY_CONC, "burst_s": $DECAY_BURST_S, "decay_s": $DECAY_S,
    "body_bytes": ${#BODY}, "path": "$REQ_PATH",
    "status_hz": 2, "smaps_hz": 1
  },
  "gateway": {
    "binary_sha256": "$(sha256sum "$BIN" | cut -d' ' -f1)",
    "rss_idle_kb": $RSS_IDLE,
    "tpc_workers": $TPC_WORKERS
  }
}
EOF
}

# ---- burst + decay ----------------------------------------------------------
# Mock first: gateway readiness posts through to the upstream, so without a
# mock the gateway answers 502 and never reads as ready.

start_mock 0
start_gateway
write_meta

decay_leg "$DECAY_CONC" "$DECAY_BURST_S" "$DECAY_S"

[ "$HARNESS_RC" = 0 ] ||
    echo "FATAL: decay run incomplete - do not read a curve out of it" >&2
echo "== done: $OUT ==" >&2
exit "$HARNESS_RC"
