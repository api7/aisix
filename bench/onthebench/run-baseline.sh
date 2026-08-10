#!/usr/bin/env bash
# On-rig baseline runner, replicating the onthebench methodology on our own box:
# 16-core m7g.4xlarge split into three disjoint pinned groups — gateway 4 cores,
# load generator 6, mock upstream 6 — default shipped config, local mock, fixed
# concurrency grid, 25s windows, >=4 valid (fail=0) repetitions per point.
#
# Usage: run-baseline.sh <aisix-src-dir> <out-dir>
#
# Load points (ttft_ms:concurrency): 0:16 0:32 0:128 10:768 — the same grid as
# the api7/aisix#891 A/B tables. The rig floor (loadgen driving the mock
# directly) is recorded for both mock variants so gateway numbers can always be
# checked against the instrument's own ceiling. One on-CPU flamegraph is taken
# at the c=128 saturation point (perf -> inferno, the #847 workflow).
set -euo pipefail

SRC="${1:?usage: run-baseline.sh <aisix-src-dir> <out-dir>}"
OUT="${2:?usage: run-baseline.sh <aisix-src-dir> <out-dir>}"

GW_CORES="0-3"; LOAD_CORES="4-9"; MOCK_CORES="10-15"
GW_PORT=3000; MOCK_PORT=8000
WINDOW=25; WARMUP=5; REPS=4; MAX_TRIES=8
GRID="0:16 0:32 0:128 10:768"
FLOOR_REPS=2
BODY='{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}],"max_tokens":16}'
CHAT_PATH="/v1/chat/completions"
# Same header shape the public board's engine sends for the OpenAI dialect.
export OTB_LOADGEN_HEADERS='[["authorization","Bearer bench-token"]]'

BIN="$SRC/target/release/aisix"
TOOLS="$HOME/bench-tools"
CLK_TCK=$(getconf CLK_TCK)
# Non-interactive ssh shells don't source the cargo env; inferno lives there.
export PATH="$HOME/.cargo/bin:$PATH"
# Plain assignment on purpose: a heredoc swallows a failing command
# substitution under set -e, so this must fail loudly here instead.
BODY_JSON=$(printf '%s' "$BODY" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')

mkdir -p "$OUT"
RESULTS="$OUT/results.jsonl"; : > "$RESULTS"
# Flipped to 1 when any point ends with fewer valid windows than promised; the
# run still completes and collects, but exits nonzero so an incomplete run can
# never be mistaken for a baseline.
HARNESS_RC=0

GW_PID=""; MOCK_PID=""; SAMPLER_PID=""
cleanup() {
    [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null || true
    [ -n "$GW_PID" ] && kill "$GW_PID" 2>/dev/null || true
    [ -n "$MOCK_PID" ] && kill "$MOCK_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

# ---- helpers ----------------------------------------------------------------

field() { # field <key> <stats-line>  -> value or empty
    sed -n "s/.*\b$1=\([^ ]*\).*/\1/p" <<<"$2"
}

cpu_ticks() { # utime+stime of the whole process, in clock ticks
    awk '{print $14+$15}' "/proc/$1/stat"
}

rss_kb() { awk '/VmRSS/{print $2}' "/proc/$1/status"; }

wait_http_200() { # wait_http_200 <url> <label> [max_s]
    local url="$1" label="$2" max="${3:-30}" code
    for _ in $(seq 1 $((max * 2))); do
        code=$(curl -s --max-time 2 -o /dev/null -w '%{http_code}' -X POST \
            -H 'authorization: Bearer bench-token' -H 'content-type: application/json' \
            -d "$BODY" "$url" || true)
        [ "$code" = "200" ] && return 0
        sleep 0.5
    done
    echo "FATAL: $label not answering 200 at $url (last code: $code)" >&2
    return 1
}

start_mock() { # start_mock <ttft_ms>
    [ -n "$MOCK_PID" ] && { kill "$MOCK_PID" 2>/dev/null || true; wait "$MOCK_PID" 2>/dev/null || true; }
    MOCK_TTFT_MS="$1" taskset -c "$MOCK_CORES" "$TOOLS/mock" -port "$MOCK_PORT" \
        >> "$OUT/mock.log" 2>&1 &
    MOCK_PID=$!
    wait_http_200 "http://127.0.0.1:$MOCK_PORT$CHAT_PATH" "mock (ttft=${1}ms)"
    # A mock that ignores MOCK_TTFT_MS would make the delayed leg silently
    # measure a 0-delay upstream with fail=0 throughout; assert the delay is
    # actually in effect before any window runs against it.
    if [ "$1" -gt 0 ]; then
        local ttfb
        ttfb=$(curl -s --max-time 2 -o /dev/null -w '%{time_starttransfer}' -X POST \
            -H 'content-type: application/json' -d "$BODY" \
            "http://127.0.0.1:$MOCK_PORT$CHAT_PATH" || echo 0)
        awk -v t="$ttfb" -v want="$1" 'BEGIN{exit !(t*1000 >= want*0.9)}' ||
            { echo "FATAL: mock TTFT ${ttfb}s does not reflect ${1}ms"; exit 1; }
    fi
}

loadgen() { # loadgen <ip:port> <conc> <dur>  -> stats line on stdout
    taskset -c "$LOAD_CORES" "$TOOLS/otb" loadgen "$1" "$CHAT_PATH" "$2" "$3" "$BODY"
}

# One measured window against the gateway: CPU% and peak RSS sampled around the
# otb window, one JSON line appended to results.jsonl. Prints 1 if valid.
measured_window() { # measured_window <point> <ttft> <conc> <rep>
    local point="$1" ttft="$2" conc="$3" rep="$4"
    local rssfile="$OUT/.rss.$$" line t0 t1 ticks0 ticks1 elapsed cpu_pct rss_peak
    local rps fail ok p50 p99 rigref budget spawn valid

    # Self-terminating on gateway death: this function runs inside a command
    # substitution subshell, so the EXIT trap can never see SAMPLER_PID.
    ( max=0; while kill -0 "$GW_PID" 2>/dev/null; do
          v=$(awk '/VmRSS/{print $2}' "/proc/$GW_PID/status" 2>/dev/null || true)
          v="${v:-0}"
          if [ "$v" -gt "$max" ]; then max="$v"; echo "$max" > "$rssfile"; fi
          sleep 0.2
      done ) & SAMPLER_PID=$!

    ticks0=$(cpu_ticks "$GW_PID"); t0=$(date +%s.%N)
    line=$(loadgen "127.0.0.1:$GW_PORT" "$conc" "$WINDOW") || line=""
    line=${line//[\"\\]/ }
    t1=$(date +%s.%N); ticks1=$(cpu_ticks "$GW_PID")
    kill "$SAMPLER_PID" 2>/dev/null || true; wait "$SAMPLER_PID" 2>/dev/null || true; SAMPLER_PID=""
    rss_peak=$(cat "$rssfile" 2>/dev/null || echo 0); rm -f "$rssfile"

    elapsed=$(awk -v a="$t0" -v b="$t1" 'BEGIN{print b-a}')
    cpu_pct=$(awk -v d=$((ticks1 - ticks0)) -v hz="$CLK_TCK" -v e="$elapsed" \
        'BEGIN{printf "%.1f", d/hz/e*100}')

    rps=$(field rps "$line"); fail=$(field fail "$line"); ok=$(field ok "$line")
    p50=$(field p50us "$line"); p99=$(field p99us "$line")
    rigref=$(field rigrefused "$line"); budget=$(field budgetexceeded "$line"); spawn=$(field spawnfailed "$line")
    valid=true
    [ "${fail:-1}" = "0" ] && [ "${rigref:-0}" = "0" ] && [ "${budget:-0}" = "0" ] \
        && [ "${spawn:-0}" = "0" ] || valid=false

    printf '{"kind":"gateway","point":"%s","ttft_ms":%s,"conc":%s,"rep":%s,"valid":%s,"rps":%s,"fail":%s,"ok":%s,"p50_us":%s,"p99_us":%s,"gw_cpu_pct":%s,"gw_rss_peak_kb":%s,"window_s":%s,"elapsed_s":%.2f,"otb_line":"%s"}\n' \
        "$point" "$ttft" "$conc" "$rep" "$valid" "${rps:-null}" "${fail:-null}" "${ok:-null}" \
        "${p50:-null}" "${p99:-null}" "$cpu_pct" "$rss_peak" "$WINDOW" "$elapsed" "$line" >> "$RESULTS"

    echo "  [$point rep=$rep] rps=$rps fail=$fail p50us=$p50 p99us=$p99 cpu=${cpu_pct}% rss=${rss_peak}kB valid=$valid" >&2
    [ "$valid" = true ] && echo 1 || echo 0
}

# ---- sanity -----------------------------------------------------------------

[ -x "$BIN" ] || { echo "FATAL: $BIN missing - build first"; exit 1; }
[ "$(nproc)" = 16 ] || { echo "FATAL: expected a 16-core box, got $(nproc)"; exit 1; }
[ "$(uname -m)" = "aarch64" ] || { echo "FATAL: expected an aarch64 rig, got $(uname -m)"; exit 1; }
# Instance identity via IMDSv2: numbers from this harness are read as
# m7g.4xlarge baselines, so a wrong instance type must refuse to measure.
# IMDS being unreachable (non-EC2 lab box) is recorded rather than fatal.
IMDS_TOKEN=$(curl -s --max-time 2 -X PUT \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
    http://169.254.169.254/latest/api/token || true)
INSTANCE_TYPE=$(curl -s --max-time 2 -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-type || true)
if [ -n "$INSTANCE_TYPE" ]; then
    [ "$INSTANCE_TYPE" = "m7g.4xlarge" ] ||
        { echo "FATAL: this core split is calibrated for m7g.4xlarge, got $INSTANCE_TYPE"; exit 1; }
else
    INSTANCE_TYPE="unknown"
    echo "WARNING: no IMDS answer; recording instance_type=unknown" >&2
fi
SYMS=$(nm "$BIN" 2>/dev/null | grep -c ' [tT] ' || true)
[ "$SYMS" -gt 100 ] || { echo "FATAL: $BIN has $SYMS text symbols - stripped binary, flamegraph would be unreadable"; exit 1; }
ulimit -n 65536

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

# ---- rig floor (0-delay): loadgen -> mock directly, gateway not yet running --

floor_point() { # floor_point <ttft> <conc>  (mock must already run with that ttft)
    # Same validity policy as gateway points, scaled to FLOOR_REPS: an invalid
    # window is recorded, marked, and retried, and can never stand as the rig
    # reference on its own.
    local ttft="$1" conc="$2" valid_n=0 try=0 line fail valid
    loadgen "127.0.0.1:$MOCK_PORT" "$conc" 5 > /dev/null || true   # warmup
    while [ "$valid_n" -lt "$FLOOR_REPS" ] && [ "$try" -lt "$MAX_TRIES" ]; do
        try=$((try + 1))
        line=$(loadgen "127.0.0.1:$MOCK_PORT" "$conc" "$WINDOW") || line=""
        line=${line//[\"\\]/ }
        fail=$(field fail "$line")
        valid=true; [ "${fail:-1}" = "0" ] || valid=false
        [ "$valid" = true ] && valid_n=$((valid_n + 1))
        printf '{"kind":"floor","point":"floor-ttft%s-c%s","ttft_ms":%s,"conc":%s,"rep":%s,"valid":%s,"rps":%s,"fail":%s,"p50_us":%s,"p99_us":%s,"otb_line":"%s"}\n' \
            "$ttft" "$conc" "$ttft" "$conc" "$try" "$valid" "$(field rps "$line")" "${fail:-null}" \
            "$(field p50us "$line")" "$(field p99us "$line")" "$line" >> "$RESULTS"
        echo "  [floor ttft=$ttft c=$conc rep=$try] valid=$valid $line" >&2
    done
    [ "$valid_n" -ge "$FLOOR_REPS" ] ||
        { echo "WARNING: floor ttft=$ttft c=$conc got only $valid_n valid windows in $MAX_TRIES tries" >&2
          HARNESS_RC=1; }
}

echo "== mock (0-delay) + rig floor ==" >&2
start_mock 0
floor_point 0 128

# ---- gateway: default shipped config, pinned to its 4 cores -----------------

echo "== gateway ==" >&2
# aisix reads AISIX_* environment variables as config overrides; nothing from
# the harness environment may leak into the measured process.
while read -r v; do unset "$v"; done < <(compgen -v | grep '^AISIX_' || true)
BENCH_AISIX_KEY=bench-token taskset -c "$GW_CORES" "$BIN" --config "$OUT/config.yaml" \
    > "$OUT/gateway.log" 2>&1 &
GW_PID=$!
wait_http_200 "http://127.0.0.1:$GW_PORT$CHAT_PATH" "gateway"
sleep 3
RSS_IDLE=$(rss_kb "$GW_PID")
TPC_WORKERS=$(ps -T -p "$GW_PID" | grep -c 'tpc-' || true)
echo "  pid=$GW_PID idle_rss=${RSS_IDLE}kB tpc_workers=$TPC_WORKERS" >&2
# Serving-mode self-check, asserted rather than merely recorded: measuring the
# shared-runtime fallback (or a wrong worker count) while calling it the
# default would be a silently wrong baseline. The expected count is derived
# from the gateway's own affinity mask, not hardcoded.
GW_NPROC=$(taskset -c "$GW_CORES" nproc)
[ "$TPC_WORKERS" -eq "$GW_NPROC" ] ||
    { echo "FATAL: expected $GW_NPROC tpc- workers under the $GW_CORES affinity, got $TPC_WORKERS"; exit 1; }

# Metadata is written NOW, before the first measured window, so an aborted run
# still leaves results.jsonl attributable to a commit and binary; the end of
# the run rewrites it with the final memory high-water mark.
write_meta() { # write_meta <rss_hwm_kb-or-null>
    cat > "$OUT/meta.json" <<EOF
{
  "commit": "${BENCH_SRC_COMMIT:-unknown}",
  "dirty_files": ${BENCH_SRC_DIRTY:-0},
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "rig": {
    "host": "$(hostname)",
    "instance_type": "$INSTANCE_TYPE",
    "kernel": "$(uname -r)",
    "arch": "$(uname -m)",
    "nproc": $(nproc),
    "mem_total_kb": $(awk '/MemTotal/{print $2}' /proc/meminfo),
    "cpu_part": "$(awk -F': ' '/CPU part/{print $2; exit}' /proc/cpuinfo)"
  },
  "cores": {"gateway": "$GW_CORES", "load": "$LOAD_CORES", "mock": "$MOCK_CORES"},
  "instruments": {
    "engine_pin": "f3adbb1315b26129f5e317af5279decefb1cea8f (engine-v1)",
    "otb_sha256": "$(sha256sum "$TOOLS/otb" | cut -d' ' -f1)",
    "mock_sha256": "$(sha256sum "$TOOLS/mock" | cut -d' ' -f1)"
  },
  "method": {
    "window_s": $WINDOW, "warmup_s": $WARMUP, "reps": $REPS,
    "max_tries": $MAX_TRIES, "floor_reps": $FLOOR_REPS,
    "grid": "$GRID", "body": $BODY_JSON,
    "path": "$CHAT_PATH", "clk_tck": $CLK_TCK, "nofile": $(ulimit -n),
    "loadgen_headers": $(printf '%s' "$OTB_LOADGEN_HEADERS" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
    "flamegraph": {"freq_hz": 499, "callgraph": "dwarf", "window_s": 25, "conc": 128}
  },
  "gateway": {
    "binary_sha256": "$(sha256sum "$BIN" | cut -d' ' -f1)",
    "rss_idle_kb": $RSS_IDLE, "rss_hwm_kb": $1,
    "tpc_workers": $TPC_WORKERS
  }
}
EOF
}
write_meta null

# ---- 0-delay grid -----------------------------------------------------------

run_point() { # run_point <ttft> <conc>
    local ttft="$1" conc="$2" point="ttft$1-c$2" valid_n=0 try=0
    echo "== point $point ==" >&2
    loadgen "127.0.0.1:$GW_PORT" "$conc" "$WARMUP" > /dev/null || true
    while [ "$valid_n" -lt "$REPS" ] && [ "$try" -lt "$MAX_TRIES" ]; do
        try=$((try + 1))
        valid_n=$((valid_n + $(measured_window "$point" "$ttft" "$conc" "$try")))
    done
    [ "$valid_n" -ge "$REPS" ] ||
        { echo "WARNING: $point got only $valid_n valid reps in $MAX_TRIES tries" >&2
          HARNESS_RC=1; }
}

for spec in $GRID; do
    ttft="${spec%%:*}"; conc="${spec##*:}"
    [ "$ttft" = 0 ] || continue
    run_point "$ttft" "$conc"
done

# ---- flamegraph at the c=128 saturation point (#847 workflow) ---------------

echo "== flamegraph (c=128, 0-delay) ==" >&2
loadgen "127.0.0.1:$GW_PORT" 128 45 > "$OUT/flamegraph-window.txt" & LOAD_BG=$!
sleep 5
perf record -F 499 --call-graph dwarf -p "$GW_PID" -o "$OUT/perf.data" -- sleep 25 \
    >> "$OUT/perf.log" 2>&1 || echo "WARNING: perf record failed" >&2
wait "$LOAD_BG" || true
# Rendering must never kill the measurement: perf.data is kept on any failure
# so the SVG can be produced off-rig.
if [ -s "$OUT/perf.data" ]; then
    if perf script -i "$OUT/perf.data" 2>> "$OUT/perf.log" |
           inferno-collapse-perf > "$OUT/flamegraph-c128.folded" 2>> "$OUT/perf.log" &&
       inferno-flamegraph --title "aisix c=128 0-delay (4 pinned cores)" \
           < "$OUT/flamegraph-c128.folded" > "$OUT/flamegraph-c128.svg" 2>> "$OUT/perf.log"; then
        rm -f "$OUT/perf.data"
    else
        echo "WARNING: flamegraph rendering failed; perf.data kept" >&2
    fi
fi

# ---- 10ms-TTFT leg: mock restarted with the delay, floor then gateway -------

echo "== mock (10ms TTFT) + floor ==" >&2
start_mock 10
floor_point 10 768

for spec in $GRID; do
    ttft="${spec%%:*}"; conc="${spec##*:}"
    [ "$ttft" = 10 ] || continue
    run_point "$ttft" "$conc"
done

# ---- final metadata (adds the memory high-water mark) -----------------------

RSS_HWM=$(awk '/VmHWM/{print $2}' "/proc/$GW_PID/status" 2>/dev/null || true)
write_meta "${RSS_HWM:-null}"

[ "$HARNESS_RC" = 0 ] ||
    echo "FATAL: one or more points are incomplete - do not use this run as a baseline" >&2
echo "== done: $OUT ==" >&2
exit "$HARNESS_RC"
