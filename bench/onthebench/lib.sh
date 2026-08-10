# Shared measurement core for the onthebench-style harness. Sourced by
# run-baseline.sh (the aisix baseline) and run-entrant.sh (any target
# process). Every reported number comes out of the same functions here, so
# two runs are comparable by construction instead of by code review.
#
# Contract: the sourcing runner calls bench_init after setting OUT, starts
# the measured process itself, and leaves its pid in GW_PID. Method knobs
# are env-overridable (BENCH_*) so a spot-check or an extra delay tier is
# an invocation, not a script edit; every default below is exactly what
# run-baseline.sh has always run (the api7/aisix#891 grid).

GW_CORES="0-3"; LOAD_CORES="4-9"; MOCK_CORES="10-15"
GW_PORT=3000; MOCK_PORT=8000

WINDOW="${BENCH_WINDOW:-25}"
WARMUP="${BENCH_WARMUP:-5}"
REPS="${BENCH_REPS:-4}"
MAX_TRIES="${BENCH_MAX_TRIES:-8}"
GRID="${BENCH_GRID:-0:16 0:32 0:128 10:768}"
FLOOR_REPS="${BENCH_FLOOR_REPS:-2}"
FLAMEGRAPH="${BENCH_FLAMEGRAPH:-1}"

# Request shape. An entrant overrides these before sourcing when its only
# ingress speaks a different dialect (the pinned loadgen and mock both serve
# several); the defaults are the OpenAI chat shape the baseline always used.
REQ_PATH="${REQ_PATH:-/v1/chat/completions}"
DEFAULT_BODY='{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}],"max_tokens":16}'
BODY="${BODY:-$DEFAULT_BODY}"
AUTH_HEADER="${AUTH_HEADER:-authorization: Bearer bench-token}"
# Same header shape the public board's engine sends for the chosen dialect.
if [ -z "${OTB_LOADGEN_HEADERS:-}" ]; then
    OTB_LOADGEN_HEADERS='[["'"${AUTH_HEADER%%:*}"'","'"${AUTH_HEADER#*: }"'"]]'
fi
export OTB_LOADGEN_HEADERS

# The name stamped into every result line, so mixed result sets stay
# attributable; run-baseline.sh sets "aisix", run-entrant.sh requires one.
ENTRANT_NAME="${ENTRANT_NAME:-}"

TOOLS="$HOME/bench-tools"
CLK_TCK=$(getconf CLK_TCK)
# Non-interactive ssh shells don't source the cargo env; inferno lives there.
export PATH="$HOME/.cargo/bin:$PATH"

json_str() { python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'; }
# Plain assignment on purpose: a heredoc swallows a failing command
# substitution under set -e, so this must fail loudly here instead.
BODY_JSON=$(printf '%s' "$BODY" | json_str)
HEADERS_JSON=$(printf '%s' "$OTB_LOADGEN_HEADERS" | json_str)

bench_init() {
    mkdir -p "$OUT"
    RESULTS="$OUT/results.jsonl"; : > "$RESULTS"
    # Flipped to 1 when any point ends with fewer valid windows than promised;
    # the run still completes and collects, but exits nonzero so an incomplete
    # run can never be mistaken for a baseline.
    HARNESS_RC=0
    GW_PID=""; MOCK_PID=""; SAMPLER_PID=""
    trap bench_cleanup EXIT
}

bench_cleanup() {
    [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null || true
    # An entrant that needs more than a kill (a container, a supervisor)
    # provides entrant_stop; it runs first so GW_PID is still meaningful.
    type entrant_stop >/dev/null 2>&1 && { entrant_stop || true; }
    [ -n "$GW_PID" ] && kill "$GW_PID" 2>/dev/null || true
    [ -n "$MOCK_PID" ] && kill "$MOCK_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}

# ---- rig identity ------------------------------------------------------------

rig_sanity() {
    [ "$(nproc)" = 16 ] || { echo "FATAL: expected a 16-core box, got $(nproc)"; exit 1; }
    [ "$(uname -m)" = "aarch64" ] || { echo "FATAL: expected an aarch64 rig, got $(uname -m)"; exit 1; }
    # Instance identity via IMDSv2: numbers from this harness are read as
    # m7g.4xlarge baselines, so a wrong instance type must refuse to measure.
    # IMDS being unreachable (non-EC2 lab box) is recorded rather than fatal.
    local imds_token
    imds_token=$(curl -s --max-time 2 -X PUT \
        -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
        http://169.254.169.254/latest/api/token || true)
    INSTANCE_TYPE=$(curl -s --max-time 2 -H "X-aws-ec2-metadata-token: $imds_token" \
        http://169.254.169.254/latest/meta-data/instance-type || true)
    if [ -n "$INSTANCE_TYPE" ]; then
        [ "$INSTANCE_TYPE" = "m7g.4xlarge" ] ||
            { echo "FATAL: this core split is calibrated for m7g.4xlarge, got $INSTANCE_TYPE"; exit 1; }
    else
        INSTANCE_TYPE="unknown"
        echo "WARNING: no IMDS answer; recording instance_type=unknown" >&2
    fi
    ulimit -n 65536
}

require_symbols() { # require_symbols <binary>  -> fails on a stripped binary
    local syms
    syms=$(nm "$1" 2>/dev/null | grep -c ' [tT] ' || true)
    [ "$syms" -gt 100 ] ||
        { echo "FATAL: $1 has $syms text symbols - stripped binary, flamegraph would be unreadable"; exit 1; }
}

# ---- primitives --------------------------------------------------------------

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
            -H "$AUTH_HEADER" -H 'content-type: application/json' \
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
    wait_http_200 "http://127.0.0.1:$MOCK_PORT$REQ_PATH" "mock (ttft=${1}ms)"
    # A mock that ignores MOCK_TTFT_MS would make the delayed leg silently
    # measure a 0-delay upstream with fail=0 throughout; assert the delay is
    # actually in effect before any window runs against it.
    if [ "$1" -gt 0 ]; then
        local ttfb
        ttfb=$(curl -s --max-time 2 -o /dev/null -w '%{time_starttransfer}' -X POST \
            -H 'content-type: application/json' -d "$BODY" \
            "http://127.0.0.1:$MOCK_PORT$REQ_PATH" || echo 0)
        awk -v t="$ttfb" -v want="$1" 'BEGIN{exit !(t*1000 >= want*0.9)}' ||
            { echo "FATAL: mock TTFT ${ttfb}s does not reflect ${1}ms"; exit 1; }
    fi
}

loadgen() { # loadgen <ip:port> <conc> <dur>  -> stats line on stdout
    taskset -c "$LOAD_CORES" "$TOOLS/otb" loadgen "$1" "$REQ_PATH" "$2" "$3" "$BODY"
}

# One measured window against the target: CPU% and peak RSS sampled around the
# otb window, one JSON line appended to results.jsonl. Prints 1 if valid.
measured_window() { # measured_window <point> <ttft> <conc> <rep>
    local point="$1" ttft="$2" conc="$3" rep="$4"
    local rssfile="$OUT/.rss.$$" line t0 t1 ticks0 ticks1 elapsed cpu_pct rss_peak
    local rps fail ok p50 p99 rigref budget spawn valid

    # Self-terminating on target death: this function runs inside a command
    # substitution subshell, so the EXIT trap can never see SAMPLER_PID.
    # /proc existence, not kill -0: a containerized target's pid belongs to
    # another user, where kill -0 reports EPERM and would read as death.
    ( max=0; while [ -d "/proc/$GW_PID" ]; do
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

    printf '{"kind":"gateway","entrant":"%s","point":"%s","ttft_ms":%s,"conc":%s,"rep":%s,"valid":%s,"rps":%s,"fail":%s,"ok":%s,"p50_us":%s,"p99_us":%s,"gw_cpu_pct":%s,"gw_rss_peak_kb":%s,"window_s":%s,"elapsed_s":%.2f,"otb_line":"%s"}\n' \
        "$ENTRANT_NAME" "$point" "$ttft" "$conc" "$rep" "$valid" "${rps:-null}" "${fail:-null}" "${ok:-null}" \
        "${p50:-null}" "${p99:-null}" "$cpu_pct" "$rss_peak" "$WINDOW" "$elapsed" "$line" >> "$RESULTS"

    echo "  [$point rep=$rep] rps=$rps fail=$fail p50us=$p50 p99us=$p99 cpu=${cpu_pct}% rss=${rss_peak}kB valid=$valid" >&2
    [ "$valid" = true ] && echo 1 || echo 0
}

# ---- points ------------------------------------------------------------------

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
        printf '{"kind":"floor","entrant":"%s","point":"floor-ttft%s-c%s","ttft_ms":%s,"conc":%s,"rep":%s,"valid":%s,"rps":%s,"fail":%s,"p50_us":%s,"p99_us":%s,"otb_line":"%s"}\n' \
            "$ENTRANT_NAME" "$ttft" "$conc" "$ttft" "$conc" "$try" "$valid" "$(field rps "$line")" "${fail:-null}" \
            "$(field p50us "$line")" "$(field p99us "$line")" "$line" >> "$RESULTS"
        echo "  [floor ttft=$ttft c=$conc rep=$try] valid=$valid $line" >&2
    done
    [ "$valid_n" -ge "$FLOOR_REPS" ] ||
        { echo "WARNING: floor ttft=$ttft c=$conc got only $valid_n valid windows in $MAX_TRIES tries" >&2
          HARNESS_RC=1; }
}

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

# ---- grid helpers ------------------------------------------------------------

grid_ttfts() { # distinct delay tiers, in grid order
    printf '%s\n' $GRID | cut -d: -f1 | awk '!seen[$0]++'
}

grid_concs() { # grid_concs <ttft>  -> the tier's concurrencies, in grid order
    local t="$1" s
    for s in $GRID; do [ "${s%%:*}" = "$t" ] && printf '%s\n' "${s##*:}"; done
}

grid_has() { # grid_has <ttft> <conc>
    local s
    for s in $GRID; do [ "$s" = "$1:$2" ] && return 0; done
    return 1
}

floor_tier() { # floor_tier <ttft>  -> the floor points this tier needs
    # 0-delay: the instrument ceiling barely moves with concurrency, so one
    # floor at the tier's max concurrency stands for the tier. Delayed tiers:
    # the ceiling is ~conc/delay, so every concurrency gets its own floor.
    local ttft="$1" conc
    if [ "$ttft" = 0 ]; then
        conc=$(grid_concs 0 | sort -n | tail -1)
        [ -n "$conc" ] && floor_point 0 "$conc"
    else
        for conc in $(grid_concs "$ttft"); do floor_point "$ttft" "$conc"; done
    fi
}

# ---- flamegraph --------------------------------------------------------------

flamegraph_point() { # flamegraph_point <conc> <title>  (0-delay mock running)
    local conc="$1" title="$2" load_bg
    echo "== flamegraph (c=$conc, 0-delay) ==" >&2
    loadgen "127.0.0.1:$GW_PORT" "$conc" 45 > "$OUT/flamegraph-window.txt" & load_bg=$!
    sleep 5
    perf record -F 499 --call-graph dwarf -p "$GW_PID" -o "$OUT/perf.data" -- sleep 25 \
        >> "$OUT/perf.log" 2>&1 || echo "WARNING: perf record failed" >&2
    wait "$load_bg" || true
    # Rendering must never kill the measurement: perf.data is kept on any
    # failure so the SVG can be produced off-rig.
    if [ -s "$OUT/perf.data" ]; then
        if perf script -i "$OUT/perf.data" 2>> "$OUT/perf.log" |
               inferno-collapse-perf > "$OUT/flamegraph-c$conc.folded" 2>> "$OUT/perf.log" &&
           inferno-flamegraph --title "$title" \
               < "$OUT/flamegraph-c$conc.folded" > "$OUT/flamegraph-c$conc.svg" 2>> "$OUT/perf.log"; then
            rm -f "$OUT/perf.data"
        else
            echo "WARNING: flamegraph rendering failed; perf.data kept" >&2
        fi
    fi
}

# ---- shared metadata fragments -----------------------------------------------

meta_rig_json() {
    cat <<EOF
{
    "host": "$(hostname)",
    "instance_type": "$INSTANCE_TYPE",
    "kernel": "$(uname -r)",
    "arch": "$(uname -m)",
    "nproc": $(nproc),
    "mem_total_kb": $(awk '/MemTotal/{print $2}' /proc/meminfo),
    "cpu_part": "$(awk -F': ' '/CPU part/{print $2; exit}' /proc/cpuinfo)"
  }
EOF
}

meta_cores_json() {
    printf '{"gateway": "%s", "load": "%s", "mock": "%s"}' "$GW_CORES" "$LOAD_CORES" "$MOCK_CORES"
}

meta_instruments_json() {
    cat <<EOF
{
    "engine_pin": "f3adbb1315b26129f5e317af5279decefb1cea8f (engine-v1)",
    "otb_sha256": "$(sha256sum "$TOOLS/otb" | cut -d' ' -f1)",
    "mock_sha256": "$(sha256sum "$TOOLS/mock" | cut -d' ' -f1)"
  }
EOF
}

meta_method_json() {
    cat <<EOF
{
    "window_s": $WINDOW, "warmup_s": $WARMUP, "reps": $REPS,
    "max_tries": $MAX_TRIES, "floor_reps": $FLOOR_REPS,
    "grid": "$GRID", "body": $BODY_JSON,
    "path": "$REQ_PATH", "clk_tck": $CLK_TCK, "nofile": $(ulimit -n),
    "loadgen_headers": $HEADERS_JSON,
    "flamegraph": {"enabled": $([ "$FLAMEGRAPH" = 1 ] && echo true || echo false), "freq_hz": 499, "callgraph": "dwarf", "window_s": 25, "conc": 128}
  }
EOF
}
