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
# Bytes of stack perf copies per sample for dwarf unwinding. perf's 8KB
# default has produced captures on this rig where 88-96% of the sampled
# weight unwound to nothing; 32KB has not. The trigger is not isolated to a
# build profile or a commit - the two worst captures are a thin-LTO build,
# and one commit produced both a 42% and a 71% capture - so this size is a
# mitigation, not a diagnosis, and check_symbolization measures every
# capture rather than trusting it.
PERF_STACK="${BENCH_PERF_STACK:-32768}"

# Request shape. An entrant overrides these before sourcing when its only
# ingress speaks a different dialect (the pinned loadgen and mock both serve
# several); the defaults are the OpenAI chat shape the baseline always used.
REQ_PATH="${REQ_PATH:-/v1/chat/completions}"
DEFAULT_BODY='{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}],"max_tokens":16}'
BODY="${BODY:-$DEFAULT_BODY}"
AUTH_HEADER="${AUTH_HEADER:-authorization: Bearer bench-token}"
# Same header shape the public board's engine sends for the chosen dialect.
# Always derived from AUTH_HEADER — an ambient OTB_LOADGEN_HEADERS would let
# measured requests carry credentials the wait_http_200 readiness check never
# exercised. json.dumps, not string splicing: a quote or backslash in the
# header value must not produce an invalid header list.
OTB_LOADGEN_HEADERS=$(python3 -c 'import json,sys
k, _, v = sys.argv[1].partition(":")
print(json.dumps([[k.strip(), v.strip()]]))' "$AUTH_HEADER")
export OTB_LOADGEN_HEADERS

# The name stamped into every result line, so mixed result sets stay
# attributable; run-baseline.sh sets "aisix", run-entrant.sh requires one.
ENTRANT_NAME="${ENTRANT_NAME:-}"

# Overrides are operator input; nonsense must refuse to measure rather than
# succeed while collecting nothing (BENCH_REPS=0 would "pass" every point).
is_pos_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_pos_int "$WINDOW" || { echo "FATAL: BENCH_WINDOW must be a positive integer, got '$WINDOW'"; exit 1; }
[[ "$WARMUP" =~ ^[0-9]+$ ]] || { echo "FATAL: BENCH_WARMUP must be a non-negative integer, got '$WARMUP'"; exit 1; }
is_pos_int "$REPS" || { echo "FATAL: BENCH_REPS must be a positive integer, got '$REPS'"; exit 1; }
is_pos_int "$MAX_TRIES" || { echo "FATAL: BENCH_MAX_TRIES must be a positive integer, got '$MAX_TRIES'"; exit 1; }
is_pos_int "$FLOOR_REPS" || { echo "FATAL: BENCH_FLOOR_REPS must be a positive integer, got '$FLOOR_REPS'"; exit 1; }
# perf rounds a non-multiple of 8 up (callchain.c get_stack_size) and hard
# errors above round_down(USHRT_MAX, 8) = 65528. Require the exact value so
# the size recorded in meta.json is the size perf actually used, and so an
# out-of-range value fails here rather than after a 45s capture window.
is_pos_int "$PERF_STACK" && [ $((PERF_STACK % 8)) -eq 0 ] && [ "$PERF_STACK" -le 65528 ] ||
    { echo "FATAL: BENCH_PERF_STACK must be a positive multiple of 8 up to 65528, got '$PERF_STACK'"; exit 1; }
case "$FLAMEGRAPH" in 0|1) ;; *) echo "FATAL: BENCH_FLAMEGRAPH must be 0 or 1, got '$FLAMEGRAPH'"; exit 1 ;; esac
[ -n "$GRID" ] || { echo "FATAL: BENCH_GRID is empty"; exit 1; }
for _spec in $GRID; do
    [[ "$_spec" =~ ^[0-9]+:[1-9][0-9]*$ ]] ||
        { echo "FATAL: BENCH_GRID entry '$_spec' is not ttft_ms:conc"; exit 1; }
done

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

    # Tick reads are fallible on purpose: a target that dies mid-window takes
    # /proc/<pid>/stat with it, and that must record an invalid window, not
    # kill the runner before the JSONL line lands.
    ticks0=$(cpu_ticks "$GW_PID" 2>/dev/null) || ticks0=""
    t0=$(date +%s.%N)
    line=$(loadgen "127.0.0.1:$GW_PORT" "$conc" "$WINDOW") || line=""
    line=${line//[\"\\]/ }
    t1=$(date +%s.%N)
    ticks1=$(cpu_ticks "$GW_PID" 2>/dev/null) || ticks1=""
    kill "$SAMPLER_PID" 2>/dev/null || true; wait "$SAMPLER_PID" 2>/dev/null || true; SAMPLER_PID=""
    rss_peak=$(cat "$rssfile" 2>/dev/null || echo 0); rm -f "$rssfile"

    elapsed=$(awk -v a="$t0" -v b="$t1" 'BEGIN{print b-a}')
    if [ -n "$ticks0" ] && [ -n "$ticks1" ]; then
        cpu_pct=$(awk -v d=$((ticks1 - ticks0)) -v hz="$CLK_TCK" -v e="$elapsed" \
            'BEGIN{printf "%.1f", d/hz/e*100}')
    else
        cpu_pct=null
    fi

    rps=$(field rps "$line"); fail=$(field fail "$line"); ok=$(field ok "$line")
    p50=$(field p50us "$line"); p99=$(field p99us "$line")
    rigref=$(field rigrefused "$line"); budget=$(field budgetexceeded "$line"); spawn=$(field spawnfailed "$line")
    valid=true
    [ "$cpu_pct" != null ] && [ "${fail:-1}" = "0" ] && [ "${rigref:-0}" = "0" ] \
        && [ "${budget:-0}" = "0" ] && [ "${spawn:-0}" = "0" ] || valid=false

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
    local ttft="$1" conc="$2" valid_n=0 try=0
    local line fail rigref budget spawn rps p50 p99 valid
    loadgen "127.0.0.1:$MOCK_PORT" "$conc" 5 > /dev/null || true   # warmup
    while [ "$valid_n" -lt "$FLOOR_REPS" ] && [ "$try" -lt "$MAX_TRIES" ]; do
        try=$((try + 1))
        line=$(loadgen "127.0.0.1:$MOCK_PORT" "$conc" "$WINDOW") || line=""
        line=${line//[\"\\]/ }
        # Same validity fields as measured_window: a floor window that was
        # refused or over budget must not become the rig reference, and an
        # empty loadgen line must record nulls, not invalid JSON.
        rps=$(field rps "$line"); fail=$(field fail "$line")
        p50=$(field p50us "$line"); p99=$(field p99us "$line")
        rigref=$(field rigrefused "$line"); budget=$(field budgetexceeded "$line"); spawn=$(field spawnfailed "$line")
        valid=true
        [ "${fail:-1}" = "0" ] && [ "${rigref:-0}" = "0" ] && [ "${budget:-0}" = "0" ] \
            && [ "${spawn:-0}" = "0" ] || valid=false
        [ "$valid" = true ] && valid_n=$((valid_n + 1))
        printf '{"kind":"floor","entrant":"%s","point":"floor-ttft%s-c%s","ttft_ms":%s,"conc":%s,"rep":%s,"valid":%s,"rps":%s,"fail":%s,"p50_us":%s,"p99_us":%s,"otb_line":"%s"}\n' \
            "$ENTRANT_NAME" "$ttft" "$conc" "$ttft" "$conc" "$try" "$valid" "${rps:-null}" "${fail:-null}" \
            "${p50:-null}" "${p99:-null}" "$line" >> "$RESULTS"
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
    # if, not &&: with && the function returns 1 whenever the LAST grid
    # element is another tier's, and a `conc=$(grid_concs 0 | ...)`
    # assignment under set -e + pipefail then kills the run silently.
    local t="$1" s
    for s in $GRID; do
        if [ "${s%%:*}" = "$t" ]; then printf '%s\n' "${s##*:}"; fi
    done
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

# A flamegraph whose stacks did not unwind is worse than no flamegraph: it
# renders, it looks plausible, and every attribution taken from it is wrong.
# The check is on the collapsed output rather than on perf's exit code
# because that is where the failure is visible - frames that unwound to
# nothing come back as [unknown]/[dso] entries with no symbol.
#
# Two shares, because they fail differently. "no symbol at any depth" is the
# coarse one and can pass a capture whose leaves are nearly all anonymous;
# the leaf share is what a flamegraph attributes self-time to, and separates
# this repo's own captures more widely. Both thresholds come from those
# captures: usable ones measure up to 71% (any-depth) and 72% (leaf), the
# failed ones start at 88% and 91% respectively.
check_symbolization() { # check_symbolization <folded-file>
    local shares any leaf
    shares=$(awk '
        { n = $NF; total += n
          sub(/ [0-9]+$/, ""); nf = split($0, frames, ";")
          resolved = 0
          # frames[1] is the thread name and always resolves; start past it
          for (i = 2; i <= nf; i++)
              if (frames[i] !~ /^\[/) { resolved = 1; break }
          if (!resolved) bad += n
          if (nf >= 2 && frames[nf] ~ /^\[/) badleaf += n }
        END { if (total > 0) printf "%.1f %.1f", bad / total * 100, badleaf / total * 100
              else print "empty empty" }
    ' "$1") || { echo "WARNING: cannot read $1 for the symbolization check" >&2; return 0; }
    read -r any leaf <<<"$shares"
    if [ "$any" = empty ]; then
        echo "WARNING: $1 has no samples - the capture produced nothing to attribute" >&2
        return 0
    fi
    # Reported into perf.log as well as the terminal: perf.log is collected
    # off the rig, and the reader who most needs this number is the one
    # opening the SVG months later, not the operator watching the run.
    echo "  flamegraph: $any% unresolved stacks, $leaf% unresolved leaves" |
        tee -a "$OUT/perf.log" >&2
    awk -v a="$any" -v l="$leaf" 'BEGIN{exit !(a > 80 || l > 85)}' &&
        echo "WARNING: $1 unwound poorly - do not attribute from this flamegraph (try a larger BENCH_PERF_STACK)" |
            tee -a "$OUT/perf.log" >&2
    return 0
}

flamegraph_point() { # flamegraph_point <conc> <title>  (0-delay mock running)
    local conc="$1" title="$2" load_bg
    echo "== flamegraph (c=$conc, 0-delay) ==" >&2
    # A 32KB dump writes roughly 4x the perf.data of perf's 8KB default
    # (~1.5GB over this window); the rig is shared and has run out of disk
    # before. Warn rather than skip - a capture is worth attempting.
    local avail_mb
    avail_mb=$(df -Pm "$OUT" | awk 'NR==2{print $4}')
    [ "${avail_mb:-0}" -ge 4096 ] ||
        echo "WARNING: ${avail_mb}MB free at $OUT; a ${PERF_STACK}-byte dump may not fit" >&2
    loadgen "127.0.0.1:$GW_PORT" "$conc" 45 > "$OUT/flamegraph-window.txt" & load_bg=$!
    sleep 5
    # Explicit stack dump size: a stack that does not fit in the dump unwinds
    # to nothing - every frame collapses to [unknown] and the SVG renders as
    # one flat bar. The failure is silent (perf exits 0, inferno renders
    # happily), which is why the size is pinned here rather than left to the
    # default, and why the collapsed output is checked below.
    perf record -F 499 --call-graph "dwarf,$PERF_STACK" -p "$GW_PID" -o "$OUT/perf.data" -- sleep 25 \
        >> "$OUT/perf.log" 2>&1 || echo "WARNING: perf record failed" >&2
    wait "$load_bg" || true
    # Rendering must never kill the measurement: perf.data is kept on any
    # failure so the SVG can be produced off-rig.
    if [ -s "$OUT/perf.data" ]; then
        if perf script -i "$OUT/perf.data" 2>> "$OUT/perf.log" |
               inferno-collapse-perf > "$OUT/flamegraph-c$conc.folded" 2>> "$OUT/perf.log" &&
           inferno-flamegraph --title "$title" \
               < "$OUT/flamegraph-c$conc.folded" > "$OUT/flamegraph-c$conc.svg" 2>> "$OUT/perf.log"; then
            check_symbolization "$OUT/flamegraph-c$conc.folded"
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
    "flamegraph": {"enabled": $([ "$FLAMEGRAPH" = 1 ] && grid_has 0 128 && echo true || echo false), "freq_hz": 499, "callgraph": "dwarf,$PERF_STACK", "stack_bytes": $PERF_STACK, "window_s": 25, "conc": 128}
  }
EOF
}
