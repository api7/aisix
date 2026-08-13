#!/usr/bin/env bash
# PGO training run — phase B of the three-phase release build (#967).
#
# Drives the v1 shape matrix (defined in trainer/src/main.rs, single source
# of truth) through an INSTRUMENTED gateway binary, one gateway process
# lifetime per shape, so every shape leaves its own .profraw and the merged
# profile is a union of hotness. Then merges the profraws with the
# llvm-profdata that ships in the pinned rustup toolchain (llvm-tools-preview
# — exact LLVM match with rustc, zero external toolchain dependencies) and
# writes a content-addressed merged-<sha>.profdata plus train-manifest.json.
#
# The content-addressed profdata name is load-bearing: -Cprofile-use=<path>
# enters cargo's fingerprint via RUSTFLAGS, but cargo does NOT fingerprint the
# file's CONTENT — a retrained profile at an unchanged path would silently
# reuse stale phase-C artifacts from a persistent target dir. Hashing the name
# makes every retrain a fresh RUSTFLAGS value.
#
# FAIL-CLOSED (#967 hard gate 2): every failure path here exits non-zero and
# the caller must treat that as fatal. Never add a fallback that lets a
# release continue with a partial or empty profile.
#
# Usage: train.sh <instrumented-aisix-binary> <pgo-dir>
#
# Knobs (env):
#   PGO_TRAINER_BIN         trainer binary (default: trainer/target/release/pgo-trainer)
#   PGO_TRAIN_REQUESTS      requests per shape        (default 3000)
#   PGO_TRAIN_CONCURRENCY   driver connections        (default 8)
#   PGO_GW_PORT / PGO_MOCK_PORT / PGO_METRICS_PORT    (defaults 13000/18001/19090)
set -euo pipefail

BIN="${1:?usage: train.sh <instrumented-aisix-binary> <pgo-dir>}"
PGO_DIR="${2:?usage: train.sh <instrumented-aisix-binary> <pgo-dir>}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER="${PGO_TRAINER_BIN:-$HERE/trainer/target/release/pgo-trainer}"
REQS="${PGO_TRAIN_REQUESTS:-3000}"
CONC="${PGO_TRAIN_CONCURRENCY:-8}"
GW_PORT="${PGO_GW_PORT:-13000}"
MOCK_PORT="${PGO_MOCK_PORT:-18001}"
METRICS_PORT="${PGO_METRICS_PORT:-19090}"

# A profraw for this binary is megabytes; an empty one (no counters flushed)
# is a few KB. The floor catches "gateway died before flushing" without ever
# false-failing a real profile.
PROFRAW_FLOOR_BYTES=262144
PROFDATA_FLOOR_BYTES=524288

[ -x "$BIN" ] || { echo "FATAL: instrumented binary missing: $BIN" >&2; exit 1; }
[ -x "$TRAINER" ] || { echo "FATAL: trainer binary missing: $TRAINER (build trainer/ first)" >&2; exit 1; }

mkdir -p "$PGO_DIR"
rm -f "$PGO_DIR"/*.profraw "$PGO_DIR"/merged-*.profdata "$PGO_DIR"/train-manifest.json

# Shape list comes from the trainer itself — one source of truth; a drift
# between this loop and the trainer's table is impossible by construction.
mapfile -t SHAPES < <("$TRAINER" --list-shapes)
[ "${#SHAPES[@]}" -ge 12 ] || { echo "FATAL: trainer lists ${#SHAPES[@]} shapes, expected >= 12" >&2; exit 1; }

# ---- gateway training config -------------------------------------------------
# Modeled on the committed bench topology (bench/onthebench/run-baseline.sh):
# standalone resources_file mode, no etcd, one mock upstream, one api key from
# env. Default observability posture stays ON (the metrics recording path must
# train hot); only the prometheus listener address moves off :9090 so a
# training run can never collide with a gateway already running on the host.
cat > "$PGO_DIR/config.yaml" <<EOF
resources_file: "$PGO_DIR/resources.yaml"
proxy:
  addr: "127.0.0.1:$GW_PORT"
admin:
  enabled: false
observability:
  metrics:
    prometheus:
      addr: "127.0.0.1:$METRICS_PORT"
EOF
cat > "$PGO_DIR/resources.yaml" <<EOF
_format_version: "1"
provider_keys:
  - display_name: pgo-openai
    provider: openai
    adapter: openai
    api_base: "http://127.0.0.1:$MOCK_PORT/v1"
    api_key: "sk-pgo-mock"
  - display_name: pgo-anthropic
    provider: anthropic
    api_base: "http://127.0.0.1:$MOCK_PORT"
    api_key: "sk-pgo-mock"
models:
  - display_name: gpt-4o-mini
    provider: openai
    model_name: gpt-4o-mini
    provider_key: pgo-openai
  - display_name: gpt-4o-mini-b
    provider: openai
    model_name: gpt-4o-mini
    provider_key: pgo-openai
  - display_name: gpt-4o-mini-rl
    provider: openai
    model_name: gpt-4o-mini
    provider_key: pgo-openai
    rate_limit:
      rpm: 1000000
      tpm: 1000000000
  - display_name: gpt-router
    routing:
      targets:
        - model: gpt-4o-mini
        - model: gpt-4o-mini-b
  - display_name: claude-pgo
    provider: anthropic
    model_name: claude-pgo
    provider_key: pgo-anthropic
  - display_name: text-embedding-mock
    provider: openai
    model_name: text-embedding-mock
    provider_key: pgo-openai
api_keys:
  - display_name: pgo-trainer
    key_env: PGO_TRAIN_KEY
    allowed_models: ["*"]
EOF

# AISIX_* env vars are config overrides; nothing from the calling environment
# may leak into the trained process (same discipline as the bench harness).
while read -r v; do unset "$v"; done < <(compgen -v | grep '^AISIX_' || true)

# ---- one gateway lifetime per shape -------------------------------------------
for shape in "${SHAPES[@]}"; do
    echo "== shape: $shape ==" >&2
    PGO_TRAIN_KEY=pgo-token \
    LLVM_PROFILE_FILE="$PGO_DIR/aisix-$shape-%p.profraw" \
        "$BIN" --config "$PGO_DIR/config.yaml" > "$PGO_DIR/gateway-$shape.log" 2>&1 &
    GW_PID=$!

    if ! "$TRAINER" --mock-port "$MOCK_PORT" --gateway "127.0.0.1:$GW_PORT" \
        --api-key pgo-token --shape "$shape" --requests "$REQS" --concurrency "$CONC"; then
        echo "FATAL: training shape '$shape' failed; gateway log tail:" >&2
        tail -n 40 "$PGO_DIR/gateway-$shape.log" >&2 || true
        kill -KILL "$GW_PID" 2>/dev/null || true
        exit 1
    fi

    # Graceful shutdown is what flushes the profile counters to disk; a
    # non-zero exit means the flush cannot be trusted.
    kill -TERM "$GW_PID"
    rc=0; wait "$GW_PID" || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "FATAL: gateway exited rc=$rc after shape '$shape'; log tail:" >&2
        tail -n 40 "$PGO_DIR/gateway-$shape.log" >&2 || true
        exit 1
    fi

    raw_count=$(find "$PGO_DIR" -maxdepth 1 -name "aisix-$shape-*.profraw" | wc -l)
    [ "$raw_count" -ge 1 ] || { echo "FATAL: shape '$shape' left no .profraw" >&2; exit 1; }
    while read -r raw; do
        sz=$(stat -c%s "$raw")
        [ "$sz" -ge "$PROFRAW_FLOOR_BYTES" ] ||
            { echo "FATAL: $raw is ${sz}B (< ${PROFRAW_FLOOR_BYTES}B floor) - counters did not flush" >&2; exit 1; }
    done < <(find "$PGO_DIR" -maxdepth 1 -name "aisix-$shape-*.profraw")
done

TOTAL_RAW=$(find "$PGO_DIR" -maxdepth 1 -name '*.profraw' | wc -l)
[ "$TOTAL_RAW" -ge "${#SHAPES[@]}" ] ||
    { echo "FATAL: $TOTAL_RAW profraw files for ${#SHAPES[@]} shapes" >&2; exit 1; }

# ---- merge with the toolchain's own llvm-profdata ------------------------------
HOST_TUPLE=$(rustc -vV | sed -n 's/^host: //p')
LLVM_PROFDATA="${PGO_LLVM_PROFDATA:-$(rustc --print sysroot)/lib/rustlib/$HOST_TUPLE/bin/llvm-profdata}"
[ -x "$LLVM_PROFDATA" ] ||
    { echo "FATAL: llvm-profdata not found at $LLVM_PROFDATA (llvm-tools-preview component missing?)" >&2; exit 1; }

"$LLVM_PROFDATA" merge -o "$PGO_DIR/merged.profdata" "$PGO_DIR"/*.profraw

PROFDATA_SHA=$(sha256sum "$PGO_DIR/merged.profdata" | cut -d' ' -f1)
PROFDATA_BYTES=$(stat -c%s "$PGO_DIR/merged.profdata")
[ "$PROFDATA_BYTES" -ge "$PROFDATA_FLOOR_BYTES" ] ||
    { echo "FATAL: merged profile is ${PROFDATA_BYTES}B (< ${PROFDATA_FLOOR_BYTES}B floor)" >&2; exit 1; }
mv "$PGO_DIR/merged.profdata" "$PGO_DIR/merged-${PROFDATA_SHA:0:16}.profdata"

SHAPES_JSON=$(printf '"%s",' "${SHAPES[@]}")
cat > "$PGO_DIR/train-manifest.json" <<EOF
{
  "shapes": [${SHAPES_JSON%,}],
  "profraw_count": $TOTAL_RAW,
  "requests_per_shape": $REQS,
  "profdata_bytes": $PROFDATA_BYTES,
  "profdata_sha256": "$PROFDATA_SHA"
}
EOF

echo "== training done: $TOTAL_RAW profraws -> merged-${PROFDATA_SHA:0:16}.profdata (${PROFDATA_BYTES}B) ==" >&2
