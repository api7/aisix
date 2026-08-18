#!/usr/bin/env bash
# Official MCP conformance suite against the shipped /mcp gateway chain:
# conformance client → scoped gateway → EphemeralBridge → in-process
# upstream (see crates/aisix-mcp/examples/conformance_server.rs).
#
# Runs every scenario applicable to the tools-only surface. The scenarios
# NOT in the list are deliberate surface decisions, documented in the
# example's module docs: prompts/resources content scenarios (capabilities
# this gateway does not advertise), server-initiated relay scenarios
# (sampling/elicitation/progress/logging), the Host-allowlist half of
# dns-rebinding-protection (the endpoint is API-key-gated at the proxy
# layer), and session-dependent scenarios (this gateway is stateless by
# design). Growing this list is expected as the gateway's MCP surface
# grows.
#
# The suite AND its full transitive dependency graph are pinned by
# tools/mcp-conformance/package-lock.json (`npm ci`); a floating install
# would let protocol behavior drift with the suite's own SDK dependency.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS_DIR="$REPO_ROOT/tools/mcp-conformance"
ADDR="${1:-127.0.0.1:3111}"
BIN="${CONFORMANCE_TARGET:-$REPO_ROOT/target/debug/examples/conformance_server}"

SCENARIOS=(
  server-initialize
  ping
  completion-complete
  tools-list
  tools-call-simple-text
  tools-call-image
  tools-call-audio
  tools-call-embedded-resource
  tools-call-mixed-content
  tools-call-error
  # server-sse-multiple-streams is deliberately ABSENT: it requires a
  # server-minted session id, and this gateway is stateless by design —
  # the scenario runs zero checks (a 0/0 WARNING) and would gate nothing.
  resources-list
  prompts-list
)

npm ci --prefix "$HARNESS_DIR" --ignore-scripts --no-audit --no-fund >/dev/null
CONFORMANCE_BIN="$HARNESS_DIR/node_modules/.bin/conformance"
[ -x "$CONFORMANCE_BIN" ] || { echo "::error::conformance binary missing after npm ci"; exit 1; }

# Per-scenario wall-clock bound. GNU coreutils `timeout` on the CI
# runners; perl's alarm as the portable fallback (macOS dev machines) so
# a hung scenario is bounded on every platform.
if command -v timeout >/dev/null; then
  BOUND=(timeout 120)
else
  BOUND=(perl -e 'alarm shift; exec @ARGV' 120)
fi

"$BIN" "$ADDR" &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait until the endpoint answers (ping needs no handshake on any generation).
ready=0
for _ in $(seq 1 100); do
  if curl -sf --max-time 2 -o /dev/null -X POST "http://$ADDR/mcp" \
    -H 'content-type: application/json' \
    -H 'accept: application/json, text/event-stream' \
    -d '{"jsonrpc":"2.0","id":1,"method":"ping"}'; then
    ready=1
    break
  fi
  sleep 0.2
done
if [ "$ready" != 1 ]; then
  echo "::error::conformance target never became ready on $ADDR"
  exit 1
fi

failed=0
for scenario in "${SCENARIOS[@]}"; do
  echo "=== scenario: $scenario"
  rc=0
  # `${BOUND[@]+...}`: expands to nothing when the array is empty — plain
  # `"${BOUND[@]}"` trips `set -u` on bash 3.2 (macOS dev machines).
  out=$(${BOUND[@]+"${BOUND[@]}"} "$CONFORMANCE_BIN" server --url "http://$ADDR/mcp" --scenario "$scenario" 2>&1) || rc=$?
  echo "$out" | grep -A2 'Test Results:' || true
  # Pass = the CLI exited 0 AND at least one check ran AND none failed
  # AND no warnings. Each leg matters: a nonzero exit with a clean-looking
  # summary, a 0/0 not-applicable run, or a warning-only pass must not
  # count as coverage (the pinned suite is deterministic, so a new warning
  # is a real behavior change worth a red build).
  if [ "$rc" != 0 ] || ! echo "$out" | grep -qE 'Passed: [0-9]+/[1-9][0-9]*, 0 failed, 0 warnings'; then
    echo "::error::conformance scenario '$scenario' failed (exit $rc)"
    echo "$out" | tail -30
    failed=1
  fi
done

exit "$failed"
