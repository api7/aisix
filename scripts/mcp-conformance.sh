#!/usr/bin/env bash
# Official MCP conformance suite against the shipped /mcp gateway chain:
# conformance client → scoped gateway → EphemeralBridge → in-process
# upstream (see crates/aisix-mcp/examples/conformance_server.rs).
#
# Runs every scenario applicable to the tools-only surface. The scenarios
# NOT in the list are deliberate surface decisions, documented in the
# example's module docs: prompts/resources content scenarios (capabilities
# this gateway does not advertise), server-initiated relay scenarios
# (sampling/elicitation/progress/logging), and the Host-allowlist half of
# dns-rebinding-protection (the endpoint is API-key-gated at the proxy
# layer). Growing this list is expected as the gateway's MCP surface grows.
#
# The package version is pinned: the suite is young and its scenario set
# moves; bumps should be deliberate, with the scenario list re-reviewed.
set -euo pipefail

CONFORMANCE_PKG="@modelcontextprotocol/conformance@0.1.16"
ADDR="${1:-127.0.0.1:3111}"
BIN="${CONFORMANCE_TARGET:-./target/debug/examples/conformance_server}"

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
  server-sse-multiple-streams
  resources-list
  prompts-list
)

"$BIN" "$ADDR" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

# Wait until the endpoint answers (ping needs no handshake on any generation).
ready=0
for _ in $(seq 1 100); do
  if curl -sf -o /dev/null -X POST "http://$ADDR/mcp" \
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
  # The CLI's exit code is not a reliable pass signal across versions;
  # the per-run "N failed" line is. `|| true` keeps set -e out of it.
  out=$(npx -y "$CONFORMANCE_PKG" server --url "http://$ADDR/mcp" --scenario "$scenario" 2>&1) || true
  echo "$out" | sed -n '/Test Results:/,+2p'
  if ! echo "$out" | grep -q ", 0 failed"; then
    echo "::error::conformance scenario '$scenario' failed"
    echo "$out" | tail -30
    failed=1
  fi
done

exit "$failed"
