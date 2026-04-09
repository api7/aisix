#!/usr/bin/env bash
set -euo pipefail

# Smoke tests for the AISIX Docker image.
# Starts aisix + etcd via quickstart/docker-compose.yaml, runs 6 checks,
# and tears down on exit. Must be run from the repo root.
#
# Usage:
#   AISIX_IMAGE=aisix:smoke-test ./ci/smoke-test.sh
#
# When services are already running (e.g. started via quickstart), set:
#   SKIP_COMPOSE=1 ADMIN_KEY=<your-key> ./ci/smoke-test.sh

# --- Configuration ---
ADMIN_KEY="${ADMIN_KEY:-admin}"
ADMIN_URL="http://127.0.0.1:3001"
PROXY_URL="http://127.0.0.1:3000"
COMPOSE_FILE="quickstart/docker-compose.yaml"
SKIP_COMPOSE="${SKIP_COMPOSE:-0}"

# --- Helpers ---
info()  { printf '\033[1;34m[smoke]\033[0m %s\n' "$1"; }
ok()    { printf '\033[1;32m[smoke]\033[0m %s\n' "$1"; }
fail()  { printf '\033[1;31m[smoke]\033[0m %s\n' "$1" >&2; }

cleanup() {
    info "Cleaning up..."
    docker compose -f "$COMPOSE_FILE" down -v 2>/dev/null || true
}

assert_status() {
    local description="$1" expected="$2" actual="$3"
    if [ "$actual" -eq "$expected" ]; then
        ok "PASS: $description (HTTP $actual)"
    else
        fail "FAIL: $description — expected HTTP $expected, got HTTP $actual"
        if [ "$SKIP_COMPOSE" = "0" ]; then
            docker compose -f "$COMPOSE_FILE" logs aisix
        fi
        exit 1
    fi
}

# --- Start services (unless already running) ---
if [ "$SKIP_COMPOSE" = "0" ]; then
    trap cleanup EXIT
    info "Starting services with image: ${AISIX_IMAGE:-ghcr.io/api7/aisix:latest}"
    docker compose -f "$COMPOSE_FILE" up -d
else
    info "Skipping docker compose (SKIP_COMPOSE=1), using already-running services"
fi

# --- Wait for readiness ---
info "Waiting for AISIX to be ready (timeout 60s)..."
elapsed=0
while [ $elapsed -lt 60 ]; do
    if curl -sf "${ADMIN_URL}/openapi" >/dev/null 2>&1; then
        break
    fi
    elapsed=$((elapsed + 1))
    sleep 1
done

if [ $elapsed -ge 60 ]; then
    fail "AISIX did not become ready within 60 seconds"
    if [ "$SKIP_COMPOSE" = "0" ]; then
        docker compose -f "$COMPOSE_FILE" logs aisix
    fi
    exit 1
fi
ok "AISIX is ready (${elapsed}s)"

# --- Test 1: Admin server — OpenAPI endpoint ---
info "Test 1: GET /openapi"
status=$(curl -s -o /dev/null -w '%{http_code}' "${ADMIN_URL}/openapi")
assert_status "GET /openapi" 200 "$status"

# --- Test 2: Admin UI serves index.html ---
info "Test 2: GET /ui/"
status=$(curl -s -o /dev/null -w '%{http_code}' "${ADMIN_URL}/ui/")
assert_status "GET /ui/" 200 "$status"

# --- Test 3: Create a model via Admin API ---
info "Test 3: POST /aisix/admin/models (create model)"
status=$(curl -s -o /dev/null -w '%{http_code}' \
    -X POST "${ADMIN_URL}/aisix/admin/models" \
    -H "Content-Type: application/json" \
    -H "X-API-KEY: ${ADMIN_KEY}" \
    -d '{"name":"smoke-test-model","model":"openai/smoke-test-upstream","provider_config":{"api_key":"unused-smoke-key"}}')
assert_status "POST /aisix/admin/models" 201 "$status"

# --- Test 4: List models — verify model exists ---
info "Test 4: GET /aisix/admin/models (list models)"
response=$(curl -s -w '\n%{http_code}' \
    "${ADMIN_URL}/aisix/admin/models" \
    -H "X-API-KEY: ${ADMIN_KEY}")
status=$(echo "$response" | tail -1)
body=$(echo "$response" | sed '$d')
assert_status "GET /aisix/admin/models" 200 "$status"
if echo "$body" | grep -q "smoke-test-model"; then
    ok "PASS: Model 'smoke-test-model' found in list"
else
    fail "FAIL: Model 'smoke-test-model' not found in response"
    if [ "$SKIP_COMPOSE" = "0" ]; then
        docker compose -f "$COMPOSE_FILE" logs aisix
    fi
    exit 1
fi

# --- Test 5: Create an API key via Admin API ---
info "Test 5: POST /aisix/admin/apikeys (create API key)"
status=$(curl -s -o /dev/null -w '%{http_code}' \
    -X POST "${ADMIN_URL}/aisix/admin/apikeys" \
    -H "Content-Type: application/json" \
    -H "X-API-KEY: ${ADMIN_KEY}" \
    -d '{"key":"sk-smoke-test","allowed_models":["smoke-test-model"]}')
assert_status "POST /aisix/admin/apikeys" 201 "$status"

# --- Test 6: Proxy port responds with API key ---
info "Test 6: GET /v1/models via proxy (port 3000)"
status=$(curl -s -o /dev/null -w '%{http_code}' \
    "${PROXY_URL}/v1/models" \
    -H "Authorization: Bearer sk-smoke-test")
assert_status "GET /v1/models (proxy)" 200 "$status"

# --- All passed ---
echo
ok "All smoke tests passed!"
