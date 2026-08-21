#!/bin/bash
set -e
python /app/glama/echo-mcp.py &
export ANON_PRINCIPAL_KEY="${ANON_PRINCIPAL_KEY:-glama-anon-principal-not-a-secret}"
/app/glama/aisix --config /app/glama/config.yaml &
for i in $(seq 1 50); do
  curl -fsS -o /dev/null http://127.0.0.1:3000/livez 2>/dev/null && break
  sleep 0.2
done
exec mcp-remote http://127.0.0.1:3000/mcp --transport http-only
