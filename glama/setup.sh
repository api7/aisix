#!/bin/bash
# Glama build step: fetch the release binary and the stdio<->HTTP bridge.
set -euo pipefail
bash /app/glama/extract-aisix.sh
npm install -g mcp-remote
