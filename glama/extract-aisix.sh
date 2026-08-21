#!/bin/bash
# Fetch the aisix binary for the latest release out of the published GHCR
# image layers. Runs inside Glama's build sandbox, which has no Docker.
set -euo pipefail
VERSION=$(git -C /app describe --tags --abbrev=0 2>/dev/null | sed 's/^v//')
VERSION=${VERSION:-0.10.0}
echo "extracting aisix ${VERSION} from ghcr.io/api7/aisix"
TOKEN=$(curl -fsSL "https://ghcr.io/token?scope=repository:api7/aisix:pull&service=ghcr.io" | python -c 'import sys,json;print(json.load(sys.stdin)["token"])')
MANIFEST=$(curl -fsSL -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json" "https://ghcr.io/v2/api7/aisix/manifests/${VERSION}")
DIGESTS=$(printf '%s' "$MANIFEST" | python -c 'import sys,json;[print(l["digest"]) for l in reversed(json.load(sys.stdin)["layers"])]')
for D in $DIGESTS; do
  curl -fsSL -H "Authorization: Bearer $TOKEN" "https://ghcr.io/v2/api7/aisix/blobs/$D" -o /tmp/layer.tgz
  ENTRY=$(tar -tzf /tmp/layer.tgz 2>/dev/null | grep -E '(^|/)usr/local/bin/aisix$' | head -1 || true)
  if [ -n "$ENTRY" ]; then
    mkdir -p /tmp/x && tar -xzf /tmp/layer.tgz -C /tmp/x "$ENTRY"
    mv "/tmp/x/$ENTRY" /app/glama/aisix && chmod +x /app/glama/aisix
    rm -rf /tmp/layer.tgz /tmp/x
    /app/glama/aisix --version
    exit 0
  fi
  rm -f /tmp/layer.tgz
done
echo "aisix binary not found in any layer of ${VERSION}" >&2
exit 1
