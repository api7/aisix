#!/bin/bash
# Fetch the aisix binary for the latest release out of the published GHCR
# image layers. Runs inside Glama's build sandbox, which has no Docker.
set -euo pipefail
VERSION=$(git -C /app describe --tags --abbrev=0 2>/dev/null | sed 's/^v//')
VERSION=${VERSION:-0.10.0}
echo "extracting aisix ${VERSION} from ghcr.io/api7/aisix"
TOKEN=$(curl -fsSL "https://ghcr.io/token?scope=repository:api7/aisix:pull&service=ghcr.io" | python -c 'import sys,json;print(json.load(sys.stdin)["token"])')
ACCEPT="application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json"
manifest() { curl -fsSL -H "Authorization: Bearer $TOKEN" -H "Accept: $ACCEPT" "https://ghcr.io/v2/api7/aisix/manifests/$1"; }

# Published tags are multi-arch indexes, and GHCR serves the index whatever
# the Accept header asks for — so resolve this sandbox's architecture to the
# single-platform manifest that actually carries the layers.
case "$(uname -m)" in
  x86_64) ARCH=amd64 ;;
  aarch64|arm64) ARCH=arm64 ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac
MANIFEST=$(manifest "${VERSION}")
SUB=$(printf '%s' "$MANIFEST" | ARCH="$ARCH" python -c 'import os,sys,json
m = json.load(sys.stdin)
if "manifests" not in m:
    sys.exit(0)  # already a single-platform manifest
want = os.environ["ARCH"]
hit = [e["digest"] for e in m["manifests"]
       if e.get("platform", {}).get("os") == "linux"
       and e.get("platform", {}).get("architecture") == want]
if not hit:
    sys.exit("no linux/%s manifest in the index for this tag" % want)
print(hit[0])')
if [ -n "$SUB" ]; then
  MANIFEST=$(manifest "$SUB")
fi
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
