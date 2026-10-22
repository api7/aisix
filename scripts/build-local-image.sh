#!/usr/bin/env bash
# Build a local gateway image of this checkout, through the same path CI uses:
# scripts/build-image-binary.sh compiles `aisix` on the host (thin-LTO
# `release-thin` profile, never PGO) into this worktree's own target/, and the
# repository Dockerfile packages it. Tagged `aisix:local-<sha7>`, the binary
# reporting `dev+sha-<sha7>`; uncommitted changes are built but not reflected
# in the tag.
#
# Usage: scripts/build-local-image.sh [image-tag]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null && pwd)"
SHA7="$(git -C "$ROOT" rev-parse --short=7 HEAD)"
TAG="${1:-aisix:local-$SHA7}"

AISIX_BUILD_SHA="$SHA7" AISIX_BUILD_VERSION="" "$ROOT/scripts/build-image-binary.sh" thin
docker build -t "$TAG" "$ROOT"
docker run --rm --entrypoint /usr/local/bin/aisix "$TAG" --version
echo "built $TAG"
