#!/usr/bin/env bash
# Idempotent rig provisioning for the onthebench-style baseline harness.
#
# Installs everything a fresh m7g.4xlarge (Ubuntu 24.04, arm64) needs to build
# aisix and run bench/onthebench/run-baseline.sh. Safe to re-run; a rebuilt rig
# only needs this script to become a rig again.
#
# The load generator (otb) and the mock upstream are the PREBUILT, PINNED
# instruments from the public onthebench benchmark rig release — the same
# binaries every entrant on the public board is measured with. That release
# ("rig") is rolling: upstream's bench-rig workflow replaces its assets on
# every mock/ or loadgen/ change. So both are fetched by release ASSET ID,
# which names one upload and never another — a replacement gets a new id and
# the old one 404s — and then checked against the sha256 of that upload. The
# pinned assets were built from commit 2d209e76ba336c3478d3754e2bcd245b663459a3
# of https://github.com/GetBusbar/benchmarking (2026-09-21); the #891 / #902
# tables were measured with the earlier f3adbb13 build (tag engine-v1), so
# numbers from before and after this pin are not directly comparable.
set -euo pipefail

TOOLS="$HOME/bench-tools"
ASSETS="https://api.github.com/repos/GetBusbar/benchmarking/releases/assets"
OTB_ASSET=579682244
OTB_SHA256="79a31656f5ececb82954d3dc1cc86c6fcbd3c1f46ee6fcac4a348114ca03e2a7"
MOCK_ASSET=579682245
MOCK_SHA256="97e76ce45fbccb87a7f123a538b107c5566331da3b260cdce918d8671f2e3d7b"

echo "== apt packages (build deps + perf) =="
sudo -n DEBIAN_FRONTEND=noninteractive apt-get update -q
# Two transactions on purpose: apt is all-or-nothing, so bundling the
# kernel-versioned linux-tools package (absent for some AWS kernels) with the
# build deps would silently drop ALL of them and fail much later in the build.
sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
    git curl build-essential pkg-config libssl-dev protobuf-compiler \
    python3 linux-tools-common iproute2 procps
sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y -q "linux-tools-$(uname -r)" ||
    sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y -q linux-tools-aws
perf --version
# The harness asserts that the process it measures is the one it started, and
# that assertion reads ss and ps; without them the check degrades to a warning
# nobody reads and the guard is silently off. Verify like perf above.
command -v ss >/dev/null || { echo "FATAL: iproute2 (ss) missing"; exit 1; }
command -v ps >/dev/null || { echo "FATAL: procps (ps) missing"; exit 1; }

echo "== rust toolchain =="
# Pinned, checksum-verified rustup-init instead of the curl|sh installer: the
# same rule as the bench instruments — every executed third-party artifact is
# a fixed byte sequence or the setup fails loudly.
RUSTUP_VERSION="1.28.2"
RUSTUP_SHA256="e3853c5a252fca15252d07cb23a1bdd9377a8c6f3efa01531109281ae47f841c"
if ! command -v "$HOME/.cargo/bin/cargo" >/dev/null 2>&1; then
    tmp=$(mktemp -d)
    curl -fsSL -o "$tmp/rustup-init" \
        "https://static.rust-lang.org/rustup/archive/$RUSTUP_VERSION/aarch64-unknown-linux-gnu/rustup-init"
    echo "$RUSTUP_SHA256  $tmp/rustup-init" | sha256sum -c --quiet ||
        { echo "setup: rustup-init does not match its pinned sha256"; exit 1; }
    chmod +x "$tmp/rustup-init"
    "$tmp/rustup-init" -y --default-toolchain none
    rm -rf "$tmp"
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
# No default toolchain is configured; every cargo invocation below runs inside
# the source tree so the repo's rust-toolchain.toml pins the version. This also
# front-loads the toolchain download out of the build step.
SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
(cd "$SRC_ROOT" && cargo --version)

echo "== inferno (flamegraph rendering: perf script -> SVG) =="
# Version-pinned like every other instrument: --locked alone pins inferno's
# dependencies, not inferno itself, and two rigs rendering with different
# inferno versions would produce non-comparable artifacts.
INFERNO_VERSION="0.12.8"
inferno-flamegraph --version 2>/dev/null | grep -qF "$INFERNO_VERSION" ||
    (cd "$SRC_ROOT" && cargo install --locked --force --version "$INFERNO_VERSION" inferno)

echo "== pinned bench instruments (otb loadgen + mock upstream) =="
mkdir -p "$TOOLS"
fetch_pinned() {
    local name="$1" asset="$2" sha="$3" path="$TOOLS/$1"
    if [ ! -x "$path" ] || ! echo "$sha  $path" | sha256sum -c --quiet 2>/dev/null; then
        curl -fsSL -H 'Accept: application/octet-stream' -o "$path" "$ASSETS/$asset" ||
            { echo "setup: $name asset $asset is gone upstream - re-pin (see the header)"; exit 1; }
        chmod +x "$path"
    fi
    echo "$sha  $path" | sha256sum -c --quiet ||
        { echo "setup: $name does not match its pinned sha256 - refusing a divergent instrument"; exit 1; }
}
fetch_pinned otb "$OTB_ASSET" "$OTB_SHA256"
fetch_pinned mock "$MOCK_ASSET" "$MOCK_SHA256"

echo "== perf sampling permission (session-scoped, documented in README) =="
sudo -n sysctl -q kernel.perf_event_paranoid=1

echo "setup: done"
