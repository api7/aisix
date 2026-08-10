#!/usr/bin/env bash
# Idempotent rig provisioning for the onthebench-style baseline harness.
#
# Installs everything a fresh m7g.4xlarge (Ubuntu 24.04, arm64) needs to build
# aisix and run bench/onthebench/run-baseline.sh. Safe to re-run; a rebuilt rig
# only needs this script to become a rig again.
#
# The load generator (otb) and the mock upstream are the PREBUILT, PINNED
# instruments from the public onthebench benchmark rig release — the same
# binaries every entrant on the public board is measured with, and the same
# "otb loadgen" used for the api7/aisix#891 and #902 tables. The engine pin
# behind the release tag is commit f3adbb1315b26129f5e317af5279decefb1cea8f
# (tag engine-v1) of https://github.com/GetBusbar/benchmarking; the sha256s
# below freeze the exact bytes so a re-provisioned rig either gets the
# identical instrument or fails loudly.
set -euo pipefail

TOOLS="$HOME/bench-tools"
REL="https://github.com/GetBusbar/benchmarking/releases/download/rig"
OTB_SHA256="913702e09392846f5eb82ca749e5fa2d86357e818c0bbb1d047913c1ded0f82e"
MOCK_SHA256="c32b9ff470eddf87d477098dac6e19a6c75eeef6e594dfc57005310d69e9049d"

echo "== apt packages (build deps + perf) =="
sudo -n DEBIAN_FRONTEND=noninteractive apt-get update -q
# Two transactions on purpose: apt is all-or-nothing, so bundling the
# kernel-versioned linux-tools package (absent for some AWS kernels) with the
# build deps would silently drop ALL of them and fail much later in the build.
sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
    git curl build-essential pkg-config libssl-dev protobuf-compiler \
    python3 linux-tools-common
sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y -q "linux-tools-$(uname -r)" ||
    sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y -q linux-tools-aws
perf --version

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
    local name="$1" sha="$2" path="$TOOLS/$1"
    if [ ! -x "$path" ] || ! echo "$sha  $path" | sha256sum -c --quiet 2>/dev/null; then
        curl -fsSL -o "$path" "$REL/${name}-arm64"
        chmod +x "$path"
    fi
    echo "$sha  $path" | sha256sum -c --quiet ||
        { echo "setup: $name does not match its pinned sha256 - refusing a divergent instrument"; exit 1; }
}
fetch_pinned otb "$OTB_SHA256"
fetch_pinned mock "$MOCK_SHA256"

echo "== perf sampling permission (session-scoped, documented in README) =="
sudo -n sysctl -q kernel.perf_event_paranoid=1

echo "setup: done"
