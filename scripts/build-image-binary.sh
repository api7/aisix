#!/usr/bin/env bash
# Build the `aisix` binary the container image ships, on the host, for the
# host's own architecture, and stage it where the Dockerfile copies it from:
#
#   bin/aisix-<amd64|arm64>               the binary
#   bin/pgo-verified-<amd64|arm64>.json   PGO proof marker (pgo mode only)
#
# Usage: scripts/build-image-binary.sh <thin|release|pgo>
#
#   thin     `release-thin` profile (thin LTO) — main/:dev images, local images
#   release  `release` profile (fat LTO, codegen-units 1) — `-rc.N` tags
#   pgo      `release` profile, profile-guided (#967) — stable `vX.Y.Z` tags
#
# The version stamps are read from the environment and compiled in:
# AISIX_BUILD_SHA (short sha) and AISIX_BUILD_VERSION (release tags only); see
# crates/aisix-core/src/version.rs. The cargo target dir is always this
# checkout's own `target/`, never one shared with another worktree.
#
# pgo runs the three phases, FAIL-CLOSED — any phase failing fails the script
# and no binary is staged:
#   A. instrumented build (-Cprofile-generate) in target-pgo-gen/;
#   B. train.sh drives the committed shape matrix through it and merges the
#      .profraw files with the pinned toolchain's own llvm-profdata;
#   C. optimized build (-Cprofile-use) in target-pgo/, so profile builds never
#      share cargo fingerprints with each other or with plain builds. The merged profile
#      is content-addressed (merged-<sha>.profdata) because cargo fingerprints
#      the -Cprofile-use PATH, not the file content.
# The proof marker is written only after phase C succeeds; the docker-image
# workflow asserts it inside the pushed image. Training runs the instrumented
# binary, so a pgo build must run natively on the target architecture.
set -euo pipefail

MODE="${1:?usage: build-image-binary.sh <thin|release|pgo>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null && pwd)"
cd "$ROOT"

# The image is Linux, and the binary links this host's glibc, which must not
# be newer than the runtime image's (the Dockerfile's install step runs it).
[ "$(uname -s)" = Linux ] || { echo "build-image-binary.sh must run on a Linux host" >&2; exit 2; }
case "$(uname -m)" in
    x86_64) ARCH=amd64 ;;
    aarch64) ARCH=arm64 ;;
    *) echo "unsupported host architecture: $(uname -m)" >&2; exit 2 ;;
esac

# jemalloc bakes the build host's page size into the binary, so a 4K-page
# arm64 build aborts at startup on a 64K-page kernel; lg-page 16 runs on both
# (AISIX-Cloud#903, see crates/aisix-server/src/main.rs).
if [ "$ARCH" = arm64 ]; then export JEMALLOC_SYS_WITH_LG_PAGE=16; fi

export AISIX_BUILD_SHA="${AISIX_BUILD_SHA:-}"
export AISIX_BUILD_VERSION="${AISIX_BUILD_VERSION:-}"
export CARGO_TARGET_DIR="$ROOT/target"
# Set-but-empty, so a developer's cargo config `build.rustflags` (say
# -Ctarget-cpu=native) cannot reach an image; the PGO phases set their own.
export RUSTFLAGS=""
unset CARGO_BUILD_RUSTFLAGS CARGO_ENCODED_RUSTFLAGS

OUT="$ROOT/bin"
mkdir -p "$OUT"
rm -f "$OUT/aisix-$ARCH" "$OUT/pgo-verified-$ARCH.json"

# `--locked` fails fast on a stale Cargo.lock instead of resolving fresh deps.
case "$MODE" in
    thin)
        cargo build --locked --profile release-thin --bin aisix
        install -m 0755 target/release-thin/aisix "$OUT/aisix-$ARCH"
        ;;
    release)
        cargo build --locked --release --bin aisix
        install -m 0755 target/release/aisix "$OUT/aisix-$ARCH"
        ;;
    pgo)
        PGO_DIR="$ROOT/target-pgo-data"
        RUSTFLAGS="-Cprofile-generate=$PGO_DIR" CARGO_TARGET_DIR="$ROOT/target-pgo-gen" \
            cargo build --locked --release --bin aisix
        CARGO_TARGET_DIR="$ROOT/bench/pgo-training/trainer/target" cargo build --locked --release \
            --manifest-path bench/pgo-training/trainer/Cargo.toml
        bash bench/pgo-training/train.sh "$ROOT/target-pgo-gen/release/aisix" "$PGO_DIR"
        PROFDATA="$(ls "$PGO_DIR"/merged-*.profdata)"
        RUSTFLAGS="-Cprofile-use=$PROFDATA" CARGO_TARGET_DIR="$ROOT/target-pgo" \
            cargo build --locked --release --bin aisix
        install -m 0755 target-pgo/release/aisix "$OUT/aisix-$ARCH"
        install -m 0644 "$PGO_DIR/train-manifest.json" "$OUT/pgo-verified-$ARCH.json"
        ;;
    *)
        echo "unsupported mode: '$MODE' (use thin|release|pgo)" >&2
        exit 2
        ;;
esac

ls -l "$OUT"
