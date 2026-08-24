# syntax=docker/dockerfile:1.7
#
# Multi-stage build for the aisix AI gateway.
#
# The workspace pins rustc via rust-toolchain.toml (currently 1.93.1).
# We use the latest Debian-based official Rust image, then copy the
# single `aisix` binary into a slim runtime image.
#
# BuildKit is required (the `--mount=type=cache` directives rely on
# it). On recent Docker Desktop / Docker CE, BuildKit is the default;
# on older clients run:  DOCKER_BUILDKIT=1 docker build -t aisix:dev .
#
# Build:
#   docker build -t aisix:dev .
#   docker build --build-arg PGO=off -t aisix:dev .   # quick local build, skips PGO
#
# Run, standalone (mount your own config):
#   docker run --rm -v $(pwd)/config.example.yaml:/etc/aisix/config.yaml \
#     aisix:dev
#
# Run, managed (connected to AISIX Cloud with env-var overrides):
#   docker run --rm \
#     -e AISIX_CONFIG_PATH=/etc/aisix/config.managed.yaml \
#     -e AISIX_MANAGED__CP_BASE_URL \
#     -e AISIX_MANAGED__CP_ETCD_ENDPOINT \
#     -e AISIX_MANAGED__CP_CERT_PEM \
#     -e AISIX_MANAGED__CP_KEY_PEM \
#     -e AISIX_MANAGED__CP_CA_PEM \
#     -v aisix-mtls:/var/lib/aisix \
#     aisix:dev
# The volume preserves the materialized mTLS bundle and gateway identity across
# container restarts.

# --- Stage 1: build ----------------------------------------------------------
# Trixie fixes the glibc the release binary links against, and the
# runtime stage must stay on the same-or-newer glibc — so the two base
# images only ever move together.
FROM rust:1.93-trixie AS builder

# protoc is required by dependencies that use prost/tonic-build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Short git sha stamped into the binary (heartbeat `version` =
# `<version>+sha-<BUILD_SHA>`) so a running DP can be matched to
# its image tag. CI passes the same short sha that tags the image;
# plain `docker build` (no arg) produces a binary that reports the bare
# crate version.
ARG BUILD_SHA=
ENV AISIX_BUILD_SHA=$BUILD_SHA

# Release version stamped into the binary. CI derives this from the
# release tag (v0.4.0 → 0.4.0), so `aisix --version`, the `Server`
# response header, and the heartbeat dp_version always self-report the
# tagged version — no manual Cargo.toml bump at release time. Empty
# (non-release / local build) falls back to the workspace crate version.
ARG BUILD_VERSION=
ENV AISIX_BUILD_VERSION=$BUILD_VERSION

# BuildKit cache mounts carry `~/.cargo/registry` + `target/` across
# builds, so changes to source files still reuse compiled dependencies.
# We could split dep-build from source-build via a manifests-only warm
# stage, but the cache mounts give us ~95% of the same win with half
# the Dockerfile complexity. Source copy is a single layer.
COPY Cargo.toml Cargo.lock rust-toolchain.toml rustfmt.toml ./
COPY crates ./crates
# `crates/aisix-admin/src/openapi.rs` uses `include_str!` to embed
# every `schemas/resources/*.schema.json` at compile time, so the
# Docker context must carry this directory or the release build fails.
COPY schemas ./schemas

# PGO training assets (#967): trainer tool + train.sh. Copied separately from
# crates/ so editing training assets doesn't invalidate the dependency layers
# above.
COPY bench/pgo-training ./bench/pgo-training

# Profile-guided optimization gate. Default ON: release artifacts are always
# PGO-built, and a forgotten build-arg ships a PGO'd image — never a silently
# un-optimized one. CI passes PGO=off only for pull-request smoke builds.
ARG PGO=on

# `--locked` forces the build to use the exact versions in Cargo.lock —
# fails fast if the lockfile is stale rather than silently resolving
# fresh deps in CI.
#
# PGO=on runs the three-phase build (#967):
#   A. instrumented build (-Cprofile-generate) in its own target dir;
#   B. train.sh drives the committed 12-shape matrix through the
#      instrumented gateway against the trainer's local mock, then merges
#      the .profraw files with the pinned toolchain's own llvm-profdata
#      (llvm-tools-preview — exact LLVM match with rustc, no extra deps);
#   C. optimized build (-Cprofile-use) in a third target dir, so profile
#      builds never share cargo fingerprints with plain builds.
# FAIL-CLOSED: any phase failing fails this RUN and nothing is shipped.
# The proof marker (pgo-verified.json) is written only after phase C
# succeeds; the push workflows assert it before trusting the image.
# The merged profile is content-addressed (merged-<sha>.profdata) because
# cargo fingerprints the -Cprofile-use PATH, not the file content — a
# retrained profile at a fixed path would silently reuse stale artifacts
# from the persistent target cache mount.
#
# If this ever builds for linux/arm64: jemalloc bakes the build host's
# page size into the binary, and QEMU reports 4K — set
# JEMALLOC_SYS_WITH_LG_PAGE=16 here or the image aborts at startup on
# 64K-page kernels (see crates/aisix-server/src/main.rs). PGO training
# additionally requires a native arm64 builder: an instrumented binary
# cannot self-train under QEMU emulation.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    --mount=type=cache,target=/src/target-pgo-gen \
    --mount=type=cache,target=/src/target-pgo \
    set -eu; \
    mkdir -p /usr/local/share/aisix; \
    if [ "$PGO" = "on" ]; then \
        RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" CARGO_TARGET_DIR=/src/target-pgo-gen \
            cargo build --locked --release --bin aisix; \
        cargo build --locked --release \
            --manifest-path bench/pgo-training/trainer/Cargo.toml; \
        bash bench/pgo-training/train.sh /src/target-pgo-gen/release/aisix /tmp/pgo-data; \
        PROFDATA="$(ls /tmp/pgo-data/merged-*.profdata)"; \
        RUSTFLAGS="-Cprofile-use=$PROFDATA" CARGO_TARGET_DIR=/src/target-pgo \
            cargo build --locked --release --bin aisix; \
        cp /src/target-pgo/release/aisix /usr/local/bin/aisix; \
        cp /tmp/pgo-data/train-manifest.json /usr/local/share/aisix/pgo-verified.json; \
    elif [ "$PGO" = "off" ]; then \
        cargo build --locked --release --bin aisix; \
        cp target/release/aisix /usr/local/bin/aisix; \
    else \
        echo "unsupported PGO value: '$PGO' (use on|off)" >&2; \
        exit 2; \
    fi

# --- Stage 2: runtime --------------------------------------------------------
# trixie-slim to match the builder's glibc (see the builder stage note);
# a binary linked against glibc 2.41 symbols cannot run on bookworm.
FROM debian:trixie-slim AS runtime

# Ownership-verification label for the MCP Registry: when this image is
# published as an MCP server entry, the registry requires this label to
# match the server name in server.json.
LABEL io.modelcontextprotocol.server.name="io.github.api7/aisix"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin aisix \
    && mkdir -p /etc/aisix/tls /var/lib/aisix \
    && chown -R aisix:aisix /etc/aisix /var/lib/aisix

# Install the binary and grant CAP_NET_BIND_SERVICE as a file
# capability so the non-root user can bind privileged ports (e.g.
# listening on :80/:443 with Kubernetes hostNetwork). Install + setcap
# happen in one RUN (bind-mount, no COPY) so the binary isn't
# duplicated across layers by the xattr change. Caveat: with the
# effective bit set, exec fails outright if NET_BIND_SERVICE is
# missing from the container's bounding set — it is in the default
# Docker/containerd cap set, but `capabilities: {drop: [ALL]}` pod
# specs must add NET_BIND_SERVICE back.
# The PGO proof marker (#967) ships with the image: written by the builder
# only after a successful profile-optimized build, asserted by the push
# workflows before an image is trusted. Absent on PGO=off (PR smoke) builds.
RUN --mount=type=bind,from=builder,source=/usr/local/bin/aisix,target=/mnt/aisix \
    --mount=type=bind,from=builder,source=/usr/local/share/aisix,target=/mnt/aisix-share \
    apt-get update \
    && apt-get install -y --no-install-recommends libcap2-bin \
    && install -m 0755 /mnt/aisix /usr/local/bin/aisix \
    && setcap 'cap_net_bind_service=+ep' /usr/local/bin/aisix \
    && mkdir -p /usr/local/share/aisix \
    && if [ -f /mnt/aisix-share/pgo-verified.json ]; then \
         install -m 0644 /mnt/aisix-share/pgo-verified.json /usr/local/share/aisix/pgo-verified.json; \
       fi \
    && apt-get purge -y --auto-remove libcap2-bin \
    && rm -rf /var/lib/apt/lists/*

# Bake the managed-mode bootstrap config so AISIX gateways managed by
# AISIX Cloud can `docker run` without mounting a configuration file.
# Environment variables provide the control-plane endpoints and gateway mTLS
# certificate bundle.
COPY config.managed.yaml /etc/aisix/config.managed.yaml

# Entrypoint script picks the config file via AISIX_CONFIG_PATH so the
# same image serves both standalone (mount your config at the default
# path) and managed (point AISIX_CONFIG_PATH at the baked file).
COPY docker/entrypoint.sh /usr/local/bin/aisix-entrypoint
RUN chmod 0755 /usr/local/bin/aisix-entrypoint

# Proxy + admin + metrics listeners from config.example.yaml.
EXPOSE 3000 3001 9090

USER aisix

# tini forwards signals cleanly to the aisix process; entrypoint script
# resolves the config path from env, then execs the binary.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/aisix-entrypoint"]
