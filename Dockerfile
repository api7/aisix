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
#
# Run, standalone (mount your own config):
#   docker run --rm -v $(pwd)/config.example.yaml:/etc/aisix/config.yaml \
#     aisix:dev
#
# Run, managed (aisix.cloud tenant — bake config + env-var overrides):
#   docker run --rm \
#     -e AISIX_CONFIG_PATH=/etc/aisix/config.managed.yaml \
#     -e AISIX_MANAGED__REGISTRATION_TOKEN=$DEPLOYMENT_TOKEN \
#     -e AISIX_MANAGED__CP_BASE_URL=https://api.us.aisix.cloud \
#     -v aisix-mtls:/var/lib/aisix \
#     aisix:dev

# --- Stage 1: build ----------------------------------------------------------
FROM rust:1.93-bookworm AS builder

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

# `--locked` forces the build to use the exact versions in Cargo.lock —
# fails fast if the lockfile is stale rather than silently resolving
# fresh deps in CI.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --locked --release --bin aisix \
    && cp target/release/aisix /usr/local/bin/aisix

# --- Stage 2: runtime --------------------------------------------------------
FROM debian:bookworm-slim AS runtime

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
RUN --mount=type=bind,from=builder,source=/usr/local/bin/aisix,target=/mnt/aisix \
    apt-get update \
    && apt-get install -y --no-install-recommends libcap2-bin \
    && install -m 0755 /mnt/aisix /usr/local/bin/aisix \
    && setcap 'cap_net_bind_service=+ep' /usr/local/bin/aisix \
    && apt-get purge -y --auto-remove libcap2-bin \
    && rm -rf /var/lib/apt/lists/*

# Bake the managed-mode bootstrap config so aisix.cloud tenants can
# `docker run` without mounting anything — env vars carry the per-DP
# secret bits (registration token + CP base URL).
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
