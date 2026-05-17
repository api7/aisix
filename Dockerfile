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

COPY --from=builder /usr/local/bin/aisix /usr/local/bin/aisix

# Bake the managed-mode bootstrap config so aisix.cloud tenants can
# `docker run` without mounting anything — env vars carry the per-DP
# secret bits (registration token + CP base URL).
COPY config.managed.yaml /etc/aisix/config.managed.yaml

# Entrypoint script picks the config file via AISIX_CONFIG_PATH so the
# same image serves both standalone (mount your config at the default
# path) and managed (point AISIX_CONFIG_PATH at the baked file).
COPY docker/entrypoint.sh /usr/local/bin/aisix-entrypoint
RUN chmod 0755 /usr/local/bin/aisix-entrypoint

# Proxy + admin listeners from config.example.yaml.
EXPOSE 3000 3001

USER aisix

# tini forwards signals cleanly to the aisix process; entrypoint script
# resolves the config path from env, then execs the binary.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/aisix-entrypoint"]
