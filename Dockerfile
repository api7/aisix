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
# Run:
#   docker run --rm -v $(pwd)/config.example.yaml:/etc/aisix/config.yaml \
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

# Proxy + admin listeners from config.example.yaml.
EXPOSE 3000 3001

USER aisix

# tini forwards signals cleanly to the aisix process.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/aisix"]
CMD ["--config", "/etc/aisix/config.yaml"]
