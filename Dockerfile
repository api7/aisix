# Multi-stage build for the aisix AI gateway.
#
# The workspace pins rustc via rust-toolchain.toml (currently 1.93.1).
# We use the latest Debian-based official Rust image, then copy the
# single `aisix` binary into a slim runtime image.
#
# Build:
#   docker build -t aisix:dev .
#
# Run:
#   docker run --rm -v $(pwd)/config.example.yaml:/etc/aisix/config.yaml \
#     -e AISIX_CONFIG=/etc/aisix/config.yaml aisix:dev

# --- Stage 1: build ----------------------------------------------------------
FROM rust:1.93-bookworm AS builder

# protoc is required by dependencies that use prost/tonic-build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Leverage Docker layer cache: copy manifests first so dependency
# compilation only re-runs when Cargo.toml / Cargo.lock change.
COPY Cargo.toml Cargo.lock rust-toolchain.toml rustfmt.toml ./
COPY crates ./crates

# The release build is what we ship.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --bin aisix \
    && cp target/release/aisix /usr/local/bin/aisix

# --- Stage 2: runtime --------------------------------------------------------
FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin aisix

COPY --from=builder /usr/local/bin/aisix /usr/local/bin/aisix

# Proxy + admin listeners from config.example.yaml.
EXPOSE 3000 3001

USER aisix

# tini forwards signals cleanly to the aisix process.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/aisix"]
CMD ["--config", "/etc/aisix/config.yaml"]
