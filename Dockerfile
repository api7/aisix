# Stage 1: Build UI
FROM node:20-slim AS ui-builder
RUN corepack enable
WORKDIR /app
COPY ui/package.json ui/pnpm-lock.yaml ui/
RUN cd ui && pnpm install --frozen-lockfile
COPY ui/ ui/
RUN cd ui && pnpm build

# Stage 2: Build Rust binary
FROM rust:bookworm AS builder
RUN apt-get update && apt-get install -y protobuf-compiler pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY build.rs ./
COPY --from=ui-builder /app/ui/dist ui/dist
RUN cargo build --release && strip target/release/ai-gateway

# Stage 3: Runtime (Google Distroless — minimal CVE surface)
# cc-debian12 already includes OpenSSL, CA certificates, and tzdata.
FROM gcr.io/distroless/cc-debian12:nonroot

COPY --from=builder /app/target/release/ai-gateway /usr/local/bin/aisix
COPY config.yaml /etc/aisix/config.yaml

EXPOSE 3000 3001
WORKDIR /etc/aisix
ENTRYPOINT ["aisix"]
