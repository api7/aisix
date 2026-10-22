# syntax=docker/dockerfile:1.7
#
# Runtime image for the AISIX gateway.
#
# The `aisix` binary is built OUTSIDE Docker, on the host, and this file only
# packages it: scripts/build-image-binary.sh stages `bin/aisix-<arch>` (plus
# `bin/pgo-verified-<arch>.json` on a PGO build), and the install step below
# picks the one matching TARGETARCH. The docker-image workflow builds each
# architecture natively on its own runner with a persistent cargo cache.
#
# Build a local image of the current checkout (thin-LTO profile, no PGO):
#   scripts/build-local-image.sh            # -> aisix:local-<sha7>
#
# Run, standalone (mount your own config):
#   docker run --rm -v $(pwd)/config.example.yaml:/etc/aisix/config.yaml \
#     aisix:local-<sha7>
#
# Run, managed (connected to AISIX Enterprise or AISIX Cloud with env-var overrides):
#   docker run --rm \
#     -e AISIX_CONFIG_PATH=/etc/aisix/config.managed.yaml \
#     -e AISIX_MANAGED__CP_BASE_URL \
#     -e AISIX_MANAGED__CP_ETCD_ENDPOINT \
#     -e AISIX_MANAGED__CP_CERT_PEM \
#     -e AISIX_MANAGED__CP_KEY_PEM \
#     -e AISIX_MANAGED__CP_CA_PEM \
#     -v aisix-mtls:/var/lib/aisix \
#     aisix:local-<sha7>
# The volume preserves the materialized mTLS bundle and gateway identity across
# container restarts.

# The binary links the build host's glibc, so this base must carry the same
# or a newer one: trixie (2.41) against the Ubuntu 24.04 runners (2.39). The
# install step below executes the binary, so a host whose glibc is newer than
# this image's fails the build instead of producing an image that cannot start.
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
# The PGO proof marker (#967) ships with the image: staged by the build
# script only after a successful profile-optimized build, asserted by the push
# workflows before an image is trusted. Absent on non-PGO builds.
ARG TARGETARCH
RUN --mount=type=bind,source=bin,target=/mnt/bin \
    apt-get update \
    && apt-get install -y --no-install-recommends libcap2-bin \
    && install -m 0755 "/mnt/bin/aisix-${TARGETARCH}" /usr/local/bin/aisix \
    && /usr/local/bin/aisix --version \
    && setcap 'cap_net_bind_service=+ep' /usr/local/bin/aisix \
    && mkdir -p /usr/local/share/aisix \
    && if [ -f "/mnt/bin/pgo-verified-${TARGETARCH}.json" ]; then \
         install -m 0644 "/mnt/bin/pgo-verified-${TARGETARCH}.json" /usr/local/share/aisix/pgo-verified.json; \
       fi \
    && apt-get purge -y --auto-remove libcap2-bin \
    && rm -rf /var/lib/apt/lists/*

# Bake the managed-mode bootstrap config so gateways connected to AISIX
# Enterprise or AISIX Cloud can `docker run` without mounting a configuration file.
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

# Numeric, not `aisix`: kubelet resolves `runAsNonRoot: true` against the
# image's configured user, and a name it cannot prove is non-root fails
# the container at admission with CreateContainerConfigError. The uid is
# the `aisix` passwd entry's, so the default runtime identity — and the
# ownership of /etc/aisix and /var/lib/aisix — is unchanged.
USER 10001

# tini forwards signals cleanly to the aisix process; entrypoint script
# resolves the config path from env, then execs the binary.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/aisix-entrypoint"]
