#!/bin/sh
# Container entrypoint for aisix.
#
# Picks the config file based on AISIX_CONFIG_PATH (default
# /etc/aisix/config.yaml). Two intended modes:
#
#   Standalone — operator mounts their own config:
#       docker run -v ./config.yaml:/etc/aisix/config.yaml ghcr.io/moonming/ai-gateway:main
#
#   Managed (aisix.cloud tenant) — use the baked-in template + env vars:
#       docker run \
#         -e AISIX_CONFIG_PATH=/etc/aisix/config.managed.yaml \
#         -e AISIX_MANAGED__REGISTRATION_TOKEN=$DEPLOYMENT_TOKEN \
#         -e AISIX_MANAGED__CP_BASE_URL=https://api.us.aisix.cloud \
#         ghcr.io/moonming/ai-gateway:main
#
# The Rust binary's `Config::load_from_path` already layers
# `AISIX_<UPPER>__<UPPER>` env vars on top of the YAML, so any field
# is reachable without re-templating the file.

set -eu

CONFIG_PATH="${AISIX_CONFIG_PATH:-/etc/aisix/config.yaml}"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "aisix-entrypoint: config file not found at $CONFIG_PATH" >&2
    echo "aisix-entrypoint: mount one at /etc/aisix/config.yaml or set" >&2
    echo "aisix-entrypoint: AISIX_CONFIG_PATH=/etc/aisix/config.managed.yaml" >&2
    exit 64
fi

# AISIX_CONFIG_PATH is ours, not the binary's. The Rust Config
# loader scrapes every AISIX_* env var as a deserialisation override,
# so leaving CONFIG_PATH in the environment makes it try to
# deserialise a field called "config_path" (which doesn't exist on
# the root Config struct) and fail with "unknown field `config_path`".
# Unset before exec so the binary never sees it.
unset AISIX_CONFIG_PATH

exec /usr/local/bin/aisix --config "$CONFIG_PATH"
