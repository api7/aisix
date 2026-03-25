#!/usr/bin/env sh
set -e

REPO="api7/aisix"
BASE_URL="https://raw.githubusercontent.com/${REPO}/${AISIX_REF:-main}/quickstart"

# --- Colors ---
info()  { printf '\033[1;34m[aisix]\033[0m %s\n' "$1"; }
ok()    { printf '\033[1;32m[aisix]\033[0m %s\n' "$1"; }
err()   { printf '\033[1;31m[aisix]\033[0m %s\n' "$1" >&2; }

# --- Preflight checks ---
check_commands() {
    if ! command -v curl >/dev/null 2>&1; then
        err "curl is not installed. Please install curl and try again."
        exit 1
    fi
    if ! command -v docker >/dev/null 2>&1; then
        err "Docker is not installed. Please install it from https://docs.docker.com/get-docker/"
        exit 1
    fi
    if ! docker compose version >/dev/null 2>&1; then
        err "Docker Compose V2 plugin is required. Please install Docker Compose V2."
        exit 1
    fi
    if ! docker info >/dev/null 2>&1; then
        err "Docker daemon is not running. Please start Docker and try again."
        exit 1
    fi
}

# --- Portable sed -i ---
sedi() {
    if sed --version >/dev/null 2>&1; then
        sed -i "$@"
    else
        sed -i '' "$@"
    fi
}

# --- Generate random hex string ---
rand_hex() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 16
    elif [ -c /dev/urandom ]; then
        head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n'
    else
        err "Cannot generate random key. Install openssl or ensure /dev/urandom is available."
        exit 1
    fi
}

# --- Main ---
main() {
    info "AISIX Quick Start"
    info "================="
    echo

    check_commands

    AISIX_DIR="${AISIX_DIR:-$HOME/.aisix}"
    mkdir -p "$AISIX_DIR"

    ADMIN_KEY=$(rand_hex)

    info "Downloading configuration files..."
    curl -fsSL "${BASE_URL}/docker-compose.yaml" -o "${AISIX_DIR}/docker-compose.yaml"
    curl -fsSL "${BASE_URL}/config.yaml" -o "${AISIX_DIR}/config.yaml"

    # Replace default admin key with a random one
    sedi "s/key: admin/key: ${ADMIN_KEY}/" "${AISIX_DIR}/config.yaml"

    info "Starting AISIX and etcd..."
    cd "$AISIX_DIR"
    docker compose pull || { err "Failed to pull images. Check your network and try again."; exit 1; }
    docker compose up -d

    info "Waiting for AISIX to be ready..."
    i=0
    while [ $i -lt 30 ]; do
        if curl -sf http://127.0.0.1:3001/openapi >/dev/null 2>&1; then
            break
        fi
        i=$((i + 1))
        sleep 1
    done

    if [ $i -ge 30 ]; then
        err "AISIX did not start within 30 seconds. Check logs with: docker compose -f ${AISIX_DIR}/docker-compose.yaml logs"
        exit 1
    fi

    echo
    ok "AISIX is running!"
    echo
    echo "  Proxy API:   http://127.0.0.1:3000"
    echo "  Admin API:   http://127.0.0.1:3001/aisix/admin"
    echo "  Admin UI:    http://127.0.0.1:3001/ui"
    echo "  API Docs:    http://127.0.0.1:3001/openapi"
    echo "  Admin Key:   ${ADMIN_KEY}"
    echo
    echo "  Export it:    export ADMIN_KEY=${ADMIN_KEY}"
    echo
    echo "  Next steps:"
    echo "    1. Open the Admin UI to configure models and API keys."
    echo "    2. Or use the Admin API directly (see docs)."
    echo
    echo "  Documentation: https://github.com/${REPO}#readme"
    echo
    echo "  Stop AISIX:   cd ${AISIX_DIR} && docker compose down"
    echo
}

main "$@"
