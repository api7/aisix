#!/usr/bin/env bash
# Exit 0 only if <ref> currently resolves to <digest>. Run it through
# registry-retry.sh right after moving a tag: a read can still return the
# index the tag pointed at before the push.
set -euo pipefail
ref="$1" want="$2"
got="$(docker buildx imagetools inspect --format '{{json .Manifest}}' "$ref" | jq -r .digest)" || got=""
[ "$got" = "$want" ] || { echo "${ref} resolves to ${got:-nothing}, expected ${want}" >&2; exit 1; }
