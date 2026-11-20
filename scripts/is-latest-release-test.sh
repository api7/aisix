#!/usr/bin/env bash
# Usage: scripts/is-latest-release-test.sh
set -euo pipefail
script="$(dirname "$(readlink -f "$0")")/is-latest-release.sh"
fail=0
check() { # <want> <tag> <existing tags...>
  local want="$1" tag="$2"; shift 2
  local got
  got="$(printf '%s\n' "$@" | "$script" "$tag")"
  if [ "$got" = "$want" ]; then echo "ok   $tag among [$*] -> $got"
  else echo "FAIL $tag among [$*] -> $got, want $want"; fail=1; fi
}
check true  v1.11.0 v1.10.0 v1.10.1 v1.11.0
check false v1.10.1 v1.10.0 v1.11.0 v1.10.1
check true  v1.11.1 v1.11.0 v1.11.1 v1.11.2-rc.1
check true  v1.10.0 v1.9.0 v1.10.0                 # numeric, not lexical
check false v1.9.5  v1.9.5 v1.10.0
check true  v2.0.0  v1.11.0 v10.0.0-rc.1 v2.0.0     # a pre-release never outranks
check false v1.12.0-rc.1 v1.11.0 v1.12.0-rc.1
check true  v1.11.0                                # no other tags at all
check true  v1.11.0 v1.11.0 v1.11.0                # re-run of the same tag
check false v1.11.0 v1.11.0 v1.11.0.1 v1.12.0
# Callers pipe `git ls-remote` into it under pipefail, so a pre-release must
# still drain stdin rather than fail the producer with SIGPIPE.
if ! (set -o pipefail; seq 1 200000 | sed 's/^/v1.0./' | "$script" v1.1.0-rc.1 > /dev/null); then
  echo "FAIL a pre-release did not drain stdin (producer died under pipefail)"; fail=1
else
  echo "ok   a pre-release drains stdin"
fi
exit "$fail"
