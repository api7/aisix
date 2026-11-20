#!/usr/bin/env bash
# Print "true" if <tag> is a stable release tag (vX.Y.Z) and no stable tag
# read from stdin (one per line, e.g. from `git ls-remote --tags`) has a
# higher version; print "false" otherwise. Pre-releases (vX.Y.Z-rc.N) are
# never latest, and neither is a patch on an older line (v1.10.1 after
# v1.11.0), so neither may move :latest or openapi-latest.json.
set -euo pipefail
tag="$1"
# Read all of stdin first: exiting before the producer finishes would fail
# it with SIGPIPE, and a caller under pipefail would abort.
tags="$(cat)"
stable='^v[0-9]+\.[0-9]+\.[0-9]+$'
if ! [[ "$tag" =~ $stable ]]; then
  echo false
  exit 0
fi
highest="$( { grep -E "$stable" <<< "$tags" || true; echo "$tag"; } | sort -V | tail -n 1)"
if [ "$highest" = "$tag" ]; then echo true; else echo false; fi
