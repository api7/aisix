#!/usr/bin/env bash
# Run a registry command against an image or tag this workflow pushed moments
# ago. A registry can accept a push before the reference resolves ("manifest
# unknown"), so retry with backoff (~75s of sleep across 6 attempts). Each
# attempt is capped so a stalled one is retried rather than hanging the job.
# Only the successful attempt's stdout is printed, so callers can pipe it;
# progress and errors go to stderr.
set -u
trap 'exit 130' INT TERM
attempts=6
out="$(mktemp)"
trap 'rm -f "$out"' EXIT
for i in $(seq 1 "$attempts"); do
  timeout 120 "$@" > "$out" && { cat "$out"; exit 0; }
  rc=$?
  if [ "$i" -lt "$attempts" ]; then
    delay=$((i * 5))
    echo "'$*' failed with exit ${rc} (attempt ${i}/${attempts}); retrying in ${delay}s" >&2
    sleep "$delay"
  fi
done
echo "::error::'$*' still failing after ${attempts} attempts (last exit ${rc}; see the errors above)" >&2
exit "$rc"
