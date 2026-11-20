#!/usr/bin/env bash
# Run a registry command against an image or tag this workflow pushed moments
# ago. A registry can accept a push before the reference resolves ("manifest
# unknown"), so retry with backoff (~75s of sleep across 6 attempts). Each
# attempt is capped so a stalled one is retried rather than hanging the job.
# The command's stdout passes through; progress and errors go to stderr.
set -u
attempts=6
for i in $(seq 1 "$attempts"); do
  timeout 120 "$@" && exit 0
  if [ "$i" -lt "$attempts" ]; then
    delay=$((i * 5))
    echo "'$*' failed (attempt ${i}/${attempts}); retrying in ${delay}s" >&2
    sleep "$delay"
  fi
done
echo "::error::'$*' still failing after ${attempts} attempts (see the errors above)" >&2
exit 1
