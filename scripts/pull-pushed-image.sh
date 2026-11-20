#!/usr/bin/env bash
# Pull an image the build step pushed moments ago. A registry can accept a
# push before the digest resolves on pull ("manifest unknown"), so retry
# with backoff before giving up; ~75s of sleep across 6 attempts.
set -u
img="$1"
attempts=6
for i in $(seq 1 "$attempts"); do
  docker pull "$img" && exit 0
  if [ "$i" -lt "$attempts" ]; then
    delay=$((i * 5))
    echo "pull ${img} failed (attempt ${i}/${attempts}); retrying in ${delay}s"
    sleep "$delay"
  fi
done
echo "::error::${img} still not pullable after ${attempts} attempts (see the docker pull errors above)"
exit 1
