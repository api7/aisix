#!/usr/bin/env bash
# Runs ghcr-retention.sh against stubbed `gh` and `curl` to check how it
# survives GitHub API failures: a 5xx or connection error on a listing or a
# DELETE is retried, a retried DELETE answered 404 counts as done, anything
# else (a first-attempt 404, or a 5xx that outlasts the attempts) fails the run.
#
# Fixture: ghcr.io/t/img holds five main builds (MAIN_KEEP=3 deletes the two
# oldest, ids 1 and 2), ghcr.io/t/notary holds a signature for each of them
# (ids 101 and 102) and one unrelated version. The plan is therefore:
#   signature 101, main 1, signature 102, main 2.
#
# Usage: scripts/ghcr-retention-test.sh
set -euo pipefail
export LC_ALL=C

script="$(dirname "$(readlink -f "$0")")/ghcr-retention.sh"
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
mkdir -p "$root/bin"

old=$(date -u -d '10 days ago' +%Y-%m-%dT%H:%M:%SZ)
digest() { printf 'sha256:%064d' "$1"; }

# The stubbed gh answers each call from a queue file named after the call
# (list-<package>, delete-<id>): every call pops the first line, an empty
# queue means success. A line is one of 500/502/503/504/404/403 (an HTTP
# error) or "conn" (a connection error, no HTTP status).
cat > "$root/bin/gh" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
[ "$1" = api ] || exit 9
shift
if [ "$1" = rate_limit ]; then echo "API rate limit: 5000/5000 remaining"; exit 0; fi
if [ "$1" = --paginate ]; then
  pkg=$(sed -n 's|.*/packages/container/\([^/]*\)/versions.*|\1|p' <<<"$2")
  key="list-$pkg"; body="$FIXTURE/$pkg.tsv"; headers=false
else
  # -i -X DELETE orgs/<org>/packages/container/<pkg>/versions/<id>
  key="delete-${4##*/}"; body=/dev/null; headers=true
fi
echo "$key" >> "$FIXTURE/calls"
answer=""
if [ -s "$FIXTURE/$key" ]; then
  answer=$(sed -n 1p "$FIXTURE/$key")
  sed -i 1d "$FIXTURE/$key"
fi
case "$answer" in
  "")
    $headers && printf 'HTTP/2.0 204 No Content\r\n\r\n'
    cat "$body"; exit 0 ;;
  conn)
    # A paginated listing has printed its earlier pages before a later one
    # fails; those rows are not in the listing that later succeeds.
    $headers || cat "$FIXTURE/$pkg.leak"
    echo "Get \"https://api.github.com/x\": read tcp 10.0.0.1:443: connection reset by peer" >&2; exit 1 ;;
  *)
    $headers || cat "$FIXTURE/$pkg.leak"
    $headers && printf 'HTTP/2.0 %s Error\r\nContent-Type: application/json\r\n\r\n{"message":"Error"}' "$answer"
    echo "gh: Error (HTTP $answer)" >&2; exit 1 ;;
esac
STUB

# The stubbed curl serves registry tokens, manifests and cosign payloads.
cat > "$root/bin/curl" <<'STUB'
#!/usr/bin/env bash
url="${*: -1}"
case "$url" in
  *ghcr.io/token*) echo '{"token":"t"}' ;;
  */t/img/manifests/*) echo '{"mediaType":"application/vnd.oci.image.manifest.v1+json","layers":[]}' ;;
  */t/notary/manifests/*)
    echo "{\"layers\":[{\"mediaType\":\"application/vnd.dev.cosign.simplesigning.v1+json\",\"digest\":\"${url##*/}\"}]}" ;;
  */t/notary/blobs/*)
    sig=${url##*/}; n=$((10#${sig: -1}))
    printf '{"critical":{"identity":{"docker-reference":"ghcr.io/t/img"},"image":{"docker-manifest-digest":"sha256:%064d"}}}\n' "$n" ;;
  *) echo "unexpected curl $url" >&2; exit 22 ;;
esac
STUB
chmod +x "$root/bin/gh" "$root/bin/curl"

fail=0
# run <name> <expected exit> <expected DELETE order or -> [queue=answers ...]
run() {
  local name=$1 want_rc=$2 want_deletes=$3 q rc=0
  shift 3
  export FIXTURE="$root/$name"
  mkdir -p "$FIXTURE"
  : > "$FIXTURE/calls"
  for i in 1 2 3 4 5; do
    printf '%s\t%s\t%s\tsha-%s\n' "$i" "$(digest "$i")" "$(date -u -d "$((20 - i)) days ago" +%Y-%m-%dT%H:%M:%SZ)" "$i"
  done > "$FIXTURE/img.tsv"
  {
    for i in 1 2; do
      printf '%s\t%s\t%s\t%s\n' "10$i" "sha256:$(printf '%064d' "$((900 + i))")" "$old" "sha256-$(printf '%064d' "$i").sig"
    done
    printf '200\tsha256:%064d\t%s\tsomething-else\n' 999 "$old"
  } > "$FIXTURE/notary.tsv"
  # Rows a failed listing prints. Each would add a DELETE if it leaked into
  # the plan: an older main build, and another signature of image 1.
  printf '50\t%s\t%s\tsha-50\n' "$(digest 50)" "$(date -u -d '30 days ago' +%Y-%m-%dT%H:%M:%SZ)" > "$FIXTURE/img.leak"
  printf '150\tsha256:%064d\t%s\tsha256-%064d.sig\n' 950 "$old" 1 > "$FIXTURE/notary.leak"
  for q; do tr ' ' '\n' <<<"${q#*=}" > "$FIXTURE/${q%%=*}"; done
  PATH="$root/bin:$PATH" IMAGES=t/img NOTARY_IMAGE=t/notary MAIN_KEEP=3 \
    DELETE_INTERVAL=0 GH_RETRY_DELAY=0 GH_ATTEMPTS=3 GH_TOKEN=x \
    "$script" > "$FIXTURE/out" 2>&1 || rc=$?
  local got_deletes
  got_deletes=$(sed -n 's/^delete-//p' "$FIXTURE/calls" | tr '\n' ' ' | sed 's/ $//')
  if [ "$rc" != "$want_rc" ] || [ "${got_deletes:--}" != "$want_deletes" ]; then
    echo "FAIL $name: exit $rc (want $want_rc), DELETEs '${got_deletes:--}' (want '$want_deletes')"
    sed 's/^/    /' "$FIXTURE/out"
    fail=1
  else
    echo "ok   $name"
  fi
}

all="101 1 102 2"
run baseline 0 "$all"
grep -q 'signatures 2; 4 of 4 DELETEs' "$root/baseline/out" || { echo "FAIL baseline: unexpected report"; fail=1; }
run list-502-then-ok 0 "$all" list-notary=502
grep -q 'signatures 2; 4 of 4 DELETEs' "$root/list-502-then-ok/out" || { echo "FAIL list-502-then-ok: a restarted listing changed the plan"; fail=1; }
run list-conn-then-ok 0 "$all" list-img=conn
run list-404-not-retried 1 - list-notary=404
[ "$(grep -c '^list-notary$' "$root/list-404-not-retried/calls")" = 1 ] || { echo "FAIL list-404-not-retried: listing was retried"; fail=1; }
run list-502-exhausted 1 - list-img="502 503 504"
run delete-502-then-ok 0 "101 1 1 102 2" delete-1=502
run delete-502-then-404 0 "101 101 1 102 2" delete-101="502 404"
grep -q 'already gone (HTTP 404): signature 101' "$root/delete-502-then-404/out" || { echo "FAIL delete-502-then-404: 404 not reported as already gone"; fail=1; }
run delete-404-first-attempt-fails 1 "101" delete-101=404
run delete-conn-then-ok 0 "101 1 102 102 2" delete-102=conn
run delete-502-exhausted 1 "101 101 101" delete-101="502 502 502"
run delete-403-not-retried 1 "101" delete-101=403
run delete-429-stops 0 "101 1 102" delete-102=429

exit "$fail"
