#!/usr/bin/env bash
# Retention for the ghcr packages docker-image.yml pushes to, and for their
# cosign signatures in the shared ghcr.io/api7/notary package.
#
# Per image package:
#   main-built  versions whose tags are all `sha-*` / `dev`: the MAIN_KEEP
#               newest by creation time and the one carrying `dev` are kept,
#               the rest deleted. Any other tag (X.Y.Z, X.Y.Z-rc.N, latest,
#               ...) protects the whole version.
#   ci-test     versions whose tags are all `ci-test-*`, deleted once older
#               than CI_TEST_MAX_AGE_DAYS.
#   children    untagged manifests a deleted index lists, unless a surviving
#               version is or references the same digest.
#   orphans     untagged versions no surviving version references, once older
#               than ORPHAN_MIN_AGE_DAYS (in-flight builds push children before
#               their index, so a young one may still be claimed).
#
# notary is shared with other products (API7 EE signs into it too), so a
# signature is deleted only when every cosign payload in it names this image
# at a digest this run deletes. Anything else in notary is left alone.
#
# Deletion runs oldest first, one unit at a time: an index's signatures, its
# children's signatures, its children, then the index (an orphan: its
# signature, then itself). A run that stops anywhere leaks nothing: a signature
# whose image is still listed is not needed, and the image is found again by
# the next run. At most MAX_DELETES DELETEs per run, DELETE_INTERVAL seconds
# apart, so the backlog drains over several runs instead of tripping GitHub's
# rate limits; a rate-limited DELETE ends the run cleanly, any other failed
# DELETE fails it. A plan that would empty a package is refused (GitHub will
# not delete a package's last version).
#
# GitHub's package API answers a 5xx now and then. Every listing and DELETE is
# retried up to GH_ATTEMPTS times with doubling backoff on a 5xx or a
# connection error; a listing restarts from its first page. A DELETE answered
# 404 counts as done: a retried DELETE can land after an earlier attempt
# already removed the version.
#
# Usage: GH_TOKEN=... scripts/ghcr-retention.sh [--dry-run]
set -euo pipefail
export LC_ALL=C

IMAGES="${IMAGES:-api7/aisix}"
NOTARY_IMAGE="${NOTARY_IMAGE:-api7/notary}"
MAIN_KEEP="${MAIN_KEEP:-3}"
CI_TEST_MAX_AGE_DAYS="${CI_TEST_MAX_AGE_DAYS:-3}"
ORPHAN_MIN_AGE_DAYS="${ORPHAN_MIN_AGE_DAYS:-1}"
MAX_DELETES="${MAX_DELETES:-300}"
DELETE_INTERVAL="${DELETE_INTERVAL:-1}"
GH_ATTEMPTS="${GH_ATTEMPTS:-4}"
GH_RETRY_DELAY="${GH_RETRY_DELAY:-5}"
dry_run=false
case "${1:-}" in
  "") ;;
  --dry-run) dry_run=true ;;
  *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
esac
[[ $MAX_DELETES =~ ^[0-9]+$ ]] || { echo "MAX_DELETES must be a whole number, got '$MAX_DELETES'" >&2; exit 2; }
verb=$($dry_run && echo 'would delete' || echo 'deleted')

ci_cutoff=$(date -u -d "${CI_TEST_MAX_AGE_DAYS} days ago" +%Y-%m-%dT%H:%M:%SZ)
orphan_cutoff=$(date -u -d "${ORPHAN_MIN_AGE_DAYS} days ago" +%Y-%m-%dT%H:%M:%SZ)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# A 5xx, or no HTTP status at all (the request never got an answer).
retryable() { grep -qE 'HTTP 50[0234]\b' <<<"$1" || ! grep -qE 'HTTP [0-9]{3}' <<<"$1"; }

# gh api for reads: prints stdout of the attempt that succeeded only.
gh_read() {
  local attempt=1 delay=$GH_RETRY_DELAY what
  for what; do [ "${what#-}" = "$what" ] && break; done
  until gh api "$@" > "$work/gh.out" 2> "$work/gh.err"; do
    if [ "$attempt" -ge "$GH_ATTEMPTS" ] || ! retryable "$(cat "$work/gh.err")"; then
      cat "$work/gh.err" >&2
      return 1
    fi
    echo "::warning::gh api $what failed ($(tail -n1 "$work/gh.err")), attempt $attempt of $GH_ATTEMPTS; retrying in ${delay}s" >&2
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
  cat "$work/gh.out"
}

gh_read rate_limit --jq '.resources.core | "API rate limit: \(.remaining)/\(.limit) remaining"'

package_of() { echo "orgs/${1%%/*}/packages/container/${1#*/}"; }

# id, digest, created_at, comma-joined tags
list_versions() {
  gh_read --paginate "$(package_of "$1")/versions?per_page=100" \
    --jq '.[] | [.id, .name, .created_at, (.metadata.container.tags | join(","))] | @tsv' \
    | sort -u -t$'\t' -k1,1
}

registry_token() {
  local url="https://ghcr.io/token?scope=repository:$1:pull&service=ghcr.io"
  if [ -n "${GH_TOKEN:-}" ]; then
    curl -fsS --retry 3 --retry-all-errors -u "x:${GH_TOKEN}" "$url" | jq -r .token
  else
    curl -fsS --retry 3 --retry-all-errors "$url" | jq -r .token
  fi
}

manifest() {
  curl -fsS --retry 3 --retry-all-errors -H "Authorization: Bearer ${registry_token}" \
    -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json' \
    "https://ghcr.io/v2/${IMAGE}/manifests/$1"
}

# Digests of the manifests an index lists (empty for a single image manifest).
children_of() {
  set -o pipefail
  manifest "$1" | jq -r '.manifests[]?.digest'
}

# "<index>\t<child>" for every manifest the index lists.
index_children() {
  set -o pipefail
  children_of "$1" | awk -v i="$1" '{ print i "\t" $0 }'
}

# Prints the signature row back when every layer of the cosign signature
# manifest is a simple-signing payload naming ghcr.io/<image> at the image
# digest the signature tag carries; prints nothing otherwise.
attribute_signature() {
  set -o pipefail
  local id=$1 digest=$2 created=$3 tag=$4 image=$5 image_digest=$6 m layer ok=0
  local payload
  m=$(manifest "$digest") || return 1
  for layer in $(jq -r '.layers[]? | "\(.mediaType)|\(.digest)"' <<<"$m"); do
    [ "${layer%%|*}" = application/vnd.dev.cosign.simplesigning.v1+json ] || return 0
    payload=$(curl -fsSL --retry 3 --retry-all-errors -H "Authorization: Bearer ${registry_token}" \
      "https://ghcr.io/v2/${IMAGE}/blobs/${layer#*|}") || return 1
    jq -e --arg ref "ghcr.io/$image" --arg d "$image_digest" \
      '.critical.identity["docker-reference"] == $ref and .critical.image["docker-manifest-digest"] == $d' \
      <<<"$payload" > /dev/null 2>&1 || return 0
    ok=1
  done
  [ "$ok" = 1 ] && printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$digest" "$created" "$tag" "$image" "$image_digest"
  return 0
}
export -f manifest children_of index_children attribute_signature

# Returns 2 when GitHub rate-limited the DELETE; exits on any other failure.
delete_version() {
  local image=$1 kind=$2 id=$3 digest=$4 created=$5 tags=$6 out status
  local attempt=1 delay=$GH_RETRY_DELAY done_as=$verb
  if ! $dry_run; then
    sleep "$DELETE_INTERVAL"
    until out=$(gh api -i -X DELETE "$(package_of "$image")/versions/$id" 2>&1); do
      status=$(sed -n '1s|^HTTP/[0-9.]* \([0-9]*\).*|\1|p' <<<"$out")
      if [ "$status" = 404 ]; then
        done_as='already gone (HTTP 404):'
        break
      fi
      if [ "$status" = 429 ] || { [ "$status" = 403 ] && grep -qiE '^x-ratelimit-remaining: 0|rate limit' <<<"$out"; }; then
        echo "::warning::rate limited (HTTP $status) deleting $kind $id from ghcr.io/$image; stopping here, the next run continues"
        return 2
      fi
      if [ "$attempt" -ge "$GH_ATTEMPTS" ] || ! retryable "${status:+HTTP $status}"; then
        echo "::error::DELETE of $kind $id from ghcr.io/$image failed (HTTP ${status:-unknown})"
        printf '%s\n' "$out" >&2
        exit 1
      fi
      echo "::warning::DELETE of $kind $id from ghcr.io/$image failed (HTTP ${status:-unknown}), attempt $attempt of $GH_ATTEMPTS; retrying in ${delay}s"
      sleep "$delay"
      attempt=$((attempt + 1))
      delay=$((delay * 2))
    done
  fi
  echo "$done_as $kind $id $digest (created $created${tags:+, tags $tags})"
}

: > "$work/deleted_digests"
: > "$work/report"
for image in $IMAGES; do
  dir="$work/${image//\//_}"
  mkdir -p "$dir"
  list_versions "$image" > "$dir/versions.tsv"
  total=$(wc -l < "$dir/versions.tsv")
  echo "== ghcr.io/$image: $total versions"

  awk -F'\t' '$4 != ""' "$dir/versions.tsv" > "$dir/tagged.tsv"
  awk -F'\t' '$4 == ""' "$dir/versions.tsv" > "$dir/untagged.tsv"
  # Versions whose tags all match the pattern.
  only_tags() { awk -F'\t' -v re="$1" '{ n = split($4, t, ","); ok = 1; for (i = 1; i <= n; i++) if (t[i] !~ re) ok = 0; if (ok) print }' "$dir/tagged.tsv"; }
  only_tags '^(sha-.*|dev)$' | sort -t$'\t' -k3,3r -k1,1nr > "$dir/main.tsv"
  # `dev` is a consumer pointer (other repositories' CI pulls it), so its
  # version stays even when newer sha-only builds push it out of the newest few.
  awk -F'\t' -v keep="$MAIN_KEEP" 'NR <= keep || $4 ~ /(^|,)dev(,|$)/' "$dir/main.tsv" > "$dir/main_kept.tsv"
  awk -F'\t' -v keep="$MAIN_KEEP" 'NR > keep && $4 !~ /(^|,)dev(,|$)/' "$dir/main.tsv" > "$dir/main_del.tsv"
  only_tags '^ci-test-' > "$dir/ci_all.tsv"
  awk -F'\t' -v c="$ci_cutoff" '$3 < c' "$dir/ci_all.tsv" > "$dir/ci_del.tsv"
  cat "$dir/main_del.tsv" "$dir/ci_del.tsv" > "$dir/indexes.tsv"
  cut -f2 "$dir/indexes.tsv" | sort -u > "$dir/index_digests"

  IMAGE="$image"
  registry_token=$(registry_token "$image")
  export IMAGE registry_token

  cut -f2 "$dir/indexes.tsv" | xargs -r -P 16 -I{} bash -c 'index_children "$1"' _ {} \
    | sort -u > "$dir/index_children"
  cut -f2 "$dir/index_children" | sort -u > "$dir/candidate_children"
  # Untagged versions this run may delete: children of a deleted index, and
  # anything past the orphan grace period.
  awk -F'\t' -v c="$orphan_cutoff" 'FILENAME == ARGV[1] { cc[$1]; next } ($2 in cc) || $3 < c' \
    "$dir/candidate_children" "$dir/untagged.tsv" > "$dir/untagged_candidates.tsv"

  # Every surviving version keeps itself and whatever it lists. An index can
  # only reference manifests pushed before it, so versions created before the
  # oldest candidate cannot reference one and are not fetched. Untagged
  # versions inside the grace period count as surviving, so an in-flight index
  # protects its children.
  : > "$dir/kept"
  if [ -s "$dir/untagged_candidates.tsv" ]; then
    earliest=$(cut -f3 "$dir/untagged_candidates.tsv" | sort | sed -n 1p)
    awk -F'\t' -v since="$earliest" -v c="$orphan_cutoff" '
      FILENAME == ARGV[1] { gone[$1]; next }
      FILENAME == ARGV[2] { gone[$1]; next }
      $3 >= since && ($4 != "" || $3 >= c) && !($2 in gone) { print $2 }' \
      "$dir/index_digests" "$dir/candidate_children" "$dir/versions.tsv" | sort -u > "$dir/surviving"
    echo "checking $(wc -l < "$dir/surviving") surviving versions created since $earliest for references"
    {
      cat "$dir/surviving"
      # Any failed fetch aborts the run rather than risk deleting a referenced manifest.
      xargs -P 16 -I{} bash -c 'children_of "$1"' _ {} < "$dir/surviving"
    } | sort -u > "$dir/kept"
  fi
  # FILENAME, not NR == FNR: kept may be empty.
  : > "$dir/children.tsv"
  : > "$dir/orphans.tsv"
  awk -F'\t' -v ch="$dir/children.tsv" -v orf="$dir/orphans.tsv" '
    FILENAME == ARGV[1] { k[$1]; next } FILENAME == ARGV[2] { cc[$1]; next }
    !($2 in k) { print > (($2 in cc) ? ch : orf) }' \
    "$dir/kept" "$dir/candidate_children" "$dir/untagged_candidates.tsv"

  plan=$(cat "$dir/indexes.tsv" "$dir/children.tsv" "$dir/orphans.tsv" | wc -l)
  if [ "$plan" -ge "$total" ]; then
    echo "refusing: the plan would delete all $total versions of ghcr.io/$image" >&2
    exit 1
  fi
  cat "$dir/children.tsv" "$dir/indexes.tsv" "$dir/orphans.tsv" \
    | awk -F'\t' -v img="$image" '{ print $2 "\t" img }' >> "$work/deleted_digests"
done

# Attribute notary signatures before deleting anything, so a failed fetch
# aborts with nothing deleted.
: > "$work/signatures.tsv"
: > "$work/notary_matched.tsv"
if [ -s "$work/deleted_digests" ]; then
  list_versions "$NOTARY_IMAGE" > "$work/notary.tsv"
  notary_total=$(wc -l < "$work/notary.tsv")
  IMAGE="$NOTARY_IMAGE"
  registry_token=$(registry_token "$NOTARY_IMAGE")
  export IMAGE registry_token
  # cosign tags a signature sha256-<hex>.sig after the signed digest.
  awk -F'\t' 'FILENAME == ARGV[1] { d = $1; sub(/^sha256:/, "sha256-", d); want[d ".sig"] = $1 "\t" $2; next }
    ($4 in want) { print $1 "\t" $2 "\t" $3 "\t" $4 "\t" want[$4] }' \
    "$work/deleted_digests" "$work/notary.tsv" > "$work/notary_matched.tsv"
  # id digest created tag image_digest image -> attribute_signature's order.
  awk -F'\t' '{ print $1 " " $2 " " $3 " " $4 " " $6 " " $5 }' "$work/notary_matched.tsv" \
    | xargs -r -P 16 -n 6 bash -c 'attribute_signature "$@"' _ > "$work/signatures.tsv"
  if [ "$(wc -l < "$work/signatures.tsv")" -ge "$notary_total" ]; then
    echo "refusing: the plan would delete all $notary_total versions of ghcr.io/$NOTARY_IMAGE" >&2
    exit 1
  fi
fi

# The deletion plan, one line per DELETE:
#   unit, unit size, kind, package image, id, digest, created, tags
# Units run oldest first; within a unit, signatures, then children, then the
# index or orphan itself. A child two deleted indexes list goes with the older.
: > "$work/plan.tsv"
for image in $IMAGES; do
  dir="$work/${image//\//_}"
  {
    awk -F'\t' '{ print $3 "\tmain\t" $1 "\t" $2 "\t" $4 }' "$dir/main_del.tsv"
    awk -F'\t' '{ print $3 "\tci-test\t" $1 "\t" $2 "\t" $4 }' "$dir/ci_del.tsv"
    awk -F'\t' '{ print $3 "\torphan\t" $1 "\t" $2 "\t" $4 }' "$dir/orphans.tsv"
  } | sort -t$'\t' -k1,1 -k3,3n > "$dir/units.tsv"
  awk -F'\t' -v img="$image" -v NOTARY="$NOTARY_IMAGE" '
    FILENAME == ARGV[1] { if ($5 == img) sig[$6] = sig[$6] "\n" $1 "\t" $2 "\t" $3 "\t" $4; next }
    FILENAME == ARGV[2] { kids[$1] = kids[$1] " " $2; next }
    FILENAME == ARGV[3] { child[$2] = $1 "\t" $2 "\t" $3 "\t" $4; next }
    function sigs(d,   n, i, rows) {
      n = split(substr(sig[d], 2), rows, "\n")
      for (i = 1; i <= n; i++) if (rows[i] != "") out[++m] = "signature\t" NOTARY "\t" rows[i]
    }
    {
      m = 0; delete out
      sigs($4)
      nk = 0; delete mine
      if ($2 != "orphan") {
        nc = split(kids[$4], cs, " ")
        for (i = 1; i <= nc; i++) if ((cs[i] in child) && !(cs[i] in taken)) { taken[cs[i]]; mine[++nk] = cs[i]; sigs(cs[i]) }
      }
      for (i = 1; i <= nk; i++) out[++m] = "child\t" img "\t" child[mine[i]]
      out[++m] = $2 "\t" img "\t" $3 "\t" $4 "\t" $1 "\t" $5
      unit++
      for (i = 1; i <= m; i++) print img ":" unit "\t" m "\t" out[i]
    }' "$work/signatures.tsv" "$dir/index_children" "$dir/children.tsv" "$dir/units.tsv" >> "$work/plan.tsv"
done
# The oldest unit that deletes a signature goes first, so a token that cannot
# delete in notary stops the run before any image is touched.
first_signed=$(awk -F'\t' '$3 == "signature" { print $1; exit }' "$work/plan.tsv")
awk -F'\t' -v u="$first_signed" '$1 == u' "$work/plan.tsv" > "$work/plan.ordered"
awk -F'\t' -v u="$first_signed" '$1 != u' "$work/plan.tsv" >> "$work/plan.ordered"
mv "$work/plan.ordered" "$work/plan.tsv"
backlog=$(wc -l < "$work/plan.tsv")

cur=""
executed=0
stopped=""
: > "$work/executed"
while IFS=$'\t' read -r unit size kind image id digest created tags; do
  if [ "$unit" != "$cur" ]; then
    cur=$unit
    # A unit larger than the cap still runs alone, or it would block every run.
    if [ "$MAX_DELETES" -eq 0 ] || { [ "$executed" -gt 0 ] && [ $((executed + size)) -gt "$MAX_DELETES" ]; }; then
      stopped="reached the cap of $MAX_DELETES DELETEs"
      break
    fi
  fi
  rc=0
  delete_version "$image" "$kind" "$id" "$digest" "$created" "$tags" || rc=$?
  if [ "$rc" = 2 ]; then
    stopped="rate limited"
    break
  fi
  executed=$((executed + 1))
  echo "$kind" >> "$work/executed"
done < "$work/plan.tsv"

done_of() { grep -cx "$1" "$work/executed" || true; }
for image in $IMAGES; do
  dir="$work/${image//\//_}"
  {
    echo "ghcr.io/$image ($(wc -l < "$dir/versions.tsv") versions):"
    echo "  main-built: $(wc -l < "$dir/main.tsv"), kept $(wc -l < "$dir/main_kept.tsv"), past retention $(wc -l < "$dir/main_del.tsv")"
    awk -F'\t' '{ print "    kept " $4 " (created " $3 ")" }' "$dir/main_kept.tsv"
    echo "  ci-test: $(wc -l < "$dir/ci_all.tsv"), past retention $(wc -l < "$dir/ci_del.tsv") (older than ${CI_TEST_MAX_AGE_DAYS} days)"
    echo "  children of those indexes: $(wc -l < "$dir/children.tsv")"
    echo "  orphans (untagged, unreferenced, older than ${ORPHAN_MIN_AGE_DAYS} day): $(wc -l < "$dir/orphans.tsv")"
  } >> "$work/report"
done
remaining=$((backlog - executed))
{
  echo "ghcr.io/$NOTARY_IMAGE:"
  echo "  signatures tagged after a digest past retention: $(wc -l < "$work/notary_matched.tsv")"
  echo "  attributed by the cosign payload: $(wc -l < "$work/signatures.tsv")"
  awk -F'\t' '{ n[$5]++ } END { for (i in n) print "    " i ": " n[i] }' "$work/signatures.tsv" | sort
  echo "this run ($verb, cap $MAX_DELETES): main $(done_of main), ci-test $(done_of ci-test), children $(done_of child), orphans $(done_of orphan), signatures $(done_of signature); $executed of $backlog DELETEs${stopped:+, stopped: $stopped}"
  [ "$remaining" -gt 0 ] && [ "$MAX_DELETES" -gt 0 ] && echo "left for later runs: $remaining DELETEs, about $(( (remaining + MAX_DELETES - 1) / MAX_DELETES )) more daily runs at this cap"
  true
} >> "$work/report"
echo
cat "$work/report"
