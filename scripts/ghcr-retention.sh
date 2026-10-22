#!/usr/bin/env bash
# Retention for the ghcr packages docker-image.yml pushes to, and for their
# cosign signatures in the shared ghcr.io/api7/notary package.
#
# Per image package:
#   main-built  versions whose tags are all `sha-*` / `dev`: the MAIN_KEEP
#               newest by creation time are kept, the rest deleted. Any other
#               tag (X.Y.Z, X.Y.Z-rc.N, latest, ...) protects the whole version.
#   ci-test     versions whose tags are all `ci-test-*`, deleted once older
#               than CI_TEST_MAX_AGE_DAYS.
#   children    untagged manifests a deleted index lists, unless a surviving
#               version is or references the same digest.
#   orphans     untagged versions no surviving version references, once older
#               than ORPHAN_MIN_AGE_DAYS (in-flight builds push children before
#               their index, so a young one may still be claimed).
#
# notary is shared with other products (API7 EE signs into it too), so a
# signature is deleted only when its cosign payload names this image and the
# digest this run deleted, and that digest is no longer in the package. Any
# signature that cannot be attributed that way is left alone.
#
# GitHub refuses to delete a package's last version, so a plan that would empty
# a package is refused. The first failed DELETE stops the run.
#
# Usage: GH_TOKEN=... scripts/ghcr-retention.sh [--dry-run]
set -euo pipefail
export LC_ALL=C

IMAGES="${IMAGES:-api7/aisix}"
NOTARY_IMAGE="${NOTARY_IMAGE:-api7/notary}"
MAIN_KEEP="${MAIN_KEEP:-3}"
CI_TEST_MAX_AGE_DAYS="${CI_TEST_MAX_AGE_DAYS:-3}"
ORPHAN_MIN_AGE_DAYS="${ORPHAN_MIN_AGE_DAYS:-1}"
dry_run=false
[ "${1:-}" = "--dry-run" ] && dry_run=true
verb=$($dry_run && echo 'would delete' || echo 'deleted')

ci_cutoff=$(date -u -d "${CI_TEST_MAX_AGE_DAYS} days ago" +%Y-%m-%dT%H:%M:%SZ)
orphan_cutoff=$(date -u -d "${ORPHAN_MIN_AGE_DAYS} days ago" +%Y-%m-%dT%H:%M:%SZ)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

gh api rate_limit --jq '.resources.core | "API rate limit: \(.remaining)/\(.limit) remaining"'

package_of() { echo "orgs/${1%%/*}/packages/container/${1#*/}"; }

# id, digest, created_at, comma-joined tags
list_versions() {
  gh api --paginate "$(package_of "$1")/versions?per_page=100" \
    --jq '.[] | [.id, .name, .created_at, (.metadata.container.tags | join(","))] | @tsv' \
    | sort -u -t$'\t' -k1,1
}

registry_token() {
  local url="https://ghcr.io/token?scope=repository:$1:pull&service=ghcr.io"
  if [ -n "${GH_TOKEN:-}" ]; then
    curl -fsS -u "x:${GH_TOKEN}" "$url" | jq -r .token
  else
    curl -fsS "$url" | jq -r .token
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

# Prints the signature row back when every layer of the cosign signature
# manifest is a simple-signing payload naming ghcr.io/<image> at the
# image digest the signature tag carries; prints nothing otherwise.
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
export -f manifest children_of attribute_signature

delete_version() {
  local image=$1 kind=$2 id=$3 digest=$4 created=$5 tags=$6
  $dry_run || gh api -X DELETE "$(package_of "$image")/versions/$id" > /dev/null
  echo "$verb $kind $id $digest (created $created${tags:+, tags $tags})"
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
  awk -F'\t' '$4 == ""' "$dir/versions.tsv" | sort -t$'\t' -k2,2 > "$dir/untagged.tsv"
  # Versions whose tags all match the pattern.
  only_tags() { awk -F'\t' -v re="$1" '{ n = split($4, t, ","); ok = 1; for (i = 1; i <= n; i++) if (t[i] !~ re) ok = 0; if (ok) print }' "$dir/tagged.tsv"; }
  only_tags '^(sha-.*|dev)$' | sort -t$'\t' -k3,3r -k1,1nr > "$dir/main.tsv"
  head -n "$MAIN_KEEP" "$dir/main.tsv" > "$dir/main_kept.tsv"
  tail -n +"$((MAIN_KEEP + 1))" "$dir/main.tsv" > "$dir/main_del.tsv"
  only_tags '^ci-test-' > "$dir/ci_all.tsv"
  awk -F'\t' -v c="$ci_cutoff" '$3 < c' "$dir/ci_all.tsv" > "$dir/ci_del.tsv"
  cat "$dir/main_del.tsv" "$dir/ci_del.tsv" > "$dir/indexes.tsv"
  cut -f2 "$dir/indexes.tsv" | sort -u > "$dir/index_digests"

  IMAGE="$image"
  registry_token=$(registry_token "$image")
  export IMAGE registry_token

  cut -f2 "$dir/indexes.tsv" | xargs -r -P 16 -I{} bash -c 'children_of "$1"' _ {} \
    | sort -u > "$dir/candidate_children"
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
    earliest=$(cut -f3 "$dir/untagged_candidates.tsv" | sort | head -n 1)
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
  # Oldest first: within an orphaned build the children predate their index.
  sort -t$'\t' -k3,3 -o "$dir/orphans.tsv" "$dir/orphans.tsv"

  plan=$(cat "$dir/indexes.tsv" "$dir/children.tsv" "$dir/orphans.tsv" | wc -l)
  if [ "$plan" -ge "$total" ]; then
    echo "refusing: the plan would delete all $total versions of ghcr.io/$image" >&2
    exit 1
  fi

  {
    echo "ghcr.io/$image ($total versions):"
    echo "  main-built: $(wc -l < "$dir/main.tsv"), kept $(wc -l < "$dir/main_kept.tsv"), $verb $(wc -l < "$dir/main_del.tsv")"
    awk -F'\t' '{ print "    kept " $4 " (created " $3 ")" }' "$dir/main_kept.tsv"
    echo "  ci-test: $(wc -l < "$dir/ci_all.tsv"), $verb $(wc -l < "$dir/ci_del.tsv") older than ${CI_TEST_MAX_AGE_DAYS} days"
    echo "  children of deleted indexes: $verb $(wc -l < "$dir/children.tsv")"
    echo "  orphans (untagged, unreferenced, older than ${ORPHAN_MIN_AGE_DAYS} day): $verb $(wc -l < "$dir/orphans.tsv")"
  } >> "$work/report"
  cat "$dir/children.tsv" "$dir/indexes.tsv" "$dir/orphans.tsv" \
    | awk -F'\t' -v img="$image" '{ print $2 "\t" img }' >> "$work/deleted_digests"
done

# Plan the notary sweep before deleting anything, so a failed fetch aborts
# with nothing deleted. Signatures pushed while the list is paged belong to
# new images, which this run does not delete.
IMAGE="$NOTARY_IMAGE"
registry_token=$(registry_token "$NOTARY_IMAGE")
export IMAGE registry_token
: > "$work/signatures.tsv"
: > "$work/notary_matched.tsv"
if [ -s "$work/deleted_digests" ]; then
  list_versions "$NOTARY_IMAGE" > "$work/notary.tsv"
  notary_total=$(wc -l < "$work/notary.tsv")
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

for image in $IMAGES; do
  dir="$work/${image//\//_}"
  # Children first: if a run aborts midway, the index survives and the next
  # run finds the remaining children through it again.
  while IFS=$'\t' read -r id digest created tags; do delete_version "$image" child "$id" "$digest" "$created" "$tags"; done < "$dir/children.tsv"
  while IFS=$'\t' read -r id digest created tags; do delete_version "$image" index "$id" "$digest" "$created" "$tags"; done < "$dir/indexes.tsv"
  while IFS=$'\t' read -r id digest created tags; do delete_version "$image" orphan "$id" "$digest" "$created" "$tags"; done < "$dir/orphans.tsv"
done

# A signature goes only once its digest is gone from the package.
: > "$work/present"
if ! $dry_run; then
  for image in $IMAGES; do list_versions "$image" | awk -F'\t' -v img="$image" '{ print img "\t" $2 }'; done \
    | sort -u > "$work/present"
fi
awk -F'\t' 'FILENAME == ARGV[1] { p[$1 "\t" $2]; next } !(($5 "\t" $6) in p)' "$work/present" "$work/signatures.tsv" \
  > "$work/notary_del.tsv"
while IFS=$'\t' read -r id digest created tag image image_digest; do
  delete_version "$NOTARY_IMAGE" signature "$id" "$digest" "$created" "$tag"
done < "$work/notary_del.tsv"

{
  echo "ghcr.io/$NOTARY_IMAGE:"
  echo "  signatures tagged after a deleted digest: $(wc -l < "$work/notary_matched.tsv")"
  echo "  attributed to ${IMAGES// /, } by the cosign payload: $(wc -l < "$work/signatures.tsv"), $verb $(wc -l < "$work/notary_del.tsv")"
  awk -F'\t' '{ n[$5]++ } END { for (i in n) print "    " i ": " n[i] }' "$work/notary_del.tsv" | sort
} >> "$work/report"
echo
cat "$work/report"
