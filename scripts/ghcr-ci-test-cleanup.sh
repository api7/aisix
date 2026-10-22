#!/usr/bin/env bash
# Deletes the scratch images that docker-image.yml pushes for a
# workflow_dispatch on a non-main branch (tag `ci-test-<sha7>`) once they are
# older than MAX_AGE_DAYS, together with their untagged per-platform children.
#
# Never deleted: a version carrying any tag that is not `ci-test-*` (sha-*,
# dev, release tags, ...), a child some surviving tagged version still
# references, and any untagged version that is not a child of a deleted
# ci-test index. GitHub refuses to delete a package's last version, so the
# script refuses a plan that would empty the package.
#
# Usage: GH_TOKEN=... scripts/ghcr-ci-test-cleanup.sh [--dry-run]
set -euo pipefail
export LC_ALL=C

PACKAGE="${PACKAGE:-orgs/api7/packages/container/aisix}"
IMAGE="${IMAGE:-api7/aisix}"
MAX_AGE_DAYS="${MAX_AGE_DAYS:-3}"
dry_run=false
[ "${1:-}" = "--dry-run" ] && dry_run=true

cutoff=$(date -u -d "${MAX_AGE_DAYS} days ago" +%Y-%m-%dT%H:%M:%SZ)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# id, digest, created_at, comma-joined tags
gh api --paginate "$PACKAGE/versions?per_page=100" \
  --jq '.[] | [.id, .name, .created_at, (.metadata.container.tags | join(","))] | @tsv' \
  > "$work/versions.tsv"
total=$(wc -l < "$work/versions.tsv")
echo "package $PACKAGE: $total versions; deleting ci-test-* created before $cutoff"

# A ci-test index qualifies only if every tag it carries is ci-test-*.
awk -F'\t' -v cutoff="$cutoff" '
  $4 != "" && $3 < cutoff {
    n = split($4, tags, ",")
    ok = 1
    for (i = 1; i <= n; i++) if (tags[i] !~ /^ci-test-/) ok = 0
    if (ok) print
  }' "$work/versions.tsv" > "$work/indexes.tsv"

ci_test_total=$(awk -F'\t' '$4 ~ /(^|,)ci-test-/' "$work/versions.tsv" | wc -l)
echo "ci-test-* versions: $ci_test_total, older than ${MAX_AGE_DAYS} days: $(wc -l < "$work/indexes.tsv")"
if [ ! -s "$work/indexes.tsv" ]; then
  echo "nothing to delete"
  exit 0
fi

token_url="https://ghcr.io/token?scope=repository:${IMAGE}:pull&service=ghcr.io"
if [ -n "${GH_TOKEN:-}" ]; then
  registry_token=$(curl -fsS -u "x:${GH_TOKEN}" "$token_url" | jq -r .token)
else
  registry_token=$(curl -fsS "$token_url" | jq -r .token)
fi

# Digests of the manifests an index lists (empty for a single image manifest).
children_of() {
  set -o pipefail
  curl -fsS --retry 3 --retry-all-errors -H "Authorization: Bearer ${registry_token}" \
    -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json' \
    "https://ghcr.io/v2/${IMAGE}/manifests/$1" | jq -r '.manifests[]?.digest'
}

cut -f2 "$work/indexes.tsv" | sort -u > "$work/deleted_digests"
while IFS=$'\t' read -r _ digest _ _; do
  children_of "$digest"
done < "$work/indexes.tsv" | sort -u > "$work/candidate_children"

# Children are deleted only as untagged package versions.
awk -F'\t' '$4 == ""' "$work/versions.tsv" | sort -t$'\t' -k2,2 > "$work/untagged.tsv"
join -t$'\t' -1 1 -2 2 -o 2.1,2.2,2.3,2.4 "$work/candidate_children" "$work/untagged.tsv" \
  > "$work/candidates.tsv"

# A surviving tagged version keeps every child it is or references. An index
# can only reference manifests pushed before it, so tagged versions created
# before the oldest candidate child cannot reference one and are not fetched.
: > "$work/kept"
if [ -s "$work/candidates.tsv" ]; then
  earliest=$(cut -f3 "$work/candidates.tsv" | sort | head -n 1)
  awk -F'\t' -v since="$earliest" '$4 != "" && $3 >= since' "$work/versions.tsv" | cut -f2 | sort -u \
    | comm -23 - "$work/deleted_digests" > "$work/surviving_tagged"
  echo "checking $(wc -l < "$work/surviving_tagged") tagged versions created since $earliest for references"
  {
    cat "$work/surviving_tagged"
    # Any failed fetch aborts the run rather than risk deleting a referenced child.
    export -f children_of
    export registry_token IMAGE
    xargs -P 16 -I{} bash -c 'children_of "$1"' _ {} < "$work/surviving_tagged"
  } | sort -u > "$work/kept"
fi
awk -F'\t' 'NR == FNR { kept[$1]; next } !($2 in kept)' "$work/kept" "$work/candidates.tsv" \
  > "$work/children.tsv"

plan=$(( $(wc -l < "$work/indexes.tsv") + $(wc -l < "$work/children.tsv") ))
if [ "$plan" -ge "$total" ]; then
  echo "refusing: the plan would delete all $total versions of the package" >&2
  exit 1
fi

# Children first: if a run aborts midway, the index survives and the next
# run finds the remaining children through it again.
while IFS=$'\t' read -r id digest created _; do
  if $dry_run; then
    echo "would delete child $id $digest (created $created)"
  else
    gh api -X DELETE "$PACKAGE/versions/$id"
    echo "deleted child $id $digest (created $created)"
  fi
done < "$work/children.tsv"
while IFS=$'\t' read -r id digest created tags; do
  if $dry_run; then
    echo "would delete index $id $digest (created $created, tags $tags)"
  else
    gh api -X DELETE "$PACKAGE/versions/$id"
    echo "deleted index $id $digest (created $created, tags $tags)"
  fi
done < "$work/indexes.tsv"
echo "$($dry_run && echo 'would delete' || echo 'deleted') $(wc -l < "$work/indexes.tsv") ci-test indexes and $(wc -l < "$work/children.tsv") children"
