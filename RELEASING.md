# Releasing

How an AISIX AI Gateway release is cut. Order matters: downstream packaging
(AISIX Cloud and the On-Premises package, whose artifact name is
`aisix-self-hosted`) pins the exact gateway image version, so the gateway is
always tagged and published **first**.

## 1. Tag

```bash
git tag vX.Y.Z <commit>
git push origin vX.Y.Z
```

Pushing the tag triggers two workflows:

- **`docker-image.yml`** builds and publishes
  `ghcr.io/api7/aisix:X.Y.Z` (plus `:latest` and `:sha-<short>`; a version
  tag is published in full only, with no `:X.Y` or `:X` abbreviation),
  mirrors the release tag to `docker.io/api7/aisix` for private/offline
  deployments, signs the images with cosign, and stamps the version into the
  binary so a running gateway self-reports `X.Y.Z` (`--version`, `Server`
  header) and `X.Y.Z+sha-<short>` in its managed-mode heartbeat.
- **`release-draft.yml`** creates a **draft** GitHub Release for the tag. The
  draft already leads with a version-stamped **Get started + Download** header
  (from [`.github/release-notes-header.md`](.github/release-notes-header.md):
  docs, gateway quickstart, and the `docker pull` command), then a commented
  curated-notes scaffold to fill in, then GitHub's auto-generated **What's
  Changed** list as a starting skeleton.

### PGO applies to the stable release tag only — and it is fail-closed there

The `vX.Y.Z` image is profile-guided-optimized (#967): the Docker build
compiles an instrumented gateway, drives the committed training matrix
(`bench/pgo-training/`) against it, and rebuilds with the merged profile.
Any phase failing — instrumented build, training, profile merge, optimized
build — fails the image build; there is no fallback to a plain build. After
the push, the workflow asserts the `pgo-verified.json` proof marker inside
the image (shape count, profile size) before signing. If a release build
fails in a PGO phase, fix the cause; never ship around it. To inspect a
shipped image's marker:

```bash
docker run --rm --entrypoint cat ghcr.io/api7/aisix:X.Y.Z \
  /usr/local/share/aisix/pgo-verified.json
```

**`-rc.N`, `:dev` and PR images are NOT PGO'd** — PGO cost ~20 min on every
one of the ~40 main pushes per cycle and none of those images are what a
customer runs. Two things follow, and both belong to the release flow:

- **The QA'd candidate is not bit-identical to the shipped image.** PGO
  changes code layout, inlining and block ordering — never semantics; the
  workspace contains no `unsafe`, so there is no undefined behaviour for a
  different inlining decision to expose. The functional QA result therefore
  still describes the shipped build. What it does *not* describe is the
  released image's performance: any perf number must come from a `vX.Y.Z`
  image or a local `--build-arg PGO=on` build, never from `:dev` or an rc.
- **The release tag is the first build to run the PGO pipeline on that
  commit**, so a training-shape regression surfaces *after* QA has passed —
  the most expensive moment to find it, since the fix moves the commit and
  costs a fresh rc plus a re-run of QA. Pre-flight it instead, and do it
  **concurrently, not as a gate before tagging**: the check needs only the
  candidate's commit, which `vX.Y.Z-rc.N` already fixes, so fire it as soon
  as the candidate is cut and read the verdict when you come to tag. It
  builds the exact three-phase path and asserts the proof marker, publishing
  only a `:sha-<short>` tag that moves no pointer — and it finishes well
  inside the QA window, so it costs no extra wall clock. Blocking on it at
  tag time would put ~30 minutes on the critical path of every release for a
  result that was already knowable hours earlier.

`--ref` takes a **branch or tag name, never a raw commit SHA**, so dispatch on
the release line's branch — at this point its HEAD *is* the candidate commit.
Do NOT dispatch on the `vX.Y.Z-rc.N` tag: a tag ref makes metadata-action
re-emit the candidate's own image tags, republishing the very artifact QA is
testing as a PGO'd build QA never saw. A branch ref publishes one GHCR
`:sha-<short>` and nothing else.

```bash
# when the candidate is cut — fire and carry on
gh workflow run docker-image.yml --ref release/<X.Y> -f pgo=true
# confirm it caught the candidate commit and not a later push to the branch
gh run list --workflow=docker-image.yml --limit 1 --json databaseId,headSha --jq '.[0]'
# when you come to tag — a lookup, not a wait
gh run view <run-id> --json conclusion --jq .conclusion
```

A PR that touches `Dockerfile`, `Cargo.toml`, `Cargo.lock`,
`rust-toolchain.toml`, `bench/pgo-training/**` or the workflow itself still
flips to PGO=on automatically — that is the one pre-tag exercise of the
pipeline that needs no one to remember it.

Local note: each retrained profile is content-addressed, so repeated local
PGO builds accumulate build artifacts in the persistent BuildKit cache
mounts; reclaim with `docker builder prune`.

## 2. Polish the release notes

Edit the draft before publishing. The Get-started/Download header and the
full-changelog link are already in place, and a commented **curated-notes
scaffold** sits between the header's `---` divider and the What's Changed list —
fill it in, then delete the comment. House style (see the
[published releases](https://github.com/api7/aisix/releases) for examples):

- Lead with a short narrative line when the release has one (e.g. "AISIX
  becomes a gateway for AI agents"), then 3–6 **highlights** in plain
  language. Group the remainder under themed sections (routing, guardrails,
  API surface, security, observability).
- **Breaking changes get their own ⚠️ section**, with the old → new config
  spelling and what to update.
- Reference only public artifacts: this repo's PR numbers are fine; never
  cite internal issue trackers.
- Describe each feature by its own function — no comparisons against other
  products.
- Keep the download/install details in the header block; don't hand-add a
  second install snippet at the bottom.

If the header text itself needs to change (new docs URL, extra image
registry), edit `.github/release-notes-header.md` — not each release by hand.

## 3. Publish

Publish the draft and mark it **Latest**. Give it a descriptive title when the
release has a headline feature (e.g. `v0.2.0 — Semantic routing`), or just the
version for patch releases.

## 4. Downstream

Only after the images are published, downstream release flows (AISIX Cloud /
the On-Premises package named `aisix-self-hosted`) may tag the same `vX.Y.Z` —
their packaging pulls
`docker.io/api7/aisix:X.Y.Z` and fails if it does not exist yet.
