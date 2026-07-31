# Releasing

How an AISIX data-plane release is cut. Order matters: downstream packaging
(AISIX Cloud and the self-hosted bundle) pins the exact data-plane image
version, so the data plane is always tagged and published **first**.

## 1. Tag

```bash
git tag vX.Y.Z <commit>
git push origin vX.Y.Z
```

Pushing the tag triggers two workflows:

- **`docker-image.yml`** builds and publishes
  `ghcr.io/api7/aisix:X.Y.Z` (plus `:X.Y`, `:X`, `:latest`, `:sha-<short>`),
  mirrors the release tag to `docker.io/api7/aisix` for private/offline
  deployments, signs the images with cosign, and stamps the version into the
  binary so a running gateway self-reports `X.Y.Z` (`--version`, `Server`
  header) and `X.Y.Z+sha-<short>` in its managed-mode heartbeat.
- **`release-draft.yml`** creates a **draft** GitHub Release for the tag. The
  draft already leads with a version-stamped **Get started + Download** header
  (from [`.github/release-notes-header.md`](.github/release-notes-header.md):
  docs, self-hosted quickstart, and the `docker pull` command), then a commented
  curated-notes scaffold to fill in, then GitHub's auto-generated **What's
  Changed** list as a starting skeleton.

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
the self-hosted package) may tag the same `vX.Y.Z` — their packaging pulls
`docker.io/api7/aisix:X.Y.Z` and fails if it does not exist yet.
