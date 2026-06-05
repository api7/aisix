# Temporary AI Gateway Docs Update And Merge Strategy

Date: 2026-06-03

This is a temporary working note for the current docs update cycle.

## Agreed Strategy

1. `api7/ai-gateway/docs` is the only source of truth for AI Gateway docs.
2. We will update docs in `api7/ai-gateway`, not in `api7/docs`.
3. `api7/docs` PR #1659 is the docs-site integration PR. It makes the docs site render AI Gateway content from `api7/ai-gateway/docs`.
4. `api7/ai-gateway` PR #469 is the publish-trigger PR. It dispatches docs-site production publishes after `docs/**` changes land on `ai-gateway/main`.
5. `api7/ai-gateway` PR #493 is the current content-release PR. We are polishing and verifying the docs content here before the release sequence completes.
6. Keep PR #1659 and PR #469 open and unmerged until the docs content is satisfactory for release.
7. Continue updating docs content in `api7/ai-gateway/docs` and fold those changes into PR #493 until the docs are ready.
8. Because PR #469 remains unmerged during this period, merged docs changes in `ai-gateway/main` will not publish to the public docs site yet.
9. When the docs content is ready, refresh or rebase the three PRs as needed, then complete the release sequence with PR #493, PR #1659, and PR #469.
10. The landing page at `docs/index.mdx` came from separate previous work. Do not change it during the docs-polish pass unless the user explicitly asks for landing-page edits.
11. Do not commit or push docs-polish changes to PR #469 or its branch `ci/publish-docs-site-on-docs-changes`. Docs-polish commits belong on PR #493, branch `docs/quickstart-refresh`.
12. Resume docs-polish work from `/Users/test/Desktop/api7/ai-gateway-docs-quickstart` on branch `docs/quickstart-refresh`, and keep changes local unless the user explicitly asks to commit or push.
13. The AI Gateway landing page and `docs/quickstart/quickstart.md` are now scrutinized and frozen. Do not edit either page unless the user explicitly asks for a targeted change.
14. After this commit/push, resume page-by-page review at `docs/quickstart/first-model-first-key-first-request.md` (`Understand Admin Resources`). Start by giving suggestions for whether and how to update it before editing.

## Resume Checklist

- Check `docs/architecture/overview.md` before release. It has already been reviewed once for external-user tone, but keep it in the final sweep.
- Recheck the provider-upstream docs in the final release sweep, especially `docs/integration/upstream-openai-compat.md`. A focused external-doc style sweep has been done, but keep this section in the final page-by-page audit.
- Avoid meta-doc framing such as "Main docs describe verified behavior. Use Feature Status..." unless the page has a concrete external-user reason for saying it.
- The generated Admin API page at `/ai-gateway/reference/admin-api` is served by the docs-site static OpenAPI JSON in local preview. The AI Gateway OpenAPI source has been cleaned for title case and external-user wording; refresh the docs-site static OpenAPI asset before release verification so the rendered Scalar page reflects it.
- The local docs-site preview also renders generated legacy `/ai-gateway/llm-providers/*` compatibility pages. The docs-site integration creates these one-sentence "provider guide moved to ..." pages from `docusaurus.config.js`; they are not source docs in the current `api7/ai-gateway/docs` tree and are not listed in `sidebars.ai-gateway.js`. Treat them as docs-site compatibility routes, not PR #493 source-content polish scope, unless the user explicitly asks to change docs-site integration behavior.
- Continue the remaining docs polish locally only. Do not stage, commit, or push by default.
- The next active review target is `docs/quickstart/first-model-first-key-first-request.md`; compare its flow against the now-frozen Quickstart before proposing edits.

## Local Preview Strategy

- For text-only edits under `api7/ai-gateway/docs`, use the fast preview path:
  copy the changed source file into the matching file under
  `/Users/test/Desktop/api7/docs/.docusaurus/ai-gateway-docs/` and let the
  running Docusaurus server hot-rebuild.
- Use a full preview restart when sidebar items, navigation, generated docs
  wiring, or copied-docs setup changes.
- Before any commit or push, do a clean preview refresh/rebuild instead of
  relying only on the fast copied-file path.
- Keep the docs preview on `http://127.0.0.1:3002` while AISIX uses
  `3000` and `3001`.

## Source Formatting Preference

- Do not manually wrap ordinary prose into short lines. Keep sentences and short paragraphs readable in source, even though Markdown rendering is unaffected.
- When editing docs, avoid reflowing unrelated existing text. For touched prose, prefer natural paragraph lines over hard-wrapped 80-column style.
- Keep code blocks, tables, lists, and frontmatter formatted for readability and correctness.

## Scope Of This Note

- This note is only meant to preserve the update and merge strategy for the current docs effort.
- It is temporary and can be removed after the docs release process is complete.
