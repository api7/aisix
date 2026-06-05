# Temporary AI Gateway Docs Style Reference

Date: 2026-06-01

Use the polished APISIX docs in `/Users/test/Desktop/api7/docs` as the style reference for this docs cycle, especially:

- `apisix_versioned_docs/version-3.16.0/getting-started/get-apisix.md`
- `apisix_versioned_docs/version-3.16.0/getting-started/README.md`
- `apisix_versioned_docs/version-3.16.0/tutorials/README.md`

## Writing Direction

1. Optimize for a new user landing on the docs for the first time.
2. Prefer a clear end-to-end journey over a fragmented explanation.
3. Use concise setup steps, explicit prerequisites, and concrete verification points.
4. Keep the tone polished and product-facing, similar to the APISIX versioned docs, while staying accurate to the current AI Gateway codebase.
5. When the current AI Gateway behavior is rougher than the polished reference docs, keep the writing clean but do not invent smoother behavior than the product actually supports.

## Frozen Page References

- The AI Gateway landing page is frozen. Do not change it unless explicitly asked.
- The self-hosted Quickstart at `docs/quickstart/quickstart.md` is frozen. Use it as the current writing-style reference for Getting Started pages.
- Quickstart style qualities to preserve: concise opening, clear prerequisites, one sentence before every code block, concrete verification, no back-to-back admonitions, curl URLs wrapped in double quotes, and `## Next Steps` with a short completion statement before follow-up links.
- Avoid making SDK use sound mandatory. SDK quickstarts are optional follow-up paths for application-code examples.
- Keep source prose in natural paragraph lines; do not hard-wrap sentences during polishing.

## Landing Page TODO

- Revisit the first two sections later: the hero and request-flow diagram still read as adjacent cards. Move the flow diagram toward an open/canvas-style section so the landing page does not feel like stacked cards.

## Information Architecture TODO

- Keep Tutorials as a top-level section, not a child of How-To Guides. Tutorials are sequential learning paths or scenario walkthroughs; How-To Guides are task-oriented recipes.
- Do not duplicate tutorial-like material inside How-To Guides. If a page teaches a longer scenario end to end, it belongs under Tutorials; if it answers one operational/configuration task, it belongs under How-To Guides.
- If a future pass reintroduces a How-To Guides bucket, do not put Tutorials under it. That hierarchy makes the docs feel structurally confused to new users because tutorials teach a journey, while how-to guides answer one focused task.

## Local Preview / Testing Note

- The docs preview currently runs on `127.0.0.1:3000`, which conflicts with the documented AISIX proxy listener in the quickstart. For local E2E testing, either stop the docs preview while running the quickstart on documented ports, or run the gateway on alternate ports and translate the commands intentionally. Do not add this preview-only workaround to public user docs.

## API Reference Rule

- Admin API and API type/reference pages should point to the live generated OpenAPI/Scalar source of truth from the AI Gateway repo instead of hand-maintaining route and field inventories in prose.
- If route, schema, or status-code documentation is wrong in that generated reference surface, fix `crates/aisix-admin/src/openapi.rs`, the relevant route/resource source, or `schemas/resources/` rather than duplicating or patching the contract in prose docs.
- Dynamic resource schemas should use the generated `schemas/resources/*.schema.json` files from the AI Gateway repo. If schema docs are wrong, fix the Rust resource types or schema generation path, then regenerate with `cargo run -p aisix-core --bin dump-schema`.

## Source Character Rule

- Public docs Markdown and generated reference inputs should stay plain ASCII. Do not use section signs, curly punctuation, Unicode arrows, or similar special characters in `docs/`, `schemas/resources/`, or `crates/aisix-admin/src/openapi.rs`.
- Use text references such as `section`, `Step`, `->`, or a heading link instead of symbols such as section signs or Unicode arrows.

## Tables And Prose Rule

- Keep tables when they make comparisons, mappings, defaults, matrices, or choice guidance clearer.
- Use prose when the content is narrative, caveat-heavy, or short enough that a table adds scanning noise.
- Do not convert tables to paragraphs mechanically. Judge each table by whether it helps the external user understand or decide faster.

## Scope Of This Note

- This is a temporary note for the current docs rewrite effort.
- It exists so the style direction does not need to be restated repeatedly in this thread.
