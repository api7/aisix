# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Bias toward caution over speed; for trivial tasks, use judgment. Merge with project-specific instructions as needed.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

- State assumptions explicitly; if uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name what's confusing, and ask.

**Scope a customer-driven issue from the requester's quoted words, not from the rest of the body.** These issues are filed in `api7/AISIX-Cloud` (a gateway change is routinely one half of a cross-repo issue) and are drafted with AI help, and the expansion only ever runs one way — it never states the ask smaller. One customer question ("how do I tell text / image / video requests apart in api7 or SLS, and how do I query it?") arrived as AISIX-Cloud#1461: five proposed wire fields, a modality taxonomy and nine acceptance checkboxes. Working from the body as written therefore over-delivers by default — that is the failure this rule exists to prevent, not a description of what to do.

- **Find the quoted words by content, not by heading.** That repo's `feature_request.yml` collects them under 「原始需求（逐字）」, but blank issues are enabled and `gh issue create` bypasses the form, so most issues never went through it and one that imitates the template titles the section however its author chose. Read for the part that quotes the requester; only when the issue genuinely carries none, ask — never reconstruct it from the body.
- Everything else in the issue — the user story, the proposed solution, the acceptance checkboxes, the pasted industry comparison — is analysis by whoever filed it, and loses to the quoted words on conflict. Leads, not a contract. The comparison against mainstream gateways that section 7 mandates is your own, run now against upstream HEAD, never the one pasted in the issue.
- **Before writing code, state three things and get them confirmed: the quoted ask, what you will build, and what you are dropping as expansion.** Dropping silently is as wrong as building it. "How much of the acceptance criteria should I do?" is the wrong question — it takes the expansion as the baseline and negotiates only its volume.
- None of this weakens delivery once scope IS agreed: the agreed scope still ships whole, across every surface it touches.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked; no abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it. Would a senior engineer call it overcomplicated? Then simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

- Don't "improve" adjacent code, comments, or formatting; don't refactor what isn't broken; match existing style, even if you'd do it differently.
- Remove imports/variables/functions YOUR changes orphaned — but don't delete pre-existing dead code; mention it instead.
- The test: every changed line traces directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

- Turn tasks into verifiable goals: "add validation" → write tests for invalid inputs, then pass them; "fix the bug" → write a reproducing test first; "refactor X" → tests pass before and after.
- For multi-step tasks, state a brief plan as `step → verify: check` lines.
- Strong criteria let you loop independently; weak criteria ("make it work") require constant clarification.

## 5. Testing Discipline

**E2E tests are the highest-priority signal. Cover the real user journey. Never silence failures.**

- Prioritize E2E over unit/integration when coverage is limited; design cases around the user's real path and don't skip steps.
- For any frontend UI, write E2E with **Playwright**, issuing requests to a **real backend API** — no stubbed network, fixture servers, or intercepted responses.
- Don't use mock data in E2E; run against real data and services. If mocking seems unavoidable, stop and get human confirmation first.
- Never skip, disable, or `.only` a test to go green — investigate the underlying bug instead.
- **Before trusting a check, name what would make it fail — and if nothing would, it is not a check.** A verification that cannot fail is worse than none, because it reports green and buys confidence for a premise nobody examined. The shape recurs here and it never looks wrong: an assertion on a field whose ABSENT value equals its expected value (a plain `int` compared `!= 0` when a missing JSON key decodes to `0`, and `0` is also the only correct answer); a `toContainText` whose expected string is a substring of the text that would prove the opposite; a requiredness test that reads only the strict schema, which shows a field is required SOMEWHERE and can never show it is required ONLY there; a `diff a b` where `a` is a symlink to `b`. The remedy is mechanical rather than clever: mutate the thing under test — delete the line, feed the value that must be rejected, remove the field from the wire — and require the check to go RED before you rely on it going green.
- **`cargo check --workspace --all-targets` compiles integration tests without running them.** It is not a substitute for `cargo test` over the packages a change can reach, and a change to a shared schema, model, or trait reaches far more packages than it edits — `git diff --name-only` understates the blast radius by design. A suite that compiled and never executed is the same false green as the bullet above, and CI is where you will find out.
- E2E tests must be **source-blind**: design assertions from scenario reasonableness alone, never by reading product source to pick expected values. The test verifies the observable contract, not the implementation.
- **If an E2E test fails, the default conclusion is a bug in the code, not the test.** Fix the product; don't weaken the assertion, relax the expected value, change the scenario, or read the source to explain it away. Only change a failing test if a human confirms the scenario is invalid.
- If a test case itself looks wrong, flag it and ask a human — don't silently delete or rewrite it.

## 6. Research Discipline

**Verify against primary sources. Never guess or infer product behavior.**

- Confirm details only via **official documentation** and **source code**; don't speculate or fill gaps with assumptions.
- If docs and source don't answer it, say so and ask — don't invent an answer.
- Cite the specific doc URL, file path, or commit/version for any claim about third-party behavior.

## 7. Reference Implementations Before Building

**Before implementing any feature, study how the established players did it — don't drift from the ecosystem.**

Before writing the first line of a new feature, read:

- **Mainstream AI gateway implementations** — research at least three established, mainstream AI gateways and study how each solves the problem. When in doubt about a request/response transform, read their sources for the same provider + endpoint and compare.
- **Upstream provider docs** — the authoritative spec for any endpoint: OpenAI <https://platform.openai.com/docs/api-reference>, Anthropic <https://docs.anthropic.com/en/api>, Gemini <https://ai.google.dev/api>, DeepSeek <https://api-docs.deepseek.com>, Bedrock <https://docs.aws.amazon.com/bedrock/>.
- **Upstream SDK source** — the real contract when docs are vague (`usage` sub-fields, streaming event order, error envelopes): the official `openai-python`, `anthropic-sdk-python`, and `generative-ai-python` repos.

The rule:

- For any new endpoint, request transform, or response normalization, compare how at least three mainstream gateways approach it, cite one upstream-spec source, and summarize that comparison — plus where your design lands — in the design notes / PR description.
- If your design diverges from how those gateways solve it, name the divergence and justify it ("they do X but we need Y because of Z" — not "I didn't know they handled it").
- For any field, header, or status code you emit or parse, cite the upstream doc URL or SDK file/line. Don't invent names the ecosystem has already chosen.
- Refer to other products generically in shipped artifacts (code, comments, commit messages, PR descriptions) — describe the approach, not the brand. Keep brand-specific notes to internal design discussion.

## 8. Independent Audit Before Merge

**Every PR pushed must be reviewed by an independent audit agent. Merge is blocked until all HIGH/MEDIUM findings are resolved or explicitly justified.**

After every `gh pr create` or force-push, spawn a fresh `general-purpose` Agent with no shared context. Brief it cold with the PR URL and the contract the PR claims to pin. Treat each angle as blocking:

- **Correctness** — does it do what the description claims? Would a real regression fail the assertions?
- **Reliability** — races, error handling, retry/timeout, propagation timing on slow CI.
- **Security** — auth/authz, input validation at boundaries, injection, header forwarding (and what's deliberately not forwarded).
- **Sensitive-info leakage** — secrets in logs/errors, internal taxonomy or upstream-provider details in user-facing fields, tokens/PII in fixtures.
- **Breaking changes** — API shape, on-disk format, wire protocol, default shifts; if breaking, is it gated/versioned?
- **E2E coverage** — the user-visible contract, not just unit happy-path; mocks tight enough that a regression on the unverified side can't sneak through.

Output HIGH/MEDIUM/LOW per finding with **concrete suggested code**, not vague "consider". **Merge gate:** every HIGH and MEDIUM is either fixed in code or explicitly justified in the PR (e.g. "feature gap, filed as #N, agreed not to block"); silent merge is not enough. For findings that surface gateway/product-behavior gaps, file separate issues and link them. Self-review misses the author's blind spots — an independent agent catches them.

## PR Batching — One PR per Session by Default

This repo is developed end-to-end by agents — no human reviewer needs small review units — and CodeRabbit bills and rate-limits **per PR**. Fanning one effort into many small PRs burns review quota and stalls the session on throttled bot reviews. Keep ONE open PR per session and push follow-up and related work to it as additional commits (rule and doc riders included) instead of opening another. Split only when a fix must merge independently ahead of the batch, or when the user asks for separate delivery.

## A Behavior Change Is Stated in the PR Description

**Release notes are assembled at release time from what the release contains, read as PR bodies rather than commit subjects — so a behavior change written nowhere but the diff never reaches users.** State it plainly in the description: what changed, what an existing configuration or caller experiences after upgrading, and whether anything has to be edited.

This covers a validation that got stricter (an existing resource that no longer saves), a filter that got wider (a configured value that silently stops taking effect), a moved default, and any wire or schema reshape. Do not open a separate tracker for it — the PR body is the record.

## Handler Families Stay in Lockstep — Fix the Whole Class

**The client-facing endpoint handlers come in families that share dispatch, auth, routing, telemetry, and guardrail logic — `/v1/chat/completions`, `/v1/messages` (+`count_tokens`), `/v1/responses`, plus embeddings/rerank/audio/images and the jobs surface (files/batches/fine-tuning). A bug or feature landed on one almost always applies to the others, and a gap on the unfixed siblings is SILENT: nothing errors, the behavior just quietly degrades.**

- When you touch a per-request mechanism (a runtime metric, a limit, an auth check, a usage emission, header threading), grep the offending call/pattern across the whole crate and wire **every** sibling path in the same PR — both streaming and non-streaming branches — or state explicitly in the PR which sibling is deferred and why, and file the follow-up issue immediately.
- "Documented follow-up" without an issue is how gaps rot: it lives in one PR description and no one ever comes back.
- **An emit function on `Metrics` with no caller is invisible to every check we run.** Its methods are `pub`, so dead-code analysis never fires; unit tests call it directly and pass; the only symptom is a series that never appears in a scrape, which is indistinguishable from "no traffic yet". A metric family is shipped when an **e2e asserts it in `GET /metrics`** after driving real traffic — not when `Metrics` can emit it. (Twice now: `record_proxy_request` until #888, then `record_deployment_request` + `record_routing_fallback` until #972.)
- Test coverage must include each wired endpoint, not just chat: an e2e that only drives `/v1/chat/completions` will stay green forever while Anthropic-SDK (`/v1/messages`) and Codex (`/v1/responses`) traffic silently misbehaves.
- **Never gate the guardrail chain on having found text to scan.** "Nothing to scan" is not "nothing to decide": a `kind: custom` row's verdict can be independent of the text, so a call site that skips the chain when its collect walk came back empty converts an operator's block rule into a silent allow. The call site always consults the chain; only a guardrail kind may decide it needs text, and every remote kind already self-guards, so consulting costs no provider round-trip. `crates/aisix-proxy/src/guardrail_coverage.rs` enforces this: it parses the routing table out of `build_router`'s own source, so a new route fails the census until it declares a posture, and every surface declared `Enforced` is driven against an unconditional-block guardrail.
- Prefer hoisting the shared logic into one chokepoint (e.g. `resolve_attempt_models`) so the family can't drift again.

(Two recurrences of the same lesson: #471 — a Model-Group dispatch fix landed only on `/v1/messages` while `/v1/responses` and `count_tokens` had the identical gap; then #715 — `least_busy`'s in-flight counter shipped fed by chat.rs only (#684 left messages/responses as an un-filed "follow-up"), so the strategy silently degraded to declaration order for Claude Code / Codex traffic until #716. The EWMA for `least_latency` (#682) wired all three endpoints at once and never had this problem — that's the standard.)

## Usage Converts at the Client Boundary, Never at the Telemetry One

**`UsageStats` carries two *disjoint* cache representations with opposite arithmetic, and a cross-protocol handler must convert one for the client while leaving the UsageEvent in the upstream's own shape.**

- OpenAI-shape upstreams report the prompt-cache hit as `cached_prompt_tokens`, a **subset already inside** `prompt_tokens`. Anthropic-shape upstreams report `cache_creation_tokens` / `cache_read_tokens` as counters **on top of** `prompt_tokens`. Each provider's wire fills one family and zeroes the other; never map between them inside `UsageStats`, and never add `cached_prompt_tokens` into a total (`total_tokens_with_cache` deliberately takes only the additive pair).
- **The client-facing renderer converts, through the projections on `UsageStats` — never by reading its fields directly.** `openai_prompt_tokens` / `openai_cached_tokens` / `openai_total_tokens` and `anthropic_input_tokens` / `anthropic_cache_read_input_tokens` / `anthropic_cache_creation_input_tokens` are defined together, on the one type that carries the ambiguity, precisely so a new inbound protocol cannot pick up one direction's arithmetic and leave the other on the old semantics. They fold rather than pick a non-zero family, which is also what keeps an **ensemble** aggregate right — `saturating_add` can merge an OpenAI-shape member with an Anthropic-shape one, and only the additive form reports both members' cache hits. Emit a cache counter only when non-zero: a fabricated `0` reads as "the provider reported no hit" rather than "the provider doesn't report this".
- **`total_tokens` on an OpenAI-shape response is recomputed, not echoed from the upstream.** OpenAI clients decompose it as `prompt + completion`; passing through a total built under different accounting is how #1447 reached the wire. An upstream counter we *do* understand belongs in a field of its own instead — Gemini's `thoughtsTokenCount` folds into `completion_tokens` with `reasoning_tokens` naming the subset, rather than inflating a total that no longer adds up.
- **The UsageEvent does NOT convert.** It carries the upstream's own shape, so one upstream call bills identically whichever inbound protocol addressed it — and because cp-api prices it that way: it charges `prompt_tokens - cached_prompt_tokens` at the prompt rate plus `cached_prompt_tokens` at the cache-read rate, and **rejects an event whose `cached_prompt_tokens` exceeds its `prompt_tokens`**. Converting on this side is a silent cost error, not a display choice.
- A dropped counter here is invisible from the dashboard: no cache detail is indistinguishable from a provider that never cached, so nothing looks wrong while the whole prompt bills uncached.
- **Prometheus does not convert either, and keeps the two families as separate counters** — `aisix_llm_cached_input_tokens_total` for the subset shape, `aisix_llm_cache_read_input_tokens_total` / `..._cache_creation_input_tokens_total` for the additive pair. Merging the two reads into one counter looks tidier and makes every cross-protocol ratio wrong, because the correct denominator (`input + cache_read + cache_creation`) and numerator (`cached + cache_read`) both need to know which shape a sample came from.

(Lesson from AISIX-Cloud#1405: the `/v1/messages` → OpenAI-compatible bridge parsed `prompt_tokens_details.cached_tokens` correctly into `UsageStats` and then dropped it at **both** exits — `AnthropicUsageMetrics` had no field for it and the Anthropic renderer emitted `input_tokens` / `output_tokens` only. `/v1/chat/completions` had carried it since #542.)

(And from AISIX-Cloud#1447, which is why the arithmetic now lives in one place: the mirror direction this section already warned about — "the reverse renderer faces the mirror problem" — was still unwritten a release later. An Anthropic upstream's `input_tokens: 40` reached OpenAI clients verbatim beside a total that had folded all 140 input tokens in, so `/v1/chat/completions` reported `40 + 10 = 150` and dropped the cache hit, while `/v1/responses` reported `cached_tokens: 70` against `input_tokens: 40` — a subset larger than its superset. Knowing the rule was not enough; a renderer that reads the fields can always get them wrong, so it no longer reads them.)

## A Config Knob Isn't Shipped Until the Control Plane Exposes It

**A user-configurable data-plane feature is NOT delivered when the Rust side works — it's delivered when a user can reach it through the control plane. DP-only is a half-feature nobody can turn on.**

This repo reads its config from etcd, but users never write etcd directly — the **control plane** (`api7/AISIX-Cloud`, a separate repo) is the only writer. That CP is **not a passthrough**: it validates every resource against a **closed** OpenAPI schema (`AISIX-Cloud: openapi/cp-admin.yaml`) and its validator **rejects any field or enum value the spec doesn't list, before it is ever written to etcd**. So the moment you add a new config surface here — a new `RoutingStrategy` variant, a new per-target field, a new resource knob, a header-driven behavior a user is expected to configure on a resource — a DP that happily reads it from etcd is still **unreachable**, because the CP will never let that value through and no UI offers it.

- **Treat any DP PR that adds or extends a user-facing config surface as automatically implying a paired CP PR.** The DP change is not "done" on its own; it's one half of a cross-plane feature. Before calling a routing/resource/config feature complete, confirm the CP can accept and persist the new shape.
- **"Done" for such a feature spans four CP layers**, none optional: (1) the `cp-admin.yaml` schema (new enum value / field) **and its regenerated Go bindings**; (2) the Go typed model + request validation + etcd projection under `internal/cpapi/resources/`; (3) the dashboard form field(s) under `dashboard/` **plus `messages/en.json` + `zh.json` i18n**; (4) paired tests — CP↔DP Go integration in `e2e/cases/` and Playwright for the UI.
- **If you can only do the DP half in this PR, say so and file/track the CP issue in the same breath** — never let the umbrella task close on DP-only work. A merged DP PR with no CP counterpart is a latent gap, not a shipped feature.
- Pure internal DP mechanics (a new algorithm with no user-set config, an observability metric, an internal refactor) don't need CP work — this rule is specifically about **user-configurable** surfaces a customer must be able to set.

(Lesson from AISIX-Cloud#873 routing: `least_cost` / `least_latency` / `least_busy`, per-target `tags`, and `sticky` canary all shipped DP-only across #681/#682/#684/#686/#687 while `cp-admin.yaml` still pinned the closed `[round_robin, weighted, failover]` enum and the dashboard had no fields — so none of it was actually usable until the matching CP integration landed. The meta-repo `AGENTS.md` carries the same rule for cross-plane agents.)

## The Resource Model Is Canonical in cp-admin.yaml

**When this repo and the control plane disagree about a resource field's name, enum values, or nesting, the control plane's spec (`AISIX-Cloud: openapi/cp-admin.yaml`) wins by definition — this repo converges to it.**

- Adding or renaming a user-facing resource field starts by defining its name and shape in `cp-admin.yaml` (in the paired CP PR — see the config-knob rule above); the Rust model then implements exactly that name. The naming decision happens once, in the spec — never independently here.
- Renames converge with `#[serde(alias = "…")]` so stored documents and existing callers keep loading through the deprecation window; never hard-rename a shipped field in one step (an unreleased field with no consumers may rename outright, as #657 did). Regenerate `schemas/resources/` afterwards (`cargo run -p aisix-core --bin dump-schema`).
- Exactly four divergence axes are registered as intentional and allowed: reference style (names here vs UUIDs in the CP), tenancy scoping (flat here vs org/environment there), credential custody (`key_hash` in documents here vs server-generated plaintext-once there), and CP-derived fields (`cost`, `telemetry_tags`). Anything else that diverges from cp-admin.yaml is drift — the planned cross-plane contract check will fail it.
- Why the CP spec and not this repo's schemas: the CP is spec-first behind a closed validator (its spec already is the authoritative field shape on that side), the spec renders into the customer-facing API reference, and this repo's schemas are generated from the implementation — a schema that follows the implementation cannot lead it. Naming drift has already cost real churn: #644 (the generated schema advertised `rps`/`rph` the validator rejected) and #657 (a wire-breaking rename because the field was named DP-first).

## Model Kinds Stay in Lockstep — Two Identities, and the Sub-Dispatch Bypasses

**A Model is one table but five kinds (`direct` / `routing` / `ensemble` / `semantic` / `embedding`, plus wildcard display-name aliases), and every request carries TWO model identities: the caller-addressed entry (may be a virtual parent) and the dispatched target. For direct models they coincide, so a mechanism built and tested against direct models silently never decides the composite case — the most-repeated silent-bug class here (#962, #1087, #1237, #1267, #786).**

The five kinds are the cross-plane taxonomy (cp-admin.yaml `kind`); this repo's `model_one_of` implements four dispatch shapes, with `embedding` carried as the `embedding` block on the direct shape (`models/model.rs`). For a wildcard-served request three names are in play — the caller-minted alias, the wildcard row's `display_name`, and the concrete upstream model — and "caller-addressed entry" means the **resolved row** for the gate/metric family: inline rate-limit buckets, Prometheus metric labels, and health keys use the row's `display_name`, not the caller-minted string (#959). The `upstream_model` half is caller-minted too — `resolve_model` hands dispatch a synthetic Model whose `model_name` is the caller's substituted suffix — so a metric label taken off a resolved Model must go through `usage_attr::metric_model_label_pair`, which collapses BOTH halves to the row's configured identity. Usage-event attribution (`requested_model`) and `model_name` policy conditions intentionally keep the caller-supplied name.

- When you touch a model-keyed mechanism (a limit, a guard, an ACL, a config knob, usage/metric attribution, cache keying), answer in the doc comment: does it key on the **requested** entry, the **dispatched** target, or **both**, and what is the behavior for each of the six shapes.
- The per-target invariant (`crates/aisix-proxy/AGENTS.md`: "a per-model gate binds each target") is written around `resolve_attempt_models` — the routing-group trunk. **Ensemble panel/judge (`ProxyModelCaller::call`, the streaming judge) and semantic targets (`semantic::resolve`) bypass that trunk**, so a gate wired only into the trunk is silently absent there (the 2026-08 audit found member IP allowlist, health consumption, and retries all missing on the semantic path for exactly this reason — #958). A new per-target gate must be wired into the sub-dispatch paths too, or explicitly deferred with a filed issue. Prefer routing every dispatch through one shared chokepoint so the family can't drift.
- **Strict writes, lenient loads.** `model_one_of` has two variants: the **strict** schema (declarative resources file, the published `schemas/resources/model.schema.json`, every strict validator consumer) forbids a knob a kind never resolves — accepted-but-unread config is the #962 class; the **lenient** loader keeps the base XOR so stored rows written by an older build still load, with `Model::strip_kind_inapplicable` dropping the dead knob and reporting it as `inapplicable:<field>` through the partial-compat channel. The two lists MUST mirror each other exactly (strict-forbidden ⇔ lenient-stripped) — a field forbidden-but-not-stripped half-honors; stripped-but-not-forbidden vanishes on load while the write path accepts it. A knob is enforced exactly as written or rejected, never half-honored (#963).
- **Never make a field of a projected resource required at the TYPE level.** Requiredness belongs in the strict schema (`require_property` in `models/schema.rs`), never in the struct: the loader validates leniently and then deserializes, and a row it cannot deserialize is **skipped entirely** (`aisix-etcd/src/loader.rs`). Skipping is survivable for a resource the request path treats as optional, but an `api_key` row that fails to load stops authenticating **every** kind of traffic, not just the feature whose field changed — a far worse outcome than the field defaulting. So a new non-`Option` field, or one that loses `#[serde(default)]`, silently turns every already-projected row into a dead one. Give it a serde default whose meaning is fail-closed, and add it to `required` in the strict schema so the write path still refuses to guess. The control plane must also re-emit the affected collection once (`ReprojectMcpAclOnce` is the pattern) — the stored shape changed, but nothing else re-projects a row whose *content* did not. (Lesson from #993: `allow` was required at the type level in #992, which made every key still projected as `mcp_access: {"mode": "inherit"}` unloadable.)
  - **"In the strict schema" is not automatic — check that the resource HAS a strict/lenient split, and prove the split with a lenient assertion.** A `*_root_schema` that ignores its `strict` flag, or takes none, hands the same `required` list to `LENIENT_SCHEMAS`, and then the requirement you meant for the write path is deleting stored rows instead. `guardrail_root_schema` took no flag until AISIX-Cloud#1467, so `embedding_model` and the `custom` kind's `script` had both been required of the loader for their whole lives, each under a doc comment saying the opposite — though only `embedding_model` wanted the split, since a scriptless `custom` row screens nothing whichever way it fails and rejecting it is what makes it visible. Neither was caught, because the guarding test read the strict schema and asserted the field was required — which shows a field is required SOMEWHERE and can never show it is required ONLY there. The assertion that pins a split is a document validated against BOTH sets: rejected by `validate_<resource>`, accepted by `validate_<resource>_lenient`.
- **Never change a projected field's shape or value domain in place — the previous release must keep loading the row.** The supported upgrade order is control plane first, then data planes, with a window that is minutes long in practice but must still be survived; a released DP is immutable, so whatever the new CP projects must still parse one release back. Unknown *fields* are tolerated by design at every depth, nested config objects included, and reported through the partial-compat channel — but only from the release that opened the read schema all the way down (#1014); a 0.10.0 or older binary drops the whole row for a field added inside a guardrail or exporter config block, and always will, because a released binary cannot be taught anything. A malformed *known* field — a reshape, a lost default, a new enum value — fails the row and the loader skips it whole on every release (the blast radius of the rule above). Two constraints keep the tolerance real on this side: unknown-field strictness lives in the strict schema (`models/schema.rs`), never in a nested `#[serde(deny_unknown_fields)]` — the loader deserializes the same types and cannot opt out of a type-level guard; and `serde_ignored` never fires inside serde-buffered content (a `#[serde(flatten)]`ed tagged enum, an untagged variant), so those resources take their report from `unknown_field_paths` instead of loading silently. A reshape therefore ships under a NEW field name (the old one is never reused), or as a **same-name dual-generation document** when the old and new keys don't collide: the CP emits one document valid for both generations, and this side carries a consumed-and-ignored tombstone for the old selector — `#[serde(default, rename = "<old key>", skip_serializing)]` + `#[schemars(skip)]` so the strict write path still rejects it, with a `COMPAT-SINCE:` marker naming the retirement condition (see the next section). A new enum value in an existing field cannot be made safe DP-side at all (lenient parsing keeps enums closed — it row-kills every older DP), so the paired CP PR must gate it behind `dpCompatGate` until the fleet minimum reads it. Whenever new semantics are invisible to the old release, verify the old default direction there: fail-closed or no-op is required; if it is fail-open, the CP must project an old-shape tombstone at the most restrictive value.
- **Size the mitigation to what the window costs.** The window is minutes, not a season — a control plane and its data planes are upgraded back to back — and anything that merely degrades in it self-heals on the next DP build: a knob not yet honoured, a field reported instead of read, a resource serving its previous semantics. That does not earn a dual-generation document, a tombstone, or holding a finished feature back a release; ship it and let the CP warn at save time (`dpCompatGate`). Three things do not self-heal, and keep the heavy technique regardless of how short the window is: a run-once migration, which rewrites stored data permanently; a security control the old binary cannot see, where the exposure has already happened however briefly (fail-closed or an old-shape tombstone, never bare degradation); and a **core carrier row** — `api_keys`, models, provider keys — where a skipped row is not a degradation but an outage, since that key authenticates nothing at all until the DP is upgraded. If a change is none of the three, the proportionate mitigation is the report.
- **`ensemble` is an experimental surface.** Its known parity gaps — member `allowed_cidrs`/guardrail/cooldown/health consumption, Prometheus token+spend attribution, response caching, parent-level generic knobs — are deliberate TODOs under a single future design pass. Do NOT piecemeal-fix one gap ahead of that pass, and do NOT re-audit them as fresh findings. (The one exception is a marshal-family or shared-chokepoint change where covering ensemble is a one-line parallel edit, e.g. projecting an entry-level field the DP already enforces.)
- Adding a NEW kind = sweeping every existing model-keyed mechanism against it (grep the kind predicates in `models/model.rs`; every hit re-answers the questions above).

## A Guardrail's Scope Is Its Attachments and Nothing Else

There is no implicit scope. `build_index_from_snapshot` puts a guardrail in the index once per enabled attachment, and a guardrail with none governs nothing — on either source. Declaring one without an attachment is not an error, because its scope target may simply have been deleted.

It used to be the opposite: a guardrail carrying ZERO attachment rows was applied to the entire environment at priority 0, a rolling-upgrade fallback from the window where this side read attachments and the control plane had not written any yet. Keying on the ABSENCE of rows is what made it dangerous — deleting the one model a guardrail was scoped to removed its last attachment and thereby WIDENED the guardrail to all traffic, silently, on a security control (AISIX-Cloud#1450). Do not reintroduce a default of any shape, and in particular do not give the resources file one "for convenience": the file source declares `guardrail_attachments` exactly as the control plane projects them, references written as the identity each collection is keyed by, so both sources answer the scoping question the same way. A convenience default in one of them is a divergence a standalone user cannot see.

The export follows from that: `aisix export` emits every guardrail plus every attachment it can name. An attachment it cannot name — `team`, which has no file collection, or a `scope_id` missing from the snapshot — is dropped with a warning rather than emitted dangling, because losing a scope makes the guardrail govern LESS. Inventing one would make it govern more.

## Time-Boxed Compat Code Carries a `COMPAT-SINCE:` Marker

**A shim meant to live for exactly one release gets a machine-readable deadline, not a sentence in a comment. Prose deadlines never come due — both of the ones this repo was carrying were found by grep, not by process.**

Put the marker on the compat code, in whatever comment syntax the file uses:

```text
COMPAT-SINCE: 0.10.0 #1009 — what this tolerates and why it can go
```

Anchor on a release that has **already shipped**, never on a guess at the next version — "due at 0.11.0" fails silently when the next release turns out to be `1.0.0`. The gate rejects an anchor that is not in `git tag`, so a predicted number cannot be written in the first place.

`crates/aisix-core/tests/compat_debt.rs` is the gate. It runs in the required `rust unit + coverage` job on every PR, and fails once a stable tag with a higher `MAJOR.MINOR` than the anchor exists; release candidates never count as shipped, and a patch on the anchor's own line does not come due. When it fires, either remove the code and its marker and close the tracking issue, or re-anchor deliberately to the newest release and say why in the issue. Full rules, including how a `release/X.Y` maintenance line is scoped, live in that file's module docs.

## An Absent Config Block Means Off

**An optional block on a projected resource must resolve, when absent, to the feature being off — and so must the block's own `enabled` flag when the operator wrote the block and omitted the flag.** A default that switches behavior on instead is not visible from either end: the console renders the block from the stored document and shows a block nobody wrote as disabled, so the first sign is the behavior.

Three things this does not govern. The other knobs inside a block whose `enabled` was set. A resource ROW's own `enabled`, which every collection defaults to `true` — creating the row is the opt-in. And a fail-closed default, which restricts rather than enables. The startup `Config` is a separate contract: an absent `observability:` block still binds the metrics listener, deliberately.

Say what omitting the block means in `cp-admin.yaml` — the spec the published API reference renders, and the one artifact both planes read. (Precedent: `cooldown` — AISIX-Cloud#1499.)

## A Retired Startup-Config Key Is Tombstoned, Never Deleted

**`Config` and its blocks are `deny_unknown_fields`, so deleting a startup-config field stops every gateway whose `config.yaml` still carries it from booting** — including everyone who copied it out of `config.example.yaml` and never turned it on. Removing the key is a far larger break than the dead setting ever was.

Retire it instead: drop the block from `config.example.yaml` / `config.managed.yaml` so nobody discovers it again; keep the field parsing, with a doc comment saying it is consumed and ignored; make it `Option<…>`, because serde cannot otherwise tell a config that omits the block from one that wrote it out with default values; and name it in the block's `retired_settings` (see `ObservabilityConfig`) so boot warns once per key the operator actually wrote, pointing at where the capability really lives. Warn on the key being **present**, not on `enabled: true` — the copied-in disabled block is precisely the one to delete. Parsed-and-ignored is fine; parsed-and-silent is the bug (AISIX-Cloud#1380).

## AISIX Product Terminology

Use the following terms in public prose, generated API descriptions, release
notes, and configuration comments:

- **AISIX AI Gateway** is the open-source product. Use **open-source AISIX
  gateway** when the distinction from AISIX Cloud matters; after establishing
  the product, use **AISIX gateway** or **gateway**.
- **AISIX Cloud** is the commercial product umbrella. **Hybrid Cloud** is its
  API7-hosted control-plane option, and **On-Premises** is its customer-hosted
  control-plane option. Do not present **AISIX Hybrid Cloud** or **AISIX
  On-Premises** as separate products.
- An AISIX gateway is a **data plane** only within AISIX Cloud architecture. Do
  not call an independently operated open-source gateway a data plane.
- Do not use **standalone gateway** as a product label. `standalone mode` and
  `managed mode` remain valid when describing runtime behavior.
- Avoid unqualified **self-hosted**. Name the component being operated, such as
  the open-source AISIX gateway, the On-Premises control plane, or a self-hosted
  upstream service.
- The Dashboard is the control plane's user interface, not the whole control
  plane. Live AI requests pass through the gateway directly and do not pass
  through the AISIX Cloud control plane or API7.

## Documentation Lives in api7/docs

**User-facing documentation is maintained in the `api7/docs` repository (published to <https://docs.api7.ai/ai-gateway/>), not in this repo.**

- This repo's source tree intentionally carries **no** user-facing doc pages — they were migrated to `api7/docs` so one site stays authoritative and never drifts from a stale in-repo copy. Do not add or keep prose docs under `docs/` here.
- When a feature needs documentation, add or update the page in `api7/docs` and link to its `docs.api7.ai` URL (e.g. from the README) — never re-introduce a `docs/*.md` page in this repo, even temporarily or "just for now".
- Only user-facing *prose* moves out. Code-level doc comments stay with the code — including the generated API reference below.

## Generated API Documentation

**Some source comments are rendered into user-facing API references.**

When editing Admin API resource models under `crates/aisix-core/src/models` or OpenAPI assembly in `crates/aisix-admin/src/openapi.rs`:

- Write descriptions as public API reference text, not internal implementation notes.
- Avoid internal shorthand such as DP, CP, kine row, wire shape, mock server, bridge dispatch, or issue-only context.
- Avoid excessive inline code. Use it only for exact field names, enum values, routes, headers, environment variables, and literal response values.
- Do not describe stable defaults only in prose. Expose them as OpenAPI `default` values when the runtime behavior has a fixed default.
- For computed fallback behavior, describe what happens when the field is omitted instead of calling it a schema default.
- Regenerate resource schemas with `cargo run -p aisix-core --bin dump-schema` after changing model comments.
- Verify the generated Admin API OpenAPI with `cargo run -p aisix-admin --bin dump-openapi > /tmp/admin-api.openapi.json` after changing Admin API routes, OpenAPI metadata, or generated descriptions.
- Preview or inspect the served OpenAPI when changing generated descriptions.

---

**Working if:** fewer unnecessary diff lines, fewer overcomplication rewrites, and clarifying questions come before implementation rather than after mistakes.
