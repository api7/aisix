# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Testing Discipline

**E2E tests are the highest-priority signal. Cover the real user journey. Never silence failures.**

- E2E tests are the most important test type — prioritize them over unit/integration when coverage is limited.
- Design test cases around the user's actual usage path. Walk through the real flow end-to-end and do not skip steps a user would take.
- For any project with a frontend UI, write E2E tests using **Playwright**.
- Do **not** skip, disable, or `.only` any test case to make things green. A passing CI is not an excuse to hide flakiness — investigate the underlying bug in the code.
- If a test case itself appears unreasonable or incorrect, do **not** silently delete or rewrite it. Flag it and ask a human to confirm before changing it.
- Do **not** use mock data in E2E tests — they must run against real data and real services. If real data is genuinely unavailable and mocking is unavoidable, stop and get human confirmation before introducing any mock.
- Every frontend E2E test **must** issue requests to a real backend API. No stubbed network layers, no fixture servers, no intercepted responses — the test must exercise the real backend end-to-end.

## 6. Research Discipline

**Verify against primary sources. Never guess or infer product behavior.**

- When researching a product or feature, confirm details only via **official documentation** and **source code**.
- Do not speculate, infer, or fill gaps with assumptions about how a product or API behaves.
- If official docs and source code do not answer the question, say so explicitly and ask — do not invent an answer.
- Cite the specific doc URL, file path, or commit/version when stating a fact about third-party behavior.

## 7. Reference Implementations Before Building

**Before implementing any feature, study how the established players did it. Avoid building a parallel reality that drifts from the ecosystem.**

This gateway sits between customer SDKs and upstream LLM providers. Both
sides have decades of accumulated wire-shape decisions, edge cases, and
backward-compat constraints. Re-deriving them from first principles is
how you ship a half-spec-compliant version that breaks on the first SDK
the customer tries.

**Before writing the first line of a new feature, read:**

- **Reference gateway implementations** — at minimum
  [LiteLLM](https://github.com/BerriAI/litellm) and
  [Portkey](https://github.com/Portkey-AI/gateway). Both are open
  source, both solve the multi-provider proxy problem, and both
  encode years of "this provider returns X but actually means Y"
  fixes. When in doubt about a request/response transform, search
  their source for the same provider + endpoint and see how they
  handle it.
- **Upstream provider docs** — for any endpoint or feature, the
  authoritative spec is the upstream provider's own docs:
  - OpenAI: <https://platform.openai.com/docs/api-reference>
  - Anthropic: <https://docs.anthropic.com/en/api>
  - Google Gemini: <https://ai.google.dev/api>
  - DeepSeek: <https://api-docs.deepseek.com>
  - AWS Bedrock: <https://docs.aws.amazon.com/bedrock/>
- **Upstream SDK source** — when the docs are vague (and they often
  are about edge cases like `usage` block sub-fields, streaming
  event ordering, or error envelope shape), the official SDK source
  is the actual contract:
  - <https://github.com/openai/openai-python>
  - <https://github.com/anthropics/anthropic-sdk-python>
  - <https://github.com/google-gemini/generative-ai-python>

**The rule:**

- Cite at least one reference implementation and one upstream-spec
  source in the design notes / PR description for any new endpoint,
  request transform, or response normalization.
- If your design diverges from how LiteLLM / Portkey solve the same
  problem, name the divergence explicitly and justify it. "I didn't
  know they handled it" is not a reason; "they do X but we need Y
  because of Z" is.
- For any field, header, or status code you emit or parse, cite the
  upstream doc URL or SDK file/line. Don't invent field names that
  "feel right" — the ecosystem has already chosen one.

## 8. Independent Audit Before Merge

**Every PR pushed must be reviewed by an independent third-party audit agent. Merge is blocked until all HIGH/MEDIUM findings are resolved or explicitly justified.**

After every `gh pr create` or force-push that updates an existing PR, immediately spawn an `Agent` (`general-purpose` subagent) with no shared context. Brief it cold with the PR URL and the contract the PR claims to pin. The agent must review from these angles, each treated as a blocking concern:

- **Correctness** — does the change actually do what the description claims? Are assertions strong enough that a real regression would fail?
- **Reliability** — race conditions, error handling, retry/timeout behavior, propagation timing on slow CI runners
- **Security** — auth/authz checks, input validation at boundaries, injection vectors, header forwarding (and what's deliberately *not* forwarded)
- **Sensitive-information leakage** — secrets in logs/error messages, internal taxonomy bleeding into client-visible output, upstream-provider details in user-facing fields, tokens/PII in test fixtures
- **Breaking changes** — public API shape, on-disk format, wire protocol, default-behavior shifts; if the change is breaking, is it gated/versioned?
- **E2E test coverage** — does the change have e2e coverage of the user-visible contract, not just unit-level happy-path? Are mocks tight enough that a regression on the unverified side (request body, headers) couldn't sneak through?

Audit agent output: HIGH / MEDIUM / LOW per finding, with **concrete suggested code** (the actual line to add), not vague "consider".

**Merge gate:** every HIGH and MEDIUM must be either (a) addressed by a code change, or (b) explicitly justified in the PR — e.g. "this exposes a feature gap, filed as #N for separate work, agreed not to block merge". "Looks fine" or silent merge is not enough.

For findings that surface gateway/product behavior gaps (not test-author oversight), file separate feature/bug issues and link from the PR.

This rule exists because the author cannot see their own blind spots — self-review without an independent agent has shipped real gaps (e.g. a cross-provider e2e PR that verified response translation while never asserting the gateway sent a request in the upstream's wire shape).

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
