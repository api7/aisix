# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Bias toward caution over speed; for trivial tasks, use judgment. Merge with project-specific instructions as needed.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

- State assumptions explicitly; if uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name what's confusing, and ask.

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

---

**Working if:** fewer unnecessary diff lines, fewer overcomplication rewrites, and clarifying questions come before implementation rather than after mistakes.
