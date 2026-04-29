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

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
