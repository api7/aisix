# aisix-guardrails

## A provider's per-call size limit is handled by `chunk::chunk_text`, never by truncating

Every remote kind talks to an API that caps how much text one call may
carry. When you add or change one, do not re-decide what happens at that
cap — the family already answered it, and answering it again in isolation
is how the same defect shipped twice (#448, then AISIX-Cloud#1381):

- **Split, never clip.** Over-limit content is chunked and *every* chunk
  is submitted. Truncating hands the caller a bypass they control: the
  text is assembled oldest-message-first, so clipping drops the newest
  turn — the one being screened — and the call still returns a clean
  verdict, so nothing logs, counts, or looks wrong.
- **No cap on chunk count.** A per-request chunk budget is unscanned
  content through the back door. Cost scales with content; that is the
  trade.
- **The split is lossless** (`chunks.concat() == text`). Kinds that write
  masked text back rebuild the caller's content from per-chunk
  replacements, so any "clever" normalising split silently corrupts
  bodies rather than failing.
- **Count characters, not bytes.** The vendor limits are documented in
  characters, and byte slicing halves a multi-byte character.

A kind whose provider documents no limit (bedrock, lakera, presidio,
openai_moderation) submits whole — its bound is the provider's own. Do
not invent a local one for it.

## Fail policies default closed

`fail_open`, `output_fail_open` and `on_buffer_exceeded` all default to
the blocking side: a check that could not run must not release the
request. Any new failure path gets the same default, and an operator who
prefers availability opts in explicitly.

A guardrail that could not evaluate reaches the request through **two**
shapes, and a change to either must keep both intact: an explicitly
fail-open row emits `Bypass` (which `MandatoryGuardrail` upgrades on the
way out), while a fail-closed row emits `Block { unavailable: Some(tag) }`
(which `MonitorGuardrail` must not downgrade for a `mandatory` row). Both
carry the same bounded per-kind failure tag; neither may carry matched
content (#153).
