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

A kind whose provider documents no usable limit submits whole — its
bound is the provider's own. Do not invent a local one for it, and do
not treat "we could not find the number" as "nobody looked": the numbers
were looked for, and they are not there to hold (AISIX-Cloud#1386).
Bedrock's ceiling is a service quota that varies by region, policy type
and tier and is adjustable per account; Lakera publishes no size limit
and no error shape at all; OpenAI documents no input limit for
`/moderations` (the real constraint is the tokens-per-minute budget, so
its refusal is a genuine 429 and chunking would not help); Presidio's
ceiling is whichever spaCy pipeline the operator deployed. `crate::
too_large` carries the sources.

When a provider refuses a payload for its size, that is its own failure
class — never `Throttled` and never `ConfigError`. Both mislead: waiting
does not help, and neither does fixing credentials. Bedrock goes further
and re-sends the content in pieces rather than failing; the recursion is
only safe because every step is guaranteed to make progress (batch →
halve the slots → halve the text), so keep that property if you touch
it. A batch budget is not a limit — it is what to try after a refusal,
and it may be larger than the account's real ceiling.

## Fail policies default closed

`fail_open`, `output_fail_open` and `on_buffer_exceeded` all default to
the blocking side: a check that could not run must not release the
request. Any new failure path gets the same default, and an operator who
prefers availability opts in explicitly.

A guardrail that could not evaluate reaches the request through two
shapes, both carrying the same bounded per-kind failure tag and neither
allowed to carry matched content (#153): an explicitly fail-open row
emits `Bypass`, a fail-closed row emits `Block { unavailable: Some(tag) }`.

**Carry that tag all the way to the caller.** A fail-closed availability
block and a content block are the same 422 with the same `error.type`, so
the tag is the only thing that separates "your policy fired" from "your
guardrail is broken" — drop it and an operator debugs a policy that is
fine while their traffic is refused. Every block site in `aisix-proxy`
therefore builds its message through `error::guardrail_block_message` /
`guardrail_block_error` and passes the verdict's `unavailable` through;
the tag also lands on `error.code = "guardrail_unavailable"`, the audit
hit's `blocked_unavailable`, and the histogram's `error_type`. A refusal
the proxy raises on a guardrail's behalf (a hold-back cap, a failed mask
splice) carries one too — see `error::TAG_*`.

**Give each failure cause its own tag.** Tags are what a dashboard shows,
so collapsing distinct operator mistakes into one catch-all costs the
operator the diagnosis: `custom_unknown_action` (a word we do not know)
and `custom_no_verdict` (no decision at all) need different fixes, and
neither is `custom_script_error` (their service is down). And a verdict
we cannot read is always a FAILURE, never an Allow — reading silence as
consent is the open door `fail_open: false` exists to close.

**`enforcement_mode: monitor` is unconditional.** A monitored row never
blocks, for any reason — not a content match, not a provider outage, not
a failure policy. Do not add an exception: the value of the mode is that
it is safe to turn on, and one edge that refuses traffic destroys it.
Wanting an unreachable provider to refuse traffic is wanting
enforcement, and `block` mode with `fail_open: false` is how that is
spelled.
