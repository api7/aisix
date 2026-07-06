//! Application helpers for PII redaction (#932 / AISIX-Cloud#932).
//!
//! `aisix-guardrails` owns detection and the text→text rewrite
//! ([`Guardrail::redact_input_text`] / [`Guardrail::redact_output_text`]);
//! this module owns WHERE the rewrite is applied on each wire shape:
//!
//! - request side: the normalised [`ChatFormat`] (chat/completions), the
//!   Anthropic-native `/v1/messages` body, the `/v1/responses` body, the
//!   legacy completions `prompt`, and embeddings `input` — message text
//!   only, mirroring the scan surface of `check_input`;
//! - response side: [`ChatResponse`] content + tool-call arguments, the
//!   Anthropic-native response JSON, and buffered streamed chunks
//!   (channel-reassembly: a masked span can cross chunk boundaries, so
//!   each content channel is concatenated, rewritten once, and the full
//!   rewritten text re-emitted on the channel's first chunk).
//!
//! Every helper returns per-detector match counts (detector names only,
//! never values) which callers merge into `usage_events
//! .redacted_entity_counts`.

use std::collections::BTreeMap;

use aisix_gateway::{ChatChunk, ChatFormat, ChatResponse};
use aisix_guardrails::Guardrail;
use serde_json::Value;

/// detector name → masked-span count. Mirrors
/// `UsageEvent::redacted_entity_counts`.
pub type RedactionCounts = BTreeMap<String, u32>;

/// Merge `from` into `into` (repeated small helper; counts are tiny maps).
pub fn merge_counts(into: &mut RedactionCounts, from: RedactionCounts) {
    for (k, v) in from {
        *into.entry(k).or_insert(0) += v;
    }
}

/// Which side's redactor to run. The two sides can be configured
/// independently (`hook_point`), so every JSON-walking helper takes the
/// direction rather than hardcoding one.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Input,
    Output,
}

fn redact_str(
    chain: &dyn Guardrail,
    dir: Direction,
    text: &str,
) -> Option<aisix_guardrails::Redaction> {
    match dir {
        Direction::Input => chain.redact_input_text(text),
        Direction::Output => chain.redact_output_text(text),
    }
}

// ─── Remote segment moderation (kind=bedrock mask write-back) ────────────────
//
// A Bedrock guardrail whose PII action is ANONYMIZE returns the masked
// replacement text from the SAME `ApplyGuardrail` call that yields the
// verdict — an async, whole-request rewrite that can't implement the sync
// per-field redact contract above. The bridge works in three walker
// passes over one wire body, all using the SAME wire-shape walker so slot
// enumeration order is identical by construction:
//
//   1. collect: a probe guardrail records every text slot the walker
//      offers (rewriting nothing);
//   2. one remote call: the chain's segment fold sends the slots as one
//      content block each and returns verdict + positionally-aligned
//      masked texts;
//   3. apply: a second probe guardrail replaces slot i with masked[i].
//
// Call sites pair this with `check_*_non_segment` so a segment-moderating
// member is consulted exactly once per hook. Families without a wire
// walker (embeddings, rerank, images, audio, passthrough, MCP) keep the
// plain `check_*` path, where an ANONYMIZE disposition still maps to
// Block — there is no write-back channel, and releasing the un-masked
// content would defeat the operator's policy.

/// Marker count key [`SegmentApplier`] attaches to each rewritten slot.
/// Several walkers discard a rewrite whose counts are empty (their
/// "did anything change" gate); the marker makes those gates fire. It is
/// never surfaced: [`moderate_body`] discards the apply-walk's returned
/// counts and reports the provider's entity counts instead.
const SEGMENT_APPLY_MARKER: &str = "__segment_apply__";

/// Pass-1 probe: records every text slot the walker offers. Never
/// rewrites, so the body is bit-identical after the collect walk.
#[derive(Default)]
struct SegmentCollector {
    texts: std::sync::Mutex<Vec<String>>,
}

impl SegmentCollector {
    fn take(&self) -> Vec<String> {
        std::mem::take(&mut self.texts.lock().expect("collector poisoned"))
    }

    fn record(&self, text: &str) -> Option<aisix_guardrails::Redaction> {
        self.texts
            .lock()
            .expect("collector poisoned")
            .push(text.to_owned());
        None
    }
}

impl Guardrail for SegmentCollector {
    fn name(&self) -> &'static str {
        "segment-collector"
    }
    fn redacts_input(&self) -> bool {
        true
    }
    fn redacts_output(&self) -> bool {
        true
    }
    fn redact_input_text(&self, text: &str) -> Option<aisix_guardrails::Redaction> {
        self.record(text)
    }
    fn redact_output_text(&self, text: &str) -> Option<aisix_guardrails::Redaction> {
        self.record(text)
    }
}

/// Pass-3 probe: replaces the i-th offered slot with `masked[i]`.
/// Positional by construction — the walker offers slots in the same
/// order the collector recorded them (same walker, same body state).
struct SegmentApplier {
    state: std::sync::Mutex<ApplierState>,
}

struct ApplierState {
    masked: Vec<String>,
    cursor: usize,
}

impl SegmentApplier {
    fn new(masked: Vec<String>) -> Self {
        Self {
            state: std::sync::Mutex::new(ApplierState { masked, cursor: 0 }),
        }
    }

    fn apply(&self, original: &str) -> Option<aisix_guardrails::Redaction> {
        let mut st = self.state.lock().expect("applier poisoned");
        let i = st.cursor;
        st.cursor += 1;
        match st.masked.get(i) {
            Some(m) if m != original => Some(aisix_guardrails::Redaction {
                text: m.clone(),
                counts: std::iter::once((SEGMENT_APPLY_MARKER.to_owned(), 1)).collect(),
            }),
            _ => None,
        }
    }

    /// Warn when the apply walk offered a different slot count than the
    /// mask carries — the extra/missing slots kept their originals (the
    /// per-slot `get` above never misassigns), this is diagnostics only.
    fn warn_if_misaligned(&self) {
        let st = self.state.lock().expect("applier poisoned");
        if st.cursor != st.masked.len() {
            tracing::warn!(
                offered = st.cursor,
                masked = st.masked.len(),
                "segment apply walk drifted from collect walk; \
                 unmatched slots kept their original text",
            );
        }
    }
}

impl Guardrail for SegmentApplier {
    fn name(&self) -> &'static str {
        "segment-applier"
    }
    fn redacts_input(&self) -> bool {
        true
    }
    fn redacts_output(&self) -> bool {
        true
    }
    fn redact_input_text(&self, text: &str) -> Option<aisix_guardrails::Redaction> {
        self.apply(text)
    }
    fn redact_output_text(&self, text: &str) -> Option<aisix_guardrails::Redaction> {
        self.apply(text)
    }
}

/// Complete one hook's moderation over a wire body: fold the already-run
/// `check_*_non_segment` verdict with the remote segment pass. The
/// segment pass is skipped when the check already blocked (the request
/// is dead — don't burn a provider call) or when the chain has no
/// segment-moderating member (zero overhead for non-Bedrock chains).
/// Masked replacements are written back through `walk`; the provider's
/// entity counts merge into `counts_out` (they feed
/// `redacted_entity_counts`, names only — #932 no-leak).
pub async fn moderate_body(
    chain: &dyn Guardrail,
    dir: Direction,
    non_segment_verdict: aisix_guardrails::GuardrailVerdict,
    counts_out: &mut RedactionCounts,
    monitor_hits_out: &mut Vec<aisix_core::models::GuardrailMonitorHit>,
    mut walk: impl FnMut(&dyn Guardrail) -> RedactionCounts,
) -> aisix_guardrails::GuardrailVerdict {
    if non_segment_verdict.is_block() || !chain.moderates_segments() {
        return non_segment_verdict;
    }
    let collector = SegmentCollector::default();
    walk(&collector);
    let texts = collector.take();
    if texts.is_empty() {
        return non_segment_verdict;
    }
    let mut outcome = match dir {
        Direction::Input => chain.moderate_input_segments(&texts).await,
        Direction::Output => chain.moderate_output_segments(&texts).await,
    };
    monitor_hits_out.append(&mut outcome.monitor_hits);
    if !outcome.verdict.is_block() {
        if let Some(masked) = outcome.masked {
            let applier = SegmentApplier::new(masked);
            // Marker counts are plumbing (see SEGMENT_APPLY_MARKER) —
            // discard them; the provider counts below are the real ones.
            let _ = walk(&applier);
            applier.warn_if_misaligned();
            merge_counts(counts_out, outcome.counts);
        }
    }
    non_segment_verdict.merged_with(outcome.verdict)
}

/// Rewrite one owned text field in place. No-op (and no allocation) when
/// nothing matches.
fn apply_to_string(
    chain: &dyn Guardrail,
    dir: Direction,
    field: &mut String,
    counts: &mut RedactionCounts,
) {
    if field.is_empty() {
        return;
    }
    if let Some(r) = redact_str(chain, dir, field) {
        *field = r.text;
        merge_counts(counts, r.counts);
    }
}

/// Rewrite a `Value::String` in place (helper for JSON-tree walking).
fn apply_to_value_string(
    chain: &dyn Guardrail,
    dir: Direction,
    v: &mut Value,
    counts: &mut RedactionCounts,
) {
    if let Value::String(s) = v {
        if !s.is_empty() {
            if let Some(r) = redact_str(chain, dir, s) {
                *s = r.text;
                merge_counts(counts, r.counts);
            }
        }
    }
}

/// Recursively rewrite every string VALUE in a JSON tree (object values,
/// array elements). Keys and non-string scalars are untouched, so the
/// tree stays structurally valid — a phone number stored as a JSON number
/// is out of scope by design (rewriting it to a mask token would corrupt
/// the document).
pub fn redact_value_strings(
    chain: &dyn Guardrail,
    dir: Direction,
    v: &mut Value,
    counts: &mut RedactionCounts,
) {
    match v {
        Value::String(_) => apply_to_value_string(chain, dir, v, counts),
        Value::Array(items) => {
            for item in items {
                redact_value_strings(chain, dir, item, counts);
            }
        }
        Value::Object(map) => {
            for (_, val) in map.iter_mut() {
                redact_value_strings(chain, dir, val, counts);
            }
        }
        _ => {}
    }
}

/// Mask-rewrite an already-assembled OUTPUT text buffer in place — the
/// content-capture accumulator a streaming hold-back path hands to
/// content-capturing exporters (#932 × AISIX-Cloud#947). The wire-side
/// SSE/chunk redaction rewrites only the held bytes released to the client;
/// the capture accumulator collects raw deltas, so without this the exported
/// content would carry PII the client never saw. Counts are deliberately
/// discarded — the wire-side redaction already tallied them, and tallying
/// the same matches again would double-count.
pub fn redact_captured_output(chain: &dyn Guardrail, text: &mut String) {
    let mut discard = RedactionCounts::new();
    apply_to_string(chain, Direction::Output, text, &mut discard);
}

/// Rewrite a JSON-*encoded* string (OpenAI `function.arguments`): parse,
/// walk the string values, re-serialise — so a mask token can't corrupt
/// the embedded document (e.g. a phone number as a JSON number value
/// stays untouched rather than becoming invalid JSON). Falls back to a
/// raw text rewrite when the payload doesn't parse (a provider emitted
/// malformed/partial args — best effort beats leaking).
pub fn redact_json_encoded(
    chain: &dyn Guardrail,
    dir: Direction,
    encoded: &mut String,
    counts: &mut RedactionCounts,
) {
    if encoded.is_empty() {
        return;
    }
    match serde_json::from_str::<Value>(encoded) {
        Ok(mut v) => {
            let mut local = RedactionCounts::new();
            redact_value_strings(chain, dir, &mut v, &mut local);
            if !local.is_empty() {
                if let Ok(s) = serde_json::to_string(&v) {
                    *encoded = s;
                    merge_counts(counts, local);
                }
            }
        }
        Err(_) => apply_to_string(chain, dir, encoded, counts),
    }
}

// ─── Request side ────────────────────────────────────────────────────────────

/// Mask the request messages of a normalised [`ChatFormat`] in place:
/// the flat `content` string and the `text` field of typed content
/// blocks — the same surface `check_input` scans (`message_scan_text`).
/// Tool-call arguments replayed in history are covered too (they reach
/// the upstream verbatim). Returns the merged counts (empty = untouched).
pub fn redact_chat_format(chain: &dyn Guardrail, req: &mut ChatFormat) -> RedactionCounts {
    let mut counts = RedactionCounts::new();
    if !chain.redacts_input() {
        return counts;
    }
    for msg in &mut req.messages {
        if let Some(content) = msg.content.as_mut() {
            apply_to_string(chain, Direction::Input, content, &mut counts);
        }
        if let Some(blocks) = msg.content_blocks.as_mut() {
            for block in blocks {
                if block.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(text) = block.get_mut("text") {
                        apply_to_value_string(chain, Direction::Input, text, &mut counts);
                    }
                }
            }
        }
        // History-replay tool calls: arguments travel to the upstream
        // verbatim through `extra`, so mask them like fresh content.
        if let Some(tool_calls) = msg.extra.get_mut("tool_calls") {
            redact_tool_call_arguments(chain, Direction::Input, tool_calls, &mut counts);
        }
    }
    counts
}

/// Mask `function.arguments` (JSON-encoded string) on each element of an
/// OpenAI-shaped `tool_calls` array. Names/ids are structural, not
/// content, and stay untouched.
fn redact_tool_call_arguments(
    chain: &dyn Guardrail,
    dir: Direction,
    tool_calls: &mut Value,
    counts: &mut RedactionCounts,
) {
    let Some(items) = tool_calls.as_array_mut() else {
        return;
    };
    for tc in items {
        if let Some(Value::String(s)) = tc.get_mut("function").and_then(|f| f.get_mut("arguments"))
        {
            let mut owned = std::mem::take(s);
            redact_json_encoded(chain, dir, &mut owned, counts);
            *s = owned;
        }
    }
}

/// Mask an Anthropic-native `/v1/messages` request body in place:
/// `system` (string or text blocks) and `messages[].content` (string or
/// blocks — `text` blocks and nested `tool_result` content). `tool_use`
/// input objects in history are walked as JSON strings.
pub fn redact_anthropic_request(chain: &dyn Guardrail, body: &mut Value) -> RedactionCounts {
    let mut counts = RedactionCounts::new();
    if !chain.redacts_input() {
        return counts;
    }
    if let Some(system) = body.get_mut("system") {
        redact_anthropic_content(chain, Direction::Input, system, &mut counts);
    }
    if let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
        for msg in messages {
            if let Some(content) = msg.get_mut("content") {
                redact_anthropic_content(chain, Direction::Input, content, &mut counts);
            }
        }
    }
    counts
}

/// Anthropic `content` is either a bare string or an array of typed
/// blocks. Rewrites `text` blocks, `tool_result` nested content, and
/// `tool_use` input objects; leaves image/document blocks alone.
fn redact_anthropic_content(
    chain: &dyn Guardrail,
    dir: Direction,
    content: &mut Value,
    counts: &mut RedactionCounts,
) {
    match content {
        Value::String(_) => apply_to_value_string(chain, dir, content, counts),
        Value::Array(blocks) => {
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(text) = block.get_mut("text") {
                            apply_to_value_string(chain, dir, text, counts);
                        }
                    }
                    Some("tool_result") => {
                        if let Some(inner) = block.get_mut("content") {
                            redact_anthropic_content(chain, dir, inner, counts);
                        }
                    }
                    Some("tool_use") => {
                        if let Some(input) = block.get_mut("input") {
                            redact_value_strings(chain, dir, input, counts);
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

/// Mask an Anthropic-native `/v1/messages` RESPONSE body in place (the
/// non-streaming passthrough JSON): top-level `content` blocks (`text` +
/// `tool_use` input).
pub fn redact_anthropic_response(chain: &dyn Guardrail, body: &mut Value) -> RedactionCounts {
    let mut counts = RedactionCounts::new();
    if !chain.redacts_output() {
        return counts;
    }
    if let Some(content) = body.get_mut("content") {
        redact_anthropic_content(chain, Direction::Output, content, &mut counts);
    }
    counts
}

/// Mask a `/v1/responses` request body in place: `instructions` and
/// `input` (bare string, or item list whose `message` items carry
/// `content` as a string or `input_text` parts). Function-call outputs
/// replayed as `function_call_output` items are walked too.
pub fn redact_responses_request(chain: &dyn Guardrail, body: &mut Value) -> RedactionCounts {
    let mut counts = RedactionCounts::new();
    if !chain.redacts_input() {
        return counts;
    }
    if let Some(instructions) = body.get_mut("instructions") {
        apply_to_value_string(chain, Direction::Input, instructions, &mut counts);
    }
    match body.get_mut("input") {
        Some(v @ Value::String(_)) => {
            apply_to_value_string(chain, Direction::Input, v, &mut counts)
        }
        Some(Value::Array(items)) => {
            for item in items {
                redact_responses_item(chain, Direction::Input, item, &mut counts);
            }
        }
        _ => {}
    }
    counts
}

/// One `/v1/responses` input/output item. `message` items carry
/// string-or-parts content (`input_text` / `output_text` / plain `text`);
/// `function_call` carries JSON-encoded `arguments`;
/// `function_call_output` carries a string `output`.
fn redact_responses_item(
    chain: &dyn Guardrail,
    dir: Direction,
    item: &mut Value,
    counts: &mut RedactionCounts,
) {
    match item.get("type").and_then(Value::as_str) {
        // An item without a `type` defaults to `message` on this API.
        Some("message") | None => match item.get_mut("content") {
            Some(v @ Value::String(_)) => apply_to_value_string(chain, dir, v, counts),
            Some(Value::Array(parts)) => {
                for part in parts {
                    if matches!(
                        part.get("type").and_then(Value::as_str),
                        Some("input_text") | Some("output_text") | Some("text")
                    ) {
                        if let Some(text) = part.get_mut("text") {
                            apply_to_value_string(chain, dir, text, counts);
                        }
                    }
                }
            }
            _ => {}
        },
        Some("function_call") => {
            if let Some(Value::String(args)) = item.get_mut("arguments") {
                let mut owned = std::mem::take(args);
                redact_json_encoded(chain, dir, &mut owned, counts);
                *args = owned;
            }
        }
        Some("function_call_output") => {
            if let Some(output) = item.get_mut("output") {
                apply_to_value_string(chain, dir, output, counts);
            }
        }
        _ => {}
    }
}

/// Mask a `/v1/responses` non-streaming RESPONSE body in place: every
/// item in `output` (message `output_text` parts, `function_call`
/// arguments) — the same surface the output check scans.
pub fn redact_responses_response(chain: &dyn Guardrail, body: &mut Value) -> RedactionCounts {
    let mut counts = RedactionCounts::new();
    if !chain.redacts_output() {
        return counts;
    }
    if let Some(Value::Array(items)) = body.get_mut("output") {
        for item in items {
            redact_responses_item(chain, Direction::Output, item, &mut counts);
        }
    }
    counts
}

/// Mask a legacy `/v1/completions` request body in place: `prompt` as a
/// bare string or an array of strings (token-id arrays carry no text).
pub fn redact_completions_request(chain: &dyn Guardrail, body: &mut Value) -> RedactionCounts {
    let mut counts = RedactionCounts::new();
    if !chain.redacts_input() {
        return counts;
    }
    match body.get_mut("prompt") {
        Some(v @ Value::String(_)) => {
            apply_to_value_string(chain, Direction::Input, v, &mut counts)
        }
        Some(Value::Array(items)) => {
            for item in items {
                if item.is_string() {
                    apply_to_value_string(chain, Direction::Input, item, &mut counts);
                }
            }
        }
        _ => {}
    }
    counts
}

/// Mask a `/v1/rerank` request body in place: `query` plus `documents[]`
/// (plain strings or `{"text": ...}` objects) — the same surface
/// `check_input` scans via `rerank_input_to_chat` (#696).
pub fn redact_rerank_request(chain: &dyn Guardrail, body: &mut Value) -> RedactionCounts {
    let mut counts = RedactionCounts::new();
    if !chain.redacts_input() {
        return counts;
    }
    if let Some(q) = body.get_mut("query") {
        apply_to_value_string(chain, Direction::Input, q, &mut counts);
    }
    if let Some(Value::Array(docs)) = body.get_mut("documents") {
        for doc in docs {
            match doc {
                Value::String(_) => {
                    apply_to_value_string(chain, Direction::Input, doc, &mut counts)
                }
                Value::Object(_) => {
                    if let Some(text) = doc.get_mut("text") {
                        apply_to_value_string(chain, Direction::Input, text, &mut counts);
                    }
                }
                _ => {}
            }
        }
    }
    counts
}

/// Mask a `/v1/images/generations` request body in place: the `prompt`
/// field — the same surface `check_input` scans via `images_input_to_chat`
/// (#696).
pub fn redact_images_request(chain: &dyn Guardrail, body: &mut Value) -> RedactionCounts {
    let mut counts = RedactionCounts::new();
    if !chain.redacts_input() {
        return counts;
    }
    if let Some(p) = body.get_mut("prompt") {
        apply_to_value_string(chain, Direction::Input, p, &mut counts);
    }
    counts
}

/// Mask a `/v1/audio/speech` request body in place: the `input` field (the
/// text to synthesize) — the same surface `check_input` scans via
/// `speech_input_to_chat` (#696).
pub fn redact_speech_request(chain: &dyn Guardrail, body: &mut Value) -> RedactionCounts {
    let mut counts = RedactionCounts::new();
    if !chain.redacts_input() {
        return counts;
    }
    if let Some(input) = body.get_mut("input") {
        apply_to_value_string(chain, Direction::Input, input, &mut counts);
    }
    counts
}

/// Mask an audio transcription/translation RESPONSE in place (#696). The
/// wire body is either JSON (`json` / `verbose_json` response_format:
/// top-level `text` + per-segment `segments[].text`) or raw text
/// (`text` / `srt` / `vtt` formats). Returns the rewritten bytes + counts,
/// or `None` when nothing matched (caller keeps the original body). A
/// non-UTF-8 body is left untouched.
pub fn redact_transcription_response(
    chain: &dyn Guardrail,
    body: &[u8],
) -> Option<(Vec<u8>, RedactionCounts)> {
    if !chain.redacts_output() {
        return None;
    }
    let mut counts = RedactionCounts::new();
    if let Ok(mut json) = serde_json::from_slice::<Value>(body) {
        if let Some(text) = json.get_mut("text") {
            apply_to_value_string(chain, Direction::Output, text, &mut counts);
        }
        if let Some(Value::Array(segments)) = json.get_mut("segments") {
            for seg in segments {
                if let Some(text) = seg.get_mut("text") {
                    apply_to_value_string(chain, Direction::Output, text, &mut counts);
                }
            }
        }
        if counts.is_empty() {
            return None;
        }
        return serde_json::to_vec(&json).ok().map(|b| (b, counts));
    }
    let text = std::str::from_utf8(body).ok()?;
    let r = chain.redact_output_text(text)?;
    Some((r.text.into_bytes(), r.counts))
}

/// Mask a legacy `/v1/completions` RESPONSE body in place: `choices[].text`.
pub fn redact_completions_response(chain: &dyn Guardrail, body: &mut Value) -> RedactionCounts {
    let mut counts = RedactionCounts::new();
    if !chain.redacts_output() {
        return counts;
    }
    if let Some(Value::Array(choices)) = body.get_mut("choices") {
        for choice in choices {
            if let Some(text) = choice.get_mut("text") {
                apply_to_value_string(chain, Direction::Output, text, &mut counts);
            }
        }
    }
    counts
}

// ─── Response side (non-streaming) ───────────────────────────────────────────

/// Mask a normalised [`ChatResponse`] in place: assistant `content` plus
/// `tool_calls` function arguments (the same surface
/// `guardrail_output_text` scans). Reasoning content is excluded from
/// guardrail scope by design and stays untouched.
pub fn redact_chat_response(chain: &dyn Guardrail, resp: &mut ChatResponse) -> RedactionCounts {
    let mut counts = RedactionCounts::new();
    if !chain.redacts_output() {
        return counts;
    }
    if let Some(content) = resp.message.content.as_mut() {
        apply_to_string(chain, Direction::Output, content, &mut counts);
    }
    if let Some(tool_calls) = resp.message.extra.get_mut("tool_calls") {
        redact_tool_call_arguments(chain, Direction::Output, tool_calls, &mut counts);
    }
    counts
}

// ─── Response side (streamed, buffered) ──────────────────────────────────────

/// Mask a fully-buffered stream of normalised [`ChatChunk`]s in place —
/// the hold-back release path (BufferFull), where the whole response is
/// available before any byte reaches the wire.
///
/// A masked span can cross chunk boundaries, so per-chunk rewriting would
/// miss it. Instead each content channel (delta content, and each
/// tool-call's streamed `arguments`) is concatenated across the buffered
/// chunks, rewritten once, and the FULL rewritten text re-emitted on the
/// channel's first carrying chunk; later chunks in that channel become
/// empty deltas. The stream is already released en bloc at this point, so
/// chunk-size distribution is not client-observable. Non-content fields
/// (ids, usage, finish_reason, reasoning) are untouched.
pub fn redact_chat_chunks(chain: &dyn Guardrail, chunks: &mut [ChatChunk]) -> RedactionCounts {
    let mut counts = RedactionCounts::new();
    if !chain.redacts_output() {
        return counts;
    }

    // Content channel: all chunks stream one assistant message.
    let content_sites: Vec<usize> = chunks
        .iter()
        .enumerate()
        .filter(|(_, c)| c.delta.content.as_deref().is_some_and(|t| !t.is_empty()))
        .map(|(i, _)| i)
        .collect();
    if !content_sites.is_empty() {
        let joined: String = content_sites
            .iter()
            .map(|&i| chunks[i].delta.content.as_deref().unwrap_or(""))
            .collect();
        if let Some(r) = chain.redact_output_text(&joined) {
            let mut first = true;
            for &i in &content_sites {
                chunks[i].delta.content = Some(if first {
                    first = false;
                    r.text.clone()
                } else {
                    String::new()
                });
            }
            merge_counts(&mut counts, r.counts);
        }
    }

    // Tool-call channels: fragments carry an `index` discriminator; the
    // concatenation of each channel's `function.arguments` strings is the
    // complete JSON-encoded argument document.
    let mut channels: BTreeMap<u64, Vec<(usize, usize)>> = BTreeMap::new();
    for (ci, chunk) in chunks.iter().enumerate() {
        if let Some(tcs) = chunk.delta.tool_calls.as_ref() {
            for (ti, tc) in tcs.iter().enumerate() {
                let idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0);
                if tc
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.is_empty())
                {
                    channels.entry(idx).or_default().push((ci, ti));
                }
            }
        }
    }
    for sites in channels.values() {
        let joined: String = sites
            .iter()
            .map(|&(ci, ti)| {
                chunks[ci].delta.tool_calls.as_ref().unwrap()[ti]
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
            })
            .collect();
        let mut rewritten = joined.clone();
        let mut local = RedactionCounts::new();
        redact_json_encoded(chain, Direction::Output, &mut rewritten, &mut local);
        if local.is_empty() {
            continue;
        }
        let mut first = true;
        for &(ci, ti) in sites {
            let args = chunks[ci].delta.tool_calls.as_mut().unwrap()[ti]
                .get_mut("function")
                .and_then(|f| f.get_mut("arguments"))
                .expect("site was selected for having arguments");
            *args = Value::String(if first {
                first = false;
                rewritten.clone()
            } else {
                String::new()
            });
        }
        merge_counts(&mut counts, local);
    }

    counts
}

// ─── Anthropic-native SSE (passthrough) rewrite ──────────────────────────────

/// One parsed SSE frame from a buffered Anthropic-native byte stream.
struct SseFrame {
    /// Original frame bytes (no trailing separator). Emitted verbatim
    /// unless `data` was modified.
    raw: Vec<u8>,
    /// Parsed `data:` payload, when the frame carries one.
    data: Option<Value>,
    dirty: bool,
}

impl SseFrame {
    /// Re-render the frame: the first `data:` line is replaced with the
    /// re-serialised payload (subsequent `data:` lines dropped); every
    /// other line passes through untouched.
    fn render(&self) -> Vec<u8> {
        if !self.dirty {
            return self.raw.clone();
        }
        let Some(data) = self.data.as_ref() else {
            return self.raw.clone();
        };
        let text = String::from_utf8_lossy(&self.raw);
        let mut out = String::new();
        let mut data_written = false;
        for line in text.split('\n') {
            if line.starts_with("data:") {
                if !data_written {
                    out.push_str("data: ");
                    out.push_str(&serde_json::to_string(data).unwrap_or_default());
                    out.push('\n');
                    data_written = true;
                }
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
        // Drop the final artificial newline added by the loop; the caller
        // re-adds the frame separator.
        if out.ends_with('\n') {
            out.pop();
        }
        out.into_bytes()
    }
}

/// Split a buffered SSE byte stream into frames on the blank-line
/// separator. Returns `(frames, trailing)` where `trailing` is a
/// partial frame with no terminator yet (forwarded verbatim).
fn split_sse_frames(raw: &[u8]) -> (Vec<SseFrame>, &[u8]) {
    let mut frames = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i + 1 < raw.len() {
        if raw[i] == b'\n' && raw[i + 1] == b'\n' {
            let frame_raw = &raw[start..i];
            let data = String::from_utf8_lossy(frame_raw)
                .split('\n')
                .find(|l| l.starts_with("data:"))
                .and_then(|l| serde_json::from_str::<Value>(l["data:".len()..].trim()).ok());
            frames.push(SseFrame {
                raw: frame_raw.to_vec(),
                data,
                dirty: false,
            });
            start = i + 2;
            i += 2;
        } else {
            i += 1;
        }
    }
    (frames, &raw[start..])
}

/// Mask a fully-buffered Anthropic-native SSE response (the `/v1/messages`
/// passthrough hold-back). Text deltas are reassembled per content-block
/// `index` (a masked span can cross frame boundaries), masked once, and
/// the full masked text re-emitted on the channel's first frame;
/// `input_json_delta` (tool-use arguments) channels are masked as complete
/// JSON documents. `None` = nothing matched, forward the original bytes
/// byte-identical.
pub fn redact_anthropic_sse(
    chain: &dyn Guardrail,
    raw: &[u8],
) -> Option<(Vec<u8>, RedactionCounts)> {
    if !chain.redacts_output() {
        return None;
    }
    let (mut frames, trailing) = split_sse_frames(raw);
    let mut counts = RedactionCounts::new();

    // channel key → ordered (frame_idx, kind) sites. Kind distinguishes the
    // JSON path to rewrite inside the frame payload.
    #[derive(Clone, Copy)]
    enum Site {
        DeltaText,
        DeltaPartialJson,
        BlockStartText,
    }
    let mut text_channels: BTreeMap<u64, Vec<(usize, Site)>> = BTreeMap::new();
    let mut json_channels: BTreeMap<u64, Vec<(usize, Site)>> = BTreeMap::new();

    for (fi, frame) in frames.iter().enumerate() {
        let Some(data) = frame.data.as_ref() else {
            continue;
        };
        let index = data.get("index").and_then(Value::as_u64).unwrap_or(0);
        match data.get("type").and_then(Value::as_str) {
            Some("content_block_delta") => {
                match data
                    .get("delta")
                    .and_then(|d| d.get("type"))
                    .and_then(Value::as_str)
                {
                    Some("text_delta") => {
                        if data
                            .get("delta")
                            .and_then(|d| d.get("text"))
                            .and_then(Value::as_str)
                            .is_some_and(|t| !t.is_empty())
                        {
                            text_channels
                                .entry(index)
                                .or_default()
                                .push((fi, Site::DeltaText));
                        }
                    }
                    Some("input_json_delta") => {
                        if data
                            .get("delta")
                            .and_then(|d| d.get("partial_json"))
                            .and_then(Value::as_str)
                            .is_some_and(|t| !t.is_empty())
                        {
                            json_channels
                                .entry(index)
                                .or_default()
                                .push((fi, Site::DeltaPartialJson));
                        }
                    }
                    _ => {}
                }
            }
            Some("content_block_start") => {
                // A `text` block may open with non-empty initial text; it
                // belongs at the head of the same channel as its deltas.
                if data
                    .get("content_block")
                    .and_then(|b| b.get("text"))
                    .and_then(Value::as_str)
                    .is_some_and(|t| !t.is_empty())
                {
                    text_channels
                        .entry(index)
                        .or_default()
                        .push((fi, Site::BlockStartText));
                }
            }
            _ => {}
        }
    }

    fn site_text(data: &Value, site: Site) -> &str {
        let path = match site {
            Site::DeltaText => data.get("delta").and_then(|d| d.get("text")),
            Site::DeltaPartialJson => data.get("delta").and_then(|d| d.get("partial_json")),
            Site::BlockStartText => data.get("content_block").and_then(|b| b.get("text")),
        };
        path.and_then(Value::as_str).unwrap_or("")
    }

    fn site_slot(data: &mut Value, site: Site) -> Option<&mut Value> {
        match site {
            Site::DeltaText => data.get_mut("delta").and_then(|d| d.get_mut("text")),
            Site::DeltaPartialJson => data
                .get_mut("delta")
                .and_then(|d| d.get_mut("partial_json")),
            Site::BlockStartText => data
                .get_mut("content_block")
                .and_then(|b| b.get_mut("text")),
        }
    }

    let rewrite = |frames: &mut Vec<SseFrame>, sites: &[(usize, Site)], new_text: String| {
        let mut first = true;
        for &(fi, site) in sites {
            let frame = &mut frames[fi];
            if let Some(slot) = frame.data.as_mut().and_then(|d| site_slot(d, site)) {
                *slot = Value::String(if first {
                    first = false;
                    new_text.clone()
                } else {
                    String::new()
                });
                frame.dirty = true;
            }
        }
    };

    for sites in text_channels.values() {
        let joined: String = sites
            .iter()
            .map(|&(fi, site)| site_text(frames[fi].data.as_ref().unwrap(), site))
            .collect();
        if let Some(r) = chain.redact_output_text(&joined) {
            rewrite(&mut frames, sites, r.text);
            merge_counts(&mut counts, r.counts);
        }
    }
    for sites in json_channels.values() {
        let joined: String = sites
            .iter()
            .map(|&(fi, site)| site_text(frames[fi].data.as_ref().unwrap(), site))
            .collect();
        let mut rewritten = joined.clone();
        let mut local = RedactionCounts::new();
        redact_json_encoded(chain, Direction::Output, &mut rewritten, &mut local);
        if !local.is_empty() {
            rewrite(&mut frames, sites, rewritten);
            merge_counts(&mut counts, local);
        }
    }

    if counts.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(raw.len());
    for frame in &frames {
        out.extend_from_slice(&frame.render());
        out.extend_from_slice(b"\n\n");
    }
    out.extend_from_slice(trailing);
    Some((out, counts))
}

/// The concatenated TEXT-channel content of a buffered Anthropic-native
/// SSE stream (per content-block `index` order, `content_block_start`
/// head text included). Used to rebuild the content-capture accumulator
/// after a segment (provider-side) mask rewrote the held bytes — the
/// sync redactor can't reproduce a provider mask (#932 × AISIX-Cloud#947).
pub fn anthropic_sse_text(raw: &[u8]) -> String {
    let (frames, _) = split_sse_frames(raw);
    let mut channels: BTreeMap<u64, String> = BTreeMap::new();
    for frame in &frames {
        let Some(data) = frame.data.as_ref() else {
            continue;
        };
        let index = data.get("index").and_then(Value::as_u64).unwrap_or(0);
        let text = match data.get("type").and_then(Value::as_str) {
            Some("content_block_delta") => data
                .get("delta")
                .filter(|d| d.get("type").and_then(Value::as_str) == Some("text_delta"))
                .and_then(|d| d.get("text"))
                .and_then(Value::as_str),
            Some("content_block_start") => data
                .get("content_block")
                .and_then(|b| b.get("text"))
                .and_then(Value::as_str),
            _ => None,
        };
        if let Some(t) = text {
            channels.entry(index).or_default().push_str(t);
        }
    }
    channels.into_values().collect()
}

/// The concatenated `output_text` delta content of a buffered
/// `/v1/responses` SSE stream (channel order). Same capture-rebuild role
/// as [`anthropic_sse_text`].
pub fn responses_sse_text(raw: &[u8]) -> String {
    let (frames, _) = split_sse_frames(raw);
    // First-seen channel order (NOT key order): the rebuilt capture must
    // read in the order the client saw the channels emitted.
    let mut channels: Vec<(String, String)> = Vec::new();
    for frame in &frames {
        let Some(data) = frame.data.as_ref() else {
            continue;
        };
        if data.get("type").and_then(Value::as_str) != Some("response.output_text.delta") {
            continue;
        }
        let Some(t) = data.get("delta").and_then(Value::as_str) else {
            continue;
        };
        let key = match data.get("item_id").and_then(Value::as_str) {
            Some(id) => format!(
                "{id}/{}",
                data.get("content_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
            ),
            None => format!(
                "{}/{}",
                data.get("output_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                data.get("content_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
            ),
        };
        match channels.iter_mut().find(|(k, _)| *k == key) {
            Some((_, buf)) => buf.push_str(t),
            None => channels.push((key, t.to_owned())),
        }
    }
    channels.into_iter().map(|(_, text)| text).collect()
}

// ─── Responses-API SSE rewrite ───────────────────────────────────────────────

/// Mask a fully-buffered Responses-API SSE byte stream (the `/v1/responses`
/// verbatim hold-back and the cross-provider bridge release). Delta events
/// are reassembled per channel (`output_text.delta` by item, `function_call
/// _arguments.delta` by item), masked once, and re-emitted on the channel's
/// first frame; the aggregate events (`*.done`, `output_item.done`,
/// `response.completed`) carry complete texts and are masked directly —
/// deterministic masking keeps them consistent with the delta channels.
/// `None` = nothing matched, forward the original bytes byte-identical.
pub fn redact_responses_sse(
    chain: &dyn Guardrail,
    raw: &[u8],
) -> Option<(Vec<u8>, RedactionCounts)> {
    if !chain.redacts_output() {
        return None;
    }
    let (mut frames, trailing) = split_sse_frames(raw);
    let mut counts = RedactionCounts::new();

    // Delta channels: (event-type discriminant, channel key) → frame sites.
    let mut text_channels: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut args_channels: BTreeMap<String, Vec<usize>> = BTreeMap::new();

    fn channel_key(data: &Value) -> String {
        // item_id is the stable discriminator; fall back to output_index +
        // content_index for encoders that omit it.
        match data.get("item_id").and_then(Value::as_str) {
            Some(id) => format!(
                "{id}/{}",
                data.get("content_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
            ),
            None => format!(
                "{}/{}",
                data.get("output_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                data.get("content_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
            ),
        }
    }

    for (fi, frame) in frames.iter().enumerate() {
        let Some(data) = frame.data.as_ref() else {
            continue;
        };
        match data.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => {
                if data
                    .get("delta")
                    .and_then(Value::as_str)
                    .is_some_and(|t| !t.is_empty())
                {
                    text_channels.entry(channel_key(data)).or_default().push(fi);
                }
            }
            Some("response.function_call_arguments.delta") => {
                if data
                    .get("delta")
                    .and_then(Value::as_str)
                    .is_some_and(|t| !t.is_empty())
                {
                    args_channels.entry(channel_key(data)).or_default().push(fi);
                }
            }
            _ => {}
        }
    }

    // Rewrite the delta channels (first frame gets the full masked text).
    let rewrite_channel = |frames: &mut Vec<SseFrame>, sites: &[usize], new_text: String| {
        let mut first = true;
        for &fi in sites {
            let frame = &mut frames[fi];
            if let Some(slot) = frame.data.as_mut().and_then(|d| d.get_mut("delta")) {
                *slot = Value::String(if first {
                    first = false;
                    new_text.clone()
                } else {
                    String::new()
                });
                frame.dirty = true;
            }
        }
    };
    for sites in text_channels.values() {
        let joined: String = sites
            .iter()
            .map(|&fi| {
                frames[fi]
                    .data
                    .as_ref()
                    .and_then(|d| d.get("delta"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
            })
            .collect();
        if let Some(r) = chain.redact_output_text(&joined) {
            rewrite_channel(&mut frames, sites, r.text);
            merge_counts(&mut counts, r.counts);
        }
    }
    for sites in args_channels.values() {
        let joined: String = sites
            .iter()
            .map(|&fi| {
                frames[fi]
                    .data
                    .as_ref()
                    .and_then(|d| d.get("delta"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
            })
            .collect();
        let mut rewritten = joined.clone();
        let mut local = RedactionCounts::new();
        redact_json_encoded(chain, Direction::Output, &mut rewritten, &mut local);
        if !local.is_empty() {
            rewrite_channel(&mut frames, sites, rewritten);
            merge_counts(&mut counts, local);
        }
    }

    // Aggregate events carry complete texts — mask them in place. Their
    // counts are NOT merged into the totals: they duplicate the delta
    // channels' matches (the audit count is per span served, not per
    // wire occurrence). Only count them when the delta channel was absent
    // (e.g. a `.done`-only encoder).
    for frame in frames.iter_mut() {
        let Some(data) = frame.data.as_mut() else {
            continue;
        };
        let ty = data
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let mut local = RedactionCounts::new();
        match ty.as_str() {
            "response.output_text.done" => {
                if let Some(text) = data.get_mut("text") {
                    apply_to_value_string(chain, Direction::Output, text, &mut local);
                }
            }
            "response.content_part.done" => {
                if let Some(text) = data.get_mut("part").and_then(|p| p.get_mut("text")) {
                    apply_to_value_string(chain, Direction::Output, text, &mut local);
                }
            }
            "response.function_call_arguments.done" => {
                if let Some(Value::String(args)) = data.get_mut("arguments") {
                    let mut owned = std::mem::take(args);
                    redact_json_encoded(chain, Direction::Output, &mut owned, &mut local);
                    *args = owned;
                }
            }
            "response.output_item.done" => {
                if let Some(item) = data.get_mut("item") {
                    redact_responses_item(chain, Direction::Output, item, &mut local);
                }
            }
            "response.completed" | "response.incomplete" | "response.failed" => {
                if let Some(Value::Array(items)) =
                    data.get_mut("response").and_then(|r| r.get_mut("output"))
                {
                    for item in items {
                        redact_responses_item(chain, Direction::Output, item, &mut local);
                    }
                }
            }
            _ => {}
        }
        if !local.is_empty() {
            frame.dirty = true;
            if counts.is_empty() {
                merge_counts(&mut counts, local);
            }
        }
    }

    let any_dirty = frames.iter().any(|f| f.dirty);
    if !any_dirty {
        return None;
    }
    let mut out = Vec::with_capacity(raw.len());
    for frame in &frames {
        out.extend_from_slice(&frame.render());
        out.extend_from_slice(b"\n\n");
    }
    out.extend_from_slice(trailing);
    Some((out, counts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_gateway::{ChatDelta, ChatMessage};
    use aisix_guardrails::{builtin_rule, GuardrailChain, PiiAction, PiiGuardrail};
    use serde_json::json;
    use std::sync::Arc;

    fn mask_chain(hook: aisix_core::models::GuardrailHookPoint) -> Arc<dyn Guardrail> {
        let g = PiiGuardrail::new(
            vec![
                builtin_rule("email", PiiAction::Mask).unwrap(),
                builtin_rule("china_mobile", PiiAction::Mask).unwrap(),
            ],
            hook,
            262_144,
            false,
        );
        Arc::new(GuardrailChain::new(vec![Arc::new(g)]))
    }

    fn both() -> Arc<dyn Guardrail> {
        mask_chain(aisix_core::models::GuardrailHookPoint::Both)
    }

    #[test]
    fn chat_format_masks_content_blocks_and_history_tool_args() {
        let chain = both();
        let mut req: ChatFormat = serde_json::from_value(json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "mail a@x.com"},
                {"role": "user", "content": "", "content_blocks": [
                    {"type": "text", "text": "call 13800138000"},
                    {"type": "image_url", "image_url": {"url": "http://x"}}
                ]},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"index": 0, "function": {"name": "send", "arguments": "{\"to\":\"b@y.org\"}"}}
                ]}
            ]
        }))
        .unwrap();
        let counts = redact_chat_format(chain.as_ref(), &mut req);
        assert_eq!(
            req.messages[0].content.as_deref(),
            Some("mail [EMAIL_REDACTED]")
        );
        let blocks = req.messages[1].content_blocks.as_ref().unwrap();
        assert_eq!(
            blocks[0].get("text").unwrap().as_str().unwrap(),
            "call [CHINA_MOBILE_REDACTED]",
        );
        let args = req.messages[2].extra["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap();
        assert_eq!(args, "{\"to\":\"[EMAIL_REDACTED]\"}");
        assert_eq!(counts.get("email"), Some(&2));
        assert_eq!(counts.get("china_mobile"), Some(&1));
    }

    #[test]
    fn input_only_chain_skips_output_and_vice_versa() {
        let input_only = mask_chain(aisix_core::models::GuardrailHookPoint::Input);
        let mut resp = ChatResponse {
            id: "r".into(),
            model: "m".into(),
            message: ChatMessage::assistant("mail a@x.com"),
            finish_reason: aisix_gateway::FinishReason::Stop,
            usage: aisix_gateway::UsageStats::new(0, 0),
        };
        assert!(redact_chat_response(input_only.as_ref(), &mut resp).is_empty());
        assert_eq!(resp.message.content.as_deref(), Some("mail a@x.com"));

        let output_only = mask_chain(aisix_core::models::GuardrailHookPoint::Output);
        let mut req: ChatFormat = serde_json::from_value(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "mail a@x.com"}]
        }))
        .unwrap();
        assert!(redact_chat_format(output_only.as_ref(), &mut req).is_empty());
        assert_eq!(req.messages[0].content.as_deref(), Some("mail a@x.com"));
    }

    #[test]
    fn chat_response_masks_content_and_tool_args_json_safely() {
        let chain = both();
        let mut msg = ChatMessage::assistant("reach me at a@x.com");
        msg.extra.insert(
            "tool_calls".into(),
            json!([{
                "id": "call_1", "type": "function",
                // A number-typed phone stays untouched (JSON preserved);
                // the string email is masked.
                "function": {"name": "f", "arguments": "{\"phone\":13800138000,\"mail\":\"b@y.org\"}"}
            }]),
        );
        let mut resp = ChatResponse {
            id: "r".into(),
            model: "m".into(),
            message: msg,
            finish_reason: aisix_gateway::FinishReason::Stop,
            usage: aisix_gateway::UsageStats::new(0, 0),
        };
        let counts = redact_chat_response(chain.as_ref(), &mut resp);
        assert_eq!(
            resp.message.content.as_deref(),
            Some("reach me at [EMAIL_REDACTED]")
        );
        let args = resp.message.extra["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap();
        let parsed: Value = serde_json::from_str(args).expect("args stay valid JSON");
        assert_eq!(parsed["phone"], json!(13800138000u64));
        assert_eq!(parsed["mail"], json!("[EMAIL_REDACTED]"));
        assert_eq!(counts.get("email"), Some(&2));
    }

    #[test]
    fn anthropic_request_masks_system_text_blocks_and_tool_result() {
        let chain = both();
        let mut body = json!({
            "model": "claude",
            "system": [{"type": "text", "text": "user email a@x.com"}],
            "messages": [
                {"role": "user", "content": "call 13800138000"},
                {"role": "user", "content": [
                    {"type": "text", "text": "and b@y.org"},
                    {"type": "tool_result", "tool_use_id": "t1", "content": [
                        {"type": "text", "text": "result c@z.io"}
                    ]}
                ]}
            ]
        });
        let counts = redact_anthropic_request(chain.as_ref(), &mut body);
        assert_eq!(body["system"][0]["text"], "user email [EMAIL_REDACTED]");
        assert_eq!(
            body["messages"][0]["content"],
            "call [CHINA_MOBILE_REDACTED]"
        );
        assert_eq!(
            body["messages"][1]["content"][0]["text"],
            "and [EMAIL_REDACTED]"
        );
        assert_eq!(
            body["messages"][1]["content"][1]["content"][0]["text"],
            "result [EMAIL_REDACTED]",
        );
        assert_eq!(counts.get("email"), Some(&3));
    }

    #[test]
    fn anthropic_response_masks_text_and_tool_use_input() {
        let chain = both();
        let mut body = json!({
            "content": [
                {"type": "text", "text": "email a@x.com"},
                {"type": "tool_use", "id": "t", "name": "send",
                 "input": {"to": "b@y.org", "count": 3}}
            ]
        });
        let counts = redact_anthropic_response(chain.as_ref(), &mut body);
        assert_eq!(body["content"][0]["text"], "email [EMAIL_REDACTED]");
        assert_eq!(body["content"][1]["input"]["to"], "[EMAIL_REDACTED]");
        assert_eq!(body["content"][1]["input"]["count"], 3);
        assert_eq!(counts.get("email"), Some(&2));
    }

    #[test]
    fn responses_request_masks_string_and_item_forms() {
        let chain = both();
        let mut body = json!({
            "model": "m",
            "instructions": "never leak a@x.com",
            "input": [
                {"type": "message", "role": "user", "content": "call 13800138000"},
                {"role": "user", "content": [
                    {"type": "input_text", "text": "mail b@y.org"}
                ]},
                {"type": "function_call_output", "call_id": "c", "output": "from c@z.io"}
            ]
        });
        let counts = redact_responses_request(chain.as_ref(), &mut body);
        assert_eq!(body["instructions"], "never leak [EMAIL_REDACTED]");
        assert_eq!(body["input"][0]["content"], "call [CHINA_MOBILE_REDACTED]");
        assert_eq!(
            body["input"][1]["content"][0]["text"],
            "mail [EMAIL_REDACTED]"
        );
        assert_eq!(body["input"][2]["output"], "from [EMAIL_REDACTED]");
        assert_eq!(counts.get("email"), Some(&3));

        let mut simple = json!({"model": "m", "input": "mail a@x.com"});
        redact_responses_request(chain.as_ref(), &mut simple);
        assert_eq!(simple["input"], "mail [EMAIL_REDACTED]");
    }

    fn content_chunk(text: &str) -> ChatChunk {
        ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta {
                content: Some(text.to_string()),
                ..ChatDelta::default()
            },
            finish_reason: None,
            usage: None,
        }
    }

    #[test]
    fn stream_chunks_mask_span_split_across_chunk_boundary() {
        let chain = both();
        // "a@x.com" split across three chunks — per-chunk masking would miss it.
        let mut chunks = vec![
            content_chunk("mail a@"),
            content_chunk("x.c"),
            content_chunk("om now"),
        ];
        let counts = redact_chat_chunks(chain.as_ref(), &mut chunks);
        assert_eq!(counts.get("email"), Some(&1));
        let reassembled: String = chunks
            .iter()
            .map(|c| c.delta.content.as_deref().unwrap_or(""))
            .collect();
        assert_eq!(reassembled, "mail [EMAIL_REDACTED] now");
        // Full text lands on the first content chunk; the rest are empty.
        assert_eq!(
            chunks[0].delta.content.as_deref(),
            Some("mail [EMAIL_REDACTED] now")
        );
        assert_eq!(chunks[1].delta.content.as_deref(), Some(""));
    }

    #[test]
    fn stream_chunks_mask_tool_call_arguments_channel() {
        let chain = both();
        let mut chunks = vec![
            ChatChunk {
                id: "c".into(),
                model: "m".into(),
                delta: ChatDelta {
                    tool_calls: Some(vec![json!({
                        "index": 0, "id": "call_1", "type": "function",
                        "function": {"name": "send", "arguments": "{\"to\":\"a@"}
                    })]),
                    ..ChatDelta::default()
                },
                finish_reason: None,
                usage: None,
            },
            ChatChunk {
                id: "c".into(),
                model: "m".into(),
                delta: ChatDelta {
                    tool_calls: Some(vec![json!({
                        "index": 0,
                        "function": {"arguments": "x.com\"}"}
                    })]),
                    ..ChatDelta::default()
                },
                finish_reason: None,
                usage: None,
            },
        ];
        let counts = redact_chat_chunks(chain.as_ref(), &mut chunks);
        assert_eq!(counts.get("email"), Some(&1));
        let first_args = chunks[0].delta.tool_calls.as_ref().unwrap()[0]["function"]["arguments"]
            .as_str()
            .unwrap()
            .to_string();
        let second_args = chunks[1].delta.tool_calls.as_ref().unwrap()[0]["function"]["arguments"]
            .as_str()
            .unwrap();
        assert_eq!(first_args, "{\"to\":\"[EMAIL_REDACTED]\"}");
        assert_eq!(second_args, "");
    }

    #[test]
    fn stream_chunks_untouched_when_nothing_matches() {
        let chain = both();
        let mut chunks = vec![content_chunk("hello "), content_chunk("world")];
        assert!(redact_chat_chunks(chain.as_ref(), &mut chunks).is_empty());
        assert_eq!(chunks[0].delta.content.as_deref(), Some("hello "));
        assert_eq!(chunks[1].delta.content.as_deref(), Some("world"));
    }

    #[test]
    fn anthropic_sse_masks_text_delta_across_frames() {
        let chain = both();
        let raw = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"mail a@\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"x.com ok\"}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        );
        let (out, counts) = redact_anthropic_sse(chain.as_ref(), raw.as_bytes()).unwrap();
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains("mail [EMAIL_REDACTED] ok"), "out: {out}");
        assert!(!out.contains("a@x.com"));
        // Second delta emptied; frame structure + unrelated frames intact.
        assert!(
            out.contains("{\"type\":\"text_delta\",\"text\":\"\"}")
                || out.contains("\"text\":\"\"")
        );
        assert!(out.contains("message_start"));
        assert!(out.contains("message_stop"));
        assert_eq!(counts.get("email"), Some(&1));
    }

    #[test]
    fn anthropic_sse_masks_tool_use_input_json_channel() {
        let chain = both();
        let raw = concat!(
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t\",\"name\":\"send\",\"input\":{}}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"to\\\":\\\"a@\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"x.com\\\"}\"}}\n\n",
        );
        let (out, counts) = redact_anthropic_sse(chain.as_ref(), raw.as_bytes()).unwrap();
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains("[EMAIL_REDACTED]"), "out: {out}");
        assert!(!out.contains("a@"), "no split original fragments: {out}");
        assert_eq!(counts.get("email"), Some(&1));
    }

    #[test]
    fn responses_sse_masks_delta_channel_and_aggregate_events() {
        let chain = both();
        let raw = concat!(
            "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"mail a@\"}\n\n",
            "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"x.com ok\"}\n\n",
            "event: response.output_text.done\ndata: {\"type\":\"response.output_text.done\",\"item_id\":\"msg_1\",\"text\":\"mail a@x.com ok\"}\n\n",
            "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"mail a@x.com ok\"}]}]}}\n\n",
        );
        let (out, counts) = redact_responses_sse(chain.as_ref(), raw.as_bytes()).unwrap();
        let out = String::from_utf8(out).unwrap();
        assert!(!out.contains("a@x.com"), "original must be gone: {out}");
        // Delta channel: full masked text on the first delta; done +
        // completed events masked consistently.
        assert!(
            out.contains("\"delta\":\"mail [EMAIL_REDACTED] ok\""),
            "out: {out}"
        );
        assert!(
            out.contains("\"text\":\"mail [EMAIL_REDACTED] ok\""),
            "out: {out}"
        );
        // Aggregate events don't double-count the same span.
        assert_eq!(counts.get("email"), Some(&1));
    }

    #[test]
    fn responses_sse_masks_function_call_args_channel() {
        let chain = both();
        let raw = concat!(
            "event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\"{\\\"to\\\":\\\"a@\"}\n\n",
            "event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\"x.com\\\"}\"}\n\n",
            "event: response.function_call_arguments.done\ndata: {\"type\":\"response.function_call_arguments.done\",\"item_id\":\"fc_1\",\"arguments\":\"{\\\"to\\\":\\\"a@x.com\\\"}\"}\n\n",
        );
        let (out, counts) = redact_responses_sse(chain.as_ref(), raw.as_bytes()).unwrap();
        let out = String::from_utf8(out).unwrap();
        assert!(!out.contains("a@"), "original fragments gone: {out}");
        assert!(out.contains("[EMAIL_REDACTED]"), "out: {out}");
        assert_eq!(counts.get("email"), Some(&1));
    }

    #[test]
    fn responses_sse_returns_none_when_clean() {
        let chain = both();
        let raw = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"m\",\"delta\":\"hello\"}\n\n";
        assert!(redact_responses_sse(chain.as_ref(), raw.as_bytes()).is_none());
    }

    #[test]
    fn responses_response_json_masks_output_items() {
        let chain = both();
        let mut body = json!({
            "id": "resp_1",
            "output": [
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "mail a@x.com"}
                ]},
                {"type": "function_call", "call_id": "c", "name": "send",
                 "arguments": "{\"to\":\"b@y.org\"}"}
            ]
        });
        let counts = redact_responses_response(chain.as_ref(), &mut body);
        assert_eq!(
            body["output"][0]["content"][0]["text"],
            "mail [EMAIL_REDACTED]"
        );
        assert_eq!(
            body["output"][1]["arguments"],
            "{\"to\":\"[EMAIL_REDACTED]\"}"
        );
        assert_eq!(counts.get("email"), Some(&2));
    }

    #[test]
    fn anthropic_sse_returns_none_when_clean() {
        let chain = both();
        let raw = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n";
        assert!(redact_anthropic_sse(chain.as_ref(), raw.as_bytes()).is_none());
    }

    #[test]
    fn malformed_tool_args_fall_back_to_raw_text_masking() {
        let chain = both();
        let mut encoded = String::from("not json but has a@x.com inside");
        let mut counts = RedactionCounts::new();
        redact_json_encoded(chain.as_ref(), Direction::Output, &mut encoded, &mut counts);
        assert_eq!(encoded, "not json but has [EMAIL_REDACTED] inside");
        assert_eq!(counts.get("email"), Some(&1));
    }

    /// #696: rerank request masking covers `query` + both document shapes.
    #[test]
    fn rerank_request_masks_query_and_documents() {
        let chain = both();
        let mut body = json!({
            "model": "m",
            "query": "who is a@x.com",
            "documents": ["contact b@y.org", {"text": "reach c@z.io"}, 42]
        });
        let counts = redact_rerank_request(chain.as_ref(), &mut body);
        assert_eq!(body["query"], "who is [EMAIL_REDACTED]");
        assert_eq!(body["documents"][0], "contact [EMAIL_REDACTED]");
        assert_eq!(body["documents"][1]["text"], "reach [EMAIL_REDACTED]");
        assert_eq!(counts.get("email"), Some(&3));
    }

    /// #696: images request masking covers `prompt`.
    #[test]
    fn images_request_masks_prompt() {
        let chain = both();
        let mut body = json!({"model": "m", "prompt": "portrait of a@x.com"});
        let counts = redact_images_request(chain.as_ref(), &mut body);
        assert_eq!(body["prompt"], "portrait of [EMAIL_REDACTED]");
        assert_eq!(counts.get("email"), Some(&1));
    }

    /// #696: speech (TTS) request masking covers `input`.
    #[test]
    fn speech_request_masks_input() {
        let chain = both();
        let mut body = json!({"model": "m", "input": "read a@x.com aloud", "voice": "alloy"});
        let counts = redact_speech_request(chain.as_ref(), &mut body);
        assert_eq!(body["input"], "read [EMAIL_REDACTED] aloud");
        assert_eq!(counts.get("email"), Some(&1));
    }

    /// #696: transcription response masking rewrites the JSON `text` +
    /// `segments[].text` (verbose_json) and reports counts.
    #[test]
    fn transcription_response_masks_json_text_and_segments() {
        let chain = both();
        let body = json!({
            "text": "mail a@x.com",
            "segments": [{"id": 0, "text": "mail a@x.com"}]
        });
        let (rewritten, counts) =
            redact_transcription_response(chain.as_ref(), &serde_json::to_vec(&body).unwrap())
                .expect("must rewrite");
        let v: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(v["text"], "mail [EMAIL_REDACTED]");
        assert_eq!(v["segments"][0]["text"], "mail [EMAIL_REDACTED]");
        assert_eq!(counts.get("email"), Some(&2));
    }

    /// #696: the raw-text response formats (`text` / `srt` / `vtt`) are
    /// masked as plain text; a clean body returns None (kept as-is).
    #[test]
    fn transcription_response_masks_raw_text_formats() {
        let chain = both();
        let (rewritten, counts) =
            redact_transcription_response(chain.as_ref(), b"speaker: a@x.com\n").expect("rewrite");
        assert_eq!(rewritten, b"speaker: [EMAIL_REDACTED]\n");
        assert_eq!(counts.get("email"), Some(&1));
        assert!(redact_transcription_response(chain.as_ref(), b"all clean\n").is_none());
    }

    // ── remote segment moderation (#932 bedrock follow-up) ──────────────

    use aisix_guardrails::{GuardrailVerdict, SegmentsOutcome};

    /// Stub of a Bedrock-style segment moderator: masks slot i to
    /// `"<M{i}:UPPER(text)>"` — index-stamped so a positional mix-up is
    /// unmissable — and reports a fixed entity count. `verdict` lets the
    /// block/bypass paths be exercised; `panic_if_called` pins the
    /// skip-when-already-blocked contract.
    struct StubSegments {
        verdict: GuardrailVerdict,
        mask: bool,
        panic_if_called: bool,
    }

    impl StubSegments {
        fn masker() -> Self {
            Self {
                verdict: GuardrailVerdict::Allow,
                mask: true,
                panic_if_called: false,
            }
        }
    }

    #[async_trait::async_trait]
    impl Guardrail for StubSegments {
        fn name(&self) -> &'static str {
            "stub-segments"
        }
        fn moderates_segments(&self) -> bool {
            true
        }
        async fn moderate_input_segments(&self, texts: &[String]) -> SegmentsOutcome {
            self.moderate(texts)
        }
        async fn moderate_output_segments(&self, texts: &[String]) -> SegmentsOutcome {
            self.moderate(texts)
        }
    }

    impl StubSegments {
        fn moderate(&self, texts: &[String]) -> SegmentsOutcome {
            if self.panic_if_called {
                panic!("segment moderator must not be called on this path");
            }
            let mut counts = RedactionCounts::new();
            counts.insert("EMAIL".to_owned(), 1);
            SegmentsOutcome {
                verdict: self.verdict.clone(),
                masked: self.mask.then(|| {
                    texts
                        .iter()
                        .enumerate()
                        .map(|(i, t)| format!("<M{i}:{}>", t.to_uppercase()))
                        .collect()
                }),
                counts,
                monitor_hits: Vec::new(),
            }
        }
    }

    fn seg_chain(stub: StubSegments) -> GuardrailChain {
        GuardrailChain::new(vec![Arc::new(stub)])
    }

    /// The collect→call→apply round trip over the chat walker: every slot
    /// kind (flat content, text block, tool-call JSON argument) gets its
    /// OWN positionally-matched mask, and the provider counts — not the
    /// applier's plumbing marker — land in `counts_out`.
    #[tokio::test]
    async fn moderate_body_masks_chat_slots_positionally() {
        let chain = seg_chain(StubSegments::masker());
        let mut req: ChatFormat = serde_json::from_value(json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "first slot"},
                {"role": "user", "content": "", "content_blocks": [
                    {"type": "text", "text": "second slot"}
                ]},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"index": 0, "function": {"name": "send", "arguments": "{\"to\":\"third slot\"}"}}
                ]}
            ]
        }))
        .unwrap();
        let mut counts = RedactionCounts::new();
        let verdict = moderate_body(
            &chain,
            Direction::Input,
            GuardrailVerdict::Allow,
            &mut counts,
            &mut Vec::new(),
            |g| redact_chat_format(g, &mut req),
        )
        .await;
        assert_eq!(verdict, GuardrailVerdict::Allow);
        assert_eq!(
            req.messages[0].content.as_deref(),
            Some("<M0:FIRST SLOT>"),
            "flat content = slot 0",
        );
        assert_eq!(
            req.messages[1].content_blocks.as_ref().unwrap()[0]["text"],
            "<M1:SECOND SLOT>",
            "text block = slot 1",
        );
        assert_eq!(
            req.messages[2].extra["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
            "{\"to\":\"<M2:THIRD SLOT>\"}",
            "tool-arg inner string = slot 2 (marker counts must fire the \
             json-encoded rewrite gate)",
        );
        assert_eq!(counts.get("EMAIL"), Some(&1), "provider counts merged");
        assert!(
            !counts.keys().any(|k| k.starts_with("__")),
            "the applier's plumbing marker must never leak into telemetry counts",
        );
    }

    /// A Block from the segment pass leaves the body untouched (no mask
    /// write-back on a dead request) and propagates the verdict.
    #[tokio::test]
    async fn moderate_body_block_leaves_body_untouched() {
        let chain = seg_chain(StubSegments {
            verdict: GuardrailVerdict::block("pii blocked"),
            mask: true,
            panic_if_called: false,
        });
        let mut req: ChatFormat = serde_json::from_value(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "original"}]
        }))
        .unwrap();
        let mut counts = RedactionCounts::new();
        let verdict = moderate_body(
            &chain,
            Direction::Input,
            GuardrailVerdict::Allow,
            &mut counts,
            &mut Vec::new(),
            |g| redact_chat_format(g, &mut req),
        )
        .await;
        assert!(verdict.is_block());
        assert_eq!(req.messages[0].content.as_deref(), Some("original"));
        assert!(counts.is_empty(), "no counts on a blocked request");
    }

    /// An already-blocked prior verdict skips the remote call entirely
    /// (the request is dead — don't burn a provider call), and a chain
    /// with no segment member is a no-op.
    #[tokio::test]
    async fn moderate_body_skips_remote_when_blocked_or_absent() {
        let chain = seg_chain(StubSegments {
            verdict: GuardrailVerdict::Allow,
            mask: false,
            panic_if_called: true,
        });
        let mut req: ChatFormat = serde_json::from_value(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "x"}]
        }))
        .unwrap();
        let mut counts = RedactionCounts::new();
        let verdict = moderate_body(
            &chain,
            Direction::Input,
            GuardrailVerdict::block("already blocked"),
            &mut counts,
            &mut Vec::new(),
            |g| redact_chat_format(g, &mut req),
        )
        .await;
        assert!(verdict.is_block(), "prior Block passes through");

        // A sync-only (non-segment) chain never enters the pass.
        let sync_only = both();
        let verdict = moderate_body(
            sync_only.as_ref(),
            Direction::Input,
            GuardrailVerdict::Allow,
            &mut counts,
            &mut Vec::new(),
            |_| panic!("walk must not run when no segment member exists"),
        )
        .await;
        assert_eq!(verdict, GuardrailVerdict::Allow);
    }

    /// The round trip through the Anthropic SSE walker: the masked
    /// channel text lands on the channel's first frame (later frames
    /// empty) even though the applier only returns marker counts — the
    /// gate that discards count-less SSE rewrites must fire on them.
    #[tokio::test]
    async fn moderate_body_masks_anthropic_sse_channels() {
        let chain = seg_chain(StubSegments::masker());
        let mut held: Vec<u8> = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello \"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"world\"}}\n\n",
        )
        .as_bytes()
        .to_vec();
        let mut counts = RedactionCounts::new();
        let verdict = moderate_body(
            &chain,
            Direction::Output,
            GuardrailVerdict::Allow,
            &mut counts,
            &mut Vec::new(),
            |g| match redact_anthropic_sse(g, &held) {
                Some((rewritten, c)) => {
                    held = rewritten;
                    c
                }
                None => RedactionCounts::new(),
            },
        )
        .await;
        assert_eq!(verdict, GuardrailVerdict::Allow);
        let out = String::from_utf8(held.clone()).unwrap();
        assert!(
            out.contains("<M0:HELLO WORLD>"),
            "channel text masked as one positional slot: {out}",
        );
        assert_eq!(counts.get("EMAIL"), Some(&1));
        // The capture-rebuild helper reads the masked channel back.
        assert_eq!(anthropic_sse_text(&held), "<M0:HELLO WORLD>");
    }

    /// `responses_sse_text` assembles `output_text` deltas per channel —
    /// the capture-rebuild source after a segment mask.
    #[test]
    fn responses_sse_text_assembles_channels() {
        let raw = concat!(
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"i1\",\"content_index\":0,\"delta\":\"foo \"}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"i1\",\"content_index\":0,\"delta\":\"bar\"}\n\n",
        );
        assert_eq!(responses_sse_text(raw.as_bytes()), "foo bar");
    }

    /// Channels concatenate in first-seen (emission) order, not item-id
    /// lexicographic order — the rebuilt capture must read like the
    /// stream the client saw.
    #[test]
    fn responses_sse_text_preserves_emission_order() {
        let raw = concat!(
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"zzz\",\"content_index\":0,\"delta\":\"first \"}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"aaa\",\"content_index\":0,\"delta\":\"second\"}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"zzz\",\"content_index\":0,\"delta\":\"more\"}\n\n",
        );
        assert_eq!(responses_sse_text(raw.as_bytes()), "first moresecond");
    }
}
