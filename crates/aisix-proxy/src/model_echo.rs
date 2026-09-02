//! Restamp the client-facing `model` field onto a native passthrough response.
//!
//! The wire contract is stated once, in [`crate::render`]: `response.model`
//! carries the model name the CALLER addressed — the alias they put on the
//! request — never the upstream provider's own id. That is what makes an
//! alias an alias: the caller names one thing, and every answer they get back
//! names the same thing, whichever provider or routing target actually served
//! it.
//!
//! The bridged dispatch paths satisfy that by construction: they build the
//! client-facing body themselves and stamp the caller's name into it. The
//! NATIVE passthrough paths do not — they forward the upstream's own document,
//! which carries the upstream's own id. This module is where that difference
//! is repaired, so the paths that use it cannot drift apart again.
//!
//! `/v1/fine_tuning/jobs` is deliberately outside the family: its request
//! half forwards the caller's `model` to the provider verbatim, so naming the
//! upstream base model on both halves is the symmetric answer there (#1089).
//!
//! Three shapes, because a native path answers in three:
//!
//! - [`restamp_body`] for a parsed JSON response.
//! - [`restamp_json_bytes`] for a response the path relays as raw bytes
//!   rather than re-serialising (`/v1/rerank`), and for one WebSocket text
//!   frame (`/v1/realtime`).
//! - [`restamp_sse_frame`] for one frame of a streamed response.
//!
//! The last two splice the value through [`crate::json_splice`] rather than
//! re-serialising the document, so every byte the gateway is not deliberately
//! changing — key order, whitespace, number spellings, escape choices —
//! reaches the client exactly as the provider wrote it.
//!
//! All three are deliberately no-ops when the document carries no `model` at
//! the selected path. A response that never had the field does not acquire
//! one.

use serde_json::Value;

use crate::json_splice::{self, PathSeg};

/// Restamp the top-level `model` of a parsed response body.
///
/// Unconditional: whatever the upstream reported is replaced. It must NOT be
/// conditioned on the upstream having echoed the configured `model_name` —
/// that guard is what let the alias leak whenever a provider answered with a
/// different id than the one it was asked for, which is the common case
/// (a dated snapshot id, or a name the provider remaps server-side).
pub(crate) fn restamp_body(body: &mut Value, client_facing_model: &str) {
    if let Some(m) = body.get_mut("model") {
        *m = Value::String(client_facing_model.to_string());
    }
}

/// Restamp the `model` value inside a raw JSON document, byte-preserving.
///
/// For the paths that relay the provider's own bytes instead of
/// re-serialising a parsed body — `/v1/rerank`, and one `/v1/realtime`
/// WebSocket text frame. Returns `None` when the document carries no value
/// at the selected path, or when the scanner refuses it; a `None` caller
/// forwards the original bytes.
pub(crate) fn restamp_json_bytes(
    doc: &[u8],
    client_facing_model: &str,
    selects_model: fn(&[PathSeg]) -> bool,
) -> Option<Vec<u8>> {
    splice_model_value(doc, selects_model, |_| {
        Some(client_facing_model.to_string())
    })
}

/// Splice new text into the `model` values `selects_model` picks out.
///
/// The one failure policy every caller shares: the scanner fails whole
/// rather than half-rewriting, and a document it cannot read is forwarded
/// verbatim — leaking the upstream id on one frame is a smaller harm than
/// corrupting the stream.
///
/// `rewrite` sees the value as the document wrote it, so a caller that only
/// wants to translate ONE name back (the realtime relay's client-to-upstream
/// direction) answers `None` for every other and leaves those bytes alone.
pub(crate) fn splice_model_value(
    doc: &[u8],
    selects_model: fn(&[PathSeg]) -> bool,
    rewrite: impl FnMut(&str) -> Option<String>,
) -> Option<Vec<u8>> {
    match json_splice::rewrite_string_values(doc, selects_model, rewrite) {
        Ok(Some(bytes)) => Some(bytes),
        Ok(None) => None,
        Err(error) => {
            tracing::debug!(%error, "model restamp skipped; forwarding verbatim");
            None
        }
    }
}

/// Restamp the `model` value inside one SSE frame's `data:` payload.
///
/// Returns the rewritten frame, or `None` when the frame carries no value at
/// the selected path (the common case — most frames in a stream have no
/// `model` at all) or when its payload is not splice-able JSON. A `None`
/// caller forwards the original bytes.
pub(crate) fn restamp_sse_frame(
    frame: &[u8],
    client_facing_model: &str,
    selects_model: fn(&[PathSeg]) -> bool,
) -> Option<Vec<u8>> {
    let range = crate::messages::extract_sse_data_range(frame)?;
    let payload = &frame[range.clone()];
    // The terminal `[DONE]` sentinel is not JSON.
    if payload == b"[DONE]" {
        return None;
    }
    let spliced = restamp_json_bytes(payload, client_facing_model, selects_model)?;
    let mut out = Vec::with_capacity(frame.len() - payload.len() + spliced.len());
    out.extend_from_slice(&frame[..range.start]);
    out.extend_from_slice(&spliced);
    out.extend_from_slice(&frame[range.end..]);
    Some(out)
}

/// Restamp every frame of a COMPLETE, already-buffered SSE document.
///
/// The streaming relays splice frame-by-frame as they drain, but a
/// block-capable output guardrail buffers the whole response and returns it
/// as one body without ever reaching the relay — so that branch needs the
/// same pass applied in one go, or the identical request answers with the
/// upstream id whenever such a guardrail is attached.
pub(crate) fn restamp_sse_buffer(
    buf: &[u8],
    client_facing_model: &str,
    selects_model: fn(&[PathSeg]) -> bool,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(buf.len());
    let mut rest = buf;
    while let Some(end) = crate::messages::find_frame_end(rest) {
        let (frame, tail) = rest.split_at(end);
        match restamp_sse_frame(frame, client_facing_model, selects_model) {
            Some(rewritten) => out.extend_from_slice(&rewritten),
            None => out.extend_from_slice(frame),
        }
        rest = tail;
    }
    // The document is COMPLETE, so a trailing fragment is not a frame still
    // arriving — it is a final frame the upstream never terminated, and it is
    // the one carrying `response.completed` when a provider omits the last
    // blank line. Splice it too. (The streaming relays are the opposite case
    // and must keep holding their tail: more bytes may still come.) The other
    // readers of this same buffer parse it line-by-line, so they already see
    // this frame; skipping it here would make the echo the only thing that
    // misses it.
    match restamp_sse_frame(rest, client_facing_model, selects_model) {
        Some(rewritten) => out.extend_from_slice(&rewritten),
        None => out.extend_from_slice(rest),
    }
    out
}

/// A `model` at the document's TOP level.
///
/// The shape `/v1/rerank` answers in: among the supported rerank backends
/// only Jina's response names a model, and it names it here. The Cohere and
/// OpenAI-compatible shapes carry none, so the splice is a no-op on them
/// rather than inventing the field.
///
/// Depth-exact, so a `model` nested inside a result or a `meta` block — not
/// the caller-facing field — is left alone.
pub(crate) fn top_level_model(path: &[PathSeg]) -> bool {
    path.len() == 1 && path[0].is_key("model")
}

/// `session.model` on a Realtime `session.created` / `session.updated` frame.
///
/// Those two are the only Realtime server events that name a model: the
/// Response object a `response.created` / `response.done` frame carries has
/// no `model` field at all.
///
/// Depth-exact for a second reason here. A session's audio config can name a
/// SEPARATE transcription model at
/// `session.audio.input.transcription.model`, which the client chose itself
/// and the gateway never aliased — restamping that one would rename a model
/// the caller is entitled to see.
pub(crate) fn realtime_session_model(path: &[PathSeg]) -> bool {
    path.len() == 2 && path[0].is_key("session") && path[1].is_key("model")
}

/// `message.model` on an Anthropic `message_start` frame.
///
/// The path alone identifies the frame: `message_start` is the only Anthropic
/// event whose payload nests a `message` object, so no type check is needed.
pub(crate) fn anthropic_message_model(path: &[PathSeg]) -> bool {
    path.len() == 2 && path[0].is_key("message") && path[1].is_key("model")
}

/// `response.model` on a Responses-API snapshot frame.
///
/// Exact-depth, so it selects the six events that carry a whole `response`
/// object — `response.created` / `.in_progress` / `.completed` /
/// `.incomplete` / `.failed` / `.queued` — and nothing nested deeper inside
/// one (an output item, a tool definition).
pub(crate) fn responses_snapshot_model(path: &[PathSeg]) -> bool {
    path.len() == 2 && path[0].is_key("response") && path[1].is_key("model")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restamp_body_replaces_whatever_the_upstream_reported() {
        // The upstream answered with a dated snapshot id, not the name it was
        // asked for. The caller still addressed `ds-native`.
        let mut body = serde_json::json!({"id": "msg_1", "model": "deepseek-v4-flash"});
        restamp_body(&mut body, "ds-native");
        assert_eq!(body["model"], "ds-native");
    }

    #[test]
    fn restamp_body_does_not_invent_the_field() {
        let mut body = serde_json::json!({"input_tokens": 7});
        restamp_body(&mut body, "ds-native");
        assert!(body.get("model").is_none());
    }

    #[test]
    fn restamp_sse_frame_rewrites_message_start_and_leaves_the_rest_byte_identical() {
        let frame = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-x-20250101\",\"usage\":{\"input_tokens\":1e2}}}\n\n";
        let out = restamp_sse_frame(frame, "my-claude", anthropic_message_model)
            .expect("message_start carries message.model");
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains("\"model\":\"my-claude\""));
        // Everything outside the spliced value survives as written — the
        // event line, the key order, and the number spelling a
        // `serde_json` round-trip would have canonicalised to `100.0`.
        assert!(out.starts_with("event: message_start\ndata: {\"type\":\"message_start\""));
        assert!(out.contains("\"input_tokens\":1e2"));
        assert!(out.ends_with("}}}\n\n"));
    }

    #[test]
    fn restamp_sse_frame_skips_frames_without_the_path() {
        // `message_delta` has no `message` object; `content_block_delta` has
        // no model anywhere. Both forward verbatim.
        for frame in [
            &b"event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":9}}\n\n"[..],
            &b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hi\"}}\n\n"[..],
            &b"data: [DONE]\n\n"[..],
        ] {
            assert!(restamp_sse_frame(frame, "my-claude", anthropic_message_model).is_none());
        }
    }

    #[test]
    fn restamp_sse_buffer_rewrites_every_snapshot_frame_and_preserves_the_rest() {
        let buf = concat!(
            "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"model\":\"up-1\"}}\n\n",
            "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n",
            "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"model\":\"up-1\",\"usage\":{\"input_tokens\":1}}}\n\n",
            "data: [DONE]\n\n",
        )
        .as_bytes();
        let out =
            String::from_utf8(restamp_sse_buffer(buf, "alias", responses_snapshot_model)).unwrap();
        assert_eq!(out.matches("\"model\":\"alias\"").count(), 2);
        assert!(!out.contains("up-1"));
        // Untouched frames survive byte-for-byte, terminator included.
        assert!(
            out.contains("data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n")
        );
        assert!(out.ends_with("data: [DONE]\n\n"));
    }

    /// The no-match path must be an identity function on bytes: this runs
    /// over a whole buffered response, so anything it perturbs reaches the
    /// client. Includes shapes the frame walk could mishandle — an empty
    /// buffer, a lone unterminated fragment, and a final frame whose
    /// terminator is missing.
    #[test]
    fn restamp_sse_buffer_is_byte_identity_when_nothing_matches() {
        for buf in [
            &b""[..],
            &b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n"[..],
            &b"event: ping\ndata: {}\n\ndata: [DONE]\n\n"[..],
            // No terminator at all, and nothing to match inside it.
            &b"data: {\"type\":\"response.created\""[..],
            // A complete frame followed by an unterminated remainder.
            &b"data: {\"a\":1}\n\ndata: {\"b\":2"[..],
            // CRLF framing.
            &b"event: x\r\ndata: {\"a\":1}\r\n\r\n"[..],
        ] {
            assert_eq!(
                restamp_sse_buffer(buf, "alias", responses_snapshot_model),
                buf.to_vec(),
                "no match must not perturb a byte: {:?}",
                String::from_utf8_lossy(buf),
            );
        }
    }

    /// A provider that omits the final blank line still gets its last frame
    /// restamped. That frame is `response.completed` — the one an SDK builds
    /// its final Response object from — so skipping it would leak the
    /// upstream id on exactly the event that matters most, and only when a
    /// buffering guardrail is attached.
    #[test]
    fn restamp_sse_buffer_splices_a_final_frame_that_was_never_terminated() {
        let buf = concat!(
            "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"model\":\"up-1\"}}\n\n",
            "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"model\":\"up-1\"}}",
        )
        .as_bytes();
        let out =
            String::from_utf8(restamp_sse_buffer(buf, "alias", responses_snapshot_model)).unwrap();
        assert_eq!(out.matches("\"model\":\"alias\"").count(), 2);
        assert!(
            !out.contains("up-1"),
            "the unterminated final frame too: {out}"
        );
        // Still no terminator invented for it.
        assert!(out.ends_with("}}"));
    }

    /// The first depth-1 predicate in production code (`/v1/rerank`). The
    /// depth-2 ones above cannot exercise a top-level match, so this pins it
    /// directly rather than assuming the walker offers a root-level value
    /// the same way it offers a nested one.
    #[test]
    fn top_level_predicate_selects_the_root_model_only() {
        // A Jina rerank response: the model is named at the root.
        let body = br#"{"model":"jina-reranker-v2-base-multilingual","results":[{"index":0,"relevance_score":0.9}],"usage":{"total_tokens":11}}"#;
        let out = restamp_json_bytes(body, "my-reranker", top_level_model)
            .expect("a top-level `model` is selected");
        let out = String::from_utf8(out).unwrap();
        assert!(out.starts_with(r#"{"model":"my-reranker","results":"#));
        // Everything outside the replaced value is byte-identical, the
        // score's spelling included.
        assert!(out.contains(r#""relevance_score":0.9"#));
        assert!(out.ends_with(r#""usage":{"total_tokens":11}}"#));

        // A Cohere-shape response names no model — nothing to rewrite, and
        // the field is not invented.
        assert!(restamp_json_bytes(
            br#"{"id":"rr-1","results":[{"index":0}]}"#,
            "my-reranker",
            top_level_model,
        )
        .is_none());

        // A `model` nested anywhere deeper is NOT the caller-facing field.
        assert!(restamp_json_bytes(
            br#"{"results":[{"model":"aux"}],"meta":{"model":"aux"}}"#,
            "my-reranker",
            top_level_model,
        )
        .is_none());
    }

    #[test]
    fn realtime_predicate_selects_the_session_model_only() {
        let frame = br#"{"type":"session.created","event_id":"e1","session":{"id":"sess_1","model":"gpt-realtime-2026-01-01","output_modalities":["audio"]}}"#;
        let out = restamp_json_bytes(frame, "my-realtime", realtime_session_model)
            .expect("session.created carries session.model");
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains(r#""model":"my-realtime""#));
        assert!(out.starts_with(r#"{"type":"session.created","event_id":"e1""#));
        assert!(out.ends_with(r#""output_modalities":["audio"]}}"#));

        // The transcription model the CLIENT configured sits deeper and is
        // not an alias the gateway handed out — leave it as written.
        assert!(restamp_json_bytes(
            br#"{"type":"session.updated","session":{"audio":{"input":{"transcription":{"model":"gpt-4o-transcribe"}}}}}"#,
            "my-realtime",
            realtime_session_model,
        )
        .is_none());

        // A `response.done` frame names no model at all.
        assert!(restamp_json_bytes(
            br#"{"type":"response.done","response":{"id":"resp_1","usage":{"input_tokens":9}}}"#,
            "my-realtime",
            realtime_session_model,
        )
        .is_none());
    }

    /// The reverse direction the realtime relay uses on its way UP: only the
    /// alias the gateway handed out is translated back, so a client naming
    /// anything else still reaches the upstream with its own words.
    #[test]
    fn splice_model_value_rewrites_only_what_the_closure_answers_for() {
        let update = |model: &str| {
            format!(
                r#"{{"type":"session.update","session":{{"model":"{model}","voice":"alloy"}}}}"#
            )
        };
        let translate = |v: &str| (v == "my-realtime").then(|| "gpt-realtime".to_string());

        let out = splice_model_value(
            update("my-realtime").as_bytes(),
            realtime_session_model,
            translate,
        )
        .expect("the alias is translated back");
        assert!(String::from_utf8(out)
            .unwrap()
            .contains(r#""model":"gpt-realtime""#));

        assert!(splice_model_value(
            update("some-other-model").as_bytes(),
            realtime_session_model,
            translate,
        )
        .is_none());
    }

    #[test]
    fn responses_predicate_selects_the_snapshot_frames_only() {
        let snapshot = b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4o-mini-2024-07-18\"}}\n\n";
        let out = restamp_sse_frame(snapshot, "gpt4o-mini", responses_snapshot_model)
            .expect("response.completed carries response.model");
        assert!(String::from_utf8(out)
            .unwrap()
            .contains("\"model\":\"gpt4o-mini\""));

        // A model named deeper inside the response object is NOT the
        // caller-facing field and must not be touched.
        let nested = b"data: {\"type\":\"response.output_item.done\",\"item\":{\"model\":\"whisper-1\"}}\n\n";
        assert!(restamp_sse_frame(nested, "gpt4o-mini", responses_snapshot_model).is_none());
        let deep = b"data: {\"type\":\"response.completed\",\"response\":{\"tools\":[{\"model\":\"aux\"}]}}\n\n";
        assert!(restamp_sse_frame(deep, "gpt4o-mini", responses_snapshot_model).is_none());
    }
}
