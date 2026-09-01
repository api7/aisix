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
//! which carries the upstream's own id. This module is the one place that
//! difference is repaired, so the family cannot drift again.
//!
//! Two shapes, because a native path answers in two:
//!
//! - [`restamp_body`] for a parsed JSON response.
//! - [`restamp_sse_frame`] for one frame of a streamed response. It splices
//!   the value through [`crate::json_splice`] rather than re-serialising the
//!   frame, so every byte the gateway is not deliberately changing — key
//!   order, whitespace, number spellings, escape choices — reaches the client
//!   exactly as the provider wrote it.
//!
//! Both are deliberately no-ops when the document carries no `model` at the
//! selected path. A response that never had the field does not acquire one.

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
    let spliced = match json_splice::rewrite_string_values(payload, selects_model, |_| {
        Some(client_facing_model.to_string())
    }) {
        Ok(Some(bytes)) => bytes,
        Ok(None) => return None,
        Err(error) => {
            // The scanner fails whole rather than half-rewriting. A frame it
            // cannot read forwards verbatim: leaking the upstream id on one
            // frame is a smaller harm than corrupting the stream.
            tracing::debug!(%error, "sse frame model restamp skipped; forwarding verbatim");
            return None;
        }
    };
    let mut out = Vec::with_capacity(frame.len() - payload.len() + spliced.len());
    out.extend_from_slice(&frame[..range.start]);
    out.extend_from_slice(&spliced);
    out.extend_from_slice(&frame[range.end..]);
    Some(out)
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
