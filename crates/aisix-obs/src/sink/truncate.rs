//! Structure-preserving truncation for captured prompt/response content.
//!
//! Captured content is frequently serialized JSON (the chat prompt is the
//! serialized request; image/embedding responses are serialized JSON too).
//! A blunt byte cut at `content_max_bytes` breaks the JSON mid-value, so a
//! log console can no longer render the field hierarchy of a large payload
//! (AISIX-Cloud#1014). When oversized content parses as JSON it is instead
//! reduced *structurally*, so the logged field stays valid JSON:
//!
//! - long string values keep a prefix plus an inline
//!   `...[aisix: truncated, N bytes total]` marker inside the string;
//! - base64 data URIs collapse to a compact placeholder (base64 media in a
//!   captured prompt already did, whatever its size: [`omit_base64_media`]);
//! - long arrays keep head+tail samples around an
//!   `{"_aisix_truncated": true, "omitted_items": N}` placeholder element;
//! - objects keep every key so the shape stays navigable.
//!
//! Reduction walks [`LADDER`] until the serialized result fits the cap.
//! Non-JSON content — and pathological JSON that cannot fit even at the
//! tightest rung — falls back to the historical UTF-8-safe byte cut, so
//! the byte cap is always honored. Re-serialization orders object keys
//! alphabetically (serde_json map ordering); accepted for over-cap
//! payloads in exchange for keeping the structure intact.

use std::borrow::Cow;

use serde_json::Value;

/// Reduction rungs, tried in order: (per-string byte cap, array head keep,
/// array tail keep). Ends tight enough that anything structurally
/// reducible fits well below the smallest real-world cap.
const LADDER: &[(usize, usize, usize)] = &[
    (8192, 64, 16),
    (2048, 24, 8),
    (512, 10, 4),
    (128, 5, 2),
    (32, 2, 1),
];

/// Truncate `s` to at most `max_bytes`. Oversized valid JSON is reduced
/// structurally (stays valid JSON); anything else falls back to a
/// UTF-8-boundary byte cut. Returns the content and whether it was cut.
///
/// Worst-case cost (only when the content exceeds the cap AND a
/// full-content exporter is enabled): one parse, one stats pass, and up
/// to |LADDER| build+serialize rounds over the tree. Structurally
/// irreducible payloads (no long strings, no long arrays) skip every
/// rung and pay parse + one compact serialize before the byte-cut
/// fallback.
pub(crate) fn truncate_content(s: &str, max_bytes: usize) -> (Cow<'_, str>, bool) {
    if s.len() <= max_bytes {
        return (Cow::Borrowed(s), false);
    }
    if let Ok(root) = serde_json::from_str::<Value>(s) {
        // Whitespace-heavy (pretty-printed) input may already fit once
        // compacted, with nothing dropped; this is also the baseline the
        // rung skip below compares against.
        if let Ok(compact) = serde_json::to_string(&root) {
            if compact.len() <= max_bytes {
                return (Cow::Owned(compact), true);
            }
        }
        let (mut max_str, mut max_arr) = (0usize, 0usize);
        measure(&root, &mut max_str, &mut max_arr);
        for &(string_cap, head, tail) in LADDER {
            // A rung that caps no string and samples no array reproduces
            // the compact serialization — already known not to fit.
            if string_cap >= max_str && head + tail + 1 >= max_arr {
                continue;
            }
            let reduced = shrink(&root, string_cap, head, tail);
            if let Ok(out) = serde_json::to_string(&reduced) {
                if out.len() <= max_bytes {
                    return (Cow::Owned(out), true);
                }
            }
        }
    }
    let (cut, _) = truncate_on_char_boundary(s, max_bytes);
    (Cow::Borrowed(cut), true)
}

/// One pass over the tree: the largest string (bytes) and largest array
/// (element count) anywhere — used to skip ladder rungs that cannot
/// change anything.
fn measure(v: &Value, max_str: &mut usize, max_arr: &mut usize) {
    match v {
        Value::String(s) => *max_str = (*max_str).max(s.len()),
        Value::Array(items) => {
            *max_arr = (*max_arr).max(items.len());
            for i in items {
                measure(i, max_str, max_arr);
            }
        }
        Value::Object(map) => {
            for val in map.values() {
                measure(val, max_str, max_arr);
            }
        }
        _ => {}
    }
}

/// Truncate `s` to at most `max_bytes`, backing up to the previous UTF-8 char
/// boundary so the slice stays valid. Returns the slice and whether it was cut.
pub(crate) fn truncate_on_char_boundary(s: &str, max_bytes: usize) -> (&str, bool) {
    if s.len() <= max_bytes {
        return (s, false);
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    (&s[..end], true)
}

/// Build a reduced copy of `v`: strings capped at `string_cap` bytes,
/// arrays sampled to `head`+`tail` items around a placeholder element,
/// object keys and scalar values kept as-is.
fn shrink(v: &Value, string_cap: usize, head: usize, tail: usize) -> Value {
    match v {
        Value::String(s) => Value::String(shrink_string(s, string_cap)),
        Value::Array(items) => {
            if items.len() > head + tail + 1 {
                let mut out = Vec::with_capacity(head + tail + 1);
                out.extend(
                    items[..head]
                        .iter()
                        .map(|i| shrink(i, string_cap, head, tail)),
                );
                // A dropped element that is itself a placeholder from an
                // earlier (larger-cap) truncation pass stands for
                // `omitted_items` originals, not 1 — fold its count in so
                // the accounting survives two-stage capture→exporter
                // truncation.
                let omitted: u64 = items[head..items.len() - tail]
                    .iter()
                    .map(|i| placeholder_omitted(i).unwrap_or(1))
                    .sum();
                out.push(serde_json::json!({
                    "_aisix_truncated": true,
                    "omitted_items": omitted,
                }));
                out.extend(
                    items[items.len() - tail..]
                        .iter()
                        .map(|i| shrink(i, string_cap, head, tail)),
                );
                Value::Array(out)
            } else {
                Value::Array(
                    items
                        .iter()
                        .map(|i| shrink(i, string_cap, head, tail))
                        .collect(),
                )
            }
        }
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, val)| (k.clone(), shrink(val, string_cap, head, tail)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// `Some(count)` when `v` is an array-sampling placeholder produced by
/// [`shrink`]; the count it stands for (1 if the field is malformed).
fn placeholder_omitted(v: &Value) -> Option<u64> {
    (v.get("_aisix_truncated") == Some(&Value::Bool(true)))
        .then(|| v.get("omitted_items").and_then(|n| n.as_u64()).unwrap_or(1))
}

const BASE64_OMITTED_PREFIX: &str = "[aisix: base64 data uri omitted,";

fn base64_omitted(original_bytes: usize) -> String {
    format!("{BASE64_OMITTED_PREFIX} {original_bytes} bytes]")
}

/// Replace every base64 media payload under `v` with the omitted-payload
/// placeholder, whatever its type and size; returns whether anything changed.
/// A data URI keeps its head (`data:image/png;base64,`), and a bare payload's
/// siblings stay, so the mime type survives beside the placeholder. Running
/// this twice keeps the original count.
///
/// Recognised by exact form or known field, never by what a string looks like:
/// a base64 data URI anywhere; `source.data` of a `type: base64` source (Anthropic)
/// and `source.bytes` (Converse); `input_audio.data`; `file_data`; `inlineData.data`
/// (Gemini); and a replayed Responses `image_generation_call`'s `result`.
pub fn omit_base64_media(v: &mut Value) -> bool {
    match v {
        Value::String(s) => omit_data_uri(s),
        Value::Array(items) => items
            .iter_mut()
            .fold(false, |changed, item| omit_base64_media(item) | changed),
        Value::Object(map) => {
            let mut changed = false;
            if map.get("type").and_then(Value::as_str) == Some("image_generation_call") {
                changed |= map.get_mut("result").is_some_and(omit_bare_base64);
            }
            for (key, child) in map.iter_mut() {
                changed |= match key.as_str() {
                    "file_data" => omit_bare_base64(child),
                    "input_audio" | "inlineData" | "inline_data" => {
                        child.get_mut("data").is_some_and(omit_bare_base64)
                    }
                    "source" => {
                        let base64 = child.get("type").and_then(Value::as_str) == Some("base64");
                        child
                            .get_mut(if base64 { "data" } else { "bytes" })
                            .is_some_and(omit_bare_base64)
                    }
                    _ => false,
                };
                changed |= omit_base64_media(child);
            }
            changed
        }
        _ => false,
    }
}

fn omit_data_uri(s: &mut String) -> bool {
    if !is_base64_data_uri(s) {
        return false;
    }
    let Some((head, payload)) = s.split_once(',') else {
        return false;
    };
    if payload.starts_with(BASE64_OMITTED_PREFIX) {
        return false;
    }
    *s = format!("{head},{}", base64_omitted(s.len()));
    true
}

fn omit_bare_base64(v: &mut Value) -> bool {
    let Value::String(s) = v else {
        return false;
    };
    if s.is_empty() || s.starts_with(BASE64_OMITTED_PREFIX) || is_base64_data_uri(s) {
        // A data URI is the general walk's.
        return false;
    }
    *s = base64_omitted(s.len());
    true
}

/// Text a prompt must contain for [`omit_base64_media`] to find anything, so a
/// prompt with no media is not parsed on the request path.
const MEDIA_NEEDLES: [&str; 6] = [
    "base64",
    "\"bytes\"",
    "input_audio",
    "file_data",
    "inline",
    "image_generation_call",
];

/// [`omit_base64_media`] over serialized content. Text that is not JSON, or
/// holds no base64 media, comes back untouched, byte for byte.
pub(crate) fn omit_base64_media_in_json(s: &str) -> Cow<'_, str> {
    if !MEDIA_NEEDLES.iter().any(|needle| s.contains(needle)) {
        return Cow::Borrowed(s);
    }
    let Ok(mut root) = serde_json::from_str::<Value>(s) else {
        return Cow::Borrowed(s);
    };
    if !omit_base64_media(&mut root) {
        return Cow::Borrowed(s);
    }
    Cow::Owned(root.to_string())
}

/// Replace every `b64_json` an image API answer carries with the placeholder.
fn omit_b64_json(v: &mut Value) {
    match v {
        Value::Array(items) => items.iter_mut().for_each(omit_b64_json),
        Value::Object(map) => {
            for (key, child) in map.iter_mut() {
                if key == "b64_json" {
                    omit_bare_base64(child);
                } else {
                    omit_b64_json(child);
                }
            }
        }
        _ => {}
    }
}

/// An image API response as an exporter's captured response: `b64_json` is the
/// placeholder, every other field as the upstream sent it.
pub fn image_response_for_capture(response: &Value) -> String {
    let mut response = response.clone();
    omit_b64_json(&mut response);
    response.to_string()
}

/// [`image_response_for_capture`] over a response of unknown kind, as a
/// passthrough route captures it: anything without a `b64_json` comes back
/// untouched, byte for byte.
pub fn image_response_text_for_capture(s: &str) -> Cow<'_, str> {
    if !s.contains("b64_json") {
        return Cow::Borrowed(s);
    }
    match serde_json::from_str::<Value>(s) {
        Ok(response) => Cow::Owned(image_response_for_capture(&response)),
        Err(_) => Cow::Borrowed(s),
    }
}

/// Cap one string value. Base64 data URIs carry no information a log
/// reader can use, so they collapse to a placeholder (detection is exact
/// — bare base64 is left alone rather than guessed at); text keeps a
/// prefix plus an inline marker naming the original size.
fn shrink_string(s: &str, cap: usize) -> String {
    if s.len() <= cap {
        return s.to_owned();
    }
    if is_base64_data_uri(s) {
        return match s.split_once(',') {
            // Omitted at capture: the count there is the original's.
            Some((_, payload)) if payload.starts_with(BASE64_OMITTED_PREFIX) => payload.to_owned(),
            _ => base64_omitted(s.len()),
        };
    }
    let (prefix, _) = truncate_on_char_boundary(s, cap);
    let marked = format!("{prefix}...[aisix: truncated, {} bytes total]", s.len());
    // A string barely over the cap can come out LONGER with the marker
    // appended; keeping the original is then strictly better.
    if marked.len() >= s.len() {
        return s.to_owned();
    }
    marked
}

/// True for `data:<mime>;base64,...` payloads (inline images/audio).
fn is_base64_data_uri(s: &str) -> bool {
    if !s.starts_with("data:") {
        return false;
    }
    let (head, _) = truncate_on_char_boundary(s, 256);
    head.contains(";base64,")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn assert_valid_json_within(out: &str, cap: usize) -> Value {
        assert!(
            out.len() <= cap,
            "output must honor the byte cap: {} > {cap}",
            out.len()
        );
        serde_json::from_str(out).expect("truncated JSON content must stay valid JSON")
    }

    #[test]
    fn under_cap_content_is_untouched() {
        let (out, cut) = truncate_content("{\"a\":1}", 128);
        assert_eq!(out, "{\"a\":1}");
        assert!(!cut);
    }

    #[test]
    fn large_array_is_sampled_head_and_tail_with_placeholder() {
        let messages: Vec<Value> = (0..200)
            .map(|i| json!({"role": "user", "content": format!("message number {i} padded {}", "x".repeat(40))}))
            .collect();
        let s = serde_json::to_string(&json!({ "messages": messages })).unwrap();
        assert!(s.len() > 4096);

        let (out, cut) = truncate_content(&s, 4096);
        assert!(cut);
        let v = assert_valid_json_within(&out, 4096);

        let arr = v["messages"].as_array().unwrap();
        // Head kept, tail kept, one placeholder in between.
        assert!(arr.first().unwrap()["content"]
            .as_str()
            .unwrap()
            .contains("message number 0"));
        assert!(arr.last().unwrap()["content"]
            .as_str()
            .unwrap()
            .contains("message number 199"));
        let placeholder = arr
            .iter()
            .find(|e| e["_aisix_truncated"] == json!(true))
            .expect("array sampling must leave an explicit placeholder");
        let omitted = placeholder["omitted_items"].as_u64().unwrap();
        assert!(omitted > 0);
        // kept + omitted accounts for every original element.
        assert_eq!(arr.len() as u64 - 1 + omitted, 200);
    }

    #[test]
    fn long_string_field_keeps_prefix_and_names_original_size() {
        let s = serde_json::to_string(
            &json!({ "content": format!("The quick brown fox. {}", "words and more words ".repeat(600)) }),
        )
        .unwrap();
        let orig_content_len = s.len();
        let (out, cut) = truncate_content(&s, 2048);
        assert!(cut);
        let v = assert_valid_json_within(&out, 2048);
        let content = v["content"].as_str().unwrap();
        assert!(
            content.starts_with("The quick brown fox."),
            "prefix must survive"
        );
        assert!(
            content.contains("[aisix: truncated,"),
            "inline marker must say the field was cut: {content}"
        );
        assert!(content.contains("bytes total]"));
        assert!(orig_content_len > 2048);
    }

    #[test]
    fn nested_object_keeps_keys_and_hierarchy() {
        let s = serde_json::to_string(&json!({
            "model": "m1",
            "options": {
                "system": "s".repeat(3000),
                "temperature": 0.5,
                "nested": { "deep": { "text": "t".repeat(3000), "keep": 42 } }
            }
        }))
        .unwrap();
        let (out, cut) = truncate_content(&s, 1024);
        assert!(cut);
        let v = assert_valid_json_within(&out, 1024);
        // Every level of the hierarchy is still addressable.
        assert_eq!(v["model"], "m1");
        assert_eq!(v["options"]["temperature"], 0.5);
        assert_eq!(v["options"]["nested"]["deep"]["keep"], 42);
        assert!(v["options"]["nested"]["deep"]["text"]
            .as_str()
            .unwrap()
            .contains("[aisix: truncated,"));
    }

    #[test]
    fn base64_media_is_omitted_whatever_its_type_and_size_and_only_once() {
        let uri = format!("data:image/png;base64,{}", "A".repeat(8));
        let pdf = format!("data:application/pdf;base64,{}", "A".repeat(8));
        let omitted = |n: usize| format!("[aisix: base64 data uri omitted, {n} bytes]");
        let mut v = json!({"messages": [
            {"type": "image_url", "image_url": {"url": uri}},
            {"type": "image_url", "image_url": {"url": "data:;base64,QUJD"}},
            {"type": "input_image", "image_url": uri},
            {"images": [{"type": "image_url", "image_url": {"url": uri}}]},
            {"type": "tool_result", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}
            ]},
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "QUJD"}},
            {"type": "image_generation_call", "result": "QUJD"},
            {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}},
            {"type": "file", "file": {"file_data": pdf, "filename": "a.pdf"}},
            {"type": "input_file", "file_data": "QUJD"},
            {"inlineData": {"mimeType": "image/png", "data": "QUJD"}},
            {"image": {"format": "png", "source": {"bytes": "QUJD"}}},
        ]});
        assert!(omit_base64_media(&mut v));
        let once = v.clone();
        assert!(!omit_base64_media(&mut v), "a second pass finds nothing");
        assert_eq!(v, once);

        let m = &v["messages"];
        let kept = format!("data:image/png;base64,{}", omitted(uri.len()));
        assert_eq!(m[0]["image_url"]["url"], kept);
        assert_eq!(
            m[1]["image_url"]["url"],
            format!("data:;base64,{}", omitted(17))
        );
        assert_eq!(m[2]["image_url"], kept);
        assert_eq!(m[3]["images"][0]["image_url"]["url"], kept);
        let source = &m[4]["content"][0]["source"];
        assert_eq!(source["media_type"], "image/png");
        assert_eq!(source["data"], omitted(4));
        assert_eq!(m[5]["source"]["media_type"], "application/pdf");
        assert_eq!(m[5]["source"]["data"], omitted(4));
        assert_eq!(m[6]["result"], omitted(4));
        assert_eq!(m[7]["input_audio"]["data"], omitted(4));
        assert_eq!(m[7]["input_audio"]["format"], "wav");
        assert_eq!(
            m[8]["file"]["file_data"],
            format!("data:application/pdf;base64,{}", omitted(pdf.len()))
        );
        assert_eq!(m[8]["file"]["filename"], "a.pdf");
        assert_eq!(m[9]["file_data"], omitted(4));
        assert_eq!(m[10]["inlineData"]["data"], omitted(4));
        assert_eq!(m[11]["image"]["source"]["bytes"], omitted(4));
    }

    #[test]
    fn urls_and_ordinary_fields_are_not_base64_media() {
        let mut v = json!({
            "a": {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            "b": {"type": "image_url", "image_url": {"url": "data:image/svg+xml,<svg/>"}},
            "c": {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
            "d": {"type": "text", "text": "QUJDQUJDQUJDQUJDQUJD"},
            "e": {"data": "QUJDQUJDQUJDQUJDQUJD", "result": "QUJD"},
        });
        let before = v.clone();
        assert!(!omit_base64_media(&mut v));
        assert_eq!(v, before);
    }

    #[test]
    fn serialized_content_without_base64_media_comes_back_byte_identical() {
        for s in [
            r#"{ "z": 1, "a": "no media" }"#,
            r#"{ "z": 1, "a": "mentions base64" }"#,
            "not json, mentions base64",
        ] {
            assert!(matches!(omit_base64_media_in_json(s), Cow::Borrowed(b) if b == s));
        }
    }

    #[test]
    fn every_recognised_field_passes_the_needle_scan() {
        for s in [
            r#"{"x": {"source": {"bytes": "QUJD"}}}"#,
            r#"{"input_audio": {"data": "QUJD"}}"#,
            r#"{"file_data": "QUJD"}"#,
            r#"{"inlineData": {"data": "QUJD"}}"#,
            r#"{"inline_data": {"data": "QUJD"}}"#,
            r#"{"type": "image_generation_call", "result": "QUJD"}"#,
            r#"{"source": {"type": "base64", "data": "QUJD"}}"#,
            r#"{"u": "data:image/png;base64,QUJD"}"#,
        ] {
            assert!(
                omit_base64_media_in_json(s).contains("[aisix: base64 data uri omitted, "),
                "{s}"
            );
        }
    }

    #[test]
    fn image_response_capture_omits_only_b64_json() {
        let response = json!({"created": 1, "output_format": "png", "data": [
            {"b64_json": "QUJD", "revised_prompt": "a cat"},
            {"url": "https://example.com/a.png"},
        ]});
        let captured: Value = serde_json::from_str(&image_response_for_capture(&response)).unwrap();
        assert_eq!(
            captured,
            json!({"created": 1, "output_format": "png", "data": [
                {"b64_json": "[aisix: base64 data uri omitted, 4 bytes]", "revised_prompt": "a cat"},
                {"url": "https://example.com/a.png"},
            ]})
        );
    }

    #[test]
    fn passthrough_response_text_omits_b64_json_and_nothing_else() {
        let frame = r#"{"type":"image_generation.completed","b64_json":"QUJD"}"#;
        let captured: Value =
            serde_json::from_str(&image_response_text_for_capture(frame)).unwrap();
        assert_eq!(
            captured["b64_json"],
            "[aisix: base64 data uri omitted, 4 bytes]"
        );
        for s in [r#"{ "text": "no images" }"#, "plain text naming b64_json"] {
            assert!(matches!(image_response_text_for_capture(s), Cow::Borrowed(b) if b == s));
        }
    }

    #[test]
    fn the_tightest_rung_keeps_the_original_count_of_an_omitted_image() {
        let scrubbed = "data:image/png;base64,[aisix: base64 data uri omitted, 123456 bytes]";
        assert_eq!(
            shrink_string(scrubbed, 32),
            "[aisix: base64 data uri omitted, 123456 bytes]"
        );
    }

    #[test]
    fn data_uri_collapses_to_placeholder() {
        let blob = format!("data:image/png;base64,{}", "AAAA".repeat(2000));
        let s = serde_json::to_string(&json!({ "image_url": { "url": blob } })).unwrap();
        let (out, cut) = truncate_content(&s, 1024);
        assert!(cut);
        let v = assert_valid_json_within(&out, 1024);
        let url = v["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("[aisix: base64 data uri omitted,"), "{url}");
    }

    #[test]
    fn bare_base64_gets_prefix_marker_not_placeholder() {
        // Bare base64 (no data-URI wrapper) is indistinguishable from a
        // long opaque token, so it takes the generic prefix+marker path —
        // detection stays exact rather than heuristic.
        let blob = "QUJDREVGR0g=".repeat(500);
        let s = serde_json::to_string(&json!({ "audio": blob })).unwrap();
        let (out, _) = truncate_content(&s, 1024);
        let v = assert_valid_json_within(&out, 1024);
        assert!(v["audio"].as_str().unwrap().contains("[aisix: truncated,"));
    }

    #[test]
    fn multibyte_utf8_stays_valid_through_string_truncation() {
        let s =
            serde_json::to_string(&json!({ "content": "非常长的中文内容".repeat(800) })).unwrap();
        let (out, cut) = truncate_content(&s, 2048);
        assert!(cut);
        let v = assert_valid_json_within(&out, 2048);
        // Parsing back proves both valid JSON and valid UTF-8 inside the value.
        assert!(v["content"]
            .as_str()
            .unwrap()
            .contains("[aisix: truncated,"));
    }

    #[test]
    fn non_json_text_keeps_the_byte_cut_behavior() {
        let text = "plain log line ".repeat(200);
        let (out, cut) = truncate_content(&text, 100);
        assert!(cut);
        assert_eq!(out.len(), 100);
        assert!(text.starts_with(&*out));
    }

    #[test]
    fn barely_over_cap_json_still_comes_back_valid() {
        // A payload a few bytes over the cap must not come back as broken
        // JSON — the old behavior would slice mid-token.
        let s = serde_json::to_string(&json!({
            "messages": [{ "role": "user", "content": "c".repeat(600) }]
        }))
        .unwrap();
        let cap = s.len() - 3;
        let (out, cut) = truncate_content(&s, cap);
        assert!(cut);
        assert_valid_json_within(&out, cap);
    }

    #[test]
    fn impossible_cap_falls_back_to_byte_cut() {
        // Even the tightest rung can't fit a JSON document into 8 bytes;
        // the byte cap still wins via the fallback cut.
        let s = serde_json::to_string(&json!({ "a": "b".repeat(100) })).unwrap();
        let (out, cut) = truncate_content(&s, 8);
        assert!(cut);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn structured_truncation_composes_across_two_caps() {
        // Handler-side capture runs at the largest exporter cap; each
        // exporter then re-truncates to its own smaller cap. The second
        // pass sees the first pass's (valid JSON) output and must keep it
        // valid JSON — AND keep the omitted-items accounting true to the
        // ORIGINAL element count (a dropped stage-1 placeholder folds its
        // count into the stage-2 placeholder instead of counting as 1).
        let messages: Vec<Value> = (0..300)
            .map(|i| json!({"role": "user", "content": format!("msg {i} {}", "y".repeat(100))}))
            .collect();
        let s = serde_json::to_string(&json!({ "messages": messages })).unwrap();
        let (first, _) = truncate_content(&s, 8192);
        assert_valid_json_within(&first, 8192);
        let (second, cut) = truncate_content(&first, 2048);
        assert!(cut);
        let v = assert_valid_json_within(&second, 2048);

        let arr = v["messages"].as_array().unwrap();
        assert!(arr.len() >= 3);
        let (mut kept, mut omitted) = (0u64, 0u64);
        for item in arr {
            match placeholder_omitted(item) {
                Some(n) => omitted += n,
                None => kept += 1,
            }
        }
        assert_eq!(
            kept + omitted,
            300,
            "kept + omitted must account for every original element \
             across both truncation stages (got kept={kept}, omitted={omitted})",
        );
    }
}
