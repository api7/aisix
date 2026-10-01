//! Byte-splicing rewrite of JSON string VALUES (AISIX-Cloud#1330).
//!
//! The MCP write-back channel must return every byte outside a masked
//! span verbatim. A `serde_json::Value` round-trip cannot promise that:
//! this workspace's `Map` is a BTreeMap (keys re-sort), and numbers
//! re-serialise canonically (`1e3` → `1000.0`). So this module never
//! re-serialises the document — it scans the raw bytes once, decodes
//! only the string values a path predicate selects, and splices the
//! re-encoded replacements back into the original buffer. Everything
//! else — key order, whitespace, number spellings, escape choices —
//! survives byte-for-byte.
//!
//! Object KEYS are never offered for rewrite (they are schema, not
//! data — same rule as `collect_string_leaves` in the MCP scan path),
//! but they ARE decoded to build the path handed to the predicate.
//!
//! The scanner is iterative, so deeply nested JSON does not consume the
//! Rust call stack. [`MAX_JSON_DEPTH`] bounds its per-request traversal
//! allocations. It still fails safe: any unexpected byte, overrun, or
//! depth excess returns an error rather than a partially rewritten document.
//! Callers decide the failure policy (the MCP output hook fails closed).

use std::ops::Range;

/// One step of the path from the document root to a value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathSeg {
    /// Object member, key decoded (escapes resolved).
    Key(String),
    /// Array element index.
    Index(usize),
}

impl PathSeg {
    /// `true` when this segment is `Key(name)`.
    pub fn is_key(&self, name: &str) -> bool {
        matches!(self, PathSeg::Key(k) if k == name)
    }
}

/// Scanner failure. Carries no document content (the byte offset only),
/// so an error can be logged without leaking the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpliceErrorKind {
    Invalid,
    DepthExceeded,
    Unevaluable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("json splice scan failed at byte {at}")]
pub struct SpliceError {
    at: usize,
    kind: SpliceErrorKind,
}

impl SpliceError {
    /// Whether the scanner stopped at its bounded traversal limit rather than
    /// because the input was malformed.
    pub(crate) fn is_depth_exceeded(self) -> bool {
        self.kind == SpliceErrorKind::DepthExceeded
    }

    /// Whether source selection could not safely establish the fields a
    /// guardrail is allowed to inspect. Unlike an invalid raw JSON document,
    /// this must not fall back to scanning the whole body: that could expose
    /// an opaque media carrier to an external guardrail.
    pub(crate) fn is_unevaluable(self) -> bool {
        matches!(
            self.kind,
            SpliceErrorKind::DepthExceeded | SpliceErrorKind::Unevaluable
        )
    }

    /// Report an exhausted JSON traversal budget from a selector which uses
    /// the same bounded raw-JSON policy as this scanner. Selectors retain raw
    /// fragments rather than source offsets, so the synthetic error uses the
    /// start of that fragment as its safe, non-payload-bearing location.
    pub(crate) fn depth_exceeded() -> Self {
        Self {
            at: 0,
            kind: SpliceErrorKind::DepthExceeded,
        }
    }

    /// Report a typed carrier that cannot be safely selected without falling
    /// back to raw source text.
    pub(crate) fn unevaluable() -> Self {
        Self {
            at: 0,
            kind: SpliceErrorKind::Unevaluable,
        }
    }
}

/// Traversal depth cap. This keeps the iterative scanner stack-safe without
/// allowing an unbounded JSON path/frame allocation from an unbounded raw
/// passthrough body. It remains well beyond serde_json's usual recursion cap.
pub(crate) const MAX_JSON_DEPTH: usize = 4_096;

/// Bounded collection policy for decoded JSON text passed to guardrails.
/// Source selectors use the same limits so a wide document cannot shift its
/// allocation from selector bookkeeping into decoded scan text.
pub(crate) const MAX_JSON_SCAN_VALUES: usize = 1_024;
pub(crate) const MAX_JSON_SCAN_TEXT_BYTES: usize = 256 * 1024;

/// Validate JSON-number syntax without materializing the value. serde_json
/// can reject a syntactically valid number outside its runtime numeric range.
pub(crate) fn is_json_number(token: &[u8]) -> bool {
    let mut pos = 0;
    if token.get(pos) == Some(&b'-') {
        pos += 1;
    }
    match token.get(pos) {
        Some(b'0') => pos += 1,
        Some(b'1'..=b'9') => {
            pos += 1;
            while token.get(pos).is_some_and(|byte| byte.is_ascii_digit()) {
                pos += 1;
            }
        }
        _ => return false,
    }
    if token.get(pos) == Some(&b'.') {
        pos += 1;
        let fraction_start = pos;
        while token.get(pos).is_some_and(|byte| byte.is_ascii_digit()) {
            pos += 1;
        }
        if pos == fraction_start {
            return false;
        }
    }
    if matches!(token.get(pos), Some(b'e' | b'E')) {
        pos += 1;
        if matches!(token.get(pos), Some(b'+' | b'-')) {
            pos += 1;
        }
        let exponent_start = pos;
        while token.get(pos).is_some_and(|byte| byte.is_ascii_digit()) {
            pos += 1;
        }
        if pos == exponent_start {
            return false;
        }
    }
    pos == token.len()
}

/// Rewrite the string values of `input` selected by `should_rewrite`,
/// leaving every other byte untouched.
///
/// For each string VALUE (never a key) whose path satisfies the
/// predicate, the decoded text is offered to `rewrite`; `Some(new)`
/// replaces that value's bytes with the JSON encoding of `new`.
///
/// Returns `Ok(None)` when nothing changed (callers keep the original
/// buffer — the no-hit case allocates nothing), `Ok(Some(bytes))` with
/// the spliced document otherwise.
pub fn rewrite_string_values(
    input: &[u8],
    mut should_rewrite: impl FnMut(&[PathSeg]) -> bool,
    mut rewrite: impl FnMut(&str) -> Option<String>,
) -> Result<Option<Vec<u8>>, SpliceError> {
    enum Frame {
        Object,
        Array,
    }

    let err = |at: usize| SpliceError {
        at,
        kind: SpliceErrorKind::Invalid,
    };
    std::str::from_utf8(input).map_err(|error| err(error.valid_up_to()))?;
    let mut splices: Vec<(Range<usize>, String)> = Vec::new();
    let mut path: Vec<PathSeg> = Vec::new();
    let mut frames: Vec<Frame> = Vec::new();
    let mut pos = 0usize;

    let skip_ws = |pos: &mut usize| {
        while *pos < input.len() && matches!(input[*pos], b' ' | b'\t' | b'\n' | b'\r') {
            *pos += 1;
        }
    };
    // Span of the string token starting at `start` (must be `"`),
    // inclusive of both quotes.
    let scan_string = |start: usize| -> Result<usize, SpliceError> {
        let mut i = start + 1;
        while i < input.len() {
            match input[i] {
                b'\\' => match input.get(i + 1).copied() {
                    Some(b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') => i += 2,
                    Some(b'u') => {
                        let Some(hex) = input.get(i + 2..i + 6) else {
                            return Err(err(i));
                        };
                        if !hex.iter().all(|byte| byte.is_ascii_hexdigit()) {
                            return Err(err(i));
                        }
                        i += 6;
                    }
                    _ => return Err(err(i)),
                },
                b'"' => return Ok(i + 1),
                0..=0x1f => return Err(err(i)),
                _ => i += 1,
            }
        }
        Err(err(start))
    };
    let decode_str = |range: Range<usize>| -> Result<String, SpliceError> {
        let at = range.start;
        serde_json::from_slice::<String>(&input[range]).map_err(|_| SpliceError {
            at,
            kind: SpliceErrorKind::Invalid,
        })
    };

    // `true` → the loop continues at a VALUE position; `false` → the
    // value just ended and the closer/comma logic below runs.
    'value: loop {
        skip_ws(&mut pos);
        let b = *input.get(pos).ok_or_else(|| err(pos))?;
        match b {
            b'{' => {
                frames.push(Frame::Object);
                if frames.len() > MAX_JSON_DEPTH {
                    return Err(SpliceError {
                        at: pos,
                        kind: SpliceErrorKind::DepthExceeded,
                    });
                }
                pos += 1;
                skip_ws(&mut pos);
                match input.get(pos) {
                    Some(b'}') => {
                        pos += 1;
                        frames.pop();
                        // fall through to after-value
                    }
                    Some(b'"') => {
                        let end = scan_string(pos)?;
                        path.push(PathSeg::Key(decode_str(pos..end)?));
                        pos = end;
                        skip_ws(&mut pos);
                        if input.get(pos) != Some(&b':') {
                            return Err(err(pos));
                        }
                        pos += 1;
                        continue 'value;
                    }
                    _ => return Err(err(pos)),
                }
            }
            b'[' => {
                frames.push(Frame::Array);
                if frames.len() > MAX_JSON_DEPTH {
                    return Err(SpliceError {
                        at: pos,
                        kind: SpliceErrorKind::DepthExceeded,
                    });
                }
                pos += 1;
                skip_ws(&mut pos);
                if input.get(pos) == Some(&b']') {
                    pos += 1;
                    frames.pop();
                    // fall through to after-value
                } else {
                    path.push(PathSeg::Index(0));
                    continue 'value;
                }
            }
            b'"' => {
                let end = scan_string(pos)?;
                if should_rewrite(&path) {
                    let decoded = decode_str(pos..end)?;
                    if let Some(new) = rewrite(&decoded) {
                        // to_string of a String is infallible.
                        let encoded = serde_json::to_string(&new).map_err(|_| err(pos))?;
                        splices.push((pos..end, encoded));
                    }
                }
                pos = end;
            }
            // Number / true / false / null.
            b'-' | b'0'..=b'9' | b't' | b'f' | b'n' => {
                let start = pos;
                while pos < input.len()
                    && matches!(input[pos],
                        b'-' | b'+' | b'.' | b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z')
                {
                    pos += 1;
                }
                let token = &input[start..pos];
                if token != b"true"
                    && token != b"false"
                    && token != b"null"
                    && !is_json_number(token)
                {
                    return Err(err(start));
                }
            }
            _ => return Err(err(pos)),
        }

        // A value just ended: unwind closers, then either continue with
        // the next member/element or finish.
        loop {
            skip_ws(&mut pos);
            let Some(frame) = frames.last() else {
                // Root value complete: only trailing whitespace may follow.
                if pos != input.len() {
                    return Err(err(pos));
                }
                break 'value;
            };
            match (frame, input.get(pos)) {
                (Frame::Object, Some(b',')) => {
                    pos += 1;
                    path.pop();
                    skip_ws(&mut pos);
                    if input.get(pos) != Some(&b'"') {
                        return Err(err(pos));
                    }
                    let end = scan_string(pos)?;
                    path.push(PathSeg::Key(decode_str(pos..end)?));
                    pos = end;
                    skip_ws(&mut pos);
                    if input.get(pos) != Some(&b':') {
                        return Err(err(pos));
                    }
                    pos += 1;
                    continue 'value;
                }
                (Frame::Object, Some(b'}')) => {
                    pos += 1;
                    path.pop();
                    frames.pop();
                }
                (Frame::Array, Some(b',')) => {
                    pos += 1;
                    match path.last_mut() {
                        Some(PathSeg::Index(i)) => *i += 1,
                        _ => return Err(err(pos)),
                    }
                    continue 'value;
                }
                (Frame::Array, Some(b']')) => {
                    pos += 1;
                    path.pop();
                    frames.pop();
                }
                _ => return Err(err(pos)),
            }
        }
    }

    if splices.is_empty() {
        return Ok(None);
    }
    // Splices were recorded in scan order (strictly ascending, disjoint).
    let mut out = Vec::with_capacity(input.len());
    let mut copied = 0usize;
    for (range, replacement) in splices {
        out.extend_from_slice(&input[copied..range.start]);
        out.extend_from_slice(replacement.as_bytes());
        copied = range.end;
    }
    out.extend_from_slice(&input[copied..]);
    Ok(Some(out))
}

/// Decode and collect every JSON string **value** in source order.
///
/// This reuses the iterative splice scanner with a no-op rewrite, so it
/// preserves duplicate keys and works beyond serde_json's default
/// container-recursion limit, up to [`MAX_JSON_DEPTH`]. Object keys are
/// decoded only to maintain the scanner's structure and are never included in
/// the returned text.
pub fn collect_string_values(input: &[u8]) -> Result<String, SpliceError> {
    collect_string_values_where(input, |_| true)
}

/// Validate one UTF-8 JSON document with the same iterative depth limit as
/// the guardrail selector, without retaining any of its string values.
pub(crate) fn validate_json(input: &[u8]) -> Result<(), SpliceError> {
    rewrite_string_values(input, |_| false, |_| None).map(|_| ())
}

/// Decode and collect selected JSON string **values** in source order.
///
/// Like [`collect_string_values`], this preserves duplicate keys and stays
/// stack-safe within the bounded nesting limit. The predicate sees the
/// decoded path of each string value, never an object key.
pub fn collect_string_values_where(
    input: &[u8],
    mut include: impl FnMut(&[PathSeg]) -> bool,
) -> Result<String, SpliceError> {
    let mut out = String::new();
    let mut collect_error = None;
    rewrite_string_values(
        input,
        |path| include(path),
        |value| {
            if collect_error.is_none() {
                let separator = if out.is_empty() { 0 } else { 1 };
                let Some(next_len) = out
                    .len()
                    .checked_add(separator)
                    .and_then(|len| len.checked_add(value.len()))
                else {
                    collect_error = Some(SpliceError::unevaluable());
                    return None;
                };
                if next_len > MAX_JSON_SCAN_TEXT_BYTES
                    || out.try_reserve(next_len - out.len()).is_err()
                {
                    collect_error = Some(SpliceError::unevaluable());
                    return None;
                }
                if separator != 0 {
                    out.push('\n');
                }
                out.push_str(value);
            }
            None
        },
    )?;
    collect_error.map_or(Ok(out), Err)
}

/// Decode selected JSON string values as separate source-order entries.
///
/// Stream guardrails use this form to keep separate repeated carrier fields
/// in independent continuation channels rather than inserting separators
/// into a literal split across frames.
pub fn collect_string_values_where_vec(
    input: &[u8],
    mut include: impl FnMut(&[PathSeg]) -> bool,
) -> Result<Vec<String>, SpliceError> {
    let mut out = Vec::new();
    let mut source_bytes = 0usize;
    let mut collect_error = None;
    rewrite_string_values(
        input,
        |path| include(path),
        |value| {
            if collect_error.is_none() {
                let Some(next_bytes) = source_bytes.checked_add(value.len()) else {
                    collect_error = Some(SpliceError::unevaluable());
                    return None;
                };
                if out.len() >= MAX_JSON_SCAN_VALUES
                    || next_bytes > MAX_JSON_SCAN_TEXT_BYTES
                    || out.try_reserve(1).is_err()
                {
                    collect_error = Some(SpliceError::unevaluable());
                    return None;
                }
                let mut decoded = String::new();
                if decoded.try_reserve(value.len()).is_err() {
                    collect_error = Some(SpliceError::unevaluable());
                    return None;
                }
                decoded.push_str(value);
                source_bytes = next_bytes;
                out.push(decoded);
            }
            None
        },
    )?;
    collect_error.map_or(Ok(out), Err)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rewrite_all(input: &str, f: impl FnMut(&str) -> Option<String>) -> Option<String> {
        rewrite_string_values(input.as_bytes(), |_| true, f)
            .unwrap()
            .map(|b| String::from_utf8(b).unwrap())
    }

    #[test]
    fn rewrites_only_the_selected_leaf_bytes() {
        // Deliberately hostile formatting: odd whitespace, exotic number
        // spellings, escape choices — none of it may change.
        let doc = "{ \"a\" :  1e3,\"b\":[ true, \"secret\" ,null] , \"c\": 0.1000 }";
        let out = rewrite_all(doc, |s| (s == "secret").then(|| "MASK".to_string())).unwrap();
        assert_eq!(
            out,
            "{ \"a\" :  1e3,\"b\":[ true, \"MASK\" ,null] , \"c\": 0.1000 }"
        );
    }

    #[test]
    fn no_change_returns_none() {
        let doc = r#"{"a": "x", "b": 2}"#;
        assert!(rewrite_string_values(doc.as_bytes(), |_| true, |_| None)
            .unwrap()
            .is_none());
    }

    #[test]
    fn validates_large_exponent_without_materializing_a_number() {
        assert!(validate_json(br#"{"n":1e400}"#).is_ok());
    }

    #[test]
    fn keys_are_never_offered_but_shape_the_path() {
        let doc = r#"{"secret": {"inner": "value"}}"#;
        let mut offered = Vec::new();
        let mut paths = Vec::new();
        rewrite_string_values(
            doc.as_bytes(),
            |p| {
                paths.push(p.to_vec());
                true
            },
            |s| {
                offered.push(s.to_owned());
                None
            },
        )
        .unwrap();
        // Only the value is offered — the keys "secret"/"inner" are not.
        assert_eq!(offered, vec!["value"]);
        assert_eq!(
            paths,
            vec![vec![
                PathSeg::Key("secret".into()),
                PathSeg::Key("inner".into())
            ]],
        );
    }

    #[test]
    fn array_indices_and_nesting_track_correctly() {
        let doc = r#"{"params":{"arguments":{"xs":["a",{"y":"b"},[],"c"],"n":7}},"id":"z"}"#;
        let mut seen = Vec::new();
        rewrite_string_values(
            doc.as_bytes(),
            |p| {
                p.first().is_some_and(|s| s.is_key("params"))
                    && p.get(1).is_some_and(|s| s.is_key("arguments"))
            },
            |s| {
                seen.push(s.to_owned());
                None
            },
        )
        .unwrap();
        // "z" (outside params.arguments) is filtered by the predicate.
        assert_eq!(seen, vec!["a", "b", "c"]);
    }

    #[test]
    fn collects_deep_string_values_without_recursion() {
        let depth = 512;
        let mut doc = "{\"v\":".repeat(depth);
        doc.push_str(r#""\u0042LOCKME""#);
        doc.push_str(&"}".repeat(depth));

        assert_eq!(collect_string_values(doc.as_bytes()).unwrap(), "BLOCKME");
    }

    #[test]
    fn collects_selected_paths_with_duplicate_keys_and_nested_values() {
        let doc = r#"{"model":"routing-only","messages":[{"content":"first","metadata":{"note":"nested"}}],"messages":[{"content":"second"}]}"#;
        assert_eq!(
            collect_string_values_where(doc.as_bytes(), |path| {
                !path.first().is_some_and(|segment| segment.is_key("model"))
            })
            .unwrap(),
            "first\nnested\nsecond"
        );
    }

    #[test]
    fn collects_selected_values_as_separate_source_ordered_entries() {
        let doc = r#"{"type":"response.output_text.delta","delta":"FOR","delta":"ok"}"#;
        assert_eq!(
            collect_string_values_where_vec(doc.as_bytes(), |path| {
                path.first().is_some_and(|segment| segment.is_key("delta"))
            })
            .unwrap(),
            vec!["FOR", "ok"]
        );
    }

    #[test]
    fn collected_json_text_over_the_shared_cap_is_unevaluable() {
        let doc = format!(
            r#"{{"text":"{}"}}"#,
            "x".repeat(MAX_JSON_SCAN_TEXT_BYTES + 1)
        );
        let error = collect_string_values(doc.as_bytes())
            .expect_err("a JSON text collection must stay bounded");
        assert!(error.is_unevaluable(), "{error}");
        let error = collect_string_values_where_vec(doc.as_bytes(), |_| true)
            .expect_err("vector collection shares the text cap");
        assert!(error.is_unevaluable(), "{error}");

        let values = format!("[{}]", "\"\",".repeat(MAX_JSON_SCAN_VALUES) + "\"\"",);
        let error = collect_string_values_where_vec(values.as_bytes(), |_| true)
            .expect_err("vector collection also bounds empty values");
        assert!(error.is_unevaluable(), "{error}");
    }

    #[test]
    fn escaped_key_decodes_for_the_predicate() {
        // `param\u0073` decodes to "params" — the predicate must see the
        // decoded spelling or a smuggled escape would bypass the scope.
        let doc = r#"{"param\u0073":{"arguments":{"t":"hit"}}}"#;
        let out = rewrite_string_values(
            doc.as_bytes(),
            |p| p.first().is_some_and(|s| s.is_key("params")),
            |s| (s == "hit").then(|| "X".to_string()),
        )
        .unwrap()
        .unwrap();
        // The key's original escape spelling is untouched; only the value changed.
        assert_eq!(
            String::from_utf8(out).unwrap(),
            r#"{"param\u0073":{"arguments":{"t":"X"}}}"#
        );
    }

    #[test]
    fn escaped_and_multibyte_values_reencode_correctly() {
        let doc = r#"{"a":"line\nbreak \"q\" 版本","b":"清 洁"}"#;
        let out = rewrite_all(doc, |s| {
            (s == "line\nbreak \"q\" 版本").then(|| "打码\"了\"".to_string())
        })
        .unwrap();
        // serde_json re-encodes the replacement; the untouched leaf keeps
        // its original bytes.
        assert_eq!(out, r#"{"a":"打码\"了\"","b":"清 洁"}"#);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["a"], "打码\"了\"");
    }

    #[test]
    fn multiple_rewrites_splice_in_order() {
        let doc = r#"["one","keep","two"]"#;
        let out = rewrite_all(doc, |s| match s {
            "one" => Some("1".into()),
            "two" => Some("2".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(out, r#"["1","keep","2"]"#);
    }

    #[test]
    fn empty_containers_and_scalars_pass_through() {
        for doc in [
            r#"{}"#,
            r#"[]"#,
            r#"{"a":[],"b":{}}"#,
            "42",
            "null",
            r#""s""#,
        ] {
            let got = rewrite_string_values(doc.as_bytes(), |_| true, |_| None).unwrap();
            assert!(got.is_none(), "{doc}");
        }
        // A bare root string IS a value and can be rewritten.
        let out = rewrite_all(r#""s""#, |_| Some("t".into())).unwrap();
        assert_eq!(out, r#""t""#);
    }

    #[test]
    fn malformed_input_errors_instead_of_partial_output() {
        for doc in [
            r#"{"a": }"#,
            r#"{"a":"x""#,
            r#"{"a":"x"} trailing"#,
            r#"{'a':1}"#,
        ] {
            assert!(
                rewrite_string_values(doc.as_bytes(), |_| true, |_| Some("m".into())).is_err(),
                "{doc}",
            );
        }
    }

    #[test]
    fn deep_nesting_is_stack_safe() {
        let mut doc = String::new();
        for _ in 0..300 {
            doc.push('[');
        }
        for _ in 0..300 {
            doc.push(']');
        }
        assert!(rewrite_string_values(doc.as_bytes(), |_| true, |_| None)
            .expect("valid deep JSON")
            .is_none());
    }

    #[test]
    fn nesting_beyond_the_cap_errors() {
        let depth = MAX_JSON_DEPTH + 1;
        let doc = format!("{}\"value\"{}", "[".repeat(depth), "]".repeat(depth));
        let err = rewrite_string_values(doc.as_bytes(), |_| true, |_| None).unwrap_err();
        assert!(err.is_depth_exceeded());
    }
}
