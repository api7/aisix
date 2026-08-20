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
//! The scanner assumes syntactically valid JSON (callers run it on
//! bytes `serde_json` has already parsed) and still fails safe: any
//! unexpected byte, overrun, or depth blow-up returns an error rather
//! than a partially rewritten document. Callers decide the failure
//! policy (the MCP output hook fails closed).

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
#[derive(Debug, thiserror::Error)]
#[error("json splice scan failed at byte {at}")]
pub struct SpliceError {
    at: usize,
}

/// Depth cap. `serde_json` refuses documents deeper than 128, so bytes
/// that reached a splice call can never hit this; it bounds the scanner
/// on its own anyway.
const MAX_DEPTH: usize = 256;

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

    let err = |at: usize| SpliceError { at };
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
                b'\\' => i += 2, // skips the escaped byte; `\uXXXX` needs no care (hex only)
                b'"' => return Ok(i + 1),
                _ => i += 1,
            }
        }
        Err(SpliceError { at: start })
    };
    let decode_str = |range: Range<usize>| -> Result<String, SpliceError> {
        let at = range.start;
        serde_json::from_slice::<String>(&input[range]).map_err(|_| SpliceError { at })
    };

    // `true` → the loop continues at a VALUE position; `false` → the
    // value just ended and the closer/comma logic below runs.
    'value: loop {
        skip_ws(&mut pos);
        let b = *input.get(pos).ok_or_else(|| err(pos))?;
        match b {
            b'{' => {
                frames.push(Frame::Object);
                if frames.len() > MAX_DEPTH {
                    return Err(err(pos));
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
                if frames.len() > MAX_DEPTH {
                    return Err(err(pos));
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
            // Number / true / false / null. The scanner does not
            // re-validate the token — the bytes already parsed upstream —
            // it only needs the token's extent.
            b'-' | b'0'..=b'9' | b't' | b'f' | b'n' => {
                while pos < input.len()
                    && matches!(input[pos],
                        b'-' | b'+' | b'.' | b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z')
                {
                    pos += 1;
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
    fn depth_cap_errors() {
        let mut doc = String::new();
        for _ in 0..300 {
            doc.push('[');
        }
        for _ in 0..300 {
            doc.push(']');
        }
        assert!(rewrite_string_values(doc.as_bytes(), |_| true, |_| None).is_err());
    }
}
