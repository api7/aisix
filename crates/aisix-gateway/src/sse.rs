//! Server-Sent Events framing, shared by every stream reader in the gateway.
//!
//! One implementation of the event-stream rules
//! (<https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation>),
//! at two levels:
//!
//! - **Framing** — [`find_frame_end`], [`split_frame_end`] and the
//!   incremental [`SseFrameSplitter`] cut a byte stream into frames. A line
//!   ends in `\r\n`, `\n` or a bare `\r`, and a blank line ends a frame. A
//!   frame is handed back as the upstream's own bytes, terminator included,
//!   so a relay that forwards frames forwards the stream unchanged.
//! - **Fields** — [`lines`], [`parse_field`], [`data_ranges`] and
//!   [`parse_frame`] read one frame: `:` comment lines, `field: value` with
//!   one leading space stripped, a line without a colon as a field with an
//!   empty value, `data` lines joined with `\n` (an empty one included), and
//!   the last `event` / `id` / `retry`.
//!
//! Both work on bytes. Nothing here decodes UTF-8: a caller decodes a
//! complete frame or payload, never a network chunk, under its own policy.
//! Nor does anything here know `[DONE]` — that sentinel is OpenAI's payload,
//! not SSE framing, and [`SseDecoder`] is the layer that recognises it.
//!
//! Two liberties, both only reachable by a non-conformant upstream:
//!
//! - A leading UTF-8 BOM is skipped at the start of ANY frame, not only the
//!   stream's first: a frame parsed on its own cannot know whether it opened
//!   the stream, and a BOM mid-stream has no other reading.
//! - When the blank line that ends a frame is a `\r` that is the last
//!   buffered byte, the frame is complete — the spec reads a lone `\r` as a
//!   line — so it is released rather than held for a `\n` that may never
//!   come. If that `\n` does follow, it is the tail of the same `\r\n`: the
//!   splitter keeps it at the head of the next frame's bytes (so forwarded
//!   bytes are unchanged) without reading it as a blank line. The stateless
//!   [`find_frame_end`] cannot know, and returns it as an empty frame of its
//!   own, which parses to nothing.

use std::borrow::Cow;
use std::ops::Range;

/// Default bound on the bytes one frame may buffer before its terminating
/// blank line arrives. A stream has no total size, so the bound is per
/// frame: it is the only thing an upstream that never ends a frame can grow.
/// Generous on purpose — a terminal Responses-API event carries the whole
/// response, and an image model can stream base64 — so only a stream that
/// has stopped framing reaches it.
pub const MAX_SSE_FRAME_BYTES: usize = 16 * 1024 * 1024;

const UTF8_BOM: &[u8] = b"\xEF\xBB\xBF";

/// A frame outgrew its bound before its terminating blank line arrived.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("SSE frame exceeded {limit} bytes without a terminating blank line")]
pub struct SseFrameTooLarge {
    pub limit: usize,
}

/// Scan `buf` from `from` for the blank line that ends a frame. Returns
/// `(blank_start, frame_end)`: where that blank line begins, and the offset
/// just past it.
///
/// The line state at `from` is read off the byte before it, which is all a
/// resumed scan needs: `from` is at a line start iff that byte ended a line,
/// and a `\n` at `from` is the tail of a `\r\n` iff that byte was `\r`.
fn scan(buf: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut i = from;
    let mut at_line_start = match from.checked_sub(1).map(|p| buf[p]) {
        None | Some(b'\n') => true,
        Some(b'\r') => {
            if buf.get(i) == Some(&b'\n') {
                i += 1;
            }
            true
        }
        Some(_) => false,
    };
    while let Some(off) = buf[i..].iter().position(|&b| b == b'\r' || b == b'\n') {
        let at = i + off;
        if off > 0 {
            at_line_start = false;
        }
        let len = if buf[at] == b'\r' && buf.get(at + 1) == Some(&b'\n') {
            2
        } else {
            1
        };
        if at_line_start {
            return Some((at, at + len));
        }
        at_line_start = true;
        i = at + len;
    }
    None
}

/// The offset just past the blank line that ends the first frame in `buf`
/// — the number of bytes to take as that frame — or `None` while no frame
/// is complete.
pub fn find_frame_end(buf: &[u8]) -> Option<usize> {
    scan(buf, 0).map(|(_, end)| end)
}

/// Like [`find_frame_end`], also saying where the frame's content ends:
/// `buf[..content_end]` is its lines without the last one's terminator, and
/// `buf[content_end..frame_end]` is that terminator plus the blank line —
/// the frame's separator, as the upstream wrote it.
pub fn split_frame_end(buf: &[u8]) -> Option<(usize, usize)> {
    let (blank, end) = scan(buf, 0)?;
    let content_end = if blank >= 2 && &buf[blank - 2..blank] == b"\r\n" {
        blank - 2
    } else {
        blank.saturating_sub(1)
    };
    Some((content_end, end))
}

/// Incremental splitter: push network chunks in, take complete frames out.
///
/// The scan resumes where the previous one stopped, so a frame that arrives
/// in many chunks costs one pass over its bytes, not one per chunk.
#[derive(Debug)]
pub struct SseFrameSplitter {
    buf: Vec<u8>,
    /// Bytes of `buf` already scanned without finding a frame end.
    scanned: usize,
    /// The last frame ended on a `\r` that was the last buffered byte, so a
    /// `\n` arriving next is the rest of that `\r\n`, not a blank line.
    after_cr: bool,
    max_frame_bytes: usize,
}

impl SseFrameSplitter {
    pub fn new(max_frame_bytes: usize) -> Self {
        Self {
            buf: Vec::new(),
            scanned: 0,
            after_cr: false,
            max_frame_bytes,
        }
    }

    pub fn push(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
    }

    /// The next complete frame, as the upstream's bytes up to and including
    /// its blank line. `Ok(None)` while the next frame is incomplete; `Err`
    /// once the incomplete frame has outgrown the bound. The bytes stay
    /// buffered either way — [`Self::take_rest`] hands them over.
    pub fn next_frame(&mut self) -> Result<Option<Vec<u8>>, SseFrameTooLarge> {
        if self.after_cr && !self.buf.is_empty() {
            self.after_cr = false;
            if self.buf[0] == b'\n' && self.scanned == 0 {
                self.scanned = 1;
            }
        }
        match scan(&self.buf, self.scanned) {
            Some((_, end)) => {
                let frame: Vec<u8> = self.buf.drain(..end).collect();
                self.scanned = 0;
                self.after_cr = self.buf.is_empty() && frame.last() == Some(&b'\r');
                Ok(Some(frame))
            }
            None => {
                self.scanned = self.buf.len();
                if self.buf.len() > self.max_frame_bytes {
                    return Err(SseFrameTooLarge {
                        limit: self.max_frame_bytes,
                    });
                }
                Ok(None)
            }
        }
    }

    /// The buffered bytes of the incomplete frame.
    pub fn buffered(&self) -> &[u8] {
        &self.buf
    }

    /// Take the buffered bytes — the unterminated tail at end-of-stream, or
    /// an oversized frame a caller releases rather than refuses.
    pub fn take_rest(&mut self) -> Vec<u8> {
        self.scanned = 0;
        self.after_cr = false;
        std::mem::take(&mut self.buf)
    }
}

/// One line of a frame: `frame[start..end]` is its content,
/// `frame[end..term_end]` its terminator (empty on an unterminated last
/// line).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseLine {
    pub start: usize,
    pub end: usize,
    pub term_end: usize,
}

/// The lines of one frame, blank lines included. A leading BOM belongs to
/// no line; an empty unterminated tail is not a line.
pub fn lines(frame: &[u8]) -> impl Iterator<Item = SseLine> + '_ {
    let mut pos = if frame.starts_with(UTF8_BOM) {
        UTF8_BOM.len()
    } else {
        0
    };
    std::iter::from_fn(move || {
        if pos >= frame.len() {
            return None;
        }
        let start = pos;
        let (end, term_end) = match frame[start..]
            .iter()
            .position(|&b| b == b'\r' || b == b'\n')
        {
            Some(off) => {
                let end = start + off;
                let len = if frame[end] == b'\r' && frame.get(end + 1) == Some(&b'\n') {
                    2
                } else {
                    1
                };
                (end, end + len)
            }
            None => (frame.len(), frame.len()),
        };
        pos = term_end;
        Some(SseLine {
            start,
            end,
            term_end,
        })
    })
}

/// A line's field name and value. `None` for a blank line and a `:`
/// comment. A line with no colon is a field with an empty value; one space
/// after the colon is framing, not value.
pub fn parse_field(line: &[u8]) -> Option<(&[u8], &[u8])> {
    if line.is_empty() || line[0] == b':' {
        return None;
    }
    match line.iter().position(|&b| b == b':') {
        Some(colon) => {
            let value = &line[colon + 1..];
            Some((&line[..colon], value.strip_prefix(b" ").unwrap_or(value)))
        }
        None => Some((line, &[])),
    }
}

/// Each `data` field's value in `frame`, as a byte range, in order — for a
/// caller that writes back into the frame rather than only reading it.
pub fn data_ranges(frame: &[u8]) -> Vec<Range<usize>> {
    lines(frame)
        .filter_map(|l| {
            let (name, value) = parse_field(&frame[l.start..l.end])?;
            (name == b"data").then(|| {
                let from = l.end - value.len();
                from..l.end
            })
        })
        .collect()
}

/// The fields of one frame.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct SseFields<'a> {
    /// The `data` values joined with `\n`. `None` when the frame has no
    /// `data` field — the spec dispatches no event then — and `Some` of an
    /// empty value when it has one that is empty. Borrowed from the frame
    /// when there is a single `data` line.
    pub data: Option<Cow<'a, [u8]>>,
    /// The last `event` value.
    pub event: Option<&'a [u8]>,
    /// The last `id` value; one containing NUL is ignored.
    pub id: Option<&'a [u8]>,
    /// The last `retry` value that is all ASCII digits.
    pub retry: Option<u64>,
}

/// Read one frame's fields. Unknown fields are ignored.
pub fn parse_frame(frame: &[u8]) -> SseFields<'_> {
    let mut f = SseFields::default();
    for l in lines(frame) {
        let Some((name, value)) = parse_field(&frame[l.start..l.end]) else {
            continue;
        };
        match name {
            b"data" => {
                f.data = Some(match f.data.take() {
                    None => Cow::Borrowed(value),
                    Some(prev) => {
                        let mut joined = prev.into_owned();
                        joined.push(b'\n');
                        joined.extend_from_slice(value);
                        Cow::Owned(joined)
                    }
                });
            }
            b"event" => f.event = Some(value),
            b"id" if !value.contains(&0) => f.id = Some(value),
            b"retry" if !value.is_empty() && value.iter().all(u8::is_ascii_digit) => {
                f.retry = std::str::from_utf8(value).ok().and_then(|s| s.parse().ok());
            }
            _ => {}
        }
    }
    f
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseEvent {
    /// A frame's `data` payload — its `data` lines joined with `\n`.
    Data(String),
    /// The OpenAI-style sentinel `[DONE]`. Called out separately so
    /// Bridges don't have to string-match.
    Done,
}

/// Feed-driven decoder for the provider Bridges: push raw chunks from the
/// HTTP body, pull [`SseEvent`]s out.
///
/// UTF-8 is decoded per complete frame, lossily: an invalid sequence
/// becomes U+FFFD and the stream goes on. A frame with no `data`, or only
/// empty `data`, yields nothing — no Bridge has a use for an empty payload.
#[derive(Debug)]
pub struct SseDecoder {
    frames: SseFrameSplitter,
}

impl Default for SseDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::with_max_frame_bytes(MAX_SSE_FRAME_BYTES)
    }

    pub fn with_max_frame_bytes(max_frame_bytes: usize) -> Self {
        Self {
            frames: SseFrameSplitter::new(max_frame_bytes),
        }
    }

    /// Feed a chunk of bytes. Returns every event this feed completed, or
    /// the error that ends the stream: a frame outgrew the bound.
    pub fn feed<'a>(
        &mut self,
        bytes: impl Into<Cow<'a, [u8]>>,
    ) -> Result<Vec<SseEvent>, SseFrameTooLarge> {
        self.frames.push(&bytes.into());
        let mut events = Vec::new();
        while let Some(frame) = self.frames.next_frame()? {
            events.extend(decode_event(&frame));
        }
        Ok(events)
    }

    /// Flush the unterminated trailing frame, if any. Call once the body
    /// has ended.
    pub fn finish(&mut self) -> Option<SseEvent> {
        decode_event(&self.frames.take_rest())
    }
}

fn decode_event(frame: &[u8]) -> Option<SseEvent> {
    let data = parse_frame(frame).data?;
    if data.is_empty() {
        return None;
    }
    let text = String::from_utf8_lossy(&data);
    Some(if text.trim() == "[DONE]" {
        SseEvent::Done
    } else {
        SseEvent::Data(text.into_owned())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_event_is_emitted_on_terminator() {
        let mut d = SseDecoder::new();
        let ev = d.feed(b"data: {\"x\":1}\n\n".as_slice()).unwrap();
        assert_eq!(ev, vec![SseEvent::Data(r#"{"x":1}"#.into())]);
    }

    /// OpenAI's streamed transcription is CRLF-framed. A decoder that
    /// only knows `\n\n` unlocks nothing on it until end-of-stream, which
    /// is how a streamed transcription came to bill zero tokens (#998
    /// follow-up): the terminal event carrying `usage` was never decoded.
    #[test]
    fn crlf_framed_events_are_decoded_as_they_arrive() {
        let raw = [
            "data: {\"type\":\"transcript.text.delta\",\"delta\":\"hello\"}\r\n\r\n",
            "data: {\"type\":\"transcript.text.done\",\"usage\":{\"input_tokens\":26}}\r\n\r\n",
            "data: [DONE]\r\n\r\n",
        ]
        .concat();
        let mut d = SseDecoder::new();
        assert_eq!(
            d.feed(raw.as_bytes()).unwrap(),
            vec![
                SseEvent::Data(r#"{"type":"transcript.text.delta","delta":"hello"}"#.into()),
                SseEvent::Data(
                    r#"{"type":"transcript.text.done","usage":{"input_tokens":26}}"#.into()
                ),
                SseEvent::Done,
            ],
        );
    }

    /// The spec allows a bare `\r` as a line terminator too, and nothing
    /// stops a stream from mixing the three.
    #[test]
    fn cr_only_and_mixed_terminators_are_decoded() {
        let mut d = SseDecoder::new();
        assert_eq!(
            d.feed("data: {\"a\":1}\r\r".as_bytes()).unwrap(),
            vec![SseEvent::Data(r#"{"a":1}"#.into())],
        );
        assert_eq!(
            d.feed("data: {\"b\":2}\n\r\n".as_bytes()).unwrap(),
            vec![SseEvent::Data(r#"{"b":2}"#.into())],
        );
        assert_eq!(
            d.feed("data: {\"c\":3}\r\n\n".as_bytes()).unwrap(),
            vec![SseEvent::Data(r#"{"c":3}"#.into())],
        );
    }

    /// A chunk boundary can fall inside a `\r\n` pair. However the halves
    /// are framed, the caller must see each event exactly once and never
    /// a spurious empty one.
    #[test]
    fn a_terminator_split_across_feeds_yields_each_event_once() {
        let mut d = SseDecoder::new();
        let mut seen = Vec::new();
        for chunk in ["data: {\"x\":1}\r", "\n\r", "\ndata: {\"y\":2}\r\n", "\r\n"] {
            seen.extend(d.feed(chunk.as_bytes()).unwrap());
        }
        seen.extend(d.finish());
        assert_eq!(
            seen,
            vec![
                SseEvent::Data(r#"{"x":1}"#.into()),
                SseEvent::Data(r#"{"y":2}"#.into()),
            ],
        );
    }

    #[test]
    fn done_sentinel_is_decoded_separately() {
        let mut d = SseDecoder::new();
        let ev = d.feed(b"data: [DONE]\n\n".as_slice()).unwrap();
        assert_eq!(ev, vec![SseEvent::Done]);
    }

    #[test]
    fn events_split_across_feeds_are_reassembled() {
        let mut d = SseDecoder::new();
        let first = d.feed(b"data: {\"x".as_slice()).unwrap();
        let second = d.feed(b"\":1}\n\n".as_slice()).unwrap();
        assert!(first.is_empty());
        assert_eq!(second, vec![SseEvent::Data(r#"{"x":1}"#.into())]);
    }

    #[test]
    fn multiple_data_lines_concatenate_with_newline() {
        let mut d = SseDecoder::new();
        let ev = d.feed(b"data: line1\ndata: line2\n\n".as_slice()).unwrap();
        assert_eq!(ev, vec![SseEvent::Data("line1\nline2".into())]);
    }

    #[test]
    fn non_data_fields_are_skipped() {
        let mut d = SseDecoder::new();
        let ev = d
            .feed(b"event: ping\ndata: payload\nid: 42\n\n".as_slice())
            .unwrap();
        assert_eq!(ev, vec![SseEvent::Data("payload".into())]);
    }

    #[test]
    fn multiple_events_in_one_feed() {
        let mut d = SseDecoder::new();
        let ev = d
            .feed(b"data: a\n\ndata: b\n\ndata: [DONE]\n\n".as_slice())
            .unwrap();
        assert_eq!(
            ev,
            vec![
                SseEvent::Data("a".into()),
                SseEvent::Data("b".into()),
                SseEvent::Done,
            ]
        );
    }

    #[test]
    fn crlf_line_endings_are_tolerated() {
        let mut d = SseDecoder::new();
        // A stray \r at end-of-line must not leak into the payload.
        let ev = d.feed(b"data: hello\r\n\n".as_slice()).unwrap();
        assert_eq!(ev, vec![SseEvent::Data("hello".into())]);
    }

    #[test]
    fn finish_emits_trailing_unterminated_event() {
        let mut d = SseDecoder::new();
        let mid = d.feed(b"data: tail-only".as_slice()).unwrap();
        assert!(mid.is_empty());
        let finale = d.finish();
        assert_eq!(finale, Some(SseEvent::Data("tail-only".into())));
    }

    #[test]
    fn finish_on_empty_buffer_returns_none() {
        let mut d = SseDecoder::new();
        assert!(d.finish().is_none());
    }

    // ---- regression coverage for issue #111 -------------------------
    // Multi-byte UTF-8 (CJK = 3 bytes, emoji = 4 bytes) split across
    // HTTP chunk boundaries used to surface as U+FFFD because each
    // chunk got passed through `from_utf8_lossy` independently. The
    // tests below split a known-good payload at every internal byte
    // index — only an actual byte-buffered decoder passes all of them.

    #[test]
    fn cjk_split_across_feeds_round_trips_intact() {
        // "你好世界" — 12 bytes in UTF-8, every codepoint is 3 bytes.
        let payload = "你好世界";
        let event = format!("data: {{\"content\":\"{payload}\"}}\n\n");
        let raw = event.as_bytes();
        // Split at every interior byte; every split must produce the
        // same final event. The bug: splitting between bytes 2-3 of any
        // codepoint corrupts that codepoint into U+FFFD.
        for split in 1..raw.len() {
            let mut d = SseDecoder::new();
            d.feed(&raw[..split]).unwrap();
            let events = d.feed(&raw[split..]).unwrap();
            assert_eq!(events.len(), 1, "split={split}");
            let SseEvent::Data(s) = &events[0] else {
                panic!("expected Data event at split={split}");
            };
            assert_eq!(
                s,
                &format!("{{\"content\":\"{payload}\"}}"),
                "split={split} corrupted CJK",
            );
            assert!(
                !s.contains('\u{FFFD}'),
                "split={split} produced U+FFFD: {s}",
            );
        }
    }

    #[test]
    fn emoji_split_across_feeds_round_trips_intact() {
        // 4-byte emoji per codepoint.
        let payload = "🙂🚀";
        let event = format!("data: {{\"content\":\"{payload}\"}}\n\n");
        let raw = event.as_bytes();
        for split in 1..raw.len() {
            let mut d = SseDecoder::new();
            d.feed(&raw[..split]).unwrap();
            let events = d.feed(&raw[split..]).unwrap();
            assert_eq!(events.len(), 1, "split={split}");
            let SseEvent::Data(s) = &events[0] else {
                panic!("expected Data event at split={split}");
            };
            assert_eq!(s, &format!("{{\"content\":\"{payload}\"}}"));
            assert!(!s.contains('\u{FFFD}'), "split={split} produced U+FFFD");
        }
    }

    #[test]
    fn truncated_multibyte_at_eof_is_replaced_on_finish() {
        // Stream ends mid-codepoint (e.g. upstream connection dropped).
        // The pending byte cannot complete; finish() must surface what
        // we have (U+FFFD for the truncated bytes) rather than dropping
        // them silently.
        let mut d = SseDecoder::new();
        d.feed(b"data: ".as_slice()).unwrap();
        // First two bytes of the 3-byte sequence for "你" — no third byte ever arrives.
        d.feed(&[0xE4_u8, 0xBD][..]).unwrap();
        let final_event = d.finish().expect("trailing bytes should surface");
        let SseEvent::Data(s) = final_event else {
            panic!("expected Data event");
        };
        assert!(
            s.contains('\u{FFFD}'),
            "truncated tail at EOF should be replaced with U+FFFD: {s:?}"
        );
    }

    #[test]
    fn invalid_utf8_is_lossily_decoded() {
        let mut d = SseDecoder::new();
        let bytes: Vec<u8> = b"data: "
            .iter()
            .copied()
            .chain([0xff_u8, b'\n', b'\n'])
            .collect();
        let ev = d.feed(bytes.as_slice()).unwrap();
        // 0xff becomes U+FFFD.
        assert_eq!(ev.len(), 1);
        if let SseEvent::Data(payload) = &ev[0] {
            assert!(payload.contains('\u{FFFD}'));
        } else {
            panic!("expected Data event");
        }
    }

    // ---- the shared frame splitter -----------------------------------

    /// Every frame `SseFrameSplitter` yields for `chunks`, plus the
    /// unterminated tail.
    fn split_all(chunks: &[&[u8]]) -> (Vec<Vec<u8>>, Vec<u8>) {
        let mut s = SseFrameSplitter::new(MAX_SSE_FRAME_BYTES);
        let mut frames = Vec::new();
        for c in chunks {
            s.push(c);
            while let Some(f) = s.next_frame().unwrap() {
                frames.push(f);
            }
        }
        (frames, s.take_rest())
    }

    fn data_of(frame: &[u8]) -> Option<Vec<u8>> {
        parse_frame(frame).data.map(Cow::into_owned)
    }

    #[test]
    fn any_line_terminator_in_any_mix_ends_lines_and_frames() {
        for (raw, content_end, frame_end) in [
            (&b"data: a\n\nrest"[..], 7, 9),
            (b"data: a\r\n\r\nrest", 7, 11),
            (b"data: a\r\rrest", 7, 9),
            (b"data: a\n\r\nrest", 7, 10),
            (b"data: a\r\n\nrest", 7, 10),
            (b"data: a\r\r\nrest", 7, 10),
            (b"event: x\rdata: a\r\rrest", 16, 18),
            // A blank line with nothing before it is an (empty) frame.
            (b"\ndata: a\n\n", 0, 1),
        ] {
            assert_eq!(
                split_frame_end(raw),
                Some((content_end, frame_end)),
                "{:?}",
                String::from_utf8_lossy(raw)
            );
        }
        assert_eq!(find_frame_end(b"data: a\r\ndata: b\r\n"), None);
        assert_eq!(find_frame_end(b"data: a\r"), None);
    }

    /// A chunk boundary at every byte of a CRLF stream: the frames always
    /// concatenate back to the input, each event is read exactly once, and
    /// no blank line is invented from the halves of a split `\r\n`.
    #[test]
    fn crlf_split_across_chunks_at_every_offset() {
        let raw: &[u8] = b"event: a\r\ndata: 1\r\n\r\nevent: b\r\ndata: 2\r\n\r\n";
        for cut in 0..=raw.len() {
            let (frames, rest) = split_all(&[&raw[..cut], &raw[cut..]]);
            assert_eq!([frames.concat(), rest.clone()].concat(), raw, "cut={cut}");
            // At most the `\n` of a `\r\n` whose `\r` ended the last frame.
            assert!(rest.is_empty() || rest == b"\n", "cut={cut}");
            let data: Vec<_> = frames.iter().filter_map(|f| data_of(f)).collect();
            assert_eq!(data, vec![b"1".to_vec(), b"2".to_vec()], "cut={cut}");
        }
    }

    /// A frame whose blank line is a `\r` at the end of the buffer is
    /// complete; a `\n` arriving next is the rest of that `\r\n` and stays
    /// in the byte stream without reading as a blank line.
    #[test]
    fn a_frame_ending_on_a_buffered_cr_is_released_and_its_lf_is_not_a_blank_line() {
        let (frames, rest) = split_all(&[b"data: a\r\n\r", b"\ndata: b\r\n\r\n"]);
        assert!(rest.is_empty());
        assert_eq!(
            frames,
            vec![b"data: a\r\n\r".to_vec(), b"\ndata: b\r\n\r\n".to_vec()]
        );
        assert_eq!(data_of(&frames[1]).as_deref(), Some(&b"b"[..]));
    }

    #[test]
    fn byte_at_a_time_matches_whole_buffer() {
        let raw: &[u8] =
            b"\xEF\xBB\xBF: hi\rdata: a\r\rdata:\ndata: b\n\nevent: e\r\ndata\r\n\r\ndata: \xE4\xBD\xA0\r\n\r\ndata: tail";
        let bytes: Vec<&[u8]> = raw.chunks(1).collect();
        for (frames, rest) in [split_all(&[raw]), split_all(&bytes)] {
            // Trickled, a frame whose blank line is `\r\n` is released at
            // the `\r`, so the `\n` rides at the head of the next frame:
            // boundaries may move, the bytes and the events may not.
            assert_eq!([frames.concat(), rest.clone()].concat(), raw);
            let data: Vec<_> = frames.iter().filter_map(|f| data_of(f)).collect();
            assert_eq!(
                data,
                vec![
                    b"a".to_vec(),
                    b"\nb".to_vec(),
                    b"".to_vec(),
                    "你".as_bytes().to_vec()
                ]
            );
            assert_eq!(data_of(&rest).as_deref(), Some(&b"tail"[..]));
        }
    }

    #[test]
    fn one_leading_bom_is_skipped() {
        let mut d = SseDecoder::new();
        assert_eq!(
            d.feed(&b"\xEF\xBB\xBFdata: first\n\n"[..]).unwrap(),
            vec![SseEvent::Data("first".into())]
        );
        // Only one: a second BOM is part of the field name.
        assert_eq!(
            parse_frame(b"\xEF\xBB\xBF\xEF\xBB\xBFdata: x\n\n").data,
            None
        );
    }

    #[test]
    fn comments_and_unknown_fields_are_ignored() {
        let f = parse_frame(b": keep-alive\rfoo: bar\rdata: x\r\r");
        assert_eq!(f.data.as_deref(), Some(&b"x"[..]));
        assert_eq!(parse_frame(b": only a comment\n\n"), SseFields::default());
    }

    /// Every `data` line appends its value and a `\n` — an empty value too
    /// — and the final `\n` is dropped at dispatch.
    #[test]
    fn empty_data_lines_still_contribute_their_newline() {
        assert_eq!(data_of(b"data:\ndata: x\n\n").as_deref(), Some(&b"\nx"[..]));
        assert_eq!(data_of(b"data: x\ndata:\n\n").as_deref(), Some(&b"x\n"[..]));
        assert_eq!(data_of(b"data\ndata\n\n").as_deref(), Some(&b"\n"[..]));
        assert_eq!(data_of(b"data:\n\n").as_deref(), Some(&b""[..]));
        assert_eq!(data_of(b"event: x\n\n"), None);
        // The decoder hands no Bridge an empty payload.
        let mut d = SseDecoder::new();
        assert!(d.feed(&b"data:\n\n"[..]).unwrap().is_empty());
        assert_eq!(
            d.feed(&b"data:\ndata: x\n\n"[..]).unwrap(),
            vec![SseEvent::Data("\nx".into())]
        );
    }

    #[test]
    fn a_field_without_a_colon_has_an_empty_value() {
        assert_eq!(parse_field(b"data"), Some((&b"data"[..], &b""[..])));
        assert_eq!(parse_frame(b"event\ndata: x\n\n").event, Some(&b""[..]));
    }

    #[test]
    fn exactly_one_space_after_the_colon_is_stripped() {
        assert_eq!(parse_field(b"data:x"), Some((&b"data"[..], &b"x"[..])));
        assert_eq!(parse_field(b"data: x"), Some((&b"data"[..], &b"x"[..])));
        assert_eq!(parse_field(b"data:  x"), Some((&b"data"[..], &b" x"[..])));
        assert_eq!(
            parse_field(b"data: x: y"),
            Some((&b"data"[..], &b"x: y"[..]))
        );
    }

    #[test]
    fn event_id_and_retry_take_the_last_valid_value() {
        let f = parse_frame(
            b"event: a\nevent: b\nid: 1\nid: 2\x00\nretry: 100\nretry: 5s\ndata: x\n\n",
        );
        assert_eq!(f.event, Some(&b"b"[..]));
        assert_eq!(f.id, Some(&b"1"[..]));
        assert_eq!(f.retry, Some(100));
    }

    #[test]
    fn data_ranges_point_at_each_value_in_place() {
        let frame: &[u8] = b"\xEF\xBB\xBFevent: m\rdata: {\"a\":\r\ndata\ndata:1}\r\n\r\n";
        let got: Vec<&[u8]> = data_ranges(frame).into_iter().map(|r| &frame[r]).collect();
        assert_eq!(got, vec![&b"{\"a\":"[..], b"", b"1}"]);
    }

    #[test]
    fn an_unterminated_frame_past_the_bound_is_an_error() {
        let mut s = SseFrameSplitter::new(16);
        s.push(b"data: 0123456789");
        assert_eq!(s.next_frame(), Ok(None));
        s.push(b"a");
        assert_eq!(s.next_frame(), Err(SseFrameTooLarge { limit: 16 }));
        // The bytes stay for a caller that releases rather than refuses.
        assert_eq!(s.take_rest(), b"data: 0123456789a");

        // Complete frames drain before the bound is judged.
        let mut s = SseFrameSplitter::new(16);
        s.push(b"data: 0123456789\n\ndata: 1\n\n");
        assert!(s.next_frame().unwrap().is_some());
        assert!(s.next_frame().unwrap().is_some());
        assert_eq!(s.next_frame(), Ok(None));

        let mut d = SseDecoder::with_max_frame_bytes(16);
        assert_eq!(
            d.feed(&b"data: 0123456789abc"[..]),
            Err(SseFrameTooLarge { limit: 16 })
        );
    }

    /// Bytes are decoded per complete frame, so a multibyte character split
    /// across chunks on a CR-framed stream survives intact.
    #[test]
    fn multibyte_split_across_chunks_on_a_cr_framed_stream() {
        let raw = "data: 你好🙂\r\r".as_bytes();
        for cut in 1..raw.len() {
            let mut d = SseDecoder::new();
            let mut events = d.feed(&raw[..cut]).unwrap();
            events.extend(d.feed(&raw[cut..]).unwrap());
            assert_eq!(events, vec![SseEvent::Data("你好🙂".into())], "cut={cut}");
        }
    }
}
