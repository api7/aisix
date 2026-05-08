//! Server-Sent Events (SSE) line decoder used by streaming Bridges.
//!
//! All four target providers (OpenAI, Anthropic, Gemini, DeepSeek) emit
//! completions as SSE with the OpenAI-style shape:
//!
//! ```text
//! data: {"choices":[…]}
//! data: {"choices":[…]}
//! data: [DONE]
//! ```
//!
//! The decoder is feed-driven: callers push raw chunks from the HTTP body
//! stream and pull typed [`SseEvent`]s back out. State survives partial
//! messages that straddle chunk boundaries. We intentionally don't use
//! `eventsource-stream` directly here — its interface is reqwest-flavoured
//! and forces a particular body shape; this decoder works against any
//! `&[u8]` and keeps the Bridge trait HTTP-client-agnostic.
//!
//! The decoder handles the subset of the SSE spec that all four providers
//! actually emit:
//! - `data:` lines (everything else is ignored)
//! - `\n\n` separator marking a complete event
//! - UTF-8 only (all four providers are JSON-over-SSE)

use std::borrow::Cow;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseEvent {
    /// A `data:` payload (everything between `data: ` and the message
    /// terminator, concatenated if the provider split over multiple
    /// lines — per the SSE spec).
    Data(String),
    /// The OpenAI-style sentinel `[DONE]`. Called out separately so
    /// Bridges don't have to string-match.
    Done,
}

#[derive(Debug, Default)]
pub struct SseDecoder {
    /// Raw bytes that have not yet decoded into a complete UTF-8
    /// sequence. We hold these across feeds so a multi-byte codepoint
    /// (CJK = 3 bytes, emoji = 4 bytes) split across HTTP chunk
    /// boundaries doesn't get corrupted into U+FFFD. The *previous*
    /// implementation called `String::from_utf8_lossy` per chunk,
    /// which replaced every truncated head-of-sequence byte with the
    /// replacement character — visible mid-stream as `好` rendering as
    /// `�` in any Chinese / Japanese / Korean / emoji response. See
    /// issue #111.
    byte_buf: Vec<u8>,
    /// Already-decoded text awaiting the `\n\n` event terminator.
    buffer: String,
    /// Concatenated `data:` payload of the in-progress event.
    current_data: String,
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk of bytes. Returns every complete event that was
    /// unlocked by this feed. Partial messages remain buffered until a
    /// subsequent call supplies the rest.
    pub fn feed<'a>(&mut self, bytes: impl Into<Cow<'a, [u8]>>) -> Vec<SseEvent> {
        let bytes = bytes.into();
        self.byte_buf.extend_from_slice(&bytes);
        self.decode_buffered_bytes();

        let mut events = Vec::new();
        // Event terminator is \n\n; process one message at a time.
        while let Some(idx) = self.buffer.find("\n\n") {
            let message: String = self.buffer.drain(..idx + 2).collect();
            self.decode_message(&message, &mut events);
        }
        events
    }

    /// Flush any buffered trailing bytes as the final event. Call once
    /// the HTTP body has ended; returns `None` if nothing is buffered.
    pub fn finish(&mut self) -> Option<SseEvent> {
        // Force a lossy decode of any trailing bytes we kept around
        // hoping for the rest of a multi-byte sequence — at end-of-
        // stream they're definitively malformed, so replace them with
        // U+FFFD now and surface whatever we have. This matches the
        // pre-fix behaviour for genuinely-invalid UTF-8 at EOF (no
        // information loss compared to the old per-chunk lossy decode).
        if !self.byte_buf.is_empty() {
            let tail = String::from_utf8_lossy(&self.byte_buf).into_owned();
            self.byte_buf.clear();
            self.buffer.push_str(&tail);
        }

        if self.buffer.trim().is_empty() && self.current_data.is_empty() {
            return None;
        }
        // Treat the tail as a terminated message so the decoder emits
        // whatever it had collected.
        let tail = std::mem::take(&mut self.buffer);
        let mut events = Vec::new();
        self.decode_message(&format!("{tail}\n\n"), &mut events);
        events.into_iter().next()
    }

    /// Drain `byte_buf` of every byte that's part of a fully-formed
    /// UTF-8 sequence (push them onto `buffer`), and keep any trailing
    /// truncated multi-byte sequence in `byte_buf` for the next feed.
    /// Mid-buffer invalid bytes (0xFF, lone 0x80, etc.) collapse to
    /// U+FFFD and we keep going — matching the pre-fix lossy contract
    /// for genuinely-invalid encodings.
    fn decode_buffered_bytes(&mut self) {
        let mut start = 0;
        loop {
            let remaining = &self.byte_buf[start..];
            match std::str::from_utf8(remaining) {
                Ok(s) => {
                    self.buffer.push_str(s);
                    start = self.byte_buf.len();
                    break;
                }
                Err(e) => {
                    let valid_up_to = e.valid_up_to();
                    // The crate forbids `unsafe`, so we re-validate
                    // the prefix even though `Utf8Error::valid_up_to`
                    // guarantees it. This is a single re-scan over a
                    // bounded-size leftover buffer per chunk, not per
                    // byte, so the cost is negligible compared to the
                    // bug it would otherwise reintroduce.
                    let valid_prefix = std::str::from_utf8(&remaining[..valid_up_to])
                        .expect("Utf8Error::valid_up_to bytes must be valid UTF-8");
                    self.buffer.push_str(valid_prefix);
                    match e.error_len() {
                        Some(len) => {
                            // Definitive decode error mid-buffer (e.g.
                            // a stray 0xFF). Emit U+FFFD and step past
                            // the offending sequence; continue trying
                            // to decode the rest of byte_buf.
                            self.buffer.push('\u{FFFD}');
                            start += valid_up_to + len;
                        }
                        None => {
                            // Truncated multi-byte sequence at the
                            // tail of the buffer. Keep those bytes for
                            // the next feed — that's the whole point
                            // of the byte buffer.
                            start += valid_up_to;
                            break;
                        }
                    }
                }
            }
        }
        self.byte_buf.drain(..start);
    }

    fn decode_message(&mut self, message: &str, out: &mut Vec<SseEvent>) {
        for line in message.lines() {
            let line = line.trim_end_matches('\r');
            if line.is_empty() {
                continue;
            }
            // Only `data:` lines are relevant for our providers.
            if let Some(rest) = line.strip_prefix("data:") {
                let data = rest.strip_prefix(' ').unwrap_or(rest);
                if !self.current_data.is_empty() {
                    self.current_data.push('\n');
                }
                self.current_data.push_str(data);
            }
            // Silently skip other field lines (event:, id:, retry:, comments).
        }

        if self.current_data.is_empty() {
            return;
        }

        let finished = std::mem::take(&mut self.current_data);
        if finished.trim() == "[DONE]" {
            out.push(SseEvent::Done);
        } else {
            out.push(SseEvent::Data(finished));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_event_is_emitted_on_terminator() {
        let mut d = SseDecoder::new();
        let ev = d.feed(b"data: {\"x\":1}\n\n".as_slice());
        assert_eq!(ev, vec![SseEvent::Data(r#"{"x":1}"#.into())]);
    }

    #[test]
    fn done_sentinel_is_decoded_separately() {
        let mut d = SseDecoder::new();
        let ev = d.feed(b"data: [DONE]\n\n".as_slice());
        assert_eq!(ev, vec![SseEvent::Done]);
    }

    #[test]
    fn events_split_across_feeds_are_reassembled() {
        let mut d = SseDecoder::new();
        let first = d.feed(b"data: {\"x".as_slice());
        let second = d.feed(b"\":1}\n\n".as_slice());
        assert!(first.is_empty());
        assert_eq!(second, vec![SseEvent::Data(r#"{"x":1}"#.into())]);
    }

    #[test]
    fn multiple_data_lines_concatenate_with_newline() {
        let mut d = SseDecoder::new();
        let ev = d.feed(b"data: line1\ndata: line2\n\n".as_slice());
        assert_eq!(ev, vec![SseEvent::Data("line1\nline2".into())]);
    }

    #[test]
    fn non_data_fields_are_skipped() {
        let mut d = SseDecoder::new();
        let ev = d.feed(b"event: ping\ndata: payload\nid: 42\n\n".as_slice());
        assert_eq!(ev, vec![SseEvent::Data("payload".into())]);
    }

    #[test]
    fn multiple_events_in_one_feed() {
        let mut d = SseDecoder::new();
        let ev = d.feed(b"data: a\n\ndata: b\n\ndata: [DONE]\n\n".as_slice());
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
        // Note: SSE event terminator is \n\n per spec; we don't promise
        // \r\n\r\n support because no target provider emits it. Here we
        // verify stray \r at end-of-line doesn't leak into the payload.
        let ev = d.feed(b"data: hello\r\n\n".as_slice());
        assert_eq!(ev, vec![SseEvent::Data("hello".into())]);
    }

    #[test]
    fn finish_emits_trailing_unterminated_event() {
        let mut d = SseDecoder::new();
        let mid = d.feed(b"data: tail-only".as_slice());
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
            d.feed(&raw[..split]);
            let events = d.feed(&raw[split..]);
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
            d.feed(&raw[..split]);
            let events = d.feed(&raw[split..]);
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
        d.feed(b"data: ".as_slice());
        // First two bytes of the 3-byte sequence for "你" — no third byte ever arrives.
        d.feed(&[0xE4_u8, 0xBD][..]);
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
        let ev = d.feed(bytes.as_slice());
        // 0xff becomes U+FFFD.
        assert_eq!(ev.len(), 1);
        if let SseEvent::Data(payload) = &ev[0] {
            assert!(payload.contains('\u{FFFD}'));
        } else {
            panic!("expected Data event");
        }
    }
}
