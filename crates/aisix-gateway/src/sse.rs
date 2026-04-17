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
    buffer: String,
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
        // Non-UTF-8 bytes are replaced rather than erroring — upstreams
        // that break encoding still surface a best-effort event so a
        // single bad byte doesn't kill the whole stream.
        let chunk = String::from_utf8_lossy(&bytes);
        self.buffer.push_str(&chunk);

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
