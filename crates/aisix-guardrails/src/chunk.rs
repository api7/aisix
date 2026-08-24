//! The one place a provider's per-call size limit is turned into calls.
//!
//! Every remote guardrail kind talks to an API that caps how much text
//! one call may carry. The family-wide contract (AISIX-Cloud#1382) is:
//!
//! - **Content is never truncated to fit a limit.** Over-limit text is
//!   split and *every* piece is submitted. Truncating lets a caller hide
//!   content past the cap behind a clean verdict, which is a bypass the
//!   caller controls — and it is silent, because the call succeeds and
//!   the row reports a pass (AISIX-Cloud#1381, and #448 before it).
//! - **There is no cap on the number of pieces.** A request costs as
//!   many provider calls as its content needs. A cap would reintroduce
//!   unscanned content through the back door.
//! - **The split is lossless**, so a kind that writes masked text back
//!   can concatenate the per-chunk replacements and reproduce the
//!   caller's content exactly.
//!
//! Kinds whose provider imposes no documented limit (`bedrock`,
//! `lakera`, `presidio`, `openai_moderation`) submit whole and do not
//! use this module; their bound is the provider's own.

/// Split `text` into chunks of at most `max_chars` **characters**,
/// preferring to break after whitespace.
///
/// Lossless by construction: `chunks.concat() == text`, so callers that
/// mask can rebuild the original from per-chunk replacements. Character-
/// counted rather than byte-counted because the provider limits this
/// module serves are documented in characters — and byte slicing would
/// split a multi-byte character in half.
///
/// A whitespace-free run longer than `max_chars` is split mid-run rather
/// than truncated: the entire token must still be evaluated (#448).
/// Empty input yields no chunks, never `[""]`.
pub(crate) fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    debug_assert!(max_chars > 0, "max_chars must be positive");
    if text.is_empty() || max_chars == 0 {
        return Vec::new();
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        return vec![text.to_owned()];
    }
    let mut chunks: Vec<String> = Vec::new();
    let mut start = 0usize;
    while start < chars.len() {
        let hard_end = (start + max_chars).min(chars.len());
        // Break after the last whitespace in the window so the separator
        // stays with the chunk it followed — dropping it would make the
        // concatenation lossy. No whitespace in range means a long token:
        // split it at the hard limit rather than overshoot.
        let end = if hard_end == chars.len() {
            hard_end
        } else {
            chars[start..hard_end]
                .iter()
                .rposition(|c| c.is_whitespace())
                .map_or(hard_end, |i| start + i + 1)
        };
        chunks.push(chars[start..end].iter().collect());
        start = end;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_yields_no_chunks() {
        assert!(chunk_text("", 100).is_empty());
    }

    #[test]
    fn text_within_the_limit_is_one_chunk() {
        assert_eq!(chunk_text("hello world", 100), vec!["hello world"]);
    }

    #[test]
    fn exactly_the_limit_is_not_split() {
        let text: String = "a".repeat(2_000);
        assert_eq!(chunk_text(&text, 2_000).len(), 1);
    }

    #[test]
    fn every_chunk_is_within_the_limit() {
        let word: String = "a".repeat(700);
        let text = format!("{word} {word} {word} {word}");
        let chunks = chunk_text(&text, 2_000);
        assert!(chunks.len() >= 2, "expected a split, got {}", chunks.len());
        for c in &chunks {
            assert!(c.chars().count() <= 2_000, "chunk over the limit");
        }
    }

    /// The property the write-back path depends on, and the one that
    /// makes "never truncated" checkable: nothing is dropped, reordered,
    /// or rewritten.
    #[test]
    fn splitting_is_lossless() {
        for text in [
            "one two three four five",
            "line one\nline two\n\nline four",
            "  leading and trailing   ",
            "\ttabs\tand\nnewlines\t",
            "nospaceatallherejustonelongrun",
            "中文没有空格所以只能按字符切分这是一个很长的句子",
            "mixed 中英文 content with spaces 和换行\n还有更多",
        ] {
            for max in [1usize, 2, 3, 7, 16] {
                let chunks = chunk_text(text, max);
                assert_eq!(
                    chunks.concat(),
                    text,
                    "lossy split of {text:?} at max={max}"
                );
                for c in &chunks {
                    assert!(
                        c.chars().count() <= max,
                        "chunk {c:?} exceeds max={max} for {text:?}"
                    );
                    assert!(!c.is_empty(), "empty chunk for {text:?} at max={max}");
                }
            }
        }
    }

    #[test]
    fn oversized_whitespace_free_run_is_split_not_truncated() {
        // The #448 shape: a single token longer than the limit. Every
        // character must still reach the provider.
        let word: String = "x".repeat(5_000);
        let chunks = chunk_text(&word, 2_000);
        assert_eq!(chunks.len(), 3, "5k chars over a 2k limit → 3 chunks");
        assert_eq!(chunks.concat(), word, "no character may be dropped");
    }

    #[test]
    fn multibyte_characters_are_never_split_in_half() {
        // Byte-slicing this (higress's shape) would corrupt the text.
        let text: String = "你好世界".repeat(1_000);
        let chunks = chunk_text(&text, 2_000);
        assert_eq!(chunks.concat(), text);
        for c in &chunks {
            assert!(c.chars().count() <= 2_000);
        }
    }

    #[test]
    fn breaks_prefer_whitespace_over_mid_word() {
        // "aaaa bb" at max 6: breaking at the hard limit would cut "bb"
        // in half; breaking after the space keeps the word whole.
        assert_eq!(chunk_text("aaaa bb", 6), vec!["aaaa ", "bb"]);
    }
}
