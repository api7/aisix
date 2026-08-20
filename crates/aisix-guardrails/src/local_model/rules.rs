//! Layer ② of the three-layer pipeline (AISIX-Cloud#1331): rule scoring
//! between the regex candidate layer (①) and the model judgement (③).
//!
//! The MVP shipped ① → ③ directly, which put the whole separation burden
//! on zero-shot cosine — measured unable to split the hard positives
//! (`升级到 2022.4`, `Virtuoso IC6.1.8`, 0.75–0.79) from the compile-log
//! hard negatives (≤0.77). This layer restores the design's division of
//! labor: candidates with decisive lexical evidence are resolved here in
//! microseconds, and ONLY the genuinely ambiguous band pays a model call.
//!
//! Per candidate span:
//!   - hotword co-occurrence RAISES the score. Three hotword classes
//!     (Chinese triggers, English triggers, EDA tool names), each counted
//!     once. A hotword ADJACENT to the span (≤ [`ADJACENT_GAP_CHARS`]
//!     chars away — `版本是 12.1`, `Virtuoso IC6.1.8`) is decisive
//!     evidence: +2 per class. Hotwords merely within the proximity
//!     window are weak evidence: +1 total, regardless of class count —
//!     distant co-occurrence alone must never mask, only route to the
//!     model.
//!   - negative patterns LOWER the score: -4 per class. A measurement
//!     unit right after the number, an IPv4-shaped span, or a source
//!     location is stronger evidence of "not a version" than any nearby
//!     hotword is of the opposite, so one negative class (-4) outweighs
//!     one decisive positive class (+2) — `版本升级后耗时 12.5s` must
//!     not mask the timing.
//!   - the double threshold turns the score into a [`RuleDecision`]:
//!     ≥ [`MASK_SCORE`] rewrites without consulting the model,
//!     ≤ [`PASS_SCORE`] releases without consulting the model, and only
//!     the band in between (no evidence, or weak/conflicting evidence)
//!     goes to layer ③.
//!
//! This is the mainstream DLP shape — hotword-proximity confidence
//! adjustment over a strong-format base pattern (Google Cloud DLP hotword
//! rules, Microsoft Purview supporting elements, AWS Macie
//! `maximumMatchDistance`, Palo Alto proximity keywords) — not an
//! invention of this crate. The proximity window default of
//! [`DEFAULT_PROXIMITY_CHARS`] sits at the tight end of the range those
//! engines use (Macie defaults to 50 chars, Palo Alto to 200, Purview
//! recommends 300, Google caps at 1000): the driving corpus is dense
//! compile logs where a wide window manufactures accidental
//! co-occurrence, and 50 also matches the ±50-char window layer ③ judges,
//! so the two layers reason about the same context. Override with
//! [`RULE_WINDOW_ENV`][super::RULE_WINDOW_ENV] (clamped to
//! [`MAX_PROXIMITY_CHARS`], the Google DLP hard cap).
//!
//! Threat-model boundary, inherited from the design issue's "只能防无意
//! 泄漏" line and inherent to score-subtraction DLP: a sender (or a
//! hostile upstream, on the output side) can defeat the rule layer by
//! FORMATTING — appending a unit-looking suffix (`升级到 2022.4s`), a
//! `:digit` tail, or fullwidth digits that never become candidates. The
//! layer scores accidental phrasing, not adversarial encoding.
//!
//! Everything here is pure text work — no model, no I/O — so the layer is
//! unit-testable standalone, which is also how the "rules alone" halves
//! of the acceptance matrix are measured.

use std::ops::Range;

use regex::Regex;

use super::window_bounds;

/// Default hotword proximity window (chars each side of the span).
pub(super) const DEFAULT_PROXIMITY_CHARS: usize = 50;

/// Upper clamp for the proximity window override.
pub(super) const MAX_PROXIMITY_CHARS: usize = 1000;

/// A hotword within this many chars of the span edge is ADJACENT —
/// decisive rather than weak evidence. 8 chars covers the connective
/// tissue of every acceptance shape (`版本是 12.1` gap 2, `Virtuoso
/// IC6.1.8` gap 3, `upgrade to 21.15` gap 1, `version: v2022.4` gap 3)
/// while staying too small for an unrelated number to drift inside.
const ADJACENT_GAP_CHARS: usize = 8;

/// Score for each hotword class with an adjacent match.
const ADJACENT_CLASS_SCORE: i32 = 2;
/// Score when hotwords exist only at window distance (flat, not per
/// class — distant co-occurrence stays weak no matter how many classes).
const WINDOW_EVIDENCE_SCORE: i32 = 1;
/// Score for each negative-pattern class that hits.
const NEGATIVE_CLASS_SCORE: i32 = -4;

/// Decision bands: `score >= MASK_SCORE` masks without the model — one
/// adjacent hotword class alone is enough.
const MASK_SCORE: i32 = 2;
/// `score <= PASS_SCORE` releases without the model — one negative class
/// alone is enough, even against an adjacent hotword (-4 + 2).
const PASS_SCORE: i32 = -1;

/// Chinese trigger hotwords (substring match — no word boundaries in
/// CJK). `版本号` is a substring of no other entry but kept explicit so
/// the list reads as the configured vocabulary.
const ZH_TRIGGERS: &[&str] = &["版本号", "版本", "升级到", "回退到"];

/// English trigger hotwords: word-bounded so `conversion` never fires
/// `version`. `upgraded? to` covers the bare and inflected forms.
const EN_TRIGGER_PATTERN: &str = r"(?i)\b(?:version|release|build|upgraded?\s+to)\b";

/// EDA tool names — the strongest anchors (`Virtuoso IC6.1.8` needs no
/// model). Word-bounded, case-insensitive (`vcs` / `VCS`).
const TOOL_PATTERN: &str = r"(?i)\b(?:virtuoso|calibre|vcs|innovus|icc2|primetime)\b";

/// Negative: a measurement unit right after the span (`12.345s`,
/// `4.2 GB`, `0.13um`, `99.9%`, `0.5ns`). Beyond the design brief's
/// minimum list, this covers the full timing-unit family (`us/ns/ps/fs`,
/// the `Hz` family, spelled-out durations) because the driving corpus —
/// STA/timing logs — is ns/ps-dense, and a unit the list misses next to
/// a tool name would RULE-MASK a slack value (audit finding on this PR).
/// Anchored to the span end with optional whitespace; letter units must
/// not continue into a longer word (`12.1 subsystem` is NOT an `s` hit),
/// checked with an explicit ASCII-alnum guard rather than `\b` because
/// the regex crate's Unicode `\b` treats a following CJK char as a word
/// char.
const UNIT_SUFFIX_PATTERN: &str = r"^\s*(?:%|(?i:ms|us|ns|ps|fs|s|secs?|seconds?|mins?|minutes?|hours?|[kmgt]i?b|um|nm|[kmg]?hz)(?:[^0-9A-Za-z]|$))";

/// Negative: the span itself is IPv4-shaped (`10.2.255.1`). Shape only —
/// no octet range check, matching how DLP engines treat dotted quads.
/// Known recall tradeoff (recorded on the design issue): real EDA
/// sub-versions can run to 4 dotted groups (`IC6.1.8.500`), and this
/// class releases them even next to an adjacent tool name. Kept anyway:
/// weakening the class when positive evidence is adjacent would mask
/// ACTUAL addresses (`Virtuoso 主机 10.2.255.1`), and a mask
/// false-positive corrupts content while a release only defers recall to
/// the sample corpus.
const IPV4_PATTERN: &str = r"^\d{1,3}(?:\.\d{1,3}){3}$";

/// Negative: source-location context — the span directly follows a
/// `file.ext:` prefix or is directly followed by `:digit` (`top.v:12.1`,
/// `12.1:3`-style diagnostics).
const FILE_COLON_PREFIX_PATTERN: &str = r"[\w.-]+\.[A-Za-z0-9]+:$";
const COLON_DIGIT_SUFFIX_PATTERN: &str = r"^:\d";

/// What layer ② decided for one candidate span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RuleDecision {
    /// Decisive positive evidence: rewrite, no model call.
    Mask,
    /// Decisive negative evidence: release, no model call.
    Pass,
    /// Uncertain band: only these consult layer ③.
    Model,
}

/// The compiled layer-② scorer. Hotword vocabulary and weights are
/// compile-time constants (the MVP's env-or-hardcoded config posture);
/// only the proximity window is operator-tunable.
pub(super) struct RuleScorer {
    proximity_chars: usize,
    en_trigger: Regex,
    tool: Regex,
    unit_suffix: Regex,
    ipv4: Regex,
    file_colon_prefix: Regex,
    colon_digit_suffix: Regex,
}

impl RuleScorer {
    pub(super) fn new(proximity_chars: usize) -> Self {
        let compile = |p: &str| Regex::new(p).expect("rule pattern must compile");
        Self {
            proximity_chars: proximity_chars.min(MAX_PROXIMITY_CHARS),
            en_trigger: compile(EN_TRIGGER_PATTERN),
            tool: compile(TOOL_PATTERN),
            unit_suffix: compile(UNIT_SUFFIX_PATTERN),
            ipv4: compile(IPV4_PATTERN),
            file_colon_prefix: compile(FILE_COLON_PREFIX_PATTERN),
            colon_digit_suffix: compile(COLON_DIGIT_SUFFIX_PATTERN),
        }
    }

    /// Score one candidate span of `text`.
    pub(super) fn score(&self, text: &str, span: &Range<usize>) -> i32 {
        let window = window_bounds(text, span, self.proximity_chars);
        let mut adjacent_classes = 0i32;
        let mut window_only = false;

        // Hotword classes: matches are searched inside the proximity
        // window; a match within ADJACENT_GAP_CHARS of the span edge
        // upgrades its class to decisive.
        let mut tally = |ranges: &mut dyn Iterator<Item = Range<usize>>| {
            let mut any = false;
            let mut adjacent = false;
            for r in ranges {
                any = true;
                if gap_chars(text, span, &r) <= ADJACENT_GAP_CHARS {
                    adjacent = true;
                    break;
                }
            }
            if adjacent {
                adjacent_classes += 1;
            } else if any {
                window_only = true;
            }
        };
        let win = &text[window.clone()];
        tally(&mut ZH_TRIGGERS.iter().flat_map(|t| {
            win.match_indices(t)
                .map(|(i, m)| window.start + i..window.start + i + m.len())
        }));
        // A clipped window edge can bisect a word and hand `\b` a false
        // boundary at the slice rim (`…conversion` clipped to a slice
        // ending in `version`), so word-bounded matches flush with a
        // CLIPPED edge are discarded. Substring (zh) matching has no
        // boundary semantics, so it needs no such guard.
        let clipped_start = window.start > 0;
        let clipped_end = window.end < text.len();
        let mut bounded = |re: &Regex| {
            tally(
                &mut re
                    .find_iter(win)
                    .filter(|m| !(clipped_start && m.start() == 0))
                    .filter(|m| !(clipped_end && m.end() == win.len()))
                    .map(|m| window.start + m.start()..window.start + m.end()),
            );
        };
        bounded(&self.en_trigger);
        bounded(&self.tool);

        let mut score = adjacent_classes * ADJACENT_CLASS_SCORE;
        if adjacent_classes == 0 && window_only {
            score += WINDOW_EVIDENCE_SCORE;
        }

        // Negative classes: span-local shape checks, independent of the
        // proximity window.
        if self.unit_suffix.is_match(&text[span.end..]) {
            score += NEGATIVE_CLASS_SCORE;
        }
        if self.ipv4.is_match(&text[span.clone()]) {
            score += NEGATIVE_CLASS_SCORE;
        }
        if self.file_colon_prefix.is_match(&text[..span.start])
            || self.colon_digit_suffix.is_match(&text[span.end..])
        {
            score += NEGATIVE_CLASS_SCORE;
        }
        score
    }

    /// Apply the double threshold to [`score`](Self::score).
    pub(super) fn decide(&self, text: &str, span: &Range<usize>) -> RuleDecision {
        let score = self.score(text, span);
        if score >= MASK_SCORE {
            RuleDecision::Mask
        } else if score <= PASS_SCORE {
            RuleDecision::Pass
        } else {
            RuleDecision::Model
        }
    }
}

/// Chars between a hotword match and the candidate span (0 when they
/// touch or overlap). Both ranges are byte ranges into `text`.
fn gap_chars(text: &str, span: &Range<usize>, hotword: &Range<usize>) -> usize {
    if hotword.end <= span.start {
        text[hotword.end..span.start].chars().count()
    } else if hotword.start >= span.end {
        text[span.end..hotword.start].chars().count()
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::super::{candidate_spans, CANDIDATE_PATTERN};
    use super::*;

    fn decisions(text: &str) -> Vec<(String, RuleDecision)> {
        let scorer = RuleScorer::new(DEFAULT_PROXIMITY_CHARS);
        let re = Regex::new(CANDIDATE_PATTERN).unwrap();
        candidate_spans(&re, text)
            .into_iter()
            .map(|s| (text[s.clone()].to_owned(), scorer.decide(text, &s)))
            .collect()
    }

    fn only(text: &str) -> RuleDecision {
        let d = decisions(text);
        assert_eq!(d.len(), 1, "expected one candidate in {text:?}: {d:?}");
        d[0].1
    }

    // ── the acceptance matrix, rules alone (no model) ────────────────────

    #[test]
    fn acceptance_positive_masks_by_rules_alone() {
        assert_eq!(
            only("这个 EDA 软件的版本是 12.1,请确认兼容性"),
            RuleDecision::Mask
        );
    }

    #[test]
    fn hard_positives_mask_by_rules_alone() {
        assert_eq!(
            only("我们把仿真工具升级到 2022.4 之后速度快了很多"),
            RuleDecision::Mask
        );
        assert_eq!(only("Virtuoso IC6.1.8 出现了崩溃"), RuleDecision::Mask);
    }

    #[test]
    fn hard_negatives_pass_by_rules_alone() {
        assert_eq!(
            decisions("Elapsed: 12.345s, Memory: 4.2 GB"),
            vec![
                ("12.345".to_owned(), RuleDecision::Pass),
                ("4.2".to_owned(), RuleDecision::Pass),
            ]
        );
        assert_eq!(only("服务器的 IP 地址是 10.2.255.1"), RuleDecision::Pass);
        assert_eq!(only("工艺节点是 0.13um,良率还行"), RuleDecision::Pass);
    }

    #[test]
    fn bare_number_stays_in_the_model_band() {
        // No evidence either way — exactly what layer ③ exists for.
        assert_eq!(only("圆周率约等于 3.14159"), RuleDecision::Model);
    }

    // ── evidence weighting ───────────────────────────────────────────────

    #[test]
    fn negative_unit_outweighs_adjacent_trigger() {
        // 版本/升级 hotwords adjacent, but the timing unit is decisive.
        assert_eq!(only("版本升级后耗时 12.5s,可以接受"), RuleDecision::Pass);
    }

    #[test]
    fn window_distance_hotword_is_weak_evidence_only() {
        // 版本 is inside the 50-char window but far from the number:
        // weak evidence routes to the model instead of masking.
        assert_eq!(
            only("新版本已经发布了。另外今天集群的负载均值是 3.5"),
            RuleDecision::Model
        );
    }

    #[test]
    fn english_triggers_are_word_bounded() {
        assert_eq!(only("the conversion rate is 3.5"), RuleDecision::Model);
        assert_eq!(only("we upgraded to 21.15 yesterday"), RuleDecision::Mask);
    }

    #[test]
    fn tool_name_is_decisive_only_when_adjacent() {
        // Adjacent tool name: the design issue's "极强锚点" case.
        assert_eq!(only("PrimeTime 2022.03 跑不过时序"), RuleDecision::Mask);
        // Tool at window distance next to a non-version number: weak
        // evidence — the model decides, not the rules.
        assert_eq!(
            only("PrimeTime reported a slack of 12.5"),
            RuleDecision::Model
        );
    }

    #[test]
    fn clipped_window_edge_cannot_fabricate_a_word_boundary() {
        // With a window sized to clip `conversion` exactly at its inner
        // `version`, the slice-start `\b` would fire and rule-mask; the
        // clipped-edge guard drops the match and the candidate stays in
        // the model band.
        let text = "big conversion 3.5 result";
        let re = Regex::new(CANDIDATE_PATTERN).unwrap();
        let span = &candidate_spans(&re, text)[0];
        let tight = RuleScorer::new(8);
        assert_eq!(tight.decide(text, span), RuleDecision::Model);
    }

    #[test]
    fn source_location_context_passes() {
        assert_eq!(
            only("see top.v:12.1 for the assignment"),
            RuleDecision::Pass
        );
    }

    #[test]
    fn unit_must_not_continue_into_a_word() {
        // `s` starts `subsystem`: not a unit hit, and 版本 is adjacent →
        // decisive positive stands.
        assert_eq!(
            only("版本是 12.1 subsystem 之外的另一个话题"),
            RuleDecision::Mask
        );
    }

    #[test]
    fn timing_units_outweigh_adjacent_tool_names() {
        // The driving corpus is ns/ps-dense STA logs: a slack value next
        // to a tool name must NOT rule-mask (audit finding on this PR).
        assert_eq!(only("PrimeTime slack 0.5ns 违例"), RuleDecision::Pass);
        assert_eq!(only("版本升级后耗时 12.5ns,可以接受"), RuleDecision::Pass);
        assert_eq!(only("clock period 1.25ps setup ok"), RuleDecision::Pass);
        assert_eq!(only("跑到 3.2GHz 依然稳定"), RuleDecision::Pass);
        assert_eq!(only("内存占用 1.5 GiB 左右"), RuleDecision::Pass);
    }

    #[test]
    fn spelled_duration_units_are_negative_evidence() {
        assert_eq!(only("the build took 12.5 minutes"), RuleDecision::Pass);
        assert_eq!(only("版本是 12.1 seconds 之外的话题"), RuleDecision::Pass);
    }

    #[test]
    fn proximity_window_bounds_the_hotword_search() {
        // Same sentence, tool name pushed outside a tiny window: the
        // evidence disappears and the candidate falls to the model band.
        let text = "Innovus 的运行日志我贴在下面了,请帮忙看看统计值 21.12";
        let tight = RuleScorer::new(4);
        let wide = RuleScorer::new(DEFAULT_PROXIMITY_CHARS);
        let re = Regex::new(CANDIDATE_PATTERN).unwrap();
        let span = &candidate_spans(&re, text)[0];
        assert_eq!(tight.decide(text, span), RuleDecision::Model);
        assert_eq!(wide.decide(text, span), RuleDecision::Model);
        // At window distance the tool is weak (+1) evidence, not a mask.
        assert_eq!(wide.score(text, span), 1);
        assert_eq!(tight.score(text, span), 0);
    }

    // ── report instrument: model-call rate over the probe corpus ────────

    /// The layer-③ call-rate accounting the design issue asks for: over
    /// the MVP probe corpus (7 windows, 8 candidates), layer ② resolves
    /// 7 of 8 candidates and only the bare-number window still pays a
    /// model call — versus 8 of 8 in the ①→③ MVP.
    #[test]
    fn probe_corpus_model_call_rate() {
        let corpus = [
            "这个 EDA 软件的版本是 12.1,请确认兼容性",
            "我们把仿真工具升级到 2022.4 之后速度快了很多",
            "Virtuoso IC6.1.8 出现了崩溃",
            "Elapsed: 12.345s, Memory: 4.2 GB",
            "服务器的 IP 地址是 10.2.255.1",
            "圆周率约等于 3.14159",
            "工艺节点是 0.13um,良率还行",
        ];
        let all: Vec<RuleDecision> = corpus
            .iter()
            .flat_map(|t| decisions(t).into_iter().map(|(_, d)| d))
            .collect();
        assert_eq!(all.len(), 8);
        let to_model = all.iter().filter(|d| **d == RuleDecision::Model).count();
        assert_eq!(to_model, 1, "decisions: {all:?}");
    }
}
