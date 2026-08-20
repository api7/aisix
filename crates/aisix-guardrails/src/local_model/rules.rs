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
//! FORMATTING — appending a unit-looking suffix (`升级到 2022.4s`) or a
//! `:digit` tail. The layer scores accidental phrasing, not adversarial
//! encoding. (Fullwidth digits left this list: layer ① now candidates
//! them, because they occur in ACCIDENTAL Chinese-IME phrasing, which is
//! in scope.)
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
/// model). Word-bounded on the left, case-insensitive (`vcs` / `VCS`).
/// On the right, either a word boundary or a DIGIT continuation: real
/// corpora fuse the tool name straight into the version (`INNOVUS211`),
/// and `\b` never fires between two word chars, so the plain `\b` form
/// was blind to exactly the fused tokens layer ① now produces. The
/// digit alternative consumes one char, which only matters to the gap
/// computation by one char inside an already-overlapping span.
const TOOL_PATTERN: &str = r"(?i)\b(?:virtuoso|calibre|vcs|innovus|icc2|primetime)(?:[0-9]|\b)";

/// Negative: a measurement unit right after the span (`12.345s`,
/// `4.2 GB`, `0.13um`, `99.9%`, `0.5ns`, `0.5 纳秒`, `3.5 小时`). Beyond
/// the design brief's minimum list, this covers the full timing-unit
/// family (`us/ns/ps/fs`, the `Hz` family, spelled-out durations)
/// because the driving corpus — STA/timing logs — is ns/ps-dense, and a
/// unit the list misses next to a tool name would RULE-MASK a slack
/// value (audit finding on PR #1005). The CHINESE unit vocabulary is the
/// adversarial-corpus finding this PR fixes: the customer's dominant
/// corpus is Chinese logs, and under the double threshold a rule-mask
/// never consults the model, so `0.5ns` released while `0.5 纳秒` was
/// rewritten. Covered families: durations (纳秒→天), lengths (纳米/微米/
/// 毫米), byte sizes (兆/吉/太字节), the Hz family (千/兆/吉赫兹), and
/// percentages (`％`, `个百分点`; the `百分之` PREFIX form is
/// [`CN_PERCENT_PREFIX_PATTERN`]).
/// Anchored to the span end with optional whitespace; ASCII letter units
/// must not continue into a longer word (`12.1 subsystem` is NOT an `s`
/// hit), checked with an explicit ASCII-alnum guard rather than `\b`
/// because the regex crate's Unicode `\b` treats a following CJK char as
/// a word char. Chinese units are substring-matched (no word boundaries
/// in CJK); durations also carry the measure-word form (`3.5 个小时`,
/// `2.5 个星期` — the audit found the bare forms alone still rule-masked
/// next to a trigger), and thermal/electrical units (`度`, `伏特`,
/// `瓦特`, `安培`) cover the power/temperature lines EDA logs are full
/// of. Single-char units with heavy compound ambiguity are deliberately
/// excluded — `分` (points vs minutes) and bare `安` (`12.1 安装之后`
/// would systematically RELEASE real versions next to the extremely
/// common 安装), and `度` must not continue into `度过` (verification
/// audit: `版本 12.1 度过了回归测试` released a real version; the
/// negated-char form costs one lookahead-free char, harmless for
/// `is_match`) — and the residual compound risk of the kept ones
/// (`12.1 天线`) is accepted: a wrong release is fail-open, a wrong
/// rewrite corrupts content.
const UNIT_SUFFIX_PATTERN: &str = r"^\s*(?:%|％|(?i:ms|us|ns|ps|fs|s|secs?|seconds?|mins?|minutes?|hours?|[kmgt]i?b|um|nm|[kmg]?hz)(?:[^0-9A-Za-z]|$)|纳秒|微秒|毫秒|秒|分钟|个?(?:小时|钟头|星期|月)|天|纳米|微米|毫米|[兆吉太]字节|[千兆吉]?赫兹|[千兆吉]赫|个?百分点|摄氏度|度(?:[^过]|$)|伏特?|瓦特?|安培|毫安)";

/// Negative: the span itself ENDS in an ASCII measurement unit. Fused
/// candidate tokens (`12.345s`, `3.2GHz`, `7nm`, `0.13um` as ONE span)
/// carry the unit evidence INSIDE the span rather than after it, so the
/// suffix check above never sees it. Same class as
/// [`UNIT_SUFFIX_PATTERN`] — either placement counts once.
const FUSED_UNIT_SUFFIX_PATTERN: &str =
    r"(?i)[0-9](?:%|ms|us|ns|ps|fs|s|secs?|seconds?|mins?|minutes?|hours?|[kmgt]i?b|um|nm|[kmg]?hz)$";

/// Negative: the Chinese percentage PREFIX form (`百分之 3.5`) — the
/// percent evidence precedes the number. Same class as the unit suffix.
const CN_PERCENT_PREFIX_PATTERN: &str = r"百分之\s*$";

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

/// Negative: time-of-day context, same class as the source location —
/// the span completes an `HH:MM:SS.mmm` clock reading. Log lines open
/// with these (`[10:23:45.123] build started`), and the trigger word
/// right after the bracket (`build`, `version`) is ADJACENT by distance,
/// so without this class the whole timestamp family rule-masks — the
/// adversarial-corpus finding. ASCII and fullwidth colons both count.
const TIME_OF_DAY_PREFIX_PATTERN: &str = r"[0-9]{1,2}[:：][0-9]{1,2}[:：]$";

/// Negative: the span is FILENAME-shaped — a fused token ending in a
/// dot plus a 2–4 letter extension (`report3.txt`, `setup2.cfg`).
/// Audit finding: everyday filenames are fused-token candidates now,
/// and `GH-2048`-class identifiers sit too close to the weakest
/// anchor-free version positives for the embedding to separate — but a
/// trailing alphabetic extension is decisive LEXICAL evidence, which is
/// this layer's job, not the model's. Version tokens never end in a
/// dot-plus-letters segment (`802.11ac`'s last segment starts with
/// digits; `…-SP2` has no dot before the letters), so the shape is
/// precise. Same class as the source-location patterns.
const FILE_EXTENSION_SUFFIX_PATTERN: &str = r"(?i)\.[a-z]{2,4}$";

/// Negative: an identifier tag right before the span — `编号 GH-2048`,
/// `编号: JIRA-1024`, optionally through `是/为`. `版本编号` is carved
/// out (the char before `编号` must not be `本`): that compound means
/// "version number" and must keep masking. Same class as the source
/// location — a tagged identifier is a locator, not a version.
const ID_TAG_PREFIX_PATTERN: &str = r"(?:^|[^本])编号[::]?(?:是|为)?\s*$";

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
    fused_unit_suffix: Regex,
    cn_percent_prefix: Regex,
    ipv4: Regex,
    file_colon_prefix: Regex,
    colon_digit_suffix: Regex,
    time_of_day_prefix: Regex,
    file_extension_suffix: Regex,
    id_tag_prefix: Regex,
}

impl RuleScorer {
    pub(super) fn new(proximity_chars: usize) -> Self {
        let compile = |p: &str| Regex::new(p).expect("rule pattern must compile");
        Self {
            // The env parse already rejects zero (a zero window finds no
            // hotword — layer ② would silently stop masking), but the
            // config fields are `pub`; clamp defensively like the lane
            // pool does.
            proximity_chars: proximity_chars.clamp(1, MAX_PROXIMITY_CHARS),
            en_trigger: compile(EN_TRIGGER_PATTERN),
            tool: compile(TOOL_PATTERN),
            unit_suffix: compile(UNIT_SUFFIX_PATTERN),
            fused_unit_suffix: compile(FUSED_UNIT_SUFFIX_PATTERN),
            cn_percent_prefix: compile(CN_PERCENT_PREFIX_PATTERN),
            ipv4: compile(IPV4_PATTERN),
            file_colon_prefix: compile(FILE_COLON_PREFIX_PATTERN),
            colon_digit_suffix: compile(COLON_DIGIT_SUFFIX_PATTERN),
            time_of_day_prefix: compile(TIME_OF_DAY_PREFIX_PATTERN),
            file_extension_suffix: compile(FILE_EXTENSION_SUFFIX_PATTERN),
            id_tag_prefix: compile(ID_TAG_PREFIX_PATTERN),
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
        // proximity window. Unit evidence counts once no matter where it
        // sits (after the span, fused into its tail, or the Chinese
        // percent prefix before it).
        if self.unit_suffix.is_match(&text[span.end..])
            || self.fused_unit_suffix.is_match(&text[span.clone()])
            || self.cn_percent_prefix.is_match(&text[..span.start])
        {
            score += NEGATIVE_CLASS_SCORE;
        }
        if self.ipv4.is_match(&text[span.clone()]) {
            score += NEGATIVE_CLASS_SCORE;
        }
        if self.file_colon_prefix.is_match(&text[..span.start])
            || self.colon_digit_suffix.is_match(&text[span.end..])
            || self.time_of_day_prefix.is_match(&text[..span.start])
            || self.file_extension_suffix.is_match(&text[span.clone()])
            || self.id_tag_prefix.is_match(&text[..span.start])
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
    use super::super::CandidateFinder;
    use super::*;

    fn decisions(text: &str) -> Vec<(String, RuleDecision)> {
        let scorer = RuleScorer::new(DEFAULT_PROXIMITY_CHARS);
        CandidateFinder::new()
            .spans(text)
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
        // `12.345s` is now ONE fused candidate (unit evidence inside the
        // span); the decision is unchanged.
        assert_eq!(
            decisions("Elapsed: 12.345s, Memory: 4.2 GB"),
            vec![
                ("12.345s".to_owned(), RuleDecision::Pass),
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
        let span = &CandidateFinder::new().spans(text)[0];
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

    // ── the Chinese-unit defect class (adversarial-corpus finding):
    //    `0.5ns` released while `0.5 纳秒` rewrote. One assertion per
    //    unit family the brief names; the duration rows carry a trigger
    //    hotword nearby, so the negative must OUTWEIGH it. ─────────────

    #[test]
    fn chinese_duration_units_are_negative_evidence() {
        assert_eq!(
            only("版本升级后这条路径 slack 变成 0.5 纳秒"),
            RuleDecision::Pass
        );
        assert_eq!(only("中断响应时间 12.5 微秒"), RuleDecision::Pass);
        assert_eq!(only("升级到新内核后延迟降到 3.5 毫秒"), RuleDecision::Pass);
        assert_eq!(only("整个 build 花了 45.5 秒"), RuleDecision::Pass);
        assert_eq!(only("全量回归跑了 90.5 分钟"), RuleDecision::Pass);
        assert_eq!(only("full chip 综合要 3.5 小时"), RuleDecision::Pass);
        assert_eq!(only("数据准备还要 2.5 天"), RuleDecision::Pass);
    }

    #[test]
    fn measure_word_durations_are_negative_evidence() {
        // The measure-word (`个`) forms, each next to a trigger — the
        // audit's finding: the bare-unit list alone still rule-masked
        // these.
        assert_eq!(only("版本升级花了 3.5 个小时"), RuleDecision::Pass);
        assert_eq!(only("升级到新机器后等了 1.5 个钟头"), RuleDecision::Pass);
        assert_eq!(only("等了 2.5 个星期才排上机时"), RuleDecision::Pass);
        assert_eq!(only("整个项目走了 4.5 个月"), RuleDecision::Pass);
    }

    #[test]
    fn thermal_and_electrical_units_are_negative_evidence() {
        assert_eq!(only("升级到新驱动后功耗 5.5 瓦"), RuleDecision::Pass);
        assert_eq!(only("结温到了 85.5 度就降频"), RuleDecision::Pass);
        assert_eq!(only("外壳温度 42.5 摄氏度"), RuleDecision::Pass);
        assert_eq!(only("核心电压是 0.75 伏"), RuleDecision::Pass);
        assert_eq!(only("满载电流 1.8 安培"), RuleDecision::Pass);
        // Bare 安 is deliberately NOT a unit: 安装 right after a version
        // must not release it (安装 is everywhere in the driving corpus).
        assert_eq!(only("版本是 12.1 安装之后报错"), RuleDecision::Mask);
        // 度 must not fire inside 度过 (verification-audit regression:
        // this real version released as a "degrees" reading).
        assert_eq!(only("版本 12.1 度过了回归测试"), RuleDecision::Mask);
    }

    #[test]
    fn chinese_size_and_frequency_units_are_negative_evidence() {
        assert_eq!(only("日志文件有 128.5 兆字节"), RuleDecision::Pass);
        assert_eq!(only("内存峰值到了 4.2 吉字节"), RuleDecision::Pass);
        assert_eq!(only("波形数据一共 1.5 太字节"), RuleDecision::Pass);
        assert_eq!(only("主频跑到 3.2 吉赫兹"), RuleDecision::Pass);
        assert_eq!(only("时钟是 800.5 兆赫兹"), RuleDecision::Pass);
        assert_eq!(only("采样率 44.1 千赫兹"), RuleDecision::Pass);
        assert_eq!(only("线宽是 0.15 微米"), RuleDecision::Pass);
        assert_eq!(only("芯片边长 8.5 毫米"), RuleDecision::Pass);
    }

    #[test]
    fn percent_forms_are_negative_evidence() {
        assert_eq!(only("覆盖率提高了 2.5 个百分点"), RuleDecision::Pass);
        assert_eq!(only("性能损失了百分之 3.5"), RuleDecision::Pass);
        assert_eq!(only("良率到了 98.5％"), RuleDecision::Pass);
        assert_eq!(only("功耗降了 12.5%,别的没变"), RuleDecision::Pass);
    }

    #[test]
    fn log_timestamps_release_even_next_to_triggers() {
        // The trigger right after the bracket is ADJACENT by distance —
        // the clock context must win (adversarial-corpus finding: the
        // whole `[HH:MM:SS.mmm]` family rule-masked).
        assert_eq!(only("[10:23:45.123] build started"), RuleDecision::Pass);
        assert_eq!(
            only("[09:01:07.500] version check passed"),
            RuleDecision::Pass
        );
        assert_eq!(
            only("10:15:30.250 upgraded to the new license server"),
            RuleDecision::Pass
        );
        assert_eq!(only("构建时间戳 [23:07:01.250] 已经记录"), RuleDecision::Pass);
    }

    // ── fused-token candidates (layer-① relaxation) through the rules ──

    #[test]
    fn fused_tokens_ending_in_units_release() {
        // One fused span each; the unit evidence is INSIDE the span.
        assert_eq!(only("7nm 工艺下功耗有点高"), RuleDecision::Pass);
        assert_eq!(only("先在 28nm 上验证流程"), RuleDecision::Pass);
        assert_eq!(only("跑到 3.2GHz 依然稳定"), RuleDecision::Pass);
    }

    #[test]
    fn fused_tool_prefix_is_decisive() {
        // The tool name fused straight into the version: the digit
        // continuation in TOOL_PATTERN makes the anchor visible.
        assert_eq!(only("INNOVUS211 的时序报告在附件里"), RuleDecision::Mask);
    }

    #[test]
    fn fused_tokens_with_adjacent_triggers_mask() {
        assert_eq!(
            only("仿真器升级到 XCELIUM2309 之后就好了"),
            RuleDecision::Mask
        );
        assert_eq!(only("回退到 E-2010.12-ICC-SP2 就不崩了"), RuleDecision::Mask);
        assert_eq!(only("装的是 v16.12-s051_1 这个版本"), RuleDecision::Mask);
        assert_eq!(only("Virtuoso IC618 又崩了"), RuleDecision::Mask);
    }

    #[test]
    fn filename_shaped_tokens_release() {
        // Trailing dot-plus-letters extension is decisive lexical
        // evidence (audit finding: `report3.txt` mis-masked in the
        // model band — but it never needed the model).
        assert_eq!(only("把 report3.txt 发我一下"), RuleDecision::Pass);
        assert_eq!(only("参数都写在 sim7.cfg 里面"), RuleDecision::Pass);
    }

    #[test]
    fn id_tag_prefix_releases_tagged_identifiers() {
        assert_eq!(only("对应 issue 编号 GH-2048"), RuleDecision::Pass);
        assert_eq!(only("工单编号: AB-3072 已建好"), RuleDecision::Pass);
        // 版本编号 is carved out — it means "version number" and must
        // keep masking (the char before 编号 is 本).
        assert_eq!(only("版本编号 12.1 别外发"), RuleDecision::Mask);
    }

    #[test]
    fn bare_fused_tokens_stay_in_the_model_band() {
        // No lexical anchor either way: exactly what layer ③ exists for.
        assert_eq!(only("XCELIUM2309 的仿真结果对不上"), RuleDecision::Model);
        assert_eq!(only("这个块是 N5 工艺的"), RuleDecision::Model);
    }

    #[test]
    fn fullwidth_version_with_adjacent_trigger_masks() {
        assert_eq!(only("版本是 １２．１,不要外传"), RuleDecision::Mask);
        assert_eq!(only("工具版本号 ２０２２．４ 见内部 wiki"), RuleDecision::Mask);
    }

    #[test]
    fn proximity_window_bounds_the_hotword_search() {
        // Same sentence, tool name pushed outside a tiny window: the
        // evidence disappears and the candidate falls to the model band.
        let text = "Innovus 的运行日志我贴在下面了,请帮忙看看统计值 21.12";
        let tight = RuleScorer::new(4);
        let wide = RuleScorer::new(DEFAULT_PROXIMITY_CHARS);
        let span = &CandidateFinder::new().spans(text)[0];
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
