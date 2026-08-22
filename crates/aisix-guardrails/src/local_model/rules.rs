//! Layer ② of the three-layer pipeline (AISIX-Cloud#1331 → #1363): rule
//! scoring between the regex candidate layer (①) and the model
//! judgement (③), compiled from a category's configured hotword groups
//! and negative patterns.
//!
//! The MVP shipped ① → ③ directly, which put the whole separation burden
//! on zero-shot cosine — measured unable to split the hard positives
//! from compile-log hard negatives. This layer restores the design's
//! division of labor: candidates with decisive lexical evidence are
//! resolved here in microseconds, and ONLY the genuinely ambiguous band
//! pays a model call.
//!
//! Per candidate span:
//!   - hotword co-occurrence RAISES the score. Each configured hotword
//!     GROUP is counted once. A group term ADJACENT to the span
//!     (≤ [`ADJACENT_GAP_CHARS`] chars away — `版本是 12.1`) is decisive
//!     evidence: +2 per group. Terms merely within the proximity window
//!     are weak evidence: +1 total, regardless of group count — distant
//!     co-occurrence alone must never mask, only route to the model.
//!   - negative patterns LOWER the score: -4 per pattern. Each pattern
//!     is tried against the span itself (always), against the text
//!     BEFORE the span when the pattern is `$`-anchored, and against the
//!     text AFTER the span when it is `^`-anchored — the anchors pin the
//!     position, so `^\s*(?:ms|s)` means "a unit right after the span"
//!     and `编号\s*$` means "a tag right before it". One negative
//!     pattern (-4) outweighs one decisive positive group (+2): a
//!     measurement unit after the number is stronger evidence of "not
//!     this category" than any nearby hotword is of the opposite.
//!   - the double threshold turns the score into a [`RuleDecision`]:
//!     ≥ [`MASK_SCORE`] rewrites without consulting the model,
//!     ≤ [`PASS_SCORE`] releases without consulting the model, and only
//!     the band in between goes to layer ③.
//!
//! This is the mainstream DLP shape — hotword-proximity confidence
//! adjustment over a strong-format base pattern — not an invention of
//! this crate. The proximity window default of
//! [`DEFAULT_PROXIMITY_CHARS`] sits at the tight end of the range those
//! engines use, and matches the ±50-char window layer ③ judges, so the
//! two layers reason about the same context.
//!
//! Term matching: terms containing any non-ASCII character are matched
//! as SUBSTRINGS (no word boundaries in CJK); pure-ASCII terms compile
//! to case-insensitive word-bounded regexes whose right edge may run
//! into a digit (`INNOVUS211` — real corpora fuse a tool name straight
//! into a version, and `\b` never fires between two word chars).
//!
//! Threat-model boundary, inherited from the design issue's "只能防无意
//! 泄漏" line and inherent to score-subtraction DLP: a sender can defeat
//! the rule layer by FORMATTING. The layer scores accidental phrasing,
//! not adversarial encoding.
//!
//! Everything here is pure text work — no model, no I/O — so the layer
//! is unit-testable standalone.

use std::ops::Range;

use regex::Regex;

use super::window_bounds;

/// Default hotword proximity window (chars each side of the span).
pub(super) const DEFAULT_PROXIMITY_CHARS: usize = 50;

/// Upper clamp for the proximity window.
pub(super) const MAX_PROXIMITY_CHARS: usize = 1000;

/// A hotword within this many chars of the span edge is ADJACENT —
/// decisive rather than weak evidence. 8 chars covers the connective
/// tissue of every acceptance shape while staying too small for an
/// unrelated number to drift inside.
const ADJACENT_GAP_CHARS: usize = 8;

/// Score for each hotword group with an adjacent match.
const ADJACENT_CLASS_SCORE: i32 = 2;
/// Score when hotwords exist only at window distance (flat, not per
/// group — distant co-occurrence stays weak no matter how many groups).
const WINDOW_EVIDENCE_SCORE: i32 = 1;
/// Score for each negative pattern that hits.
const NEGATIVE_CLASS_SCORE: i32 = -4;

/// Decision bands: `score >= MASK_SCORE` masks without the model — one
/// adjacent hotword group alone is enough.
const MASK_SCORE: i32 = 2;
/// `score <= PASS_SCORE` releases without the model — one negative
/// pattern alone is enough, even against an adjacent hotword (-4 + 2).
const PASS_SCORE: i32 = -1;

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

/// One compiled hotword group: substring terms (CJK) + word-bounded
/// regexes (ASCII). The group counts once toward the score.
struct HotwordGroup {
    substrings: Vec<String>,
    word_patterns: Vec<Regex>,
}

/// One compiled negative pattern and which slices it applies to,
/// derived from its anchors (see the module doc).
struct NegativePattern {
    regex: Regex,
    try_prefix: bool,
    try_suffix: bool,
}

/// The compiled layer-② scorer for one category.
pub(super) struct RuleScorer {
    proximity_chars: usize,
    groups: Vec<HotwordGroup>,
    negatives: Vec<NegativePattern>,
}

impl RuleScorer {
    /// Compile a category's hotword groups and negative patterns.
    /// `negatives` carries `(compiled, source)` so the anchor-derived
    /// slice selection can inspect the source text. Term-compile errors
    /// return `(offending_term, error)`.
    pub(super) fn compile(
        proximity_chars: usize,
        groups: &[Vec<String>],
        negatives: Vec<(Regex, String)>,
    ) -> Result<Self, (String, regex::Error)> {
        let mut compiled_groups = Vec::with_capacity(groups.len());
        for terms in groups {
            let mut substrings = Vec::new();
            let mut word_patterns = Vec::new();
            for term in terms {
                if term.is_empty() {
                    continue;
                }
                if term.is_ascii() {
                    // Word-bounded, case-insensitive, digit-continuation
                    // on the right (fused tool names).
                    let pattern = format!(r"(?i)\b{}(?:[0-9]|\b)", regex::escape(term));
                    word_patterns.push(Regex::new(&pattern).map_err(|e| (term.clone(), e))?);
                } else {
                    substrings.push(term.clone());
                }
            }
            compiled_groups.push(HotwordGroup {
                substrings,
                word_patterns,
            });
        }
        let negatives = negatives
            .into_iter()
            .map(|(regex, source)| NegativePattern {
                try_prefix: source.ends_with('$'),
                try_suffix: source.starts_with('^'),
                regex,
            })
            .collect();
        Ok(Self {
            proximity_chars: proximity_chars.clamp(1, MAX_PROXIMITY_CHARS),
            groups: compiled_groups,
            negatives,
        })
    }

    /// Score one candidate span of `text`.
    pub(super) fn score(&self, text: &str, span: &Range<usize>) -> i32 {
        let window = window_bounds(text, span, self.proximity_chars);
        let win = &text[window.clone()];
        // A clipped window edge can bisect a word and hand `\b` a false
        // boundary at the slice rim (`…conversion` clipped to a slice
        // ending in `version`), so word-bounded matches flush with a
        // CLIPPED edge are discarded. Substring matching has no boundary
        // semantics, so it needs no such guard.
        let clipped_start = window.start > 0;
        let clipped_end = window.end < text.len();

        let mut adjacent_groups = 0i32;
        let mut window_only = false;
        for group in &self.groups {
            let mut any = false;
            let mut adjacent = false;
            'group: {
                for term in &group.substrings {
                    for (i, m) in win.match_indices(term.as_str()) {
                        any = true;
                        let r = window.start + i..window.start + i + m.len();
                        if gap_chars(text, span, &r) <= ADJACENT_GAP_CHARS {
                            adjacent = true;
                            break 'group;
                        }
                    }
                }
                for re in &group.word_patterns {
                    for m in re
                        .find_iter(win)
                        .filter(|m| !(clipped_start && m.start() == 0))
                        .filter(|m| !(clipped_end && m.end() == win.len()))
                    {
                        any = true;
                        let r = window.start + m.start()..window.start + m.end();
                        if gap_chars(text, span, &r) <= ADJACENT_GAP_CHARS {
                            adjacent = true;
                            break 'group;
                        }
                    }
                }
            }
            if adjacent {
                adjacent_groups += 1;
            } else if any {
                window_only = true;
            }
        }

        let mut score = adjacent_groups * ADJACENT_CLASS_SCORE;
        if adjacent_groups == 0 && window_only {
            score += WINDOW_EVIDENCE_SCORE;
        }

        // Negative patterns: span-local shape checks, independent of the
        // proximity window. Each pattern counts once no matter which
        // slice it hits.
        for neg in &self.negatives {
            let hit = neg.regex.is_match(&text[span.clone()])
                || (neg.try_prefix && neg.regex.is_match(&text[..span.start]))
                || (neg.try_suffix && neg.regex.is_match(&text[span.end..]));
            if hit {
                score += NEGATIVE_CLASS_SCORE;
            }
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
    use super::super::tests::fixtures::{eda_category, eda_finder, eda_scorer};
    use super::*;

    fn decisions(text: &str) -> Vec<(String, RuleDecision)> {
        let scorer = eda_scorer(DEFAULT_PROXIMITY_CHARS);
        eda_finder()
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
        // `12.345s` is ONE fused candidate (unit evidence inside the
        // span).
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
        let span = &eda_finder().spans(text)[0];
        let tight = eda_scorer(8);
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
        // to a tool name must NOT rule-mask.
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

    // ── the Chinese-unit defect class (adversarial-corpus finding) ──────

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
        // 度 must not fire inside 度过 (verification-audit regression).
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
        assert_eq!(only("[10:23:45.123] build started"), RuleDecision::Pass);
        assert_eq!(
            only("[09:01:07.500] version check passed"),
            RuleDecision::Pass
        );
        assert_eq!(
            only("10:15:30.250 upgraded to the new license server"),
            RuleDecision::Pass
        );
        assert_eq!(
            only("构建时间戳 [23:07:01.250] 已经记录"),
            RuleDecision::Pass
        );
    }

    // ── fused-token candidates through the rules ────────────────────────

    #[test]
    fn fused_tokens_ending_in_units_release() {
        assert_eq!(only("7nm 工艺下功耗有点高"), RuleDecision::Pass);
        assert_eq!(only("先在 28nm 上验证流程"), RuleDecision::Pass);
        assert_eq!(only("跑到 3.2GHz 依然稳定"), RuleDecision::Pass);
    }

    #[test]
    fn fused_tool_prefix_is_decisive() {
        // The tool name fused straight into the version: the digit
        // continuation on ASCII terms makes the anchor visible.
        assert_eq!(only("INNOVUS211 的时序报告在附件里"), RuleDecision::Mask);
    }

    #[test]
    fn fused_tokens_with_adjacent_triggers_mask() {
        assert_eq!(
            only("仿真器升级到 XCELIUM2309 之后就好了"),
            RuleDecision::Mask
        );
        assert_eq!(
            only("回退到 E-2010.12-ICC-SP2 就不崩了"),
            RuleDecision::Mask
        );
        assert_eq!(only("装的是 v16.12-s051_1 这个版本"), RuleDecision::Mask);
        assert_eq!(only("Virtuoso IC618 又崩了"), RuleDecision::Mask);
    }

    #[test]
    fn filename_shaped_tokens_release() {
        assert_eq!(only("把 report3.txt 发我一下"), RuleDecision::Pass);
        assert_eq!(only("参数都写在 sim7.cfg 里面"), RuleDecision::Pass);
        // Decisive even against an ADJACENT trigger: the extension must
        // outweigh 版本.
        assert_eq!(only("版本是 build_2023.log 里说的那个"), RuleDecision::Pass);
    }

    #[test]
    fn id_tag_prefix_releases_tagged_identifiers() {
        assert_eq!(only("对应 issue 编号 GH-2048"), RuleDecision::Pass);
        assert_eq!(only("工单编号: AB-3072 已建好"), RuleDecision::Pass);
        assert_eq!(only("工单编号：AB-3072 已建好"), RuleDecision::Pass);
        // 版本编号 is carved out — it means "version number" and must
        // keep masking (the char before 编号 is 本).
        assert_eq!(only("版本编号 12.1 别外发"), RuleDecision::Mask);
    }

    #[test]
    fn bare_fused_tokens_stay_in_the_model_band() {
        assert_eq!(only("XCELIUM2309 的仿真结果对不上"), RuleDecision::Model);
        assert_eq!(only("这个块是 N5 工艺的"), RuleDecision::Model);
    }

    #[test]
    fn fullwidth_version_with_adjacent_trigger_masks() {
        assert_eq!(only("版本是 １２．１,不要外传"), RuleDecision::Mask);
        assert_eq!(
            only("工具版本号 ２０２２．４ 见内部 wiki"),
            RuleDecision::Mask
        );
    }

    #[test]
    fn proximity_window_bounds_the_hotword_search() {
        // Same sentence, tool name pushed outside a tiny window: the
        // evidence disappears and the candidate falls to the model band.
        let text = "Innovus 的运行日志我贴在下面了,请帮忙看看统计值 21.12";
        let tight = eda_scorer(4);
        let wide = eda_scorer(DEFAULT_PROXIMITY_CHARS);
        let span = &eda_finder().spans(text)[0];
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

    /// The fixture's decisions are pinned above; this pins that the
    /// fixture itself round-trips through the public config type — the
    /// factory template ships these exact values.
    #[test]
    fn eda_fixture_is_valid_config() {
        let cat = eda_category();
        assert!(!cat.candidate_patterns.is_empty());
        assert!(cat.candidate_patterns.len() <= 10);
        assert!(cat.negative_patterns.len() <= 20);
        assert!(cat.hotword_groups.len() <= 10);
    }
}
