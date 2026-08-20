//! Report instrument for the guardrail-defect fixes: an 88-line labeled
//! adversarial corpus covering the three measured defect classes
//! (Chinese-unit negatives, prototype-scoring form, fused-token layer-①
//! recall) plus the shipped acceptance shapes as regressions.
//!
//! This module is test-only data + one `#[ignore]` model-backed test that
//! prints the PR's three required reports:
//!   1. candidate-level rule-layer stats (model-band share, rule-mask
//!      precision) and line-level end-to-end accuracy;
//!   2. the prototype-margin comparison on the candidates that reach the
//!      model band (old absolute form vs relative form when negative
//!      prototypes are present);
//!   3. candidate counts (how much layer-① relaxation widened the funnel
//!      and where the widened candidates were resolved).
//!
//! Labels are BY LINE: `sensitive` lists the exact substrings the pipeline
//! must rewrite; everything else must return byte-identical. A candidate
//! is a true positive when it overlaps an occurrence of a sensitive
//! substring, so a partial mask (the pre-fix `IC***` shape) counts as a
//! line miss but still credits the overlapping candidate.

use std::ops::Range;

use super::rules::RuleDecision;
use super::*;

struct Case {
    cat: &'static str,
    text: &'static str,
    sensitive: &'static [&'static str],
}

/// The corpus (79 lines from the defect brief + 9 from the independent
/// audit and verification rounds: everyday identifiers, measure-word
/// durations, electrical units, the 度过 compound). Tool names and numbers are deliberately disjoint
/// from [`PROTOTYPE_SAMPLES`] (and the negative sample set once it
/// exists) so model-band scores measure shape generalization, not string
/// overlap — the same discipline as the probe matrix.
///
/// One line per case, kept single-line on purpose (grep-friendly data
/// table — the `ebml.rs` tag-table precedent).
#[rustfmt::skip]
const CORPUS: &[Case] = &[
    // ── Chinese measurement units (defect 1): must all release ──────────
    Case { cat: "zh-unit", text: "时钟周期是 0.8 纳秒", sensitive: &[] },
    Case { cat: "zh-unit", text: "这条路径的建立裕量只剩 0.5 纳秒", sensitive: &[] },
    Case { cat: "zh-unit", text: "版本升级后这条路径 slack 变成 0.5 纳秒", sensitive: &[] },
    Case { cat: "zh-unit", text: "中断响应时间 12.5 微秒", sensitive: &[] },
    Case { cat: "zh-unit", text: "升级到新内核后延迟降到 3.5 毫秒", sensitive: &[] },
    Case { cat: "zh-unit", text: "整个 build 花了 45.5 秒", sensitive: &[] },
    Case { cat: "zh-unit", text: "全量回归跑了 90.5 分钟", sensitive: &[] },
    Case { cat: "zh-unit", text: "full chip 综合要 3.5 小时", sensitive: &[] },
    Case { cat: "zh-unit", text: "数据准备还要 2.5 天", sensitive: &[] },
    Case { cat: "zh-unit", text: "日志文件有 128.5 兆字节", sensitive: &[] },
    Case { cat: "zh-unit", text: "内存峰值到了 4.2 吉字节", sensitive: &[] },
    Case { cat: "zh-unit", text: "波形数据一共 1.5 太字节", sensitive: &[] },
    Case { cat: "zh-unit", text: "主频跑到 3.2 吉赫兹", sensitive: &[] },
    Case { cat: "zh-unit", text: "时钟是 800.5 兆赫兹", sensitive: &[] },
    Case { cat: "zh-unit", text: "采样率 44.1 千赫兹", sensitive: &[] },
    Case { cat: "zh-unit", text: "覆盖率提高了 2.5 个百分点", sensitive: &[] },
    Case { cat: "zh-unit", text: "性能损失了百分之 3.5", sensitive: &[] },
    Case { cat: "zh-unit", text: "线宽是 0.15 微米", sensitive: &[] },
    Case { cat: "zh-unit", text: "芯片边长 8.5 毫米", sensitive: &[] },
    Case { cat: "zh-unit", text: "功耗降了 12.5%,别的没变", sensitive: &[] },
    // Audit round: measure-word duration and electrical units, each with
    // an ADJACENT trigger — the shapes the first fix round still masked.
    Case { cat: "zh-unit", text: "版本升级花了 3.5 个小时", sensitive: &[] },
    Case { cat: "zh-unit", text: "升级到新驱动后功耗 5.5 瓦", sensitive: &[] },
    // ── log timestamps (defect 1): must all release ──────────────────────
    Case { cat: "timestamp", text: "[10:23:45.123] build started", sensitive: &[] },
    Case { cat: "timestamp", text: "[09:01:07.500] version check passed", sensitive: &[] },
    Case { cat: "timestamp", text: "日志停在 23:59:59.999 之后就没了", sensitive: &[] },
    Case { cat: "timestamp", text: "10:15:30.250 upgraded to the new license server", sensitive: &[] },
    Case { cat: "timestamp", text: "构建时间戳 [23:07:01.250] 已经记录", sensitive: &[] },
    // ── ASCII units (regression): must keep releasing ────────────────────
    Case { cat: "en-unit", text: "Elapsed: 12.345s, Memory: 4.2 GB", sensitive: &[] },
    Case { cat: "en-unit", text: "PrimeTime slack 0.5ns 违例", sensitive: &[] },
    Case { cat: "en-unit", text: "跑到 3.2GHz 依然稳定", sensitive: &[] },
    Case { cat: "en-unit", text: "工艺节点是 0.13um,良率还行", sensitive: &[] },
    Case { cat: "en-unit", text: "the build took 12.5 minutes", sensitive: &[] },
    Case { cat: "en-unit", text: "内存占用 1.5 GiB 左右", sensitive: &[] },
    // ── IPv4 / source locations (regression): must keep releasing ────────
    Case { cat: "locator", text: "服务器的 IP 地址是 10.2.255.1", sensitive: &[] },
    Case { cat: "locator", text: "see top.v:12.1 for the assignment", sensitive: &[] },
    Case { cat: "locator", text: "Virtuoso 主机 10.2.255.1 上跑的", sensitive: &[] },
    // ── fused version tokens (defect 3): must mask the WHOLE token ───────
    Case { cat: "fused-pos", text: "Virtuoso IC618 又崩了", sensitive: &["IC618"] },
    Case { cat: "fused-pos", text: "IC618 在新工艺角下不稳定", sensitive: &["IC618"] },
    Case { cat: "fused-pos", text: "版图工具用的是 ICADV12.3", sensitive: &["ICADV12.3"] },
    Case { cat: "fused-pos", text: "XCELIUM2309 的仿真结果对不上", sensitive: &["XCELIUM2309"] },
    Case { cat: "fused-pos", text: "仿真器升级到 XCELIUM2309 之后就好了", sensitive: &["XCELIUM2309"] },
    Case { cat: "fused-pos", text: "MMSIM151 装在新机器上了", sensitive: &["MMSIM151"] },
    Case { cat: "fused-pos", text: "综合用的 T-2022.03 有已知问题", sensitive: &["T-2022.03"] },
    Case { cat: "fused-pos", text: "回退到 E-2010.12-ICC-SP2 就不崩了", sensitive: &["E-2010.12-ICC-SP2"] },
    Case { cat: "fused-pos", text: "装的是 v16.12-s051_1 这个版本", sensitive: &["v16.12-s051_1"] },
    Case { cat: "fused-pos", text: "hotfix 20.09-s003 已经推送了", sensitive: &["20.09-s003"] },
    Case { cat: "fused-pos", text: "INNOVUS211 的时序报告在附件里", sensitive: &["INNOVUS211"] },
    // ── process nodes (defect 3 visibility): candidates, but released ────
    Case { cat: "node-neg", text: "7nm 工艺下功耗有点高", sensitive: &[] },
    Case { cat: "node-neg", text: "这个块是 N5 工艺的", sensitive: &[] },
    Case { cat: "node-neg", text: "N7 和 N3 都评估过了", sensitive: &[] },
    Case { cat: "node-neg", text: "先在 28nm 上验证流程", sensitive: &[] },
    // ── fullwidth digits (defect 3): must mask ────────────────────────────
    Case { cat: "fullwidth-pos", text: "版本是 １２．１,不要外传", sensitive: &["１２．１"] },
    Case { cat: "fullwidth-pos", text: "工具版本号 ２０２２．４ 见内部 wiki", sensitive: &["２０２２．４"] },
    // ── anchored positives (regression): must keep masking ───────────────
    Case { cat: "anchor-pos", text: "这个 EDA 软件的版本是 12.1", sensitive: &["12.1"] },
    Case { cat: "anchor-pos", text: "我们把仿真工具升级到 2022.4 之后速度快了很多", sensitive: &["2022.4"] },
    Case { cat: "anchor-pos", text: "Virtuoso IC6.1.8 出现了崩溃", sensitive: &["IC6.1.8"] },
    Case { cat: "anchor-pos", text: "we upgraded to 21.15 yesterday", sensitive: &["21.15"] },
    Case { cat: "anchor-pos", text: "PrimeTime 2022.03 跑不过时序", sensitive: &["2022.03"] },
    Case { cat: "anchor-pos", text: "Innovus 21.13 的这个 bug 已经确认了", sensitive: &["21.13"] },
    Case { cat: "anchor-pos", text: "版本回退到 20.11 才恢复正常", sensitive: &["20.11"] },
    // Verification-audit regression: 度 inside 度过 released this.
    Case { cat: "anchor-pos", text: "版本 12.1 度过了回归测试", sensitive: &["12.1"] },
    Case { cat: "anchor-pos", text: "build 33.1 is broken on centos", sensitive: &["33.1"] },
    // ── model-band negatives (defect 2): decimals with non-software
    //    semantics — must release, and only the model can say so ──────────
    Case { cat: "model-neg", text: "圆周率约等于 3.14159", sensitive: &[] },
    Case { cat: "model-neg", text: "今天美元汇率 7.23", sensitive: &[] },
    Case { cat: "model-neg", text: "孩子发烧到 38.5 了", sensitive: &[] },
    Case { cat: "model-neg", text: "合同日期是 2026.3.15", sensitive: &[] },
    Case { cat: "model-neg", text: "这批晶圆一共 12.5 万片", sensitive: &[] },
    Case { cat: "model-neg", text: "等了 2.5 个星期才排上机时", sensitive: &[] },
    Case { cat: "model-neg", text: "结温到了 85.5 度就降频", sensitive: &[] },
    Case { cat: "model-neg", text: "核心电压是 0.75 伏", sensitive: &[] },
    Case { cat: "model-neg", text: "客户满意度评分 9.5", sensitive: &[] },
    Case { cat: "model-neg", text: "第 3.2 节有详细说明", sensitive: &[] },
    Case { cat: "model-neg", text: "see section 4.1.2 for details", sensitive: &[] },
    Case { cat: "model-neg", text: "今天集群负载均值是 3.5", sensitive: &[] },
    // ── everyday identifiers (audit round): the fused-token relaxation
    //    makes these candidates for the FIRST time; they carry no unit
    //    and no anchor, so only the model can release them — the
    //    negative families files/hashes/tickets/standards and
    //    product/model names exist because these mis-masked without
    //    them ────────────────────────────────────────────────────────────
    Case { cat: "ident-neg", text: "把 report3.txt 发我一下", sensitive: &[] },
    Case { cat: "ident-neg", text: "模型是 gpt-4o 那个", sensitive: &[] },
    Case { cat: "ident-neg", text: "commit deadbeef123 部署上去了", sensitive: &[] },
    Case { cat: "ident-neg", text: "构建号 a1b2c3d4e5f6", sensitive: &[] },
    Case { cat: "ident-neg", text: "对应 issue 编号 GH-2048", sensitive: &[] },
    Case { cat: "ident-neg", text: "这块板子过了 802.11ac 认证", sensitive: &[] },
    // ── model-band positives: version mentions with NO lexical anchor —
    //    only the model can mask these ─────────────────────────────────────
    Case { cat: "model-pos", text: "工具从 21.10 换到 21.12 就不崩了", sensitive: &["21.10", "21.12"] },
    Case { cat: "model-pos", text: "生产机上装的是 6.1.8", sensitive: &["6.1.8"] },
    Case { cat: "model-pos", text: "他们还在用 17.4,太老了", sensitive: &["17.4"] },
    Case { cat: "model-pos", text: "换成 2018.09 之后问题就消失了", sensitive: &["2018.09"] },
    Case { cat: "model-pos", text: "新装的 2022.4 跑不了旧工程", sensitive: &["2022.4"] },
    Case { cat: "model-pos", text: "12.1 和 13.0 都测过,后者稳定些", sensitive: &["12.1", "13.0"] },
    Case { cat: "model-pos", text: "装了 34.0 那个包就好了", sensitive: &["34.0"] },
    Case { cat: "model-pos", text: "那台机器上是 15.2", sensitive: &["15.2"] },
];

/// Byte ranges of every occurrence of every sensitive substring.
fn sensitive_ranges(case: &Case) -> Vec<Range<usize>> {
    case.sensitive
        .iter()
        .flat_map(|s| case.text.match_indices(s).map(|(i, m)| i..i + m.len()))
        .collect()
}

fn overlaps(a: &Range<usize>, b: &Range<usize>) -> bool {
    a.start < b.end && b.start < a.end
}

/// The line's expected end-to-end output: every sensitive occurrence
/// rewritten, everything else byte-identical.
fn expected_output(case: &Case) -> String {
    let mut ranges = sensitive_ranges(case);
    ranges.sort_by_key(|r| r.start);
    apply_masks(case.text, &ranges)
}

/// Best 0/1 accuracy over all thresholds for `(score, is_positive)`
/// pairs, deciding `mask ⇔ score >= t`. Returns (best accuracy, one
/// threshold achieving it).
fn best_threshold_accuracy(items: &[(f32, bool)]) -> (f64, f32) {
    let mut cuts: Vec<f32> = items.iter().map(|(s, _)| *s).collect();
    // The two degenerate classifiers bound the sweep: -inf = mask
    // everything, +inf = release everything (bot-review finding: without
    // the latter, an all-negative item set understates best accuracy).
    cuts.push(f32::NEG_INFINITY);
    cuts.push(f32::INFINITY);
    let mut best = (0.0f64, f32::NEG_INFINITY);
    for cut in cuts {
        let correct = items.iter().filter(|(s, pos)| (*s >= cut) == *pos).count();
        let acc = correct as f64 / items.len() as f64;
        if acc > best.0 {
            best = (acc, cut);
        }
    }
    best
}

fn pct(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        return 0.0;
    }
    100.0 * part as f64 / whole as f64
}

/// The report instrument. Prints per-category and total stats; asserts
/// only the instrument's own consistency (replicated decisions must
/// reproduce the real pipeline output) so the SAME test measures the
/// before and after states without encoding either as a contract.
#[tokio::test]
#[ignore = "needs GUARDRAIL_LOCAL_MODEL_DIR with model.onnx + tokenizer.json"]
async fn adversarial_corpus_report() {
    let Some(cfg) = LocalModelConfig::from_env() else {
        return;
    };
    let g = LocalModelGuardrail::load(&cfg).expect("model files present but load failed");

    let mut n_candidates = 0usize;
    let (mut rule_masked, mut rule_passed, mut model_band) = (0usize, 0usize, 0usize);
    let (mut rule_masked_correct, mut rule_passed_correct) = (0usize, 0usize);
    // Model-band scores in BOTH forms: `abs` — the old positive-only max
    // cosine; `rel` — the shipped `max_pos − max_neg`. Same embeddings,
    // so the comparison isolates the scoring FORM.
    let mut model_items_abs: Vec<(f32, bool)> = Vec::new();
    let mut model_items_rel: Vec<(f32, bool)> = Vec::new();
    let (mut lines_correct, mut cand_correct) = (0usize, 0usize);
    let mut wrong_lines: Vec<(&str, String, String)> = Vec::new();

    for case in CORPUS {
        let labels = sensitive_ranges(case);
        let spans = g.finder.spans(case.text);
        let mut hits: Vec<Range<usize>> = Vec::new();
        for span in spans {
            n_candidates += 1;
            let positive = labels.iter().any(|l| overlaps(l, &span));
            let masked = match g.rules.decide(case.text, &span) {
                RuleDecision::Mask => {
                    rule_masked += 1;
                    rule_masked_correct += usize::from(positive);
                    true
                }
                RuleDecision::Pass => {
                    rule_passed += 1;
                    rule_passed_correct += usize::from(!positive);
                    false
                }
                RuleDecision::Model => {
                    model_band += 1;
                    let window =
                        case.text[window_bounds(case.text, &span, WINDOW_CONTEXT_CHARS)].to_owned();
                    let v = g.embed_window(window).await.expect("embed failed");
                    model_items_abs.push((max_cosine(&g.prototypes.positive, &v), positive));
                    let score = g.prototypes.score(&v);
                    model_items_rel.push((score, positive));
                    score >= g.threshold
                }
            };
            cand_correct += usize::from(masked == positive);
            if masked {
                hits.push(span);
            }
        }
        // Instrument consistency: the replication above must reproduce
        // the real pipeline byte for byte.
        hits.sort_by_key(|r| r.start);
        let replicated = apply_masks(case.text, &hits);
        let outcome = g.moderate_input_segments(&[case.text.to_owned()]).await;
        let actual = outcome
            .masked
            .map_or_else(|| case.text.to_owned(), |m| m[0].clone());
        assert_eq!(
            replicated, actual,
            "instrument diverged from the pipeline on: {}",
            case.text
        );

        let want = expected_output(case);
        if actual == want {
            lines_correct += 1;
        } else {
            wrong_lines.push((case.cat, case.text.to_owned(), actual));
        }
    }

    println!(
        "── corpus: {} lines, {} candidates",
        CORPUS.len(),
        n_candidates
    );
    println!(
        "rule-masked {rule_masked} (precision {:.1}%), rule-passed {rule_passed} ({} misreleased), model band {model_band} ({:.1}% of candidates)",
        pct(rule_masked_correct, rule_masked),
        rule_passed - rule_passed_correct,
        pct(model_band, n_candidates),
    );
    println!(
        "candidate-level accuracy {:.1}%  line-level accuracy {:.1}% ({}/{})",
        pct(cand_correct, n_candidates),
        pct(lines_correct, CORPUS.len()),
        lines_correct,
        CORPUS.len(),
    );
    for (cat, text, actual) in &wrong_lines {
        println!("  WRONG [{cat}] {text:?} -> {actual:?}");
    }

    let n_pos = model_items_rel.iter().filter(|(_, p)| *p).count();
    println!(
        "── model band: {} items ({} pos / {} neg)  shipped threshold {:.4}",
        model_items_rel.len(),
        n_pos,
        model_items_rel.len() - n_pos,
        g.threshold,
    );
    for (name, items) in [("abs", &model_items_abs), ("rel", &model_items_rel)] {
        let (acc, cut) = best_threshold_accuracy(items);
        let min_pos = items
            .iter()
            .filter(|(_, p)| *p)
            .map(|(s, _)| *s)
            .fold(f32::INFINITY, f32::min);
        let max_neg = items
            .iter()
            .filter(|(_, p)| !*p)
            .map(|(s, _)| *s)
            .fold(f32::NEG_INFINITY, f32::max);
        println!(
            "  form {name}: margin (min pos − max neg) {:+.4}  best-threshold accuracy {:.1}% at {:.4}",
            min_pos - max_neg,
            100.0 * acc,
            cut,
        );
    }
    for ((abs, p), (rel, _)) in model_items_abs.iter().zip(&model_items_rel) {
        println!(
            "  {} abs {abs:.4} rel {rel:+.4}",
            if *p { "POS" } else { "NEG" }
        );
    }

    // Quality floor, asserted LAST so a regression still prints the full
    // report above (audit finding: a report that only prints lets sample
    // or model drift collapse the numbers silently while CI stays
    // green). The rule layer must be EXACT on this corpus — a rule
    // decision never consults the model, so a wrong one is an
    // unconditional mis-rewrite (mask side) or a silent leak (pass
    // side). The end-to-end floor allows exactly the two disclosed
    // model-band misses.
    assert_eq!(
        rule_masked_correct, rule_masked,
        "rule-mask precision must stay 100% on the corpus"
    );
    assert_eq!(
        rule_passed_correct, rule_passed,
        "rule-pass must not release a labeled positive"
    );
    assert!(
        lines_correct >= CORPUS.len() - 2,
        "line accuracy fell below the pinned floor: {lines_correct}/{}",
        CORPUS.len()
    );
}
