//! Unit + model-backed tests for the semantic guardrail runtime.
//!
//! Everything above the `model-backed` marker runs modelless: the rule
//! layers are pure text work, and [`SemanticRuntime::for_tests_without_model`]
//! exercises the degrade arms (engine unavailable → model-band spans
//! release). The `#[ignore]` tests need the real bundle:
//!   GUARDRAIL_LOCAL_MODEL_DIR=~/.cache/aisix-local-guardrail-mvp \
//!     cargo test -p aisix-guardrails --features local-model -- --ignored
//! (`fixtures::ensure_manifest` writes a `manifest.json` next to the
//! model files on first use, hashing whatever is there.)

use super::*;

/// The factory EDA-version template — the exact category the control
/// plane ships as its "create from template" preset, kept here as the
/// fixture every rule/corpus test pins its decisions against. Values
/// are the MVP's compile-time constants, expressed as config.
pub(crate) mod fixtures {
    use super::super::rules::RuleScorer;
    use super::*;

    pub(crate) const EDA_DOTTED: &str = r"[0-9０-９]+(?:[.．][0-9０-９]+)+";
    /// Fused version token, letter-before-digit (`IC618`, `T-2022.03`).
    /// Alnum rims by construction, so sentence punctuation never sticks.
    pub(crate) const EDA_FUSED_LD: &str =
        r"[A-Za-z][A-Za-z0-9._-]*[0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?";
    /// Fused version token, digit-before-letter (`7nm`, `20.09-s003`).
    pub(crate) const EDA_FUSED_DL: &str =
        r"[0-9][A-Za-z0-9._-]*[A-Za-z](?:[A-Za-z0-9._-]*[A-Za-z0-9])?";

    pub(crate) fn eda_category() -> SemanticCategory {
        SemanticCategory {
            name: "eda_version".into(),
            description: "EDA 软件的版本号".into(),
            candidate_patterns: vec![
                EDA_DOTTED.into(),
                EDA_FUSED_LD.into(),
                EDA_FUSED_DL.into(),
            ],
            negative_patterns: vec![
                // Measurement unit right after the span (`12.345 s`,
                // `4.2 GB`, `0.5 纳秒`, `3.5 个小时`).
                r"^\s*(?:%|％|(?i:ms|us|ns|ps|fs|s|secs?|seconds?|mins?|minutes?|hours?|[kmgt]i?b|um|nm|[kmg]?hz)(?:[^0-9A-Za-z]|$)|纳秒|微秒|毫秒|秒|分钟|个?(?:小时|钟头|星期|月)|天|纳米|微米|毫米|[兆吉太]字节|[千兆吉]?赫兹|[千兆吉]赫|个?百分点|摄氏度|度(?:[^过]|$)|伏特?|瓦特?|安培|毫安)".into(),
                // Unit fused into the span's tail (`12.345s`, `3.2GHz`).
                r"(?i)[0-9](?:%|ms|us|ns|ps|fs|s|secs?|seconds?|mins?|minutes?|hours?|[kmgt]i?b|um|nm|[kmg]?hz)$".into(),
                // Chinese percent PREFIX form (`百分之 3.5`).
                r"百分之\s*$".into(),
                // IPv4-shaped span.
                r"^\d{1,3}(?:\.\d{1,3}){3}$".into(),
                // Source location: `file.ext:` before, `:digit` after.
                r"[\w.-]+\.[A-Za-z0-9]+:$".into(),
                r"^:\d".into(),
                // Clock reading prefix (`[10:23:45.123]`).
                r"[0-9]{1,2}[:：][0-9]{1,2}[:：]$".into(),
                // Filename-shaped span (dot + 2-4 letter extension).
                r"(?i)\.[a-z]{2,4}$".into(),
                // Identifier tag before the span (`编号 GH-2048`);
                // 版本编号 is carved out and keeps masking.
                r"(?:^|[^本])编号[:：]?(?:是|为)?\s*$".into(),
            ],
            hotword_groups: vec![
                aisix_core::models::SemanticHotwordGroup {
                    terms: vec![
                        "版本号".into(),
                        "版本".into(),
                        "升级到".into(),
                        "回退到".into(),
                    ],
                },
                aisix_core::models::SemanticHotwordGroup {
                    terms: vec![
                        "version".into(),
                        "release".into(),
                        "build".into(),
                        "upgrade to".into(),
                        "upgraded to".into(),
                    ],
                },
                aisix_core::models::SemanticHotwordGroup {
                    terms: vec![
                        "virtuoso".into(),
                        "calibre".into(),
                        "vcs".into(),
                        "innovus".into(),
                        "icc2".into(),
                        "primetime".into(),
                    ],
                },
            ],
            action: Some("mask".into()),
            replacement: Some("***".into()),
            threshold: None,
        }
    }

    pub(crate) fn eda_finder() -> CandidateFinder {
        let cat = eda_category();
        CandidateFinder::new(
            cat.candidate_patterns
                .iter()
                .map(|p| Regex::new(p).expect("fixture pattern compiles"))
                .collect(),
        )
    }

    pub(crate) fn eda_scorer(proximity_chars: usize) -> RuleScorer {
        let cat = eda_category();
        let groups: Vec<Vec<String>> = cat.hotword_groups.iter().map(|g| g.terms.clone()).collect();
        let negatives = cat
            .negative_patterns
            .iter()
            .map(|p| (Regex::new(p).expect("fixture pattern compiles"), p.clone()))
            .collect();
        RuleScorer::compile(proximity_chars, &groups, negatives).expect("fixture terms compile")
    }

    /// Write a `manifest.json` next to the model files when absent,
    /// hashing whatever the directory holds — the test-side counterpart
    /// of the release bundle's pinned manifest.
    pub(crate) fn ensure_manifest(dir: &Path) {
        let path = dir.join("manifest.json");
        if path.exists() {
            return;
        }
        let mut files = serde_json::Map::new();
        for name in ["model.onnx", "tokenizer.json"] {
            let bytes = std::fs::read(dir.join(name)).expect("model file present");
            files.insert(
                name.to_owned(),
                serde_json::Value::String(format!("sha256:{}", hex_digest(&bytes))),
            );
        }
        let manifest = serde_json::json!({
            "manifest_version": 1,
            "model_id": "ibm-granite/granite-embedding-97m-multilingual-r2",
            "embedding_dim": 384,
            "files": files,
            "calibration": { "description": 0.80 },
        });
        std::fs::write(&path, serde_json::to_vec_pretty(&manifest).unwrap())
            .expect("write test manifest");
    }

    /// The real-model runtime, or `None` when [`MODEL_DIR_ENV`] is
    /// unset (the `#[ignore]` tests no-op then).
    pub(crate) fn model_runtime(lanes: usize) -> Option<SemanticRuntime> {
        let dir = PathBuf::from(std::env::var_os(MODEL_DIR_ENV)?);
        ensure_manifest(&dir);
        Some(SemanticRuntime::load(dir, lanes, None).expect("model bundle verifies"))
    }

    pub(crate) fn model_guardrail() -> Option<SemanticGuardrail> {
        let rt = Arc::new(model_runtime(1)?);
        Some(
            SemanticGuardrail::from_config(rt, &[eda_category()], GuardrailHookPoint::Both)
                .expect("fixture compiles"),
        )
    }
}

use fixtures::*;

fn spans_of(text: &str) -> Vec<Range<usize>> {
    eda_finder().spans(text)
}

fn values_of(text: &str) -> Vec<&str> {
    spans_of(text).into_iter().map(|s| &text[s]).collect()
}

fn modelless_guardrail() -> SemanticGuardrail {
    SemanticGuardrail::from_config(
        Arc::new(SemanticRuntime::for_tests_without_model()),
        &[eda_category()],
        GuardrailHookPoint::Both,
    )
    .expect("fixture compiles")
}

#[test]
fn candidate_spans_find_dotted_runs_only() {
    let text = "版本是 12.1,构建号 2022.4.1,端口 8080";
    // Plain integers are still not candidates.
    assert_eq!(values_of(text), vec!["12.1", "2022.4.1"]);
}

#[test]
fn candidate_spans_find_fused_tokens_whole() {
    // The adversarial-corpus shapes: version fused with the tool
    // name / letter affixes into ONE token — the candidate is the
    // whole token, not its dotted substring.
    for (text, want) in [
        ("Virtuoso IC618 又崩了", "IC618"),
        ("版图工具用的是 ICADV12.3", "ICADV12.3"),
        ("XCELIUM2309 的仿真结果对不上", "XCELIUM2309"),
        ("MMSIM151 装在新机器上了", "MMSIM151"),
        ("综合用的 T-2022.03 有已知问题", "T-2022.03"),
        ("回退到 E-2010.12-ICC-SP2 就不崩了", "E-2010.12-ICC-SP2"),
        ("装的是 v16.12-s051_1 这个版本", "v16.12-s051_1"),
        ("hotfix 20.09-s003 已经推送了", "20.09-s003"),
        ("7nm 工艺下功耗有点高", "7nm"),
        ("这个块是 N5 工艺的", "N5"),
    ] {
        assert_eq!(values_of(text), vec![want], "text: {text}");
    }
}

#[test]
fn candidate_spans_find_fullwidth_dotted_runs() {
    assert_eq!(values_of("版本是 １２．１,不要外传"), vec!["１２．１"]);
    // Mixed-width digits with a fullwidth dot still form one run.
    assert_eq!(values_of("旧版是 ２０２２．4"), vec!["２０２２．4"]);
}

#[test]
fn fused_tokens_exclude_rim_punctuation() {
    // Sentence punctuation must not stick to a fused token — the
    // template patterns require alphanumeric rims by construction.
    assert_eq!(values_of("pinned to v16.12-s051_1."), vec!["v16.12-s051_1"]);
    // A pure word and a pure dash-number never become fused tokens.
    assert_eq!(values_of("high-performance run -3.5 offset"), vec!["3.5"]);
}

#[test]
fn candidate_dedupe_keeps_standalone_dotted_runs_between_fused_tokens() {
    // Interleaved fused and dotted candidates: the de-overlap must drop
    // exactly the dotted runs inside fused tokens and keep the
    // standalone ones.
    let text = "v1.2-a 3.4 IC5.6 7.8 soc9.9x 10.11";
    assert_eq!(
        values_of(text),
        vec!["v1.2-a", "3.4", "IC5.6", "7.8", "soc9.9x", "10.11"]
    );
}

#[test]
fn candidate_generation_stays_linear_on_floods() {
    // The audit measured a pre-fix quadratic dedupe at 17.6 s of
    // synchronous CPU for a 1 MiB `"1.1 "` flood. Linear generation
    // does this in tens of milliseconds; the bound leaves two orders
    // of magnitude of CI headroom.
    let flood = "1.1 ".repeat(256 * 1024); // 1 MiB
    let started = Instant::now();
    let spans = eda_finder().spans(&flood);
    assert_eq!(spans.len(), 256 * 1024);
    assert!(
        started.elapsed() < std::time::Duration::from_secs(10),
        "candidate generation took {:?} on a 1 MiB flood",
        started.elapsed()
    );
}

#[test]
fn candidate_spans_drop_oversized_runs() {
    // A crafted `1.1.1...` run over the byte cap is one regex match
    // but NOT a candidate — it must vanish before budget accounting.
    let bomb = "1.1".repeat(60); // 180 bytes, single match
    let text = format!("前缀 {bomb} 中缀 12.1 后缀");
    assert_eq!(values_of(&text), vec!["12.1"]);
}

#[test]
fn window_bounds_snap_to_char_boundaries() {
    let text = "这个 EDA 软件的版本是 12.1,请勿外传";
    let span = spans_of(text).remove(0);
    // A tiny context still lands on char boundaries around CJK.
    let w = window_bounds(text, &span, 3);
    let window = &text[w];
    assert!(window.contains("12.1"), "window: {window}");
    assert_eq!(window, "本是 12.1,请勿");
}

#[test]
fn window_bounds_clamp_to_text_edges() {
    let text = "12.1 只有后文";
    let span = spans_of(text).remove(0);
    let w = window_bounds(text, &span, 50);
    assert_eq!(&text[w], text);
}

#[test]
fn apply_masks_rewrites_right_to_left() {
    let text = "从 12.1 升到 13.0 了";
    let spans = spans_of(text);
    assert_eq!(apply_masks(text, &spans, "***"), "从 *** 升到 *** 了");
}

#[test]
fn default_mask_token_folds_the_name() {
    assert_eq!(default_mask_token("eda_version"), "[EDA_VERSION_REDACTED]");
    assert_eq!(default_mask_token("身份证"), "[____REDACTED]");
}

#[test]
fn parse_lanes_defaults_and_clamps() {
    assert_eq!(parse_lanes(None), 1);
    assert_eq!(parse_lanes(Some("")), 1);
    assert_eq!(parse_lanes(Some("abc")), 1);
    assert_eq!(parse_lanes(Some("-2")), 1);
    assert_eq!(parse_lanes(Some("0")), 1);
    assert_eq!(parse_lanes(Some("4")), 4);
    assert_eq!(parse_lanes(Some("9999")), MAX_LANES);
}

// ── category compilation ─────────────────────────────────────────────────

#[test]
fn compile_rejects_duplicate_category_names() {
    let cats = vec![eda_category(), eda_category()];
    assert!(matches!(
        compile_categories(&cats, 0.8),
        Err(CategoryCompileError::DuplicateName { .. })
    ));
}

#[test]
fn compile_rejects_non_mask_actions() {
    let mut cat = eda_category();
    cat.action = Some("block".into());
    assert!(matches!(
        compile_categories(&[cat], 0.8),
        Err(CategoryCompileError::UnsupportedAction { .. })
    ));
}

#[test]
fn compile_rejects_invalid_regex_and_empty_candidates() {
    let mut cat = eda_category();
    cat.candidate_patterns = vec!["(".into()];
    assert!(matches!(
        compile_categories(&[cat], 0.8),
        Err(CategoryCompileError::InvalidRegex { .. })
    ));
    let mut cat = eda_category();
    cat.candidate_patterns.clear();
    assert!(matches!(
        compile_categories(&[cat], 0.8),
        Err(CategoryCompileError::NoCandidatePatterns { .. })
    ));
}

#[test]
fn compile_resolves_threshold_and_replacement_defaults() {
    let mut cat = eda_category();
    cat.replacement = None;
    cat.threshold = None;
    let compiled = compile_categories(&[cat], 0.75).unwrap();
    assert_eq!(compiled[0].threshold, 0.75);
    assert_eq!(compiled[0].replacement, "[EDA_VERSION_REDACTED]");
    let mut cat = eda_category();
    cat.threshold = Some(0.9);
    let compiled = compile_categories(&[cat], 0.75).unwrap();
    assert_eq!(compiled[0].threshold, 0.9);
}

// ── manifest verification ────────────────────────────────────────────────

fn write_bundle(dir: &Path, corrupt: bool) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("model.onnx"), b"fake model bytes").unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    let model_hash = if corrupt {
        "0".repeat(64)
    } else {
        hex_digest(b"fake model bytes")
    };
    let manifest = serde_json::json!({
        "manifest_version": 1,
        "model_id": "test/fake",
        "embedding_dim": 4,
        "files": {
            "model.onnx": format!("sha256:{model_hash}"),
            "tokenizer.json": format!("sha256:{}", hex_digest(b"{}")),
        },
        "calibration": { "description": 0.80 },
    });
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
}

#[test]
fn manifest_verifies_and_rejects_corruption() {
    let dir = std::env::temp_dir().join(format!(
        "aisix-semantic-manifest-test-{}",
        std::process::id()
    ));
    let good = dir.join("good");
    write_bundle(&good, false);
    let m = ModelManifest::load_and_verify(&good).expect("good bundle verifies");
    assert_eq!(m.model_id, "test/fake");
    assert_eq!(m.embedding_dim, 4);
    assert!((m.calibration.description - 0.80).abs() < f32::EPSILON);

    let bad = dir.join("bad");
    write_bundle(&bad, true);
    let err = ModelManifest::load_and_verify(&bad).unwrap_err();
    assert!(
        err.to_string().contains("sha256 mismatch"),
        "unexpected error: {err}"
    );

    let missing = dir.join("missing");
    assert!(ModelManifest::load_and_verify(&missing).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn manifest_rejects_traversal_file_names() {
    let dir = std::env::temp_dir().join(format!(
        "aisix-semantic-manifest-traversal-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let manifest = serde_json::json!({
        "manifest_version": 1,
        "model_id": "test/fake",
        "embedding_dim": 4,
        "files": {
            "model.onnx": "sha256:00",
            "tokenizer.json": "sha256:00",
            "../escape": "sha256:00",
        },
        "calibration": { "description": 0.80 },
    });
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let err = ModelManifest::load_and_verify(&dir).unwrap_err();
    assert!(
        err.to_string().contains("invalid file name"),
        "unexpected error: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── modelless guardrail behavior (rule layers + degrade arms) ────────────

#[tokio::test]
async fn rule_layer_masks_without_a_model() {
    let g = modelless_guardrail();
    let outcome = g
        .moderate_input_segments(&["这个 EDA 软件的版本是 12.1".to_owned()])
        .await;
    assert_eq!(outcome.verdict, GuardrailVerdict::Allow);
    let masked = outcome.masked.expect("rule layer masks");
    assert_eq!(masked[0], "这个 EDA 软件的版本是 ***");
    assert_eq!(outcome.counts.get("eda_version"), Some(&1));
}

#[tokio::test]
async fn model_band_releases_when_engine_unavailable() {
    // No lexical evidence → model band; the test runtime's engine never
    // initializes, so the span degrades to release (fail-open) instead
    // of blocking or stalling.
    let g = modelless_guardrail();
    let outcome = g
        .moderate_input_segments(&["圆周率约等于 3.14159".to_owned()])
        .await;
    assert_eq!(outcome.verdict, GuardrailVerdict::Allow);
    assert!(outcome.masked.is_none());
    assert!(outcome.counts.is_empty());
}

#[tokio::test]
async fn hook_point_gates_the_segment_hooks() {
    let g = SemanticGuardrail::from_config(
        Arc::new(SemanticRuntime::for_tests_without_model()),
        &[eda_category()],
        GuardrailHookPoint::Input,
    )
    .unwrap();
    let texts = vec!["版本是 12.1".to_owned()];
    assert!(g.moderate_input_segments(&texts).await.masked.is_some());
    assert!(g.moderate_output_segments(&texts).await.masked.is_none());
    assert!(!g.runs_on_output());
}

#[tokio::test]
async fn categories_compose_and_count_separately() {
    let mut second = eda_category();
    second.name = "ticket_id".into();
    second.description = "工单编号".into();
    second.candidate_patterns = vec![r"JIRA-\d+".into()];
    second.negative_patterns = vec![];
    second.hotword_groups = vec![aisix_core::models::SemanticHotwordGroup {
        terms: vec!["工单".into()],
    }];
    second.replacement = None; // default token
    let g = SemanticGuardrail::from_config(
        Arc::new(SemanticRuntime::for_tests_without_model()),
        &[eda_category(), second],
        GuardrailHookPoint::Both,
    )
    .unwrap();
    let outcome = g
        .moderate_input_segments(&["版本是 12.1,工单 JIRA-1024".to_owned()])
        .await;
    let masked = outcome.masked.expect("both categories mask");
    assert_eq!(masked[0], "版本是 ***,工单 [TICKET_ID_REDACTED]");
    assert_eq!(outcome.counts.get("eda_version"), Some(&1));
    assert_eq!(outcome.counts.get("ticket_id"), Some(&1));
}

// ── model-backed tests (need the real model files) ──────────────────────

fn load_model_guardrail() -> Option<SemanticGuardrail> {
    fixtures::model_guardrail()
}

/// A candidate flood past [`MAX_RULE_SCORED_SPANS_PER_SEGMENT`]
/// releases the tail unscored: a rule-maskable sentence hidden
/// beyond the cap stays untouched (fail-open — rewrite less, never
/// stall), and the flood burns at most the model-call budget.
#[tokio::test]
#[ignore = "needs GUARDRAIL_LOCAL_MODEL_DIR with the model bundle"]
async fn candidate_flood_releases_the_tail() {
    let Some(g) = load_model_guardrail() else {
        return;
    };
    let flood = "1.1 ".repeat(MAX_RULE_SCORED_SPANS_PER_SEGMENT);
    // Padding wider than the ±50-char context window between the
    // flood and the bait sentence, so the test measures the cap, not
    // window contamination. The padding word is letters-only — not a
    // candidate.
    let text = format!("{flood}{} 这个 EDA 软件的版本是 12.1", "x".repeat(60));
    let outcome = g.moderate_input_segments(&[text]).await;
    assert_eq!(outcome.verdict, GuardrailVerdict::Allow);
    assert!(
        outcome.masked.is_none(),
        "the over-cap tail must be released unscored"
    );
}

/// The acceptance matrix end to end through the segment hook, on the
/// default configuration: both hard positives and the MVP acceptance
/// positive are rewritten; every hard negative comes back untouched
/// (`masked == None` ⇒ byte-identical passthrough upstream).
#[tokio::test]
#[ignore = "needs GUARDRAIL_LOCAL_MODEL_DIR with the model bundle"]
async fn acceptance_matrix_end_to_end() {
    let Some(g) = load_model_guardrail() else {
        return;
    };
    let masked_cases = [
        ("这个 EDA 软件的版本是 12.1", "这个 EDA 软件的版本是 ***"),
        (
            "我们把仿真工具升级到 2022.4 之后速度快了很多",
            "我们把仿真工具升级到 *** 之后速度快了很多",
        ),
        // Whole-token rewrite: the fused `IC6.1.8` is ONE candidate,
        // so the tool-fused prefix does not survive.
        ("Virtuoso IC6.1.8 出现了崩溃", "Virtuoso *** 出现了崩溃"),
        ("Virtuoso IC618 又崩了", "Virtuoso *** 又崩了"),
        ("版本是 １２．１,不要外传", "版本是 ***,不要外传"),
    ];
    for (input, want) in masked_cases {
        let outcome = g.moderate_input_segments(&[input.to_owned()]).await;
        assert_eq!(outcome.verdict, GuardrailVerdict::Allow);
        let masked = outcome
            .masked
            .unwrap_or_else(|| panic!("must mask: {input}"));
        assert_eq!(masked[0], want);
    }
    let passthrough_cases = [
        "Elapsed: 12.345s, Memory: 4.2 GB",
        "服务器的 IP 地址是 10.2.255.1",
        "圆周率约等于 3.14159",
        "工艺节点是 0.13um,良率还行",
        "整个 build 花了 45.5 秒",
        "[10:23:45.123] build started",
    ];
    for input in passthrough_cases {
        let outcome = g.moderate_input_segments(&[input.to_owned()]).await;
        assert_eq!(outcome.verdict, GuardrailVerdict::Allow);
        assert!(
            outcome.masked.is_none(),
            "negative must pass untouched: {input} → {:?}",
            outcome.masked
        );
    }
}

/// The acceptance path in miniature: the segment hook masks the
/// version number and only the version number.
#[tokio::test]
#[ignore = "needs GUARDRAIL_LOCAL_MODEL_DIR with the model bundle"]
async fn masks_acceptance_sentence() {
    let Some(g) = load_model_guardrail() else {
        return;
    };
    let texts = vec!["这个 EDA 软件的版本是 12.1".to_owned()];
    let outcome = g.moderate_input_segments(&texts).await;
    assert_eq!(outcome.verdict, GuardrailVerdict::Allow);
    let masked = outcome.masked.expect("must mask the version number");
    assert_eq!(masked[0], "这个 EDA 软件的版本是 ***");
    assert_eq!(outcome.counts.get("eda_version"), Some(&1));
}

/// Lanes are interchangeable and safe under concurrency: with a
/// 2-lane pool, 8 concurrent embeds of the same window all succeed
/// and agree (sessions share nothing but identical weights, and a
/// single-threaded run is deterministic).
#[tokio::test]
#[ignore = "needs GUARDRAIL_LOCAL_MODEL_DIR with the model bundle"]
async fn lanes_run_concurrently_and_agree() {
    let Some(rt) = fixtures::model_runtime(2).map(Arc::new) else {
        return;
    };
    let window = "这个 EDA 软件的版本是 12.1,请确认兼容性";
    let tasks: Vec<_> = (0..8)
        .map(|_| {
            let rt = Arc::clone(&rt);
            tokio::spawn(async move { rt.embed_text(window.to_owned()).await })
        })
        .collect();
    let mut vectors = Vec::new();
    for t in tasks {
        vectors.push(t.await.unwrap().expect("concurrent embed failed"));
    }
    let reference = &vectors[0];
    for v in &vectors[1..] {
        let agreement = cosine(v, reference);
        assert!(agreement > 0.9999, "lanes disagree: cosine {agreement}");
    }
}

/// Throughput calibration: lanes come from the runtime constructor, so
/// one binary sweeps 1/2/4 lanes via GUARDRAIL_LOCAL_MODEL_LANES.
#[tokio::test]
#[ignore = "needs GUARDRAIL_LOCAL_MODEL_DIR with the model bundle"]
async fn probe_lane_throughput() {
    let lanes = parse_lanes(std::env::var(LANES_ENV).ok().as_deref());
    let Some(rt) = fixtures::model_runtime(lanes).map(Arc::new) else {
        return;
    };
    let window = "这个 EDA 软件的版本是 12.1,请确认与工艺库的兼容性之后再安排回归测试";
    for _ in 0..3 {
        rt.embed_text(window.to_owned()).await.unwrap();
    }
    // Saturating batches sized under LANE_WAIT_TIMEOUT (the runtime
    // sheds queue waits past 500ms by design, so an unbounded burst
    // would measure the shed path, not throughput).
    let total = 64usize;
    let batch = 8usize;
    let started = Instant::now();
    for _ in 0..total / batch {
        let tasks: Vec<_> = (0..batch)
            .map(|_| {
                let rt = Arc::clone(&rt);
                tokio::spawn(async move { rt.embed_text(window.to_owned()).await })
            })
            .collect();
        for t in tasks {
            t.await.unwrap().expect("embed failed");
        }
    }
    let secs = started.elapsed().as_secs_f64();
    println!(
        "lanes={lanes} total={total} wall={secs:.2}s throughput={:.1} inferences/s",
        total as f64 / secs
    );
}

/// Rough single-inference latency figure.
#[tokio::test]
#[ignore = "needs GUARDRAIL_LOCAL_MODEL_DIR with the model bundle"]
async fn probe_inference_latency() {
    let Some(rt) = fixtures::model_runtime(1) else {
        return;
    };
    let window = "这个 EDA 软件的版本是 12.1,请确认与工艺库的兼容性之后再安排回归测试".to_owned();
    for _ in 0..3 {
        rt.embed_text(window.clone()).await.unwrap();
    }
    let mut samples = Vec::new();
    for _ in 0..20 {
        let t = Instant::now();
        rt.embed_text(window.clone()).await.unwrap();
        samples.push(t.elapsed().as_micros() as u64);
    }
    samples.sort_unstable();
    println!(
        "inference us over 20 runs: p50={} min={} max={}",
        samples[samples.len() / 2],
        samples[0],
        samples[samples.len() - 1]
    );
}

/// The calibration probe matrix, kept as the report instrument for the
/// sample-based strategies (v2 material): the same 7 probe windows as
/// the MVP sweep scored against 5 single-description prototype
/// phrasings plus the two sample-set constructions, in both the
/// absolute and relative scoring forms.
#[tokio::test]
#[ignore = "needs GUARDRAIL_LOCAL_MODEL_DIR with the model bundle"]
async fn probe_similarity_matrix() {
    /// The v2 sample material (probe fixtures): positive phrasings by
    /// SHAPE and negative number-semantics families. See the design
    /// issue for the curation rules.
    const PROTOTYPE_SAMPLES: &[&str] = &[
        "布局布线工具升级到 21.15 之后跑得快多了",
        "仿真器回退到 19.03 才恢复正常",
        "这个后端工具的版本是 33.0",
        "综合工具的版本号是 2020.09,不要外传",
        "Spectre 23.1.0 在这个工艺角下会崩溃",
        "签核工具装的是 22.4 这个版本",
        "时序工具从 18.1 换到 20.2 就没再出过问题",
        "现在生产环境跑的是 16.3 那个版本的布线器",
        "形式验证工具还停在 10.6,太老了",
        "装了 31.2 之后 license 就报错",
        "版图工具的补丁版本是 QSV302",
        "提取工具升级到 QRC1921 以后内存翻倍",
        "DRC 用的签核包是 K-2019.06-SP1",
        "那台机器装的仿真器是 v14.2-p004",
        "We upgraded the place-and-route tool to 21.15",
        "The simulator crashed on release 6.2.1",
        "The sign-off tool version is 2020.09",
        "Genus 19.13 fails on this floorplan",
        "the flow needs tool build 30.4 or newer",
        "we rolled back to 17.0 after the crash",
        "they still run SPECTRE181 in production",
        "the timing box has PT-2021.06-SP3 installed",
        "our extraction flow is pinned to v19.1-s022_2",
        "the older 14.7 install still passes DRC",
    ];
    const NEGATIVE_PROTOTYPE_SAMPLES: &[&str] = &[
        "圆周率约是 3.1416",
        "自然常数 e 约等于 2.71828",
        "黄金分割比大约是 1.618",
        "根号二约等于 1.41421",
        "pi is roughly 3.1416",
        "the golden ratio is about 1.618",
        "今天美元兑人民币汇率是 7.18",
        "欧元汇率涨到 7.92",
        "股价收在 24.35",
        "年化利率是 3.65",
        "the exchange rate moved to 7.15",
        "the stock closed at 132.5 today",
        "早上量体温 36.6,正常",
        "孩子昨晚烧到 39.2",
        "空腹血糖 5.2,没问题",
        "体重降到 62.5 公斤了",
        "her temperature was 37.8 last night",
        "resting heart rate dropped to 58.5",
        "会议改到 9.28 上午十点",
        "项目截止日期是 2026.10.31",
        "10.1 假期值班表出来了",
        "发票日期写的 2025.12.05",
        "the review is scheduled for 11.20",
        "the contract was signed on 2026.4.30",
        "这批晶圆一共 8.5 万片",
        "平均每天触发 4.5 次告警",
        "样本均值 6.35,标准差 1.2",
        "库存还剩 3.5 箱",
        "we shipped 2.4 million units last year",
        "the average queue depth is 5.5",
        "排队等了 3.5 个星期",
        "面试聊了 1.5 个钟头",
        "整个流程走了 4.5 个月",
        "assembly 那步要等 2.5 个工作日",
        "it took 2.5 weeks end to end",
        "the outage lasted 3.5 days",
        "结温升到 88.5 度就限频",
        "内核电压是 0.72 伏",
        "整机功耗 5.5 瓦左右",
        "环境温度 23.5 度恒温",
        "the die temperature hit 95.5 degrees",
        "supply voltage sagged to 0.66 volts",
        "die 面积是 15.21 平方毫米",
        "键合线直径 25.4 微米",
        "这条走线长 3.6 毫米",
        "硅片厚度 0.775 毫米",
        "the package is 10.5 by 10.5 millimeters",
        "the wafer is 0.725 millimeters thick",
        "主力工艺切到 N2 了",
        "这个块还在 N16 上",
        "新项目评估 4nm 的 PDK",
        "老产品线停留在 14nm",
        "the pilot line runs N6",
        "we are qualifying the 3nm flow",
        "客户满意度打了 9.2 分",
        "评审平均分 8.65",
        "基准测试跑分 456.5",
        "良率这周爬到 91.5",
        "the benchmark scores 78.5 overall",
        "approval rating sits at 62.5",
        "详见第 2.4 节",
        "规范的 5.3.1 条款有说明",
        "图 4.2 画的是数据通路",
        "表 6.1 列出了引脚定义",
        "see section 3.4.2 of the spec",
        "chapter 12.3 covers the protocol",
        "日志停在 11:42:07.333",
        "[07:03:59.001] job finished",
        "晚上 20.45 的班车",
        "闹钟定在 6.30",
        "the cron fires at 23.55 every night",
        "the shuttle leaves at 8.15",
        "0.2 0.4 0.6 0.8 1.0 1.2",
        "数据列是 2.2 4.4 6.6 8.8",
        "表里那列全是 1.3 2.6 3.9 5.2 这种数",
        "the raw dump reads 0.5 1.5 2.5 3.5 4.5",
        "column two is 9.1 8.2 7.3 6.4",
        "坐标序列 10.5 20.5 30.5 40.5",
        "配置都在 setup2.cfg 里",
        "commit 是 f00dbabe42 那个",
        "工单编号是 JIRA-1024",
        "the log lives in run5.txt",
        "the checksum is 9f8e7d6c5b",
        "先过 802.3af 认证再说",
        "换成 gpt-5.2 再试一次",
        "这个 bug 用 llama-3.1 也能复现",
        "显卡是 RTX4090",
        "主控芯片是 BCM2712",
        "the endpoint serves claude-haiku-4.5",
        "the box ships with an RTX4080 inside",
    ];

    /// L2-renormalized mean of a set of L2-normalized vectors.
    fn centroid(vectors: &[Vec<f32>]) -> Vec<f32> {
        let dim = vectors.first().map_or(0, Vec::len);
        let mut mean = vec![0.0f32; dim];
        for v in vectors {
            for (m, x) in mean.iter_mut().zip(v) {
                *m += x;
            }
        }
        let norm = mean.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for m in &mut mean {
                *m /= norm;
            }
        }
        mean
    }

    struct PrototypeSet {
        positive: Vec<Vec<f32>>,
        negative: Vec<Vec<f32>>,
    }
    impl PrototypeSet {
        fn score(&self, v: &[f32]) -> f32 {
            let pos = max_cosine(&self.positive, v);
            if self.negative.is_empty() {
                pos
            } else {
                pos - max_cosine(&self.negative, v)
            }
        }
    }

    let Some(rt) = fixtures::model_runtime(1) else {
        return;
    };
    let phrasings = [
        "EDA 软件的版本号",
        "软件版本号",
        "芯片设计软件的版本号",
        "提到了 EDA 工具的具体版本号",
        "EDA 软件的版本信息,比如某个工具的版本是 12.1",
    ];
    let windows = [
        ("ACC", "这个 EDA 软件的版本是 12.1,请确认兼容性"),
        ("POS", "我们把仿真工具升级到 2022.4 之后速度快了很多"),
        ("POS", "Virtuoso IC6.1.8 出现了崩溃"),
        ("NEG", "Elapsed: 12.345s, Memory: 4.2 GB"),
        ("NEG", "服务器的 IP 地址是 10.2.255.1"),
        ("NEG", "圆周率约等于 3.14159"),
        ("NEG", "工艺节点是 0.13um,良率还行"),
    ];

    let mut columns: Vec<(String, PrototypeSet)> = Vec::new();
    for p in phrasings {
        let v = rt.embed_text(p.to_owned()).await.unwrap();
        columns.push((
            format!("desc:{p}"),
            PrototypeSet {
                positive: vec![v],
                negative: Vec::new(),
            },
        ));
    }
    let mut pos = Vec::new();
    for s in PROTOTYPE_SAMPLES {
        pos.push(rt.embed_text((*s).to_owned()).await.unwrap());
    }
    let mut neg = Vec::new();
    for s in NEGATIVE_PROTOTYPE_SAMPLES {
        neg.push(rt.embed_text((*s).to_owned()).await.unwrap());
    }
    columns.push((
        "samples-max".to_owned(),
        PrototypeSet {
            positive: pos.clone(),
            negative: neg.clone(),
        },
    ));
    columns.push((
        "samples-centroid".to_owned(),
        PrototypeSet {
            positive: vec![centroid(&pos)],
            negative: vec![centroid(&neg)],
        },
    ));

    for (name, set) in &columns {
        let mut margins = [(f32::INFINITY, f32::NEG_INFINITY); 2]; // abs, rel
        println!("── column: {name}");
        for (kind, text) in windows {
            let v = rt.embed_text(text.to_owned()).await.unwrap();
            let abs = max_cosine(&set.positive, &v);
            let rel = set.score(&v);
            println!("  {kind} abs {abs:.4} rel {rel:+.4}  {text}");
            for (m, s) in margins.iter_mut().zip([abs, rel]) {
                match kind {
                    "POS" => m.0 = m.0.min(s),
                    "NEG" => m.1 = m.1.max(s),
                    _ => {}
                }
            }
        }
        let [abs_m, rel_m] = margins.map(|(p, n)| p - n);
        println!("  hard margin (min POS − max NEG): abs {abs_m:+.4} rel {rel_m:+.4}");
    }
}
