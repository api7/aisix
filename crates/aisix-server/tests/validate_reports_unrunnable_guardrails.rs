//! `aisix validate` against guardrail rows that load but cannot run.
//!
//! A guardrail row is checked twice on the way to serving traffic: the
//! loader parses it, and the chain builder turns it into a runtime
//! guardrail. Only the first of those used to be reachable before deploy.
//! A row that parses but does not build — an invalid regex, an unknown
//! detector, a `kind: custom` script with a syntax error — is dropped from
//! the chain with a warn line and nothing else: the gateway serves,
//! `/status/config` stays `synced` with an empty `rejected` list, and the
//! screening the row describes never happens. A content policy that
//! silently does not exist is the worst shape this can fail in, so the
//! pre-deploy check has to reach the second step too.
//!
//! These drive the real binary because the exit code and the report on
//! stderr are the contract an operator and a CI pipeline consume.

use std::process::Command;

use tempfile::NamedTempFile;

/// Write `body` to a temp file and run `aisix validate --resources` on it.
fn validate(body: &str) -> (bool, String, String) {
    use std::io::Write;
    let mut file = NamedTempFile::new().expect("temp file");
    file.write_all(body.as_bytes()).expect("write");
    file.flush().expect("flush");

    let out = Command::new(env!("CARGO_BIN_EXE_aisix"))
        .args(["validate", "--resources"])
        .arg(file.path())
        .output()
        .expect("run aisix validate");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// A `kind: custom` row whose script is not a well-formed ES module. The
/// loader has no opinion on script contents, so this file parses.
const BROKEN_SCRIPT: &str = r#"
_format_version: "1"

guardrails:
  - name: screen-with-my-service
    kind: custom
    hook_point: input
    script: |
      export async function checkInput(ctx) { {{
        return { action: "none" };
      }
"#;

const WORKING_SCRIPT: &str = r#"
_format_version: "1"

guardrails:
  - name: screen-with-my-service
    kind: custom
    hook_point: input
    script: |
      export async function checkInput(ctx) {
        return { action: "none" };
      }
"#;

/// `kind: keyword` carrying a pattern the regex engine rejects. Same
/// class as the script error: the file is well-formed, the row is not.
const BROKEN_REGEX: &str = r#"
_format_version: "1"

guardrails:
  - name: block-secrets
    kind: keyword
    hook_point: input
    patterns:
      - kind: regex
        value: "AKIA[0-9A-Z{16}"
"#;

#[test]
fn a_custom_script_that_does_not_compile_fails_validation() {
    let (ok, _stdout, stderr) = validate(BROKEN_SCRIPT);

    assert!(
        !ok,
        "a guardrail that cannot run must not validate; stderr: {stderr}"
    );
    assert!(
        stderr.contains("screen-with-my-service"),
        "the report must name the row an operator has to fix: {stderr}"
    );
    assert!(
        stderr.contains("does not compile"),
        "the report must say why the row cannot run: {stderr}"
    );
}

#[test]
fn a_custom_script_that_compiles_validates() {
    let (ok, stdout, stderr) = validate(WORKING_SCRIPT);

    assert!(ok, "stderr: {stderr}");
    assert!(stdout.contains("OK:"), "stdout: {stdout}");
}

#[test]
fn an_invalid_keyword_regex_fails_validation() {
    let (ok, _stdout, stderr) = validate(BROKEN_REGEX);

    assert!(
        !ok,
        "the script case is one instance of a class; every row that cannot \
         build has to fail the same way; stderr: {stderr}"
    );
    assert!(
        stderr.contains("block-secrets"),
        "the report must name the row: {stderr}"
    );
}

#[test]
fn a_disabled_row_that_cannot_build_does_not_fail_validation() {
    let disabled =
        BROKEN_SCRIPT.replace("    kind: custom", "    enabled: false\n    kind: custom");
    let (ok, stdout, stderr) = validate(&disabled);

    assert!(
        ok,
        "a staged row is not screening anything either way; stderr: {stderr}"
    );
    assert!(stdout.contains("OK:"), "stdout: {stdout}");
}

/// A `kind: semantic` row is the one thing that fails to build for a
/// reason the file cannot fix: the embedding dispatcher is a runtime
/// capability, and `validate` boots nothing. Reporting it would make the
/// check cry wolf on a correct configuration, which is how a validation
/// step stops being read.
#[test]
fn a_semantic_row_does_not_fail_validation_for_the_missing_embedder() {
    let semantic = r#"
_format_version: "1"

provider_keys:
  - display_name: openai-prod
    provider: openai
    api_key: sk-not-a-real-key

models:
  - display_name: embed-small
    provider: openai
    model_name: text-embedding-3-small
    provider_key: openai-prod
    embedding:
      dimensions: 1536

guardrails:
  - name: screen-by-meaning
    kind: semantic
    hook_point: input
    embedding_model: embed-small
    deny_examples:
      - "how do I make a bomb"
"#;
    let (ok, stdout, stderr) = validate(semantic);

    assert!(ok, "stderr: {stderr}");
    assert!(stdout.contains("OK:"), "stdout: {stdout}");
}
