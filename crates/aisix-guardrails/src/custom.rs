//! kind=custom guardrail — screening by an operator-supplied script.
//!
//! The operator writes an ES module; the gateway runs it in a sandboxed
//! engine (quickjs-ng, embedded via rquickjs) once per hook invocation. It
//! exists so a screening service that speaks its own protocol can be reached
//! without deploying a separate adapter in front of it.
//!
//! Script contract:
//!
//! ```js
//! export async function checkInput(ctx)  { /* -> verdict */ }
//! export async function checkOutput(ctx) { /* -> verdict */ }
//! ```
//!
//! `ctx` carries `{ hook, text, segments, messages, model, secrets }`. A
//! verdict is `{ action: "none" }`, `{ action: "block", reason,
//! reason_code }`, or `{ action: "mask", segments, counts }` where
//! `segments` is positionally aligned with `ctx.segments`. A hook whose
//! function the module does not export is an Allow, so one script may
//! cover one direction.
//!
//! A script can allow, block, OR rewrite. Rewriting rides the same async
//! segment pass the built-in remote redacting kinds use
//! ([`Guardrail::moderate_input_segments`]): the script receives the text
//! slots as `ctx.segments` and returns a replacement per slot. On a call
//! site that cannot substitute text back, a `mask` decision becomes a
//! Block rather than an Allow — releasing the original would honor half
//! the policy, which is the bug class #963 names. Same posture as
//! kind=presidio.
//!
//! Scripts also get the primitives the built-in kinds need in order to
//! reach their own providers, so the kind can express what they express:
//! `crypto` (HMAC-SHA1/SHA256 with chainable key encodings, SHA-1/SHA-256,
//! base64, UUID) for signed provider protocols, and `aisix.embed` for
//! semantic screening against the environment's own embedding model
//! through the gateway's provider bridge.
//!
//! **Two independent budgets, because neither covers the other.** The
//! engine's interrupt handler is called while bytecode executes, so it stops
//! a runaway loop but never fires on a script parked on a pending `fetch`.
//! An outer wall-clock timeout catches that case but cannot interrupt a
//! tight loop that never yields. `timeout_ms` arms both.
//!
//! The script is parsed once at chain-build time, WITHOUT being evaluated
//! ([`validate`]) — so a syntax error is reported through
//! `rejected_resources` when the row lands, not on the first request that
//! hits it, and no operator code runs on the config-apply path.
//!
//! Each invocation gets a brand-new runtime and context (~215µs), so no
//! state survives between requests and the memory ceiling applies per call.
//! The module is re-parsed inside that fresh context rather than reloaded
//! from cached bytecode: this crate is `forbid(unsafe_code)` and
//! `Module::load` is an `unsafe` call, which is not a trade worth making
//! for a parse that costs a fraction of the sandbox it runs in.
//!
//! Behavior matrix. The effective `fail_open` is the outer
//! `Guardrail::fail_open` on the INPUT hook and the independent
//! `CustomConfig::output_fail_open` (default fail-closed) on the OUTPUT hook:
//!
//! | Outcome                                | `fail_open` | Verdict                            |
//! |----------------------------------------|-------------|------------------------------------|
//! | returns `{action:"none"}`              | n/a         | Allow                              |
//! | hook function not exported             | n/a         | Allow                              |
//! | returns `{action:"block"}`             | n/a         | Block { reason }                   |
//! | returns `{action:"mask"}`              | n/a         | Allow with rewritten segments      |
//! | `mask` where write-back is impossible  | n/a         | Block (never a silent Allow)       |
//! | `mask` with a mismatched slot count    | true        | Bypass { "custom_bad_verdict" }    |
//! | wall-clock budget elapsed              | true        | Bypass { "custom_timeout" }        |
//! | script threw                           | true        | Bypass { "custom_script_error" }   |
//! | returned a shape that is not a verdict | true        | Bypass { "custom_bad_verdict" }    |
//! | engine could not start                 | true        | Bypass { "custom_engine_error" }   |
//! | any failure                            | false       | Block { "custom script unavailable …" } |

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use aisix_core::models::{CustomConfig, GuardrailHookPoint};
use aisix_gateway::{ChatFormat, ChatResponse};
use async_trait::async_trait;
use rquickjs::{CatchResultExt, Module};

use crate::{Guardrail, GuardrailVerdict, SegmentsOutcome, StreamOutputPolicy};

/// Module name the script is compiled under. Shows up in the JS stack trace
/// a thrown error carries, so keep it recognisable to the operator.
const MODULE_NAME: &str = "guardrail.js";

/// Engine stack ceiling. Independent of `max_memory_bytes`: deep recursion
/// exhausts the native stack long before the JS heap, and the default would
/// let it reach the host's guard page.
const MAX_STACK_BYTES: usize = 512 * 1024;

/// Host surface installed into every fresh context before the operator's
/// module is loaded. `__hostFetch` and `__hostLog` are the only two Rust
/// entry points; everything an operator writes against is defined here, in
/// the familiar shape, so ordinary `fetch`/`console` code works verbatim.
///
/// `text()` and `json()` return plain values rather than promises — the body
/// has already been read by the time the response object exists, and
/// `await` on a non-thenable is a no-op, so `await resp.json()` still reads
/// naturally.
const PRELUDE: &str = r#"
globalThis.fetch = async function (url, init) {
  const raw = await __hostFetch(String(url), init === undefined || init === null ? null : JSON.stringify(init));
  const r = JSON.parse(raw);
  if (r.error) { throw new Error(r.error); }
  return {
    status: r.status,
    ok: r.ok,
    headers: r.headers,
    text: function () { return r.body; },
    json: function () { return JSON.parse(r.body); },
  };
};
const __fmt = function (args) {
  return Array.prototype.map.call(args, function (a) {
    if (typeof a === "string") { return a; }
    try { return JSON.stringify(a); } catch (e) { return String(a); }
  }).join(" ");
};
globalThis.console = {
  log:   function () { __hostLog("info",  __fmt(arguments)); },
  info:  function () { __hostLog("info",  __fmt(arguments)); },
  warn:  function () { __hostLog("warn",  __fmt(arguments)); },
  error: function () { __hostLog("error", __fmt(arguments)); },
  debug: function () { __hostLog("debug", __fmt(arguments)); },
};
globalThis.crypto = {
  // key/out encodings are what make a signing CHAIN expressible: AWS SigV4
  // feeds each HMAC's raw output in as the next one's key, so the key has
  // to be acceptable as hex, not only as text.
  hmac: function (alg, key, data, outEncoding, keyEncoding) {
    return __hostHmac(alg, String(key), keyEncoding || "utf8", String(data), outEncoding || "hex");
  },
  hash: function (alg, data, outEncoding) {
    return __hostHash(alg, String(data), outEncoding || "hex");
  },
  base64Encode: function (data) { return __hostBase64("encode", String(data)); },
  base64Decode: function (data) { return __hostBase64("decode", String(data)); },
  randomUUID: function () { return __hostUuid(); },
};
globalThis.aisix = {
  // Screening against the environment's own embedding model, through the
  // same provider bridge the built-in semantic kind uses — a script cannot
  // reach it any other way without being handed separate credentials.
  embed: async function (model, texts) {
    const raw = await __hostEmbed(String(model), JSON.stringify(texts));
    const r = JSON.parse(raw);
    if (r.error) { throw new Error(r.error); }
    return r.vectors;
  },
};
"#;

/// One `kind: custom` row, materialised into a request-time runner.
pub struct CustomGuardrail {
    /// Operator-facing row name. Used for log labels and to disambiguate
    /// rows in a block reason; the trait's static `name()` stays "custom"
    /// so metric cardinality does not follow row names.
    row_name: String,
    /// The operator's module source, validated at build time. Each
    /// invocation declares it in its own fresh context — this crate is
    /// `forbid(unsafe_code)` and reloading precompiled bytecode is an
    /// `unsafe` call, so the module is parsed per invocation instead. It
    /// costs a fraction of the sandbox setup it happens inside.
    script: Arc<String>,
    secrets: Arc<BTreeMap<String, String>>,
    hook_point: GuardrailHookPoint,
    fail_open: bool,
    output_fail_open: bool,
    budget: Duration,
    max_memory_bytes: usize,
    stream_processing_mode: String,
    window_size: usize,
    window_overlap_size: usize,
    max_buffer_bytes: usize,
    on_buffer_exceeded_fail_open: bool,
    http: Arc<reqwest::Client>,
    /// The gateway's own embedding dispatcher, so a script can screen
    /// semantically against the environment's configured embedding model.
    /// Empty when the chain was built without one — `aisix.embed` then
    /// throws, which the script can catch.
    embedder: crate::GuardrailEmbedderSlot,
}

impl CustomGuardrail {
    /// Validate the operator's script and materialise the row.
    ///
    /// Parses WITHOUT evaluating: a syntax error is returned here, on the
    /// config-apply path, while nothing the operator wrote has run yet.
    pub fn new(
        row_name: impl Into<String>,
        cfg: &CustomConfig,
        hook_point: GuardrailHookPoint,
        fail_open: bool,
        embedder: crate::GuardrailEmbedderSlot,
    ) -> Result<Self, CompileError> {
        validate(&cfg.script)?;
        // Same connection-layer settings as every provider call: a bound
        // connect phase, TCP keepalive on, and pooled connections expired
        // before a hop in front of the script's destination reaps them.
        let http = aisix_gateway::client_builder()
            .build()
            .expect("guardrail http client builds");
        Ok(Self {
            row_name: row_name.into(),
            script: Arc::new(cfg.script.clone()),
            secrets: Arc::new(cfg.secrets.clone()),
            hook_point,
            fail_open,
            output_fail_open: cfg.output_fail_open,
            budget: Duration::from_millis(u64::from(cfg.timeout_ms)),
            max_memory_bytes: usize::try_from(cfg.max_memory_bytes).unwrap_or(usize::MAX),
            stream_processing_mode: cfg.stream_processing_mode.clone(),
            window_size: cfg.window_size as usize,
            window_overlap_size: cfg.window_overlap_size as usize,
            max_buffer_bytes: usize::try_from(cfg.max_buffer_bytes).unwrap_or(usize::MAX),
            on_buffer_exceeded_fail_open: cfg.on_buffer_exceeded == "fail_open",
            http: Arc::new(http),
            embedder,
        })
    }

    fn hook_enabled(&self, want: GuardrailHookPoint) -> bool {
        matches!(self.hook_point, GuardrailHookPoint::Both) || self.hook_point == want
    }

    /// Run one exported hook function against `ctx_json`.
    async fn run_hook(&self, func: &str, ctx_json: String) -> Result<ScriptOutcome, ScriptFailure> {
        match tokio::time::timeout(self.budget, self.invoke(func, ctx_json)).await {
            // The wall-clock budget covers a script parked on a pending
            // call, which the interrupt handler never sees.
            Err(_elapsed) => Err(ScriptFailure::Timeout),
            Ok(other) => other,
        }
    }

    /// Run a hook at a call site that CANNOT write rewritten text back.
    ///
    /// A `mask` decision becomes a Block here rather than an Allow: the
    /// script asked for content to be rewritten and this site cannot do it,
    /// so releasing the original would honor half the policy — the #963
    /// class. Mirrors what kind=presidio does on the same path.
    async fn run_blocking_hook(
        &self,
        func: &str,
        ctx_json: String,
        fail_open: bool,
    ) -> GuardrailVerdict {
        match self.run_hook(func, ctx_json).await {
            Err(failure) => self.handle_failure(failure, fail_open),
            Ok(ScriptOutcome::NotExported | ScriptOutcome::Allow) => GuardrailVerdict::Allow,
            Ok(ScriptOutcome::Block(verdict)) => verdict,
            Ok(ScriptOutcome::Mask { .. }) => GuardrailVerdict::block(format!(
                "custom script asked to rewrite content on a call site that cannot \
                 apply it (row: {})",
                self.row_name
            )),
        }
    }

    /// Run a hook at a call site that CAN write rewritten text back.
    async fn run_segment_hook(
        &self,
        func: &str,
        texts: &[String],
        hook: &'static str,
        model: Option<&str>,
        fail_open: bool,
    ) -> SegmentsOutcome {
        let messages: Vec<ScriptMessage> = texts
            .iter()
            .map(|t| ScriptMessage {
                role: aisix_gateway::Role::User,
                text: t.clone(),
            })
            .collect();
        let joined = texts.join("\n");
        let ctx = ScriptContext {
            hook,
            text: &joined,
            segments: texts,
            messages: &messages,
            model,
            secrets: &self.secrets,
        };
        let Ok(json) = serde_json::to_string(&ctx) else {
            return SegmentsOutcome::from_verdict(
                self.handle_failure(ScriptFailure::Engine, fail_open),
            );
        };

        match self.run_hook(func, json).await {
            Err(failure) => SegmentsOutcome::from_verdict(self.handle_failure(failure, fail_open)),
            Ok(ScriptOutcome::NotExported | ScriptOutcome::Allow) => SegmentsOutcome::allow(),
            Ok(ScriptOutcome::Block(verdict)) => SegmentsOutcome::from_verdict(verdict),
            Ok(ScriptOutcome::Mask { segments, counts }) => {
                // Positional substitution is the whole contract; a script
                // that returns a different number of slots would silently
                // shift content from one message onto another.
                if segments.len() != texts.len() {
                    tracing::warn!(
                        row = %self.row_name,
                        returned = segments.len(),
                        expected = texts.len(),
                        "custom guardrail returned a mismatched number of masked segments",
                    );
                    return SegmentsOutcome::from_verdict(
                        self.handle_failure(ScriptFailure::BadVerdict, fail_open),
                    );
                }
                SegmentsOutcome {
                    verdict: GuardrailVerdict::Allow,
                    masked: Some(segments),
                    counts,
                    monitor_hits: Vec::new(),
                }
            }
        }
    }

    /// One invocation in a brand-new sandbox. `Ok(None)` means the module
    /// does not export `func`, which is an Allow rather than an error.
    async fn invoke(&self, func: &str, ctx_json: String) -> Result<ScriptOutcome, ScriptFailure> {
        let runtime = rquickjs::AsyncRuntime::new().map_err(|e| {
            tracing::error!(row = %self.row_name, error = %e, "custom guardrail engine start failed");
            ScriptFailure::Engine
        })?;
        runtime.set_memory_limit(self.max_memory_bytes).await;
        runtime.set_max_stack_size(MAX_STACK_BYTES).await;

        // Stops a tight loop that never yields; the outer timeout cannot,
        // because such a script never returns control to the executor.
        let deadline = Instant::now() + self.budget;
        runtime
            .set_interrupt_handler(Some(Box::new(move || Instant::now() > deadline)))
            .await;

        let context = rquickjs::AsyncContext::full(&runtime).await.map_err(|e| {
            tracing::error!(row = %self.row_name, error = %e, "custom guardrail context start failed");
            ScriptFailure::Engine
        })?;

        let script = Arc::clone(&self.script);
        let http = Arc::clone(&self.http);
        let row_name = self.row_name.clone();
        let func = func.to_owned();
        let body_cap = self.max_memory_bytes;
        let embedder = self.embedder.clone();
        let embed_budget = self.budget;

        let raw: Result<Option<String>, ScriptFailure> = context
            .async_with(async |ctx| {
                install_host(
                    &ctx,
                    http,
                    row_name.clone(),
                    body_cap,
                    embedder,
                    embed_budget,
                )?;

                let module = Module::declare(ctx.clone(), MODULE_NAME, script.as_str())
                    .catch(&ctx)
                    .map_err(|e| threw(&row_name, "parse", e))?;
                let (module, pending) = module
                    .eval()
                    .catch(&ctx)
                    .map_err(|e| threw(&row_name, "eval", e))?;
                pending
                    .into_future::<()>()
                    .await
                    .catch(&ctx)
                    .map_err(|e| threw(&row_name, "eval", e))?;

                // A script may cover one direction only; the hook it does
                // not export is an Allow, not a misconfiguration.
                let Ok(hook) = module.get::<_, rquickjs::Function<'_>>(func.as_str()) else {
                    return Ok(None);
                };

                let arg = ctx
                    .json_parse(ctx_json)
                    .catch(&ctx)
                    .map_err(|e| threw(&row_name, "context", e))?;
                let returned: rquickjs::Value<'_> = hook
                    .call((arg,))
                    .catch(&ctx)
                    .map_err(|e| threw(&row_name, "call", e))?;

                // The contract is `async function`, but a plain function
                // returning a verdict object is just as valid — resolve
                // only what is actually a promise.
                let resolved = match returned.as_promise() {
                    Some(promise) => promise
                        .clone()
                        .into_future::<rquickjs::Value<'_>>()
                        .await
                        .catch(&ctx)
                        .map_err(|e| threw(&row_name, "call", e))?,
                    None => returned,
                };

                let json = ctx
                    .json_stringify(resolved)
                    .catch(&ctx)
                    .map_err(|e| threw(&row_name, "verdict", e))?;
                // Distinguished here, while the two are still separable: a
                // hook the module never exported is an Allow above, whereas
                // a hook that returned `undefined` decided nothing and must
                // not be read as one.
                match json.and_then(|s| s.to_string().ok()) {
                    Some(json) => Ok(Some(json)),
                    None => {
                        tracing::warn!(
                            row = %row_name,
                            "custom guardrail returned undefined instead of a verdict",
                        );
                        Err(ScriptFailure::BadVerdict)
                    }
                }
            })
            .await;

        match raw? {
            None => Ok(ScriptOutcome::NotExported),
            Some(json) => self.parse_verdict(&json),
        }
    }

    /// Translate the script's return value into an outcome.
    fn parse_verdict(&self, json: &str) -> Result<ScriptOutcome, ScriptFailure> {
        let parsed: ScriptVerdict = serde_json::from_str(json).map_err(|e| {
            tracing::warn!(
                row = %self.row_name,
                error = %e,
                "custom guardrail returned a value that is not a verdict object",
            );
            ScriptFailure::BadVerdict
        })?;
        match parsed.action.as_str() {
            "none" => Ok(ScriptOutcome::Allow),
            "block" => {
                // Both fields are operator-authored and land in ops logs
                // only — `Block.reason` never reaches the wire envelope
                // (#153), so neither can leak scanned content to a caller.
                let detail = match (parsed.reason_code.as_deref(), parsed.reason.as_deref()) {
                    (Some(code), Some(reason)) => format!("{code}: {reason}"),
                    (Some(code), None) => code.to_owned(),
                    (None, Some(reason)) => reason.to_owned(),
                    (None, None) => "no reason given".to_owned(),
                };
                Ok(ScriptOutcome::Block(GuardrailVerdict::block(format!(
                    "custom script blocked ({detail}) (row: {})",
                    self.row_name
                ))))
            }
            "mask" => {
                let Some(segments) = parsed.segments else {
                    tracing::warn!(
                        row = %self.row_name,
                        "custom guardrail asked to mask without returning segments",
                    );
                    return Err(ScriptFailure::BadVerdict);
                };
                Ok(ScriptOutcome::Mask {
                    segments,
                    counts: parsed.counts.unwrap_or_default(),
                })
            }
            other => {
                tracing::warn!(
                    row = %self.row_name,
                    action = %other,
                    "custom guardrail returned an unknown action",
                );
                Err(ScriptFailure::BadVerdict)
            }
        }
    }

    fn handle_failure(&self, failure: ScriptFailure, fail_open: bool) -> GuardrailVerdict {
        let tag = failure.bypass_tag();
        tracing::warn!(
            row = %self.row_name,
            failure = ?failure,
            fail_open = fail_open,
            "custom guardrail script failed",
        );
        if fail_open {
            GuardrailVerdict::Bypass { reason: tag.into() }
        } else {
            GuardrailVerdict::block_unavailable(format!("custom script unavailable ({tag})"), tag)
        }
    }
}

#[async_trait]
impl Guardrail for CustomGuardrail {
    fn name(&self) -> &'static str {
        "custom"
    }

    fn runs_on_output(&self) -> bool {
        self.hook_enabled(GuardrailHookPoint::Output)
    }

    /// Detection-only, so a streamed response does not have to be held whole
    /// the way a masking kind does: the sliding window is the default, and
    /// content is released as each window scans clean.
    fn stream_output_policy(&self) -> StreamOutputPolicy {
        match self.stream_processing_mode.as_str() {
            "buffer_full" => StreamOutputPolicy::BufferFull {
                max_buffer_bytes: self.max_buffer_bytes,
                on_exceeded_fail_open: self.on_buffer_exceeded_fail_open,
            },
            // "window" (default) and any unexpected value → sliding window.
            _ => StreamOutputPolicy::Window {
                size_chars: self.window_size,
                overlap_chars: self.window_overlap_size,
            },
        }
    }

    async fn check_input(&self, req: &ChatFormat) -> GuardrailVerdict {
        if !self.hook_enabled(GuardrailHookPoint::Input) {
            return GuardrailVerdict::Allow;
        }
        let messages: Vec<ScriptMessage> = req
            .messages
            .iter()
            .map(|m| ScriptMessage {
                role: m.role,
                text: crate::message_scan_text(m),
            })
            .filter(|m| !m.text.is_empty())
            .collect();
        if messages.is_empty() {
            return GuardrailVerdict::Allow;
        }
        let segments: Vec<String> = messages.iter().map(|m| m.text.clone()).collect();
        let text = segments.join("\n");
        let ctx = ScriptContext {
            hook: "input",
            text: &text,
            segments: &segments,
            messages: &messages,
            model: Some(req.model.as_str()),
            secrets: &self.secrets,
        };
        let Ok(json) = serde_json::to_string(&ctx) else {
            return self.handle_failure(ScriptFailure::Engine, self.fail_open);
        };
        self.run_blocking_hook("checkInput", json, self.fail_open)
            .await
    }

    async fn check_output(&self, resp: &ChatResponse) -> GuardrailVerdict {
        if !self.hook_enabled(GuardrailHookPoint::Output) {
            return GuardrailVerdict::Allow;
        }
        let text = resp.guardrail_output_text();
        if text.is_empty() {
            return GuardrailVerdict::Allow;
        }
        let messages = [ScriptMessage {
            role: aisix_gateway::Role::Assistant,
            text: text.clone(),
        }];
        let segments = [text.clone()];
        let ctx = ScriptContext {
            hook: "output",
            text: &text,
            segments: &segments,
            messages: &messages,
            model: None,
            secrets: &self.secrets,
        };
        let Ok(json) = serde_json::to_string(&ctx) else {
            return self.handle_failure(ScriptFailure::Engine, self.output_fail_open);
        };
        self.run_blocking_hook("checkOutput", json, self.output_fail_open)
            .await
    }

    /// kind=custom moderates via the segment pass wherever the call site
    /// supports mask write-back, so a script can rewrite content the way
    /// the built-in redacting kinds do.
    fn moderates_segments(&self) -> bool {
        true
    }

    async fn moderate_input_segments(&self, texts: &[String]) -> SegmentsOutcome {
        if !self.hook_enabled(GuardrailHookPoint::Input) {
            return SegmentsOutcome::allow();
        }
        self.run_segment_hook("checkInput", texts, "input", None, self.fail_open)
            .await
    }

    async fn moderate_output_segments(&self, texts: &[String]) -> SegmentsOutcome {
        if !self.hook_enabled(GuardrailHookPoint::Output) {
            return SegmentsOutcome::allow();
        }
        self.run_segment_hook("checkOutput", texts, "output", None, self.output_fail_open)
            .await
    }
}

// ---------------------------------------------------------------------------
// Compilation
// ---------------------------------------------------------------------------

/// Parse a script and report whether it is a well-formed ES module.
///
/// Declaring a module parses it without evaluating it, so validation never
/// runs a line the operator wrote. Used by the chain builder to turn a
/// script typo into a rejected resource at save time.
pub fn validate(script: &str) -> Result<(), CompileError> {
    let runtime = rquickjs::Runtime::new().map_err(|e| CompileError::Engine(e.to_string()))?;
    let context =
        rquickjs::Context::full(&runtime).map_err(|e| CompileError::Engine(e.to_string()))?;
    context.with(|ctx| {
        Module::declare(ctx.clone(), MODULE_NAME, script)
            .catch(&ctx)
            .map_err(|e| CompileError::Syntax(e.to_string()))?;
        Ok(())
    })
}

/// Why a script could not be compiled.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CompileError {
    /// The script is not a well-formed ES module. Carries the engine's own
    /// message, which names the line and column.
    #[error("{0}")]
    Syntax(String),
    /// The engine itself could not be started — not the operator's fault.
    #[error("script engine unavailable: {0}")]
    Engine(String),
}

// ---------------------------------------------------------------------------
// Host surface
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn install_host(
    ctx: &rquickjs::Ctx<'_>,
    http: Arc<reqwest::Client>,
    row_name: String,
    body_cap: usize,
    embedder: crate::GuardrailEmbedderSlot,
    embed_budget: Duration,
) -> Result<(), ScriptFailure> {
    let globals = ctx.globals();

    let fetch_row = row_name.clone();
    let host_fetch = rquickjs::Function::new(
        ctx.clone(),
        rquickjs::function::Async(move |url: String, init: Option<String>| {
            let http = Arc::clone(&http);
            let row = fetch_row.clone();
            async move { Ok::<_, rquickjs::Error>(host_fetch(http, row, url, init, body_cap).await) }
        }),
    )
    .map_err(|_| ScriptFailure::Engine)?;
    globals
        .set("__hostFetch", host_fetch)
        .map_err(|_| ScriptFailure::Engine)?;

    let log_row = row_name.clone();
    let host_log = rquickjs::Function::new(ctx.clone(), move |level: String, message: String| {
        // Operator-authored text at operator-chosen levels. Bounded so a
        // script cannot write an unbounded line into the gateway's log.
        let message: String = message.chars().take(2048).collect();
        match level.as_str() {
            "error" => tracing::error!(row = %log_row, "custom guardrail script: {message}"),
            "warn" => tracing::warn!(row = %log_row, "custom guardrail script: {message}"),
            "debug" => tracing::debug!(row = %log_row, "custom guardrail script: {message}"),
            _ => tracing::info!(row = %log_row, "custom guardrail script: {message}"),
        }
    })
    .map_err(|_| ScriptFailure::Engine)?;
    globals
        .set("__hostLog", host_log)
        .map_err(|_| ScriptFailure::Engine)?;

    let host_hmac = rquickjs::Function::new(
        ctx.clone(),
        |alg: String, key: String, key_encoding: String, data: String, out: String| {
            hmac_hex(&alg, &key, &key_encoding, &data, &out).unwrap_or_default()
        },
    )
    .map_err(|_| ScriptFailure::Engine)?;
    globals
        .set("__hostHmac", host_hmac)
        .map_err(|_| ScriptFailure::Engine)?;

    let host_hash =
        rquickjs::Function::new(ctx.clone(), |alg: String, data: String, out: String| {
            hash_hex(&alg, data.as_bytes(), &out).unwrap_or_default()
        })
        .map_err(|_| ScriptFailure::Engine)?;
    globals
        .set("__hostHash", host_hash)
        .map_err(|_| ScriptFailure::Engine)?;

    let host_base64 = rquickjs::Function::new(ctx.clone(), |mode: String, data: String| {
        use base64::Engine as _;
        match mode.as_str() {
            "decode" => base64::engine::general_purpose::STANDARD
                .decode(data.as_bytes())
                .ok()
                .and_then(|b| String::from_utf8(b).ok())
                .unwrap_or_default(),
            _ => base64::engine::general_purpose::STANDARD.encode(data.as_bytes()),
        }
    })
    .map_err(|_| ScriptFailure::Engine)?;
    globals
        .set("__hostBase64", host_base64)
        .map_err(|_| ScriptFailure::Engine)?;

    let host_uuid = rquickjs::Function::new(ctx.clone(), || uuid::Uuid::new_v4().to_string())
        .map_err(|_| ScriptFailure::Engine)?;
    globals
        .set("__hostUuid", host_uuid)
        .map_err(|_| ScriptFailure::Engine)?;

    let embed_row = row_name.clone();
    let host_embed = rquickjs::Function::new(
        ctx.clone(),
        rquickjs::function::Async(move |model: String, texts_json: String| {
            let embedder = embedder.clone();
            let row = embed_row.clone();
            let budget = embed_budget;
            async move {
                Ok::<_, rquickjs::Error>(host_embed(embedder, row, model, texts_json, budget).await)
            }
        }),
    )
    .map_err(|_| ScriptFailure::Engine)?;
    globals
        .set("__hostEmbed", host_embed)
        .map_err(|_| ScriptFailure::Engine)?;

    ctx.eval::<(), _>(PRELUDE)
        .catch(ctx)
        .map_err(|_| ScriptFailure::Engine)?;
    Ok(())
}

/// HMAC over `data`, keyed by `key` read as `key_encoding`, rendered as
/// `out`. The key encoding matters: a signing chain (AWS SigV4) uses one
/// HMAC's raw output as the next one's key.
fn hmac_hex(alg: &str, key: &str, key_encoding: &str, data: &str, out: &str) -> Option<String> {
    use hmac::Mac;
    let key_bytes: Vec<u8> = match key_encoding {
        "hex" => hex::decode(key).ok()?,
        "base64" => {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD.decode(key).ok()?
        }
        _ => key.as_bytes().to_vec(),
    };
    let digest: Vec<u8> = match alg {
        "sha1" => {
            let mut mac = hmac::Hmac::<sha1::Sha1>::new_from_slice(&key_bytes).ok()?;
            mac.update(data.as_bytes());
            mac.finalize().into_bytes().to_vec()
        }
        "sha256" => {
            let mut mac = hmac::Hmac::<sha2::Sha256>::new_from_slice(&key_bytes).ok()?;
            mac.update(data.as_bytes());
            mac.finalize().into_bytes().to_vec()
        }
        _ => return None,
    };
    Some(encode_digest(&digest, out))
}

fn hash_hex(alg: &str, data: &[u8], out: &str) -> Option<String> {
    use sha2::Digest as _;
    let digest: Vec<u8> = match alg {
        "sha1" => {
            use sha1::Digest as _;
            sha1::Sha1::digest(data).to_vec()
        }
        "sha256" => sha2::Sha256::digest(data).to_vec(),
        _ => return None,
    };
    Some(encode_digest(&digest, out))
}

fn encode_digest(digest: &[u8], out: &str) -> String {
    match out {
        "base64" => {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD.encode(digest)
        }
        _ => hex::encode(digest),
    }
}

/// Embed on the script's behalf through the gateway's own dispatcher, so a
/// semantic screen can use the environment's configured embedding model
/// rather than a second set of credentials the operator has to manage.
async fn host_embed(
    embedder: crate::GuardrailEmbedderSlot,
    row_name: String,
    model: String,
    texts_json: String,
    budget: Duration,
) -> String {
    let Some(embedder) = embedder.get().cloned() else {
        return embed_error("no embedding dispatcher available in this build".to_owned());
    };
    let texts: Vec<String> = match serde_json::from_str(&texts_json) {
        Ok(t) => t,
        Err(e) => return embed_error(format!("invalid texts argument: {e}")),
    };
    match embedder.embed(&model, &texts, false, budget).await {
        Ok(vectors) => serde_json::to_string(&EmbedResult {
            error: None,
            vectors,
        })
        .unwrap_or_else(|e| embed_error(e.to_string())),
        Err(failure) => {
            tracing::warn!(row = %row_name, model = %model, failure = ?failure, "custom guardrail embed failed");
            embed_error(format!("{failure:?}"))
        }
    }
}

fn embed_error(message: String) -> String {
    serde_json::to_string(&EmbedResult {
        error: Some(message),
        vectors: Vec::new(),
    })
    .unwrap_or_else(|_| r#"{"error":"embed failed"}"#.to_owned())
}

#[derive(serde::Serialize)]
struct EmbedResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    vectors: Vec<Vec<f32>>,
}

/// Perform one outbound request on the script's behalf and encode the result
/// as the JSON the prelude turns into a `Response`.
///
/// The destination is deliberately unconstrained: the script is written by
/// the operator and runs in the operator's own network, so there is no
/// boundary here for the gateway to police. The one bound is the response
/// body, capped at the script's own memory ceiling — anything larger could
/// not be handed to the script regardless.
async fn host_fetch(
    http: Arc<reqwest::Client>,
    row_name: String,
    url: String,
    init: Option<String>,
    body_cap: usize,
) -> String {
    let init: FetchInit = match init.as_deref() {
        None | Some("null") => FetchInit::default(),
        Some(raw) => match serde_json::from_str(raw) {
            Ok(parsed) => parsed,
            Err(e) => return fetch_error(format!("invalid fetch options: {e}")),
        },
    };

    let method = match reqwest::Method::from_bytes(init.method.as_bytes()) {
        Ok(m) => m,
        Err(_) => return fetch_error(format!("invalid method: {}", init.method)),
    };
    let mut request = http.request(method, &url);
    for (name, value) in &init.headers {
        request = request.header(name, value);
    }
    if let Some(body) = init.body {
        request = request.body(body);
    }

    let mut response = match request.send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(row = %row_name, url = %url, error = %e, "custom guardrail fetch failed");
            return fetch_error(e.to_string());
        }
    };

    let status = response.status();
    let headers: BTreeMap<String, String> = response
        .headers()
        .iter()
        .filter_map(|(k, v)| {
            v.to_str()
                .ok()
                .map(|v| (k.as_str().to_owned(), v.to_owned()))
        })
        .collect();
    let body = crate::read_body_capped(&mut response, body_cap).await;

    serde_json::to_string(&FetchResult {
        error: None,
        status: status.as_u16(),
        ok: status.is_success(),
        headers,
        body,
    })
    .unwrap_or_else(|e| fetch_error(e.to_string()))
}

fn fetch_error(message: String) -> String {
    serde_json::to_string(&FetchResult {
        error: Some(message),
        status: 0,
        ok: false,
        headers: BTreeMap::new(),
        body: String::new(),
    })
    .unwrap_or_else(|_| r#"{"error":"fetch failed"}"#.to_owned())
}

fn threw(row_name: &str, stage: &str, err: rquickjs::CaughtError<'_>) -> ScriptFailure {
    tracing::warn!(row = %row_name, stage = %stage, error = %err, "custom guardrail script threw");
    ScriptFailure::Threw
}

// ---------------------------------------------------------------------------
// Wire shapes
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
struct ScriptContext<'a> {
    hook: &'static str,
    text: &'a str,
    /// The text slots this hook is scanning, in the order the caller will
    /// substitute them back. A masking script returns a replacement per
    /// slot; a detection-only one can ignore them and read `text`.
    segments: &'a [String],
    messages: &'a [ScriptMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    secrets: &'a BTreeMap<String, String>,
}

#[derive(serde::Serialize)]
struct ScriptMessage {
    /// Serialises to the same lowercase strings the caller sent.
    role: aisix_gateway::Role,
    text: String,
}

#[derive(serde::Deserialize)]
struct ScriptVerdict {
    action: String,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    reason_code: Option<String>,
    /// `action: "mask"` only: the rewritten segments, positionally aligned
    /// with `ctx.segments`. The whole point of aligning them is that the
    /// caller can substitute slot for slot.
    #[serde(default)]
    segments: Option<Vec<String>>,
    /// What the script detected, by name, for the usage event. Names only —
    /// never matched content (#153).
    #[serde(default)]
    counts: Option<BTreeMap<String, u32>>,
}

#[derive(serde::Deserialize)]
struct FetchInit {
    #[serde(default = "default_method")]
    method: String,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    body: Option<String>,
}

impl Default for FetchInit {
    fn default() -> Self {
        Self {
            method: default_method(),
            headers: BTreeMap::new(),
            body: None,
        }
    }
}

fn default_method() -> String {
    "GET".to_owned()
}

#[derive(serde::Serialize)]
struct FetchResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    status: u16,
    ok: bool,
    headers: BTreeMap<String, String>,
    body: String,
}

/// What one hook invocation decided, before it is mapped onto the verdict
/// the call site can actually express.
enum ScriptOutcome {
    /// The module exports no such hook — one script may cover a single
    /// direction, so this is an Allow rather than a misconfiguration.
    NotExported,
    Allow,
    Block(GuardrailVerdict),
    /// Rewrite the scanned slots. Only a call site that substitutes text
    /// back can honor this; the others turn it into a Block.
    Mask {
        segments: Vec<String>,
        counts: BTreeMap<String, u32>,
    },
}

/// Failure cause buckets. `bypass_tag()` maps to the strings stored in
/// `usage_events.guardrail_bypassed_reason` — changing them is a breaking
/// change for operators who filter on these values.
#[derive(Debug)]
enum ScriptFailure {
    /// The wall-clock budget elapsed, or the interrupt handler fired.
    Timeout,
    /// The script raised, or the module body did.
    Threw,
    /// The script returned something that is not a verdict object.
    BadVerdict,
    /// The engine could not be started, or the host surface not installed.
    Engine,
}

impl ScriptFailure {
    fn bypass_tag(&self) -> &'static str {
        match self {
            Self::Timeout => "custom_timeout",
            Self::Threw => "custom_script_error",
            Self::BadVerdict => "custom_bad_verdict",
            Self::Engine => "custom_engine_error",
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_gateway::{ChatFormat, ChatMessage};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config(script: &str) -> CustomConfig {
        CustomConfig {
            script: script.to_owned(),
            secrets: BTreeMap::new(),
            timeout_ms: 2_000,
            max_memory_bytes: 16 * 1024 * 1024,
            stream_processing_mode: "window".to_owned(),
            window_size: 10_000,
            window_overlap_size: 256,
            max_buffer_bytes: 262_144,
            on_buffer_exceeded: "fail_closed".to_owned(),
            output_fail_open: false,
        }
    }

    fn guardrail(cfg: &CustomConfig, fail_open: bool) -> CustomGuardrail {
        CustomGuardrail::new(
            "row",
            cfg,
            GuardrailHookPoint::Both,
            fail_open,
            crate::GuardrailEmbedderSlot::none(),
        )
        .expect("script compiles")
    }

    fn request(text: &str) -> ChatFormat {
        ChatFormat::new("gpt-4o", vec![ChatMessage::user(text)])
    }

    #[test]
    fn validate_rejects_a_syntax_error_with_the_engine_message() {
        let err = validate("export async function checkInput(ctx) { return {").unwrap_err();
        assert!(
            matches!(err, CompileError::Syntax(ref m) if m.contains(MODULE_NAME)),
            "syntax error should name the module: {err:?}",
        );
    }

    #[test]
    fn validate_accepts_a_well_formed_module_without_running_it() {
        // The module body would throw if it were evaluated. Validation must
        // parse only, so a row still builds and the throw surfaces per
        // request under the fail-open policy rather than at config-apply.
        validate("throw new Error('boom'); export function checkInput() {}")
            .expect("parses without evaluating");
    }

    #[tokio::test]
    async fn allow_verdict_passes() {
        let cfg = config("export function checkInput() { return { action: 'none' }; }");
        let verdict = guardrail(&cfg, false).check_input(&request("hello")).await;
        assert!(matches!(verdict, GuardrailVerdict::Allow), "{verdict:?}");
    }

    #[tokio::test]
    async fn block_verdict_composes_reason_without_leaking_scanned_text() {
        let cfg = config(
            "export function checkInput(ctx) {
               return ctx.text.includes('bomb')
                 ? { action: 'block', reason_code: 'weapons', reason: 'policy hit' }
                 : { action: 'none' };
             }",
        );
        let verdict = guardrail(&cfg, false).check_input(&request("a bomb")).await;
        let GuardrailVerdict::Block {
            reason,
            unavailable,
            ..
        } = verdict
        else {
            panic!("expected Block, got {verdict:?}");
        };
        assert!(reason.contains("weapons"), "{reason}");
        assert!(reason.contains("policy hit"), "{reason}");
        assert!(reason.contains("row: row"), "{reason}");
        assert!(
            unavailable.is_none(),
            "a content decision is not an availability failure",
        );
    }

    #[tokio::test]
    async fn a_hook_the_module_does_not_export_allows() {
        let cfg = config("export function checkOutput() { return { action: 'block' }; }");
        let verdict = guardrail(&cfg, false).check_input(&request("hello")).await;
        assert!(matches!(verdict, GuardrailVerdict::Allow), "{verdict:?}");
    }

    #[tokio::test]
    async fn a_throwing_script_blocks_when_fail_closed() {
        let cfg = config("export function checkInput() { throw new Error('boom'); }");
        let verdict = guardrail(&cfg, false).check_input(&request("hello")).await;
        let GuardrailVerdict::Block { unavailable, .. } = verdict else {
            panic!("expected Block, got {verdict:?}");
        };
        assert_eq!(unavailable.as_deref(), Some("custom_script_error"));
    }

    #[tokio::test]
    async fn a_throwing_script_bypasses_when_fail_open() {
        let cfg = config("export function checkInput() { throw new Error('boom'); }");
        let verdict = guardrail(&cfg, true).check_input(&request("hello")).await;
        let GuardrailVerdict::Bypass { reason } = verdict else {
            panic!("expected Bypass, got {verdict:?}");
        };
        assert_eq!(reason, "custom_script_error");
    }

    #[tokio::test]
    async fn a_verdict_that_is_not_a_verdict_is_a_failure_not_an_allow() {
        // Returning nothing must not read as "allow" — that would let a
        // buggy script silently disable the policy it implements.
        let cfg = config("export function checkInput() { return 42; }");
        let verdict = guardrail(&cfg, false).check_input(&request("hello")).await;
        let GuardrailVerdict::Block { unavailable, .. } = verdict else {
            panic!("expected Block, got {verdict:?}");
        };
        assert_eq!(unavailable.as_deref(), Some("custom_bad_verdict"));
    }

    #[tokio::test]
    async fn an_unknown_action_is_a_failure() {
        let cfg = config("export function checkInput() { return { action: 'maybe' }; }");
        let verdict = guardrail(&cfg, false).check_input(&request("hello")).await;
        let GuardrailVerdict::Block { unavailable, .. } = verdict else {
            panic!("expected Block, got {verdict:?}");
        };
        assert_eq!(unavailable.as_deref(), Some("custom_bad_verdict"));
    }

    #[tokio::test]
    async fn a_runaway_loop_is_cut_by_the_interrupt_handler() {
        // The outer wall-clock timeout cannot catch this: a tight loop
        // never yields to the executor.
        let mut cfg = config("export function checkInput() { for (;;) {} }");
        cfg.timeout_ms = 200;
        let started = Instant::now();
        let verdict = guardrail(&cfg, false).check_input(&request("hello")).await;
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "should not hang"
        );
        let GuardrailVerdict::Block { unavailable, .. } = verdict else {
            panic!("expected Block, got {verdict:?}");
        };
        // Whichever budget wins the race, the row must fail closed.
        assert!(
            matches!(
                unavailable.as_deref(),
                Some("custom_timeout" | "custom_script_error")
            ),
            "{unavailable:?}",
        );
    }

    #[tokio::test]
    async fn a_hung_call_is_cut_by_the_wall_clock_budget() {
        // The mirror case: the script is parked on a pending response, so
        // the interrupt handler never runs and only the outer timeout fires.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/scan"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(30)))
            .mount(&upstream)
            .await;
        let mut cfg = config(&format!(
            "export async function checkInput(ctx) {{
               await fetch('{}/scan', {{ method: 'POST', body: ctx.text }});
               return {{ action: 'none' }};
             }}",
            upstream.uri(),
        ));
        cfg.timeout_ms = 300;
        let started = Instant::now();
        let verdict = guardrail(&cfg, false).check_input(&request("hello")).await;
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "should not hang"
        );
        let GuardrailVerdict::Block { unavailable, .. } = verdict else {
            panic!("expected Block, got {verdict:?}");
        };
        assert_eq!(unavailable.as_deref(), Some("custom_timeout"));
    }

    #[tokio::test]
    async fn the_script_reaches_a_real_service_and_uses_its_answer() {
        // The whole point of the kind: an adapter written inside the
        // gateway, talking to a service with its own protocol.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/scan"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "outcome": { "deny": true, "rule": "R-17" }
            })))
            .mount(&upstream)
            .await;
        let cfg = config(&format!(
            "export async function checkInput(ctx) {{
               const resp = await fetch('{}/scan', {{
                 method: 'POST',
                 headers: {{ 'content-type': 'application/json', 'x-api-key': ctx.secrets.SCAN_KEY }},
                 body: JSON.stringify({{ text: ctx.text }}),
               }});
               if (!resp.ok) {{ throw new Error('screening unavailable: ' + resp.status); }}
               const body = await resp.json();
               return body.outcome.deny
                 ? {{ action: 'block', reason_code: body.outcome.rule }}
                 : {{ action: 'none' }};
             }}",
            upstream.uri(),
        ));
        let verdict = guardrail(&cfg, false).check_input(&request("hello")).await;
        let GuardrailVerdict::Block { reason, .. } = verdict else {
            panic!("expected Block, got {verdict:?}");
        };
        assert!(reason.contains("R-17"), "{reason}");
    }

    #[tokio::test]
    async fn secrets_reach_the_script_and_the_wire() {
        let upstream = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/whoami"))
            .and(wiremock::matchers::header("x-api-key", "sk-live-42"))
            .respond_with(ResponseTemplate::new(204))
            .mount(&upstream)
            .await;
        let mut cfg = config(&format!(
            "export async function checkInput(ctx) {{
               const resp = await fetch('{}/whoami', {{ headers: {{ 'x-api-key': ctx.secrets.SCAN_KEY }} }});
               return resp.status === 204 ? {{ action: 'none' }} : {{ action: 'block' }};
             }}",
            upstream.uri(),
        ));
        cfg.secrets
            .insert("SCAN_KEY".to_owned(), "sk-live-42".to_owned());
        let verdict = guardrail(&cfg, false).check_input(&request("hello")).await;
        assert!(
            matches!(verdict, GuardrailVerdict::Allow),
            "the mock only answers 204 when the secret arrived: {verdict:?}",
        );
    }

    #[tokio::test]
    async fn a_failed_call_is_catchable_by_the_script() {
        // fetch rejects rather than returning a sentinel, so an operator
        // can decide their own policy for an unreachable service.
        let cfg = config(
            "export async function checkInput(ctx) {
               try {
                 await fetch('http://127.0.0.1:1/nope');
                 return { action: 'none' };
               } catch (e) {
                 return { action: 'block', reason_code: 'screening_down' };
               }
             }",
        );
        let verdict = guardrail(&cfg, true).check_input(&request("hello")).await;
        let GuardrailVerdict::Block { reason, .. } = verdict else {
            panic!("expected Block, got {verdict:?}");
        };
        assert!(reason.contains("screening_down"), "{reason}");
    }

    #[tokio::test]
    async fn no_state_survives_between_invocations() {
        // A fresh sandbox per call is what makes one request unable to
        // influence the next.
        let cfg = config(
            "globalThis.seen = (globalThis.seen || 0) + 1;
             export function checkInput() {
               return globalThis.seen === 1 ? { action: 'none' } : { action: 'block' };
             }",
        );
        let g = guardrail(&cfg, false);
        for _ in 0..3 {
            let verdict = g.check_input(&request("hello")).await;
            assert!(matches!(verdict, GuardrailVerdict::Allow), "{verdict:?}");
        }
    }

    // --- parity with the built-in kinds -----------------------------------
    //
    // Each of these re-implements, in a script, something a built-in kind
    // does natively. They are the standing check that the kind stays a
    // superset rather than drifting into a half-feature.

    #[tokio::test]
    async fn a_script_can_rewrite_content_the_way_the_redacting_kinds_do() {
        // kind=pii / presidio / lakera parity: mask spans in place.
        let cfg = config(
            "export function checkInput(ctx) {
               const out = ctx.segments.map(s => s.replace(/\\d{3}-\\d{2}-\\d{4}/g, '<SSN>'));
               const hits = ctx.segments.length - out.filter((s, i) => s === ctx.segments[i]).length;
               if (hits === 0) return { action: 'none' };
               return { action: 'mask', segments: out, counts: { US_SSN: hits } };
             }",
        );
        let outcome = guardrail(&cfg, false)
            .moderate_input_segments(&["my ssn is 123-45-6789".to_owned(), "hello".to_owned()])
            .await;
        assert!(matches!(outcome.verdict, GuardrailVerdict::Allow));
        assert_eq!(
            outcome.masked.as_deref(),
            Some(["my ssn is <SSN>".to_owned(), "hello".to_owned()].as_slice()),
        );
        assert_eq!(outcome.counts.get("US_SSN"), Some(&1));
    }

    #[tokio::test]
    async fn a_rewrite_blocks_where_it_cannot_be_written_back() {
        // Never half-honor a knob (#963): a call site with no write-back
        // must not release the text the script asked to rewrite.
        let cfg = config(
            "export function checkInput(ctx) {
               return { action: 'mask', segments: ctx.segments.map(() => '<redacted>') };
             }",
        );
        let verdict = guardrail(&cfg, false).check_input(&request("secret")).await;
        let GuardrailVerdict::Block { reason, .. } = verdict else {
            panic!("expected Block, got {verdict:?}");
        };
        assert!(reason.contains("cannot"), "{reason}");
    }

    #[tokio::test]
    async fn a_mismatched_slot_count_is_a_failure_not_a_silent_shift() {
        let cfg = config(
            "export function checkInput() { return { action: 'mask', segments: ['only one'] }; }",
        );
        let outcome = guardrail(&cfg, false)
            .moderate_input_segments(&["a".to_owned(), "b".to_owned()])
            .await;
        let GuardrailVerdict::Block { unavailable, .. } = outcome.verdict else {
            panic!("expected Block, got {:?}", outcome.verdict);
        };
        assert_eq!(unavailable.as_deref(), Some("custom_bad_verdict"));
        assert!(outcome.masked.is_none(), "nothing may be substituted");
    }

    #[tokio::test]
    async fn a_script_can_sign_a_request_the_way_the_aliyun_kind_does() {
        // kind=aliyun_text_moderation parity: its RPC signature is
        // HMAC-SHA1 over a canonical string, base64-encoded.
        let cfg = config(
            "export function checkInput() {
               const sig = crypto.hmac('sha1', 'key&', 'GET&%2F&foo', 'base64');
               return sig === '' ? { action: 'block' } : { action: 'block', reason_code: sig };
             }",
        );
        let verdict = guardrail(&cfg, false).check_input(&request("x")).await;
        let GuardrailVerdict::Block { reason, .. } = verdict else {
            panic!("expected Block, got {verdict:?}");
        };
        // Cross-checked against the same primitive the built-in kind uses.
        let expect = hmac_hex("sha1", "key&", "utf8", "GET&%2F&foo", "base64").unwrap();
        assert!(reason.contains(&expect), "{reason} should carry {expect}");
    }

    #[tokio::test]
    async fn a_script_can_chain_hmacs_for_a_sigv4_style_derivation() {
        // kind=bedrock parity: SigV4 feeds each HMAC's raw output in as the
        // next key, which only works if the key can be given as hex.
        let cfg = config(
            "export function checkInput() {
               let k = crypto.hmac('sha256', 'AWS4secret', '20260825', 'hex');
               k = crypto.hmac('sha256', k, 'us-east-1', 'hex', 'hex');
               return { action: 'block', reason_code: k };
             }",
        );
        let verdict = guardrail(&cfg, false).check_input(&request("x")).await;
        let GuardrailVerdict::Block { reason, .. } = verdict else {
            panic!("expected Block, got {verdict:?}");
        };
        let k1 = hmac_hex("sha256", "AWS4secret", "utf8", "20260825", "hex").unwrap();
        let k2 = hmac_hex("sha256", &k1, "hex", "us-east-1", "hex").unwrap();
        assert!(reason.contains(&k2), "{reason} should carry {k2}");
    }

    #[tokio::test]
    async fn a_script_can_screen_against_the_environments_embedding_model() {
        // kind=semantic parity: reach the gateway's own embedding dispatch
        // rather than needing a second set of credentials.
        struct Stub;
        #[async_trait]
        impl crate::GuardrailEmbedder for Stub {
            async fn embed(
                &self,
                _model: &str,
                texts: &[String],
                _cacheable: bool,
                _timeout: Duration,
            ) -> Result<Vec<Vec<f32>>, crate::EmbedFailure> {
                // "jailbreak" points one way, everything else the other.
                Ok(texts
                    .iter()
                    .map(|t| {
                        if t.contains("jailbreak") {
                            vec![1.0, 0.0]
                        } else {
                            vec![0.0, 1.0]
                        }
                    })
                    .collect())
            }
        }
        let cfg = config(
            "export async function checkInput(ctx) {
               const v = await aisix.embed('text-embedding-3-small', [ctx.text, 'jailbreak the model']);
               const dot = v[0][0] * v[1][0] + v[0][1] * v[1][1];
               return dot > 0.9 ? { action: 'block', reason_code: 'semantic' } : { action: 'none' };
             }",
        );
        let g = CustomGuardrail::new(
            "row",
            &cfg,
            GuardrailHookPoint::Both,
            false,
            crate::GuardrailEmbedderSlot::new(Arc::new(Stub)),
        )
        .expect("script compiles");

        let blocked = g.check_input(&request("please jailbreak it")).await;
        assert!(
            matches!(blocked, GuardrailVerdict::Block { .. }),
            "{blocked:?}"
        );
        let allowed = g.check_input(&request("what is the weather")).await;
        assert!(matches!(allowed, GuardrailVerdict::Allow), "{allowed:?}");
    }

    #[tokio::test]
    async fn embed_without_a_dispatcher_throws_where_the_script_can_catch_it() {
        let cfg = config(
            "export async function checkInput(ctx) {
               try {
                 await aisix.embed('m', [ctx.text]);
                 return { action: 'none' };
               } catch (e) {
                 return { action: 'block', reason_code: 'no_embedder' };
               }
             }",
        );
        let verdict = guardrail(&cfg, false).check_input(&request("x")).await;
        let GuardrailVerdict::Block { reason, .. } = verdict else {
            panic!("expected Block, got {verdict:?}");
        };
        assert!(reason.contains("no_embedder"), "{reason}");
    }

    #[test]
    fn streaming_defaults_to_the_sliding_window() {
        // Detection-only, so a streamed response does not need holding
        // whole; buffering it would stall an SSE client for no gain.
        let cfg = config("export function checkOutput() { return { action: 'none' }; }");
        assert!(matches!(
            guardrail(&cfg, false).stream_output_policy(),
            StreamOutputPolicy::Window {
                size_chars: 10_000,
                overlap_chars: 256
            },
        ));
    }

    #[test]
    fn streaming_honors_an_explicit_buffer_full_choice() {
        let mut cfg = config("export function checkOutput() { return { action: 'none' }; }");
        cfg.stream_processing_mode = "buffer_full".to_owned();
        assert!(matches!(
            guardrail(&cfg, false).stream_output_policy(),
            StreamOutputPolicy::BufferFull {
                on_exceeded_fail_open: false,
                ..
            },
        ));
    }
}
