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
//! `ctx` carries `{ hook, text, messages, model, request_id, secrets }`. A
//! verdict is `{ action: "none" }` or
//! `{ action: "block", reason, reason_code }`. A hook whose function the
//! module does not export is an Allow, so one script may cover one direction.
//!
//! Detection-only: it blocks, never rewrites. Rewriting is a synchronous
//! path ([`Guardrail::redact_input_text`]) and a script whose purpose is to
//! await an external call cannot participate in it.
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

use crate::{Guardrail, GuardrailVerdict, StreamOutputPolicy};

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
        })
    }

    fn hook_enabled(&self, want: GuardrailHookPoint) -> bool {
        matches!(self.hook_point, GuardrailHookPoint::Both) || self.hook_point == want
    }

    /// Run one exported hook function against `ctx_json`.
    async fn run_hook(&self, func: &str, ctx_json: String, fail_open: bool) -> GuardrailVerdict {
        let outcome = tokio::time::timeout(
            self.budget,
            self.invoke(func, ctx_json),
        )
        .await;

        match outcome {
            // The wall-clock budget covers a script parked on a pending
            // call, which the interrupt handler never sees.
            Err(_elapsed) => self.handle_failure(ScriptFailure::Timeout, fail_open),
            Ok(Err(failure)) => self.handle_failure(failure, fail_open),
            Ok(Ok(None)) => GuardrailVerdict::Allow,
            Ok(Ok(Some(verdict))) => verdict,
        }
    }

    /// One invocation in a brand-new sandbox. `Ok(None)` means the module
    /// does not export `func`, which is an Allow rather than an error.
    async fn invoke(
        &self,
        func: &str,
        ctx_json: String,
    ) -> Result<Option<GuardrailVerdict>, ScriptFailure> {
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

        let raw: Result<Option<String>, ScriptFailure> = context
            .async_with(async |ctx| {
                install_host(&ctx, http, row_name.clone(), body_cap)?;

                let module = Module::declare(ctx.clone(), MODULE_NAME, script.as_str())
                    .catch(&ctx)
                    .map_err(|e| threw(&row_name, "parse", e))?;
                let (module, pending) =
                    module.eval().catch(&ctx).map_err(|e| threw(&row_name, "eval", e))?;
                pending
                    .into_future::<()>()
                    .await
                    .catch(&ctx)
                    .map_err(|e| threw(&row_name, "eval", e))?;

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
                Ok(json.and_then(|s| s.to_string().ok()))
            })
            .await;

        match raw? {
            None => Err(ScriptFailure::BadVerdict),
            Some(json) => self.parse_verdict(&json).map(Some),
        }
    }

    /// Translate the script's return value into a verdict.
    fn parse_verdict(&self, json: &str) -> Result<GuardrailVerdict, ScriptFailure> {
        let parsed: ScriptVerdict = serde_json::from_str(json).map_err(|e| {
            tracing::warn!(
                row = %self.row_name,
                error = %e,
                "custom guardrail returned a value that is not a verdict object",
            );
            ScriptFailure::BadVerdict
        })?;
        match parsed.action.as_str() {
            "none" => Ok(GuardrailVerdict::Allow),
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
                Ok(GuardrailVerdict::block(format!(
                    "custom script blocked ({detail}) (row: {})",
                    self.row_name
                )))
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
        let text = messages
            .iter()
            .map(|m| m.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let ctx = ScriptContext {
            hook: "input",
            text: &text,
            messages: &messages,
            model: Some(req.model.as_str()),
            secrets: &self.secrets,
        };
        let Ok(json) = serde_json::to_string(&ctx) else {
            return self.handle_failure(ScriptFailure::Engine, self.fail_open);
        };
        self.run_hook("checkInput", json, self.fail_open).await
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
        let ctx = ScriptContext {
            hook: "output",
            text: &text,
            messages: &messages,
            model: None,
            secrets: &self.secrets,
        };
        let Ok(json) = serde_json::to_string(&ctx) else {
            return self.handle_failure(ScriptFailure::Engine, self.output_fail_open);
        };
        self.run_hook("checkOutput", json, self.output_fail_open)
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

fn install_host(
    ctx: &rquickjs::Ctx<'_>,
    http: Arc<reqwest::Client>,
    row_name: String,
    body_cap: usize,
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

    let host_log = rquickjs::Function::new(
        ctx.clone(),
        move |level: String, message: String| {
            // Operator-authored text at operator-chosen levels. Bounded so a
            // script cannot write an unbounded line into the gateway's log.
            let message: String = message.chars().take(2048).collect();
            match level.as_str() {
                "error" => tracing::error!(row = %row_name, "custom guardrail script: {message}"),
                "warn" => tracing::warn!(row = %row_name, "custom guardrail script: {message}"),
                "debug" => tracing::debug!(row = %row_name, "custom guardrail script: {message}"),
                _ => tracing::info!(row = %row_name, "custom guardrail script: {message}"),
            }
        },
    )
    .map_err(|_| ScriptFailure::Engine)?;
    globals
        .set("__hostLog", host_log)
        .map_err(|_| ScriptFailure::Engine)?;

    ctx.eval::<(), _>(PRELUDE)
        .catch(ctx)
        .map_err(|_| ScriptFailure::Engine)?;
    Ok(())
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
        .filter_map(|(k, v)| v.to_str().ok().map(|v| (k.as_str().to_owned(), v.to_owned())))
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
