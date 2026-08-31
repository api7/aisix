//! A census of the input guardrail chain over every surface
//! [`crate::build_router`] mounts.
//!
//! This exists because the same defect kept landing on one sibling at a
//! time. `/v1/messages/count_tokens` shipped with no chain at all; `/mcp`
//! consulted the chain only when a `tools/call` happened to carry text;
//! `/v1/audio/*` and `/v1/images/edits` only when a `prompt` part happened
//! to be present; `/a2a` never. Each was invisible because the test that
//! was supposed to cover the family restated a hand-written list of
//! endpoints, and a list nobody updates agrees with itself forever.
//!
//! So nothing here is hand-listed. [`mounted_surfaces`] reads the routing
//! table out of `lib.rs` itself, and [`POSTURE`] must classify exactly
//! that set — mount a route without saying what it owes an operator's
//! guardrail chain and `posture_covers_every_mounted_surface` fails. A
//! surface classified [`Posture::Enforced`] must also carry a request
//! fixture, and `enforced_surfaces_refuse_a_blocking_guardrail` drives
//! each one through the real router against a guardrail that blocks
//! unconditionally. Nothing short of an actual refusal passes.
//!
//! The blocking guardrail is a `kind: custom` script, deliberately: it is
//! the one kind whose verdict does not depend on the text, so it is the
//! only one that can tell "the chain ran and decided" apart from "the
//! chain found nothing to match". That distinction is the whole bug class.

use std::collections::BTreeSet;
use std::sync::Arc;

use aisix_core::snapshot::SnapshotHandle;
use aisix_core::{AisixSnapshot, ApiKey, ProxyConfig, ResourceEntry};
use aisix_obs::{UsageEvent, UsageSink};
use axum::body::Body;
use axum::http::Request;
use tower::ServiceExt;

/// The source of the routing table. Parsed rather than duplicated so a new
/// `.route(...)` cannot slip past this file.
const ROUTER_SRC: &str = include_str!("lib.rs");

/// What a mounted surface owes an operator's INPUT guardrail chain.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Posture {
    /// Caller-authored content reaches an upstream here, so the chain runs
    /// and can refuse the request before the upstream is contacted. Every
    /// such surface is driven for real below.
    Enforced,
    /// The chain runs, but the surface cannot be exercised through a
    /// `oneshot` against the router. Carries why, and where it is covered.
    EnforcedNotDrivableInCrate(&'static str),
    /// Nothing a caller authored reaches an upstream, so there is nothing
    /// for an input hook to screen. Carries why.
    NoUpstreamContent(&'static str),
}

/// Every surface `build_router` mounts, and its posture. The set of keys is
/// checked against the parsed routing table, so this cannot silently fall
/// behind the router.
pub(crate) const POSTURE: &[(&str, Posture)] = &[
    // --- liveness / discovery: no caller content leaves the gateway -----
    ("/livez", Posture::NoUpstreamContent("liveness probe")),
    ("/readyz", Posture::NoUpstreamContent("readiness probe")),
    (
        "/v1/models",
        Posture::NoUpstreamContent("lists the gateway's own snapshot; contacts no upstream"),
    ),
    (
        "/.well-known/oauth-protected-resource",
        Posture::NoUpstreamContent("RFC 9728 metadata served by the gateway itself"),
    ),
    (
        "/.well-known/oauth-protected-resource/mcp",
        Posture::NoUpstreamContent("RFC 9728 metadata served by the gateway itself"),
    ),
    (
        "/a2a/:agent/.well-known/agent-card.json",
        Posture::NoUpstreamContent("serves the agent's card with the URL rewritten; no caller body"),
    ),
    // --- reads of a job/asset the caller already created ----------------
    (
        "/v1/videos/:id",
        Posture::NoUpstreamContent("polls a job by id; the prompt was screened at creation"),
    ),
    (
        "/v1/videos/:id/content",
        Posture::NoUpstreamContent("fetches a rendered asset by id; carries no caller text"),
    ),
    (
        "/v1/files/:id",
        Posture::NoUpstreamContent("GET/DELETE by id; carries no caller text"),
    ),
    (
        "/v1/files/:id/content",
        Posture::NoUpstreamContent("downloads by id; carries no caller text"),
    ),
    (
        "/v1/batches/:id",
        Posture::NoUpstreamContent("GET by id; carries no caller text"),
    ),
    (
        "/v1/batches/:id/cancel",
        Posture::NoUpstreamContent("cancels by id; carries no caller text"),
    ),
    (
        "/v1/fine_tuning/jobs/:id",
        Posture::NoUpstreamContent("GET by id; carries no caller text"),
    ),
    (
        "/v1/fine_tuning/jobs/:id/cancel",
        Posture::NoUpstreamContent("cancels by id; carries no caller text"),
    ),
    // --- content-bearing surfaces --------------------------------------
    ("/v1/chat/completions", Posture::Enforced),
    ("/v1/completions", Posture::Enforced),
    ("/v1/embeddings", Posture::Enforced),
    ("/v1/images/generations", Posture::Enforced),
    ("/v1/images/edits", Posture::Enforced),
    ("/v1/messages", Posture::Enforced),
    ("/v1/messages/count_tokens", Posture::Enforced),
    ("/v1/rerank", Posture::Enforced),
    ("/v1/responses", Posture::Enforced),
    ("/v1/audio/transcriptions", Posture::Enforced),
    ("/v1/audio/translations", Posture::Enforced),
    ("/v1/audio/speech", Posture::Enforced),
    ("/v1/videos", Posture::Enforced),
    ("/mcp", Posture::Enforced),
    ("/mcp/", Posture::Enforced),
    ("/mcp/:server", Posture::Enforced),
    ("/a2a/:agent", Posture::Enforced),
    (
        "/v1/realtime",
        Posture::EnforcedNotDrivableInCrate(
            "WebSocket upgrade; the per-frame scan (realtime.rs `guardrail_block_event`) needs a \
             live socket. Covered by the realtime e2e suite.",
        ),
    ),
    (
        "/v1/files",
        Posture::EnforcedNotDrivableInCrate(
            "`jobs::scan_input_blob` screens the uploaded blob, but reaching it needs a resolvable \
             job target plus a multipart upload the mock upstream must accept. Covered by the jobs \
             e2e suite.",
        ),
    ),
    (
        "/v1/batches",
        Posture::EnforcedNotDrivableInCrate(
            "same `jobs::scan_input_blob` gate; POST needs an `input_file_id` that resolves \
             through a prior upload.",
        ),
    ),
    (
        "/v1/fine_tuning/jobs",
        Posture::EnforcedNotDrivableInCrate(
            "same `jobs::scan_input_blob` gate; POST needs a `training_file` that resolves \
             through a prior upload.",
        ),
    ),
    (
        FALLBACK_SURFACE,
        Posture::EnforcedNotDrivableInCrate(
            "`passthrough_route::entry`; a passthrough route matches on a configured path prefix \
             or Host, so there is no fixed path to drive here. Covered by \
             passthrough-guardrail-e2e.",
        ),
    ),
];

/// The router's `.fallback(...)` seat, which has no path literal of its own.
const FALLBACK_SURFACE: &str = "<fallback>";

/// Pull every surface `build_router` mounts out of its own source: the
/// `.route("<path>", ...)` literals, plus the `.fallback(...)` seat.
///
/// Scoped to the function body so unrelated `.route(` calls elsewhere in
/// `lib.rs` (tests, doc examples) cannot pad the census.
fn mounted_surfaces() -> BTreeSet<String> {
    let start = ROUTER_SRC
        .find("pub fn build_router(")
        .expect("build_router must exist in lib.rs");
    // The function ends at the first line that closes at column 0.
    let body = &ROUTER_SRC[start..];
    let end = body
        .find("\n}\n")
        .expect("build_router must be brace-balanced at column 0");
    let body = &body[..end];

    let mut found = BTreeSet::new();
    for (idx, _) in body.match_indices(".route(") {
        let rest = &body[idx + ".route(".len()..];
        // The path is the next string literal; `.route(` is always called
        // with one in this router.
        let open = rest.find('"').expect(".route( must take a path literal");
        let close = open
            + 1
            + rest[open + 1..]
                .find('"')
                .expect("unterminated route path literal");
        found.insert(rest[open + 1..close].to_string());
    }
    if body.contains(".fallback(") {
        found.insert(FALLBACK_SURFACE.to_string());
    }
    found
}

#[test]
fn posture_covers_every_mounted_surface() {
    let mounted = mounted_surfaces();
    let declared: BTreeSet<String> = POSTURE.iter().map(|(p, _)| (*p).to_string()).collect();

    // Sanity: the parse must actually find the router, not silently yield
    // an empty set that makes both assertions below vacuous.
    assert!(
        mounted.len() > 20,
        "the routing-table parse found only {} surfaces — it has stopped tracking build_router",
        mounted.len()
    );

    let unclassified: Vec<_> = mounted.difference(&declared).collect();
    assert!(
        unclassified.is_empty(),
        "these surfaces are mounted in build_router but have no guardrail posture declared in \
         POSTURE: {unclassified:?}\n\
         Say what each owes an operator's input guardrail chain. If content a caller wrote can \
         reach an upstream through it, the answer is Posture::Enforced and it needs a fixture in \
         `drive`.",
    );

    let stale: Vec<_> = declared.difference(&mounted).collect();
    assert!(
        stale.is_empty(),
        "POSTURE classifies surfaces build_router no longer mounts: {stale:?}",
    );

    // An exemption is only worth anything if it says why. A bare
    // "not enforced here" is how the previous gaps read to every reviewer
    // who looked at them.
    for (surface, posture) in POSTURE {
        let reason = match posture {
            Posture::Enforced => continue,
            Posture::EnforcedNotDrivableInCrate(reason) | Posture::NoUpstreamContent(reason) => {
                reason
            }
        };
        assert!(
            !reason.trim().is_empty(),
            "{surface} is exempted from the enforced set with no reason given",
        );
    }
}

// ---------------------------------------------------------------------------
// The behavioural half: drive every Enforced surface against a guardrail
// that blocks unconditionally, and require an actual refusal.
// ---------------------------------------------------------------------------

const CALLER: &str = "sk-census-caller";
/// SHA-256 of `CALLER`.
const CALLER_HASH: &str = "d73b98669c1f938ed09ee3d8e81ecdbb58f7bf38c57a4f78c301e7bdadc2fdf2";
const GUARDRAIL_ROW: &str = "census-block";
const PK_ID: &str = "11111111-1111-1111-1111-111111111111";
const ANTHROPIC_PK_ID: &str = "22222222-2222-2222-2222-222222222222";
const MCP_SERVER_ID: &str = "33333333-3333-3333-3333-333333333333";

/// A script with a verdict that owes nothing to the text it is handed. A
/// keyword or PII row could not distinguish "the chain ran" from "the chain
/// matched nothing", which is precisely the confusion this census exists to
/// rule out.
const BLOCK_EVERYTHING: &str = r#"
export function checkInput() {
  return { action: "block", reason_code: "census" };
}
"#;

fn cfg() -> ProxyConfig {
    ProxyConfig {
        addr: "127.0.0.1:0".into(),
        request_body_limit_bytes: 1_048_576,
        real_ip: Default::default(),
        request_id: Default::default(),
        url_rewrites: Vec::new(),
        tls: None,
        thread_per_core: None,
        workers: None,
    }
}

/// A snapshot wired so every content-bearing surface can resolve what it
/// needs (key, models, MCP server, A2A agent) and reach its guardrail gate.
/// The upstreams deliberately point nowhere: a surface that reaches one has
/// already failed the test.
fn census_snapshot() -> AisixSnapshot {
    let snap = AisixSnapshot::new();

    let key: ApiKey = serde_json::from_value(serde_json::json!({
        "key_hash": CALLER_HASH,
        "allowed_models": ["*"],
        "allowed_routes": ["*"],
        "allowed_agents": ["*"],
        "mcp_access": { "allow": ["*"] },
    }))
    .expect("valid api key");
    snap.apikeys.insert(ResourceEntry::new("ak-census", key, 1));

    let pk: aisix_core::ProviderKey = serde_json::from_value(serde_json::json!({
        "display_name": "census-openai",
        "secret": "sk-unused",
        "api_base": "http://127.0.0.1:1/v1",
        "provider": "openai",
        "adapter": "openai",
    }))
    .expect("valid provider key");
    snap.provider_keys.insert(ResourceEntry::new(PK_ID, pk, 1));

    let anthropic_pk: aisix_core::ProviderKey = serde_json::from_value(serde_json::json!({
        "display_name": "census-anthropic",
        "secret": "sk-ant-unused",
        "api_base": "http://127.0.0.1:1",
        "provider": "anthropic",
        "adapter": "anthropic",
    }))
    .expect("valid provider key");
    snap.provider_keys
        .insert(ResourceEntry::new(ANTHROPIC_PK_ID, anthropic_pk, 1));

    for (id, name, provider, model_name, pk_id) in [
        ("m-openai", "census-openai", "openai", "gpt-4o-mini", PK_ID),
        (
            "m-anthropic",
            "census-anthropic",
            "anthropic",
            "claude-haiku-4-5-20251001",
            ANTHROPIC_PK_ID,
        ),
    ] {
        let model: aisix_core::Model = serde_json::from_value(serde_json::json!({
            "display_name": name,
            "provider": provider,
            "model_name": model_name,
            "provider_key_id": pk_id,
        }))
        .expect("valid model");
        snap.models.insert(ResourceEntry::new(id, model, 1));
    }

    let embedding: aisix_core::Model = serde_json::from_value(serde_json::json!({
        "display_name": "census-embedding",
        "provider": "openai",
        "model_name": "text-embedding-3-small",
        "provider_key_id": PK_ID,
        "kind": "embedding",
    }))
    .expect("valid embedding model");
    snap.models
        .insert(ResourceEntry::new("m-embedding", embedding, 1));

    let mcp: aisix_core::McpServer = serde_json::from_value(serde_json::json!({
        "display_name": "census",
        "url": "http://127.0.0.1:1/mcp",
        "enabled": true,
    }))
    .expect("valid mcp server");
    snap.mcp_servers
        .insert(ResourceEntry::new(MCP_SERVER_ID, mcp, 1));

    let agent: aisix_core::A2aAgent = serde_json::from_value(serde_json::json!({
        "name": "census",
        "url": "http://127.0.0.1:1/a2a",
        "enabled": true,
    }))
    .expect("valid a2a agent");
    snap.a2a_agents
        .insert(ResourceEntry::new("agent-census", agent, 1));

    let guardrail: aisix_core::Guardrail = serde_json::from_value(serde_json::json!({
        "name": GUARDRAIL_ROW,
        "enabled": true,
        "kind": "custom",
        "hook_point": "input",
        "fail_open": false,
        "script": BLOCK_EVERYTHING,
        "timeout_ms": 5000,
    }))
    .expect("valid guardrail");
    crate::seed_env_scoped_guardrail(&snap, ResourceEntry::new("g-census", guardrail, 1));

    snap
}

/// A hub with the two provider bridges the fixtures name. `/v1/embeddings`
/// resolves its bridge BEFORE the guardrail gate, so a bare hub would answer
/// 503 and the census would never reach the check it exists to make.
fn census_hub() -> Arc<aisix_gateway::Hub> {
    let hub = Arc::new(aisix_gateway::Hub::new());
    hub.register_specialized(
        "openai",
        Arc::new(aisix_provider_openai::OpenAiBridge::new()),
    );
    hub.register_specialized(
        "anthropic",
        Arc::new(aisix_provider_anthropic::AnthropicBridge::new()),
    );
    hub
}

fn census_router() -> axum::Router {
    let handle = SnapshotHandle::new(census_snapshot());
    let index = aisix_guardrails::LiveGuardrailIndex::new(handle.clone(), None);
    let state = crate::ProxyState::new(handle, census_hub(), &cfg())
        .without_cache()
        .with_guardrail_index(index);
    crate::build_router(state)
}

/// [`census_router`] plus the receiver its usage events land in.
///
/// One router per surface rather than a shared one: the sink is a single
/// channel, so surfaces driven through the same router would interleave
/// their events and "which surface emitted nothing" would stop being
/// answerable — which is the whole question below.
pub(crate) fn census_router_with_usage() -> (axum::Router, tokio::sync::mpsc::Receiver<UsageEvent>)
{
    let handle = SnapshotHandle::new(census_snapshot());
    let index = aisix_guardrails::LiveGuardrailIndex::new(handle.clone(), None);
    let (tx, rx) = tokio::sync::mpsc::channel(32);
    let state = crate::ProxyState::new(handle, census_hub(), &cfg())
        .without_cache()
        .with_guardrail_index(index)
        .with_usage_sink(UsageSink::new(tx));
    (crate::build_router(state), rx)
}

const MULTIPART_BOUNDARY: &str = "censusboundary";

/// A multipart body with the parts named, in order. Values are inline
/// bytes; nothing here needs a real file.
fn multipart(parts: &[(&str, &str)]) -> String {
    let mut out = String::new();
    for (name, value) in parts {
        out.push_str(&format!("--{MULTIPART_BOUNDARY}\r\n"));
        if *name == "file" || *name == "image" {
            out.push_str(&format!(
                "Content-Disposition: form-data; name=\"{name}\"; filename=\"a.bin\"\r\n\
                 Content-Type: application/octet-stream\r\n\r\n"
            ));
        } else {
            out.push_str(&format!(
                "Content-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            ));
        }
        out.push_str(value);
        out.push_str("\r\n");
    }
    out.push_str(&format!("--{MULTIPART_BOUNDARY}--\r\n"));
    out
}

/// One driveable request per `Posture::Enforced` surface.
///
/// Bodies are deliberately CONTENTLESS wherever the wire shape allows it —
/// no prompt, empty `arguments`, empty message text. That is the shape every
/// bug in this class hid behind, so it is the shape the census drives. A
/// guardrail with a text-independent verdict must refuse them all.
pub(crate) fn fixture(surface: &str) -> Option<Request<Body>> {
    let json = |uri: &str, body: serde_json::Value| {
        Request::builder()
            .method("POST")
            .uri(uri.to_string())
            .header("authorization", format!("Bearer {CALLER}"))
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    };
    let anthropic = |uri: &str, body: serde_json::Value| {
        Request::builder()
            .method("POST")
            .uri(uri.to_string())
            .header("x-api-key", CALLER)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    };
    let form = |uri: &str, parts: &[(&str, &str)]| {
        Request::builder()
            .method("POST")
            .uri(uri.to_string())
            .header("authorization", format!("Bearer {CALLER}"))
            .header(
                "content-type",
                format!("multipart/form-data; boundary={MULTIPART_BOUNDARY}"),
            )
            .body(Body::from(multipart(parts)))
            .unwrap()
    };
    let jsonrpc = |uri: &str, body: serde_json::Value| {
        Request::builder()
            .method("POST")
            .uri(uri.to_string())
            .header("authorization", format!("Bearer {CALLER}"))
            .header("host", "census.aisix.example.com")
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .body(Body::from(body.to_string()))
            .unwrap()
    };

    Some(match surface {
        "/v1/chat/completions" => json(
            surface,
            serde_json::json!({
                "model": "census-openai",
                "messages": [{ "role": "user", "content": "" }],
            }),
        ),
        "/v1/completions" => json(
            surface,
            serde_json::json!({ "model": "census-openai", "prompt": "" }),
        ),
        "/v1/embeddings" => json(
            surface,
            serde_json::json!({ "model": "census-embedding", "input": "" }),
        ),
        "/v1/images/generations" => json(
            surface,
            serde_json::json!({ "model": "census-openai", "prompt": "" }),
        ),
        // No `prompt` part at all — the exact shape that used to skip the
        // chain outright.
        "/v1/images/edits" => form(surface, &[("model", "census-openai"), ("image", "x")]),
        "/v1/messages" => anthropic(
            surface,
            serde_json::json!({
                "model": "census-anthropic",
                "max_tokens": 16,
                "messages": [{ "role": "user", "content": "" }],
            }),
        ),
        "/v1/messages/count_tokens" => anthropic(
            surface,
            serde_json::json!({
                "model": "census-anthropic",
                "messages": [{ "role": "user", "content": "" }],
            }),
        ),
        "/v1/rerank" => json(
            surface,
            serde_json::json!({ "model": "census-openai", "query": "", "documents": [""] }),
        ),
        "/v1/responses" => json(
            surface,
            serde_json::json!({ "model": "census-openai", "input": "" }),
        ),
        // No `prompt` part — an ordinary transcription upload.
        "/v1/audio/transcriptions" | "/v1/audio/translations" => {
            form(surface, &[("model", "census-openai"), ("file", "RIFF")])
        }
        "/v1/audio/speech" => json(
            surface,
            serde_json::json!({ "model": "census-openai", "input": "", "voice": "alloy" }),
        ),
        // The only fixture carrying text: `/v1/videos` rejects an empty
        // `prompt` at schema validation, so it has no contentless shape.
        "/v1/videos" => json(
            surface,
            serde_json::json!({ "model": "census-openai", "prompt": "a cat" }),
        ),
        // `"arguments": {}` — the reported MCP bypass.
        "/mcp" | "/mcp/" => jsonrpc(
            surface,
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": { "name": "census__tool", "arguments": {} },
            }),
        ),
        "/mcp/:server" => jsonrpc(
            "/mcp/census",
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": { "name": "tool", "arguments": {} },
            }),
        ),
        // `tasks/get` carries no message at all.
        "/a2a/:agent" => jsonrpc(
            "/a2a/census",
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tasks/get",
                "params": { "id": "task-1" },
            }),
        ),
        _ => return None,
    })
}

/// Every envelope on the proxy — OpenAI 422, the Anthropic error shape, the
/// JSON-RPC error object, and the `/mcp` `isError` tool result — renders its
/// refusal through `error::guardrail_block_message`, which names the firing
/// row. So one substring recognises a genuine guardrail refusal on all of
/// them, and cannot be satisfied by an unrelated 4xx (a missing field, an
/// unknown model, a dead upstream) — which is the failure mode a
/// status-code-only assertion would have.
fn refused_by_guardrail(body: &str) -> bool {
    body.contains(&format!("guardrail '{GUARDRAIL_ROW}'"))
}

#[tokio::test]
async fn enforced_surfaces_refuse_a_blocking_guardrail() {
    let mut missing_fixture = Vec::new();
    let mut not_refused = Vec::new();
    let router = census_router();

    for (surface, posture) in POSTURE {
        if !matches!(posture, Posture::Enforced) {
            continue;
        }
        let Some(request) = fixture(surface) else {
            missing_fixture.push(*surface);
            continue;
        };
        let response = router
            .clone()
            .oneshot(request)
            .await
            .expect("router must answer");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("body must read");
        let body = String::from_utf8_lossy(&bytes).into_owned();
        if !refused_by_guardrail(&body) {
            not_refused.push(format!("{surface} -> {status}: {body}"));
        }
    }

    assert!(
        missing_fixture.is_empty(),
        "these surfaces are declared Posture::Enforced but have no fixture in `fixture()`, so \
         nothing actually checks them: {missing_fixture:?}",
    );
    assert!(
        not_refused.is_empty(),
        "a guardrail that blocks unconditionally did NOT refuse these surfaces — the input chain \
         either did not run or did not decide:\n{}",
        not_refused.join("\n"),
    );
}

/// AISIX-Cloud#1435: refusing is half the job — the refusal also has to be
/// REPORTED, and on the same router-derived set.
///
/// `guardrail_blocked_telemetry` already pins this invariant, but against a
/// hand-written list of surfaces, and that is exactly how the gap it was
/// written for came back: `/v1/messages/count_tokens` gained the chain
/// (#1064) and the flag (#1065) in the same release, was absent from the
/// list, and emitted no usage event at all — so it refused correctly and
/// the refusal was unfindable, on a route whose whole job is to ship the
/// caller's entire payload to a provider. Here the set comes out of the
/// router, so a surface cannot be missing from it.
///
/// The counters are asserted alongside because a refusal that BILLS is the
/// other way to get this wrong: nothing ran upstream, so nothing is owed.
#[tokio::test]
async fn an_enforced_surface_reports_the_refusal_it_makes() {
    let mut wrong = Vec::new();

    for (surface, posture) in POSTURE {
        if !matches!(posture, Posture::Enforced) {
            continue;
        }
        let Some(request) = fixture(surface) else {
            // `enforced_surfaces_refuse_a_blocking_guardrail` owns the
            // missing-fixture complaint; do not duplicate it here.
            continue;
        };
        let (router, mut rx) = census_router_with_usage();
        let response = router.oneshot(request).await.expect("router must answer");
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("body must read");
        if !refused_by_guardrail(&String::from_utf8_lossy(&bytes)) {
            // Ditto: a surface that did not refuse is the sibling test's
            // finding, and reporting it twice buries the new one.
            continue;
        }

        // Drained on a short timeout rather than counted — how many events
        // a surface emits is its own business, and pinning it here would
        // make this file fail for reasons that are not the flag.
        let mut events = Vec::new();
        while let Ok(Some(event)) =
            tokio::time::timeout(std::time::Duration::from_millis(300), rx.recv()).await
        {
            events.push(event);
        }

        if events.is_empty() {
            wrong.push(format!("{surface}: refused but emitted no usage event"));
            continue;
        }
        if !events.iter().any(|e| e.guardrail_blocked) {
            wrong.push(format!(
                "{surface}: emitted {} usage event(s), none marked guardrail_blocked",
                events.len(),
            ));
            continue;
        }
        // No surface is exempt here, `/a2a` included —
        // `guardrail_blocked_telemetry` has to exempt it because its
        // counters are the gateway's own reading of the request text,
        // filled before the chain runs, and that file's fixtures carry
        // real text. These are contentless by construction, so zero is
        // the honest answer on every one of them and an exemption would
        // protect nothing while hiding a surface that started billing.
        for event in &events {
            if event.prompt_tokens != 0 || event.completion_tokens != 0 {
                wrong.push(format!(
                    "{surface}: refused request billed {}+{} tokens",
                    event.prompt_tokens, event.completion_tokens,
                ));
            }
        }
    }

    assert!(
        wrong.is_empty(),
        "a guardrail refusal must reach the Logs \"Guardrail blocks\" view \
         (usage_events.guardrail_blocked = true) on every enforced surface, and cost \
         the caller nothing:\n  {}",
        wrong.join("\n  "),
    );
}

/// Drive every enforced surface with `guardrails` holding `row` — or
/// nothing at all when `row` is `None` — and return the surfaces that
/// answered with a guardrail refusal.
async fn surfaces_refused_with(row: Option<serde_json::Value>) -> Vec<&'static str> {
    let snap = census_snapshot();
    snap.guardrails.remove("g-census");
    if let Some(row) = row {
        let guardrail: aisix_core::Guardrail =
            serde_json::from_value(row).expect("valid guardrail");
        crate::seed_env_scoped_guardrail(&snap, ResourceEntry::new("g-census", guardrail, 2));
    }
    let handle = SnapshotHandle::new(snap);
    let index = aisix_guardrails::LiveGuardrailIndex::new(handle.clone(), None);
    let state = crate::ProxyState::new(handle, census_hub(), &cfg())
        .without_cache()
        .with_guardrail_index(index);
    let router = crate::build_router(state);

    let mut refused = Vec::new();
    for (surface, posture) in POSTURE {
        if !matches!(posture, Posture::Enforced) {
            continue;
        }
        let Some(request) = fixture(surface) else {
            continue;
        };
        let response = router
            .clone()
            .oneshot(request)
            .await
            .expect("router must answer");
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("body must read");
        if refused_by_guardrail(&String::from_utf8_lossy(&bytes)) {
            refused.push(*surface);
        }
    }
    refused
}

/// The other half of the rule, and the reason the fix is not simply
/// "always block when there is no text": a guardrail whose verdict comes
/// FROM the text has matched nothing, and a textless request is clean to
/// it. Whether the chain runs is the call site's business; what it decides
/// is the guardrail's.
#[tokio::test]
async fn a_text_matching_guardrail_leaves_textless_requests_alone() {
    let refused = surfaces_refused_with(Some(serde_json::json!({
        "name": GUARDRAIL_ROW,
        "enabled": true,
        "kind": "keyword",
        "hook_point": "input",
        "patterns": [{ "kind": "literal", "value": "census-forbidden-token" }],
    })))
    .await;
    assert!(
        refused.is_empty(),
        "a keyword rule that matched nothing refused these surfaces — removing the \
         empty-text short-circuits must not turn 'no text' into a blanket block: {refused:?}",
    );
}

/// The census is only meaningful if the fixtures would otherwise succeed
/// past the guardrail gate. With no guardrail configured, none of them may
/// answer with a guardrail refusal — otherwise the test above could be
/// passing on some unrelated error text.
#[tokio::test]
async fn fixtures_do_not_self_refuse_without_a_guardrail() {
    let snap = census_snapshot();
    snap.guardrails.remove("g-census");
    let handle = SnapshotHandle::new(snap);
    let state = crate::ProxyState::new(handle, census_hub(), &cfg()).without_cache();
    let router = crate::build_router(state);

    for (surface, posture) in POSTURE {
        if !matches!(posture, Posture::Enforced) {
            continue;
        }
        let Some(request) = fixture(surface) else {
            continue;
        };
        let response = router
            .clone()
            .oneshot(request)
            .await
            .expect("router must answer");
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("body must read");
        let body = String::from_utf8_lossy(&bytes).into_owned();
        assert!(
            !refused_by_guardrail(&body),
            "{surface} reported a guardrail refusal with no guardrail configured: {body}",
        );
    }
}
