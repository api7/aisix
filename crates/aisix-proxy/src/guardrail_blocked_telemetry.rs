//! Cross-handler guard for one invariant: when a guardrail refuses a
//! request, the request's terminal `UsageEvent` says so.
//!
//! `UsageEvent::guardrail_blocked` is not decorative. cp-api indexes it
//! (`idx_dpmgr_usage_events_env_blocked`) and the dashboard's Logs
//! "Guardrail blocks" view is the exact predicate `guardrail_blocked =
//! true`, so a refusal emitted without it is a 422 the caller definitely
//! saw and the operator cannot find. The failure mode is worse than a
//! missing row: the request IS in the unfiltered feed, so the Blocked view
//! coming back empty reads as "the gateway logged no guardrail activity",
//! which is how AISIX-Cloud#1428 was reported.
//!
//! The flag was set on `/v1/chat/completions` and `/mcp` and nowhere else.
//! Every other handler builds its failure event through a different
//! emitter — `usage_attr::build_error_usage_event` for the single-attempt
//! family, `responses::emit_zero_token_event` and
//! `messages::emit_anthropic_usage_event` for the two retrying ones — and
//! each left the field at its `false` default. So this file drives the
//! surfaces themselves rather than any one emitter: an emitter test would
//! have passed for chat while nine siblings were wrong.
//!
//! Each surface is driven twice against the same keyword guardrail — once
//! with the blocking literal in the field a caller writes, once without.
//! The second run is what makes the first mean anything: these fixtures
//! point at a dead upstream, so a clean request fails too, and a flag that
//! merely tracked "the request failed" would pass the blocked run and fail
//! the clean one.
//!
//! The list below is hand-written, and AISIX-Cloud#1435 is what a
//! hand-written list costs: `/v1/messages/count_tokens` gained the chain
//! and the flag in the same release, was not on it, and emitted no usage
//! event at all. So the "no surface may be missing" half now lives in
//! `guardrail_coverage`, whose set is parsed out of the router — add a
//! route and it is checked whether or not anyone edits a list. What stays
//! here is what that census cannot express: the CLEAN control above, which
//! needs a text-dependent guardrail rather than the census's unconditional
//! script, and `/passthrough/byo`, which needs a configured route prefix
//! the census snapshot does not carry.

use std::sync::Arc;

use aisix_core::snapshot::SnapshotHandle;
use aisix_core::{AisixSnapshot, ApiKey, ProxyConfig, ResourceEntry};
use aisix_obs::{UsageEvent, UsageSink};
use axum::body::Body;
use axum::http::Request;
use tower::ServiceExt;

/// The literal the guardrail row below refuses.
const BLOCK: &str = "BLOCKME";
const CALLER: &str = "sk-caller";
/// SHA-256 of `CALLER`.
const CALLER_HASH: &str = "8b6712790a2089c67aa97a2d80022df18cc65c7814350e33baebe79aab508891";
const PK_ID: &str = "11111111-1111-1111-1111-111111111111";
const ANTHROPIC_PK_ID: &str = "22222222-2222-2222-2222-222222222222";

/// A port nothing listens on, so a request that gets past the guardrail
/// gate fails at the network instead of hanging. That failure is the point
/// of the clean run: it produces an error event whose flag must stay false.
const DEAD_UPSTREAM: &str = "http://127.0.0.1:1";

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

fn snapshot() -> AisixSnapshot {
    let snap = AisixSnapshot::new();

    let key: ApiKey = serde_json::from_value(serde_json::json!({
        "key_hash": CALLER_HASH,
        "allowed_models": ["*"],
        "allowed_routes": ["*"],
        "allowed_agents": ["*"],
    }))
    .expect("valid api key");
    snap.apikeys.insert(ResourceEntry::new("ak-1", key, 1));

    for (id, name, provider, adapter, base) in [
        (
            PK_ID,
            "openai-up",
            "openai",
            "openai",
            format!("{DEAD_UPSTREAM}/v1"),
        ),
        (
            ANTHROPIC_PK_ID,
            "anthropic-up",
            "anthropic",
            "anthropic",
            DEAD_UPSTREAM.to_string(),
        ),
    ] {
        let pk: aisix_core::ProviderKey = serde_json::from_value(serde_json::json!({
            "display_name": name,
            "secret": "sk-unused",
            "api_base": base,
            "provider": provider,
            "adapter": adapter,
        }))
        .expect("valid provider key");
        snap.provider_keys.insert(ResourceEntry::new(id, pk, 1));
    }

    for (id, name, provider, model_name, pk_id, kind) in [
        (
            "m-openai",
            "gpt-under-test",
            "openai",
            "gpt-4o",
            PK_ID,
            None,
        ),
        (
            "m-anthropic",
            "claude-under-test",
            "anthropic",
            "claude-3-haiku-20240307",
            ANTHROPIC_PK_ID,
            None,
        ),
        (
            "m-embedding",
            "embed-under-test",
            "openai",
            "text-embedding-3-small",
            PK_ID,
            Some("embedding"),
        ),
    ] {
        let mut value = serde_json::json!({
            "display_name": name,
            "provider": provider,
            "model_name": model_name,
            "provider_key_id": pk_id,
        });
        if let Some(kind) = kind {
            value["kind"] = serde_json::Value::String(kind.to_string());
        }
        let model: aisix_core::Model = serde_json::from_value(value).expect("valid model");
        snap.models.insert(ResourceEntry::new(id, model, 1));
    }

    let agent: aisix_core::A2aAgent = serde_json::from_value(serde_json::json!({
        "name": "agent-under-test",
        "url": format!("{DEAD_UPSTREAM}/a2a"),
        "enabled": true,
    }))
    .expect("valid a2a agent");
    snap.a2a_agents
        .insert(ResourceEntry::new("agent-1", agent, 1));

    let route: aisix_core::PassthroughRoute = serde_json::from_value(serde_json::json!({
        "name": "byo-tunnel",
        "path_prefix": "/passthrough/byo",
        "target_url": DEAD_UPSTREAM,
        "provider_key_id": PK_ID,
    }))
    .expect("valid passthrough route");
    snap.passthrough_routes
        .insert(ResourceEntry::new("route-1", route, 1));

    // Env-scoped (no attachment row → the backward-compat fallback applies
    // it to every request), input hook, fail-closed. A keyword row is the
    // deterministic stand-in for any input-hook kind and, unlike a
    // text-independent script, keeps the clean run genuinely clean.
    let guardrail: aisix_core::Guardrail = serde_json::from_value(serde_json::json!({
        "name": "block-literal",
        "enabled": true,
        "kind": "keyword",
        "hook_point": "input",
        "fail_open": false,
        "patterns": [{ "kind": "literal", "value": BLOCK }],
    }))
    .expect("valid guardrail");
    snap.guardrails
        .insert(ResourceEntry::new("g-1", guardrail, 1));

    snap
}

/// Build the router plus the receiver its usage events land in.
fn router() -> (axum::Router, tokio::sync::mpsc::Receiver<UsageEvent>) {
    let hub = Arc::new(aisix_gateway::Hub::new());
    hub.register_specialized(
        "openai",
        Arc::new(aisix_provider_openai::OpenAiBridge::new()),
    );
    hub.register_specialized(
        "anthropic",
        Arc::new(aisix_provider_anthropic::AnthropicBridge::new()),
    );
    let (tx, rx) = tokio::sync::mpsc::channel(32);
    let state = crate::ProxyState::new(SnapshotHandle::new(snapshot()), hub, &cfg())
        .without_cache()
        .with_usage_sink(UsageSink::new(tx));
    (crate::build_router(state), rx)
}

/// One driveable request per surface, with `text` in the field a caller
/// authors — the field an input guardrail screens.
fn fixture(surface: &str, text: &str) -> Request<Body> {
    let json = |uri: &str, body: serde_json::Value| {
        Request::builder()
            .method("POST")
            .uri(uri.to_string())
            .header("authorization", format!("Bearer {CALLER}"))
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    };
    match surface {
        "/v1/chat/completions" => json(
            surface,
            serde_json::json!({
                "model": "gpt-under-test",
                "messages": [{ "role": "user", "content": text }],
            }),
        ),
        "/v1/completions" => json(
            surface,
            serde_json::json!({ "model": "gpt-under-test", "prompt": text }),
        ),
        "/v1/responses" => json(
            surface,
            serde_json::json!({ "model": "gpt-under-test", "input": text }),
        ),
        "/v1/messages" => Request::builder()
            .method("POST")
            .uri(surface)
            .header("x-api-key", CALLER)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "model": "claude-under-test",
                    "max_tokens": 16,
                    "messages": [{ "role": "user", "content": text }],
                })
                .to_string(),
            ))
            .unwrap(),
        "/v1/embeddings" => json(
            surface,
            serde_json::json!({ "model": "embed-under-test", "input": text }),
        ),
        "/v1/rerank" => json(
            surface,
            serde_json::json!({
                "model": "gpt-under-test",
                "query": text,
                "documents": ["a document"],
            }),
        ),
        "/v1/images/generations" => json(
            surface,
            serde_json::json!({ "model": "gpt-under-test", "prompt": text }),
        ),
        "/v1/audio/speech" => json(
            surface,
            serde_json::json!({ "model": "gpt-under-test", "input": text, "voice": "alloy" }),
        ),
        "/v1/videos" => json(
            surface,
            serde_json::json!({ "model": "gpt-under-test", "prompt": text }),
        ),
        // JSON-RPC rather than an OpenAI envelope: the screened text is
        // `params.message`, the only caller-authored content A2A carries.
        "/a2a" => Request::builder()
            .method("POST")
            .uri("/a2a/agent-under-test")
            .header("authorization", format!("Bearer {CALLER}"))
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .body(Body::from(
                serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [{ "kind": "text", "text": text }],
                            "messageId": "m-1",
                        },
                    },
                })
                .to_string(),
            ))
            .unwrap(),
        // The passthrough tunnel forwards the body verbatim, so its
        // screened text is whatever the detected envelope carries.
        "/passthrough/byo" => json(
            "/passthrough/byo/v1/chat/completions",
            serde_json::json!({
                "model": "gpt-4o",
                "messages": [{ "role": "user", "content": text }],
            }),
        ),
        other => panic!("no fixture for {other}"),
    }
}

/// Every surface whose input hook a keyword row can reach today, i.e. every
/// one that carries caller-authored text in a JSON field.
///
/// The multipart surfaces (`/v1/audio/transcriptions`, `/v1/images/edits`)
/// and the blob-scanning job surfaces are deliberately absent: their
/// screened text is an uploaded file, so a keyword fixture would assert
/// nothing they don't already share with `/v1/audio/speech` — all four go
/// through the same `usage_attr::build_error_usage_event` emitter this
/// list already covers three times over.
const SURFACES: &[&str] = &[
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/v1/messages",
    "/v1/embeddings",
    "/v1/rerank",
    "/v1/images/generations",
    "/v1/audio/speech",
    "/v1/videos",
    "/passthrough/byo",
    "/a2a",
];

/// Drive one fixture and collect every usage event it emitted.
///
/// Events are drained on a short timeout rather than counted: the number a
/// surface emits is its own business (a retrying family emits one per
/// failed attempt), and pinning it here would make this file fail for
/// reasons that have nothing to do with the flag.
async fn drive(surface: &str, text: &str) -> (u16, Vec<UsageEvent>) {
    let (router, mut rx) = router();
    let response = router
        .oneshot(fixture(surface, text))
        .await
        .expect("router must answer");
    let status = response.status().as_u16();
    // Drain the body: a streaming surface emits from its end-of-stream
    // guard, which only runs once the body is polled to completion.
    let _ = axum::body::to_bytes(response.into_body(), 1 << 20).await;

    let mut events = Vec::new();
    while let Ok(Some(event)) =
        tokio::time::timeout(std::time::Duration::from_millis(300), rx.recv()).await
    {
        events.push(event);
    }
    (status, events)
}

#[tokio::test]
async fn a_guardrail_refusal_is_marked_on_every_surface() {
    let mut wrong = Vec::new();
    for surface in SURFACES {
        let (status, events) = drive(surface, &format!("please {BLOCK} now")).await;
        if status != 422 {
            wrong.push(format!("{surface}: refused with {status}, expected 422"));
            continue;
        }
        if events.is_empty() {
            wrong.push(format!("{surface}: refused but emitted no usage event"));
            continue;
        }
        // The refusal is request-scoped, so it rides the request's terminal
        // event. These fixtures refuse before any upstream is contacted, so
        // that event is the only one.
        if !events.iter().any(|e| e.guardrail_blocked) {
            wrong.push(format!(
                "{surface}: emitted {} usage event(s), none marked guardrail_blocked",
                events.len()
            ));
            continue;
        }
        // A refusal costs the caller nothing: no upstream ran. `/a2a` is
        // exempt because its counters are the gateway's own reading of the
        // words, flagged `usage_estimated` and never charged — they are
        // filled from the request before the chain even runs.
        if *surface != "/a2a" {
            for event in &events {
                if event.prompt_tokens != 0 || event.completion_tokens != 0 {
                    wrong.push(format!(
                        "{surface}: refused request billed {}+{} tokens",
                        event.prompt_tokens, event.completion_tokens
                    ));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "a guardrail refusal must reach the Logs \"Guardrail blocks\" view \
         (usage_events.guardrail_blocked = true) on every surface:\n  {}",
        wrong.join("\n  "),
    );
}

#[tokio::test]
async fn an_ordinary_failure_is_not_marked_as_a_guardrail_block() {
    let mut wrong = Vec::new();
    for surface in SURFACES {
        // Same guardrail, same fixtures, text it does not match — so the
        // request runs on and dies at the dead upstream instead.
        let (status, events) = drive(surface, "a perfectly ordinary question").await;
        if status == 422 {
            wrong.push(format!("{surface}: clean text was refused ({status})"));
            continue;
        }
        // Without this the control is vacuous: a fixture that stopped
        // before the handler ran — a stale key hash, a rejected body, a
        // missing model row — emits nothing, and "no event is marked" is
        // trivially satisfied by having no event.
        if events.is_empty() {
            wrong.push(format!(
                "{surface}: clean run emitted no usage event, so nothing was checked"
            ));
            continue;
        }
        if let Some(event) = events.iter().find(|e| e.guardrail_blocked) {
            wrong.push(format!(
                "{surface}: {} marked guardrail_blocked on a {} that no guardrail refused",
                event.error_class, event.status_code,
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "guardrail_blocked must track guardrail refusals, not failures in general — \
         a flag set by every 4xx/5xx makes the Blocked view useless:\n  {}",
        wrong.join("\n  "),
    );
}
