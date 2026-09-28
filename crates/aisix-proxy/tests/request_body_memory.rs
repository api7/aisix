//! Bytes the gateway holds per byte of request body while a proxied request
//! waits on its upstream.
//!
//! A request carrying tens of MB of base64 images is sent through the real
//! router to a mock upstream that reads the whole body, discards it, and
//! then withholds its response. While it withholds, the process's live heap
//! (counted by the allocator below, by allocated size) is compared with what
//! it was before the request body existed. Everything the gateway keeps for
//! that one request — the parsed request it retries from, the outbound body,
//! any copy in between — shows up in the difference, so the ratio to the
//! body length is the in-flight multiplier.
//!
//! One test function drives every family in turn: the counter is
//! process-wide, so two measurements must never overlap.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use aisix_core::resource::ResourceEntry;
use aisix_core::snapshot::SnapshotHandle;
use aisix_core::{AisixSnapshot, ApiKey, Model, ProxyConfig};
use aisix_gateway::Hub;
use aisix_provider_anthropic::AnthropicBridge;
use aisix_provider_openai::OpenAiBridge;
use aisix_proxy::{build_router, ProxyState};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use futures::StreamExt;
use tokio::sync::{mpsc, watch};
use tower::ServiceExt;

struct CountingAllocator;

static LIVE: AtomicIsize = AtomicIsize::new(0);

thread_local! {
    /// Set on the mock upstream's own threads: its connection buffers are
    /// not the gateway's.
    static UNCOUNTED: Cell<bool> = const { Cell::new(false) };
}

fn count(delta: isize) {
    if !UNCOUNTED.with(Cell::get) {
        LIVE.fetch_add(delta, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size() as isize);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size() as isize);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size as isize - layout.size() as isize);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        count(-(layout.size() as isize));
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn live() -> isize {
    LIVE.load(Ordering::Relaxed)
}

/// Room above the bound for bookkeeping. Every whole extra copy of the
/// request costs a full 1.0.
const MARGIN: f64 = 0.1;

const PK_ID: &str = "11111111-1111-1111-1111-111111111111";
const ANTHROPIC_PK_ID: &str = "22222222-2222-2222-2222-222222222222";

/// A mock upstream that consumes each request body chunk by chunk without
/// keeping it, reports how many bytes it read, and — while the test holds
/// it — answers only once released. It runs on a runtime of its own whose
/// threads the allocator does not count.
struct HeldUpstream {
    base: String,
    received: mpsc::UnboundedReceiver<usize>,
    hold: watch::Sender<bool>,
}

fn start_upstream() -> HeldUpstream {
    let (tx, received) = mpsc::unbounded_channel();
    let (hold, held) = watch::channel(true);
    let app = Router::new().fallback({
        move |req: Request<Body>| {
            let tx = tx.clone();
            let mut held = held.clone();
            async move {
                let path = req.uri().path().to_string();
                let mut body = req.into_body().into_data_stream();
                let mut n = 0usize;
                while let Some(chunk) = body.next().await {
                    n += chunk.expect("upstream body chunk").len();
                }
                drop(body);
                let _ = tx.send(n);
                let _ = held.wait_for(|holding| !*holding).await;
                axum::Json(upstream_response(&path))
            }
        }
    });
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        UNCOUNTED.with(|u| u.set(true));
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move {
                let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                axum::serve(listener, app).await.unwrap()
            })
    });
    HeldUpstream {
        base: format!("http://{addr}"),
        received,
        hold,
    }
}

fn upstream_response(path: &str) -> serde_json::Value {
    if path.ends_with("/messages") {
        serde_json::json!({
            "id": "msg_upstream",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 1}
        })
    } else if path.ends_with("/responses") {
        serde_json::json!({
            "id": "resp_upstream",
            "object": "response",
            "status": "completed",
            "model": "gpt-4o",
            "output": [{
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}]
            }],
            "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
        })
    } else if path.ends_with("/embeddings") {
        serde_json::json!({
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        })
    } else {
        serde_json::json!({
            "id": "cmpl-upstream",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
        })
    }
}

fn snapshot(api_base: &str) -> AisixSnapshot {
    let snap = AisixSnapshot::new();
    for (pk_id, provider) in [(PK_ID, "openai"), (ANTHROPIC_PK_ID, "anthropic")] {
        let pk: aisix_core::ProviderKey = serde_json::from_str(&format!(
            r#"{{"display_name":"{provider}-up","secret":"sk-upstream","api_base":"{api_base}","provider":"{provider}","adapter":"{provider}"}}"#
        ))
        .unwrap();
        snap.provider_keys.insert(ResourceEntry::new(pk_id, pk, 1));
    }
    for (id, name, provider, upstream, pk_id) in [
        ("model-chat", "vision", "openai", "gpt-4o", PK_ID),
        (
            "model-embed",
            "embedder",
            "openai",
            "text-embedding-3-small",
            PK_ID,
        ),
        (
            "model-claude",
            "claude",
            "anthropic",
            "claude-sonnet-4-5",
            ANTHROPIC_PK_ID,
        ),
        ("model-rerank", "reranker", "openai", "rerank-v3", PK_ID),
        ("model-chat-b", "vision-b", "openai", "gpt-4o-mini", PK_ID),
    ] {
        let model: Model = serde_json::from_str(&format!(
            r#"{{"display_name":"{name}","provider":"{provider}","model_name":"{upstream}","provider_key_id":"{pk_id}"}}"#
        ))
        .unwrap();
        snap.models.insert(ResourceEntry::new(id, model, 1));
    }
    let key: ApiKey = serde_json::from_str(&format!(
        r#"{{"key_hash":"{}","allowed_models":["vision","embedder","claude","reranker","effort","panel"]}}"#,
        ApiKey::hash_bearer("sk-caller")
    ))
    .unwrap();
    snap.apikeys.insert(ResourceEntry::new("key-1", key, 1));
    // A model whose effort mapping rewrites every request it serves.
    let effort: Model = serde_json::from_str(&format!(
        r#"{{"display_name":"effort","provider":"openai","model_name":"gpt-4o","provider_key_id":"{PK_ID}","effort_mapping":{{"":"low"}}}}"#
    ))
    .unwrap();
    snap.models
        .insert(ResourceEntry::new("model-effort", effort, 1));
    let panel: Model = serde_json::from_str(
        r#"{"display_name":"panel","ensemble":{"panel":[{"model":"vision"},{"model":"vision-b"}],"judge":{"model":"vision"}}}"#,
    )
    .unwrap();
    snap.models
        .insert(ResourceEntry::new("model-panel", panel, 1));
    // An exporter that captures full content, so every route also builds
    // its captured prompt: that copy must not be held across the wait
    // either. Nothing is delivered — the proxy only reads the setting.
    let exporter: aisix_core::models::ObservabilityExporter = serde_json::from_str(
        r#"{"name":"capture","enabled":true,"kind":"otlp_http","endpoint":"http://127.0.0.1:9/v1/traces","content_mode":"full","content_max_bytes":1048576}"#,
    )
    .unwrap();
    snap.observability_exporters
        .insert(ResourceEntry::new("exporter-1", exporter, 1));
    snap
}

fn router(api_base: &str) -> Router {
    let hub = Arc::new(Hub::new());
    hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
    hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
    let cfg = ProxyConfig {
        addr: "127.0.0.1:0".into(),
        request_body_limit_bytes: 256 * 1024 * 1024,
        real_ip: Default::default(),
        request_id: Default::default(),
        url_rewrites: Vec::new(),
        tls: None,
        listeners: Vec::new(),
        thread_per_core: None,
        workers: None,
    };
    build_router(
        ProxyState::new(SnapshotHandle::new(snapshot(api_base)), hub, &cfg).without_cache(),
    )
}

/// Base64 text of `len` bytes that no allocator or parser can share with
/// another image: every image differs.
fn base64_blob(len: usize, seed: usize) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut x = (seed as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ALPHABET[(x % 64) as usize] as char
        })
        .collect()
}

/// Builds a request body of `(blocks, bytes per block)`.
type BodyFn = fn((usize, usize)) -> String;

/// The measured request: 16 images of 1.3 MB base64 each (~21 MB).
const LARGE: (usize, usize) = (16, 1_300_000);
/// The same request shape at a size where only per-request overhead —
/// connection buffers, task state — shows, subtracted from the large one.
const SMALL: (usize, usize) = (1, 1_000);

fn chat_body(size: (usize, usize)) -> String {
    chat_body_for("vision", size)
}

fn effort_mapped_chat_body(size: (usize, usize)) -> String {
    chat_body_for("effort", size)
}

fn ensemble_body(size: (usize, usize)) -> String {
    chat_body_for("panel", size)
}

/// An Anthropic request addressed to the OpenAI model: translated into
/// the gateway's chat format for the OpenAI bridge on every attempt.
fn messages_to_openai_body(size: (usize, usize)) -> String {
    messages_body_for("vision", size)
}

/// A Responses request addressed to the Anthropic model, which does not
/// serve the Responses API: translated for the Anthropic bridge. That
/// bridge does not carry images across, so the bulk here is text.
fn responses_to_anthropic_body((parts, part_bytes): (usize, usize)) -> String {
    let content: Vec<_> = (0..parts)
        .map(|i| serde_json::json!({"type": "input_text", "text": base64_blob(part_bytes, i)}))
        .collect();
    serde_json::json!({
        "model": "claude",
        "max_output_tokens": 64,
        "input": [{"role": "user", "content": content}]
    })
    .to_string()
}

/// An OpenAI-shape request addressed to the Anthropic model, so the
/// Anthropic bridge translates it into an owned Anthropic request. That
/// bridge does not carry images across, so the bulk here is text, sent as
/// plain string messages.
fn chat_to_anthropic_body((messages, message_bytes): (usize, usize)) -> String {
    let messages: Vec<_> = (0..messages)
        .map(|i| {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            serde_json::json!({"role": role, "content": base64_blob(message_bytes, i)})
        })
        .chain(std::iter::once(
            serde_json::json!({"role": "user", "content": "Summarise the above."}),
        ))
        .collect();
    serde_json::json!({"model": "claude", "max_tokens": 64, "messages": messages}).to_string()
}

fn chat_body_for(model: &str, (images, image_bytes): (usize, usize)) -> String {
    let mut content = vec![serde_json::json!({"type": "text", "text": "Describe these images."})];
    for i in 0..images {
        content.push(serde_json::json!({
            "type": "image_url",
            "image_url": {"url": format!("data:image/jpeg;base64,{}", base64_blob(image_bytes, i))}
        }));
    }
    serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": content}]
    })
    .to_string()
}

fn messages_body(size: (usize, usize)) -> String {
    messages_body_for("claude", size)
}

fn messages_body_for(model: &str, (images, image_bytes): (usize, usize)) -> String {
    let mut content = vec![serde_json::json!({"type": "text", "text": "Describe these images."})];
    for i in 0..images {
        content.push(serde_json::json!({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_blob(image_bytes, i)}
        }));
    }
    serde_json::json!({
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": content}]
    })
    .to_string()
}

fn responses_body((images, image_bytes): (usize, usize)) -> String {
    let mut content =
        vec![serde_json::json!({"type": "input_text", "text": "Describe these images."})];
    for i in 0..images {
        content.push(serde_json::json!({
            "type": "input_image",
            "image_url": format!("data:image/jpeg;base64,{}", base64_blob(image_bytes, i))
        }));
    }
    serde_json::json!({
        "model": "vision",
        "input": [{"role": "user", "content": content}]
    })
    .to_string()
}

fn rerank_body((documents, document_bytes): (usize, usize)) -> String {
    let documents: Vec<String> = (0..documents)
        .map(|i| base64_blob(document_bytes, i))
        .collect();
    serde_json::json!({"model": "reranker", "query": "which one?", "documents": documents})
        .to_string()
}

fn embeddings_body((images, image_bytes): (usize, usize)) -> String {
    let input: Vec<String> = (0..images).map(|i| base64_blob(image_bytes, i)).collect();
    serde_json::json!({"model": "embedder", "input": input}).to_string()
}

/// Send `body` to `path` and, once `requests` upstream calls have arrived
/// and are being withheld, return the live-heap growth since before the
/// body existed, together with the body length.
async fn held_while_pending(
    app: &Router,
    upstream: &mut HeldUpstream,
    path: &str,
    requests: usize,
    make_body: impl FnOnce() -> String,
) -> (isize, usize) {
    upstream.hold.send_replace(true);
    let before = live();
    let body = make_body();
    let body_len = body.len();
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header("authorization", "Bearer sk-caller")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let call = tokio::spawn(app.clone().oneshot(req));

    for _ in 0..requests {
        let received = tokio::time::timeout(Duration::from_secs(60), upstream.received.recv())
            .await
            .expect("upstream never received the request")
            .unwrap();
        assert!(
            received >= body_len / 2,
            "{path}: upstream read only {received} bytes"
        );
    }
    // Let anything transient from the send settle before sampling.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let held = live() - before;

    upstream.hold.send_replace(false);
    let resp = call.await.unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "{path}");
    let _ = axum::body::to_bytes(resp.into_body(), usize::MAX).await;
    // Later calls of the same request (an ensemble's judge) answered
    // without being held; drain their receipts.
    while upstream.received.try_recv().is_ok() {}
    (held, body_len)
}

/// Bytes held per byte of request body: the large request's growth less
/// the small one's, over the difference in their lengths.
async fn in_flight_multiplier(app: &Router, upstream: &mut HeldUpstream, family: &Family) -> f64 {
    let (small_held, small_len) =
        held_while_pending(app, upstream, family.path, family.requests, || {
            (family.body)(SMALL)
        })
        .await;
    let (large_held, large_len) =
        held_while_pending(app, upstream, family.path, family.requests, || {
            (family.body)(LARGE)
        })
        .await;
    (large_held - small_held) as f64 / (large_len - small_len) as f64
}

struct Family {
    name: &'static str,
    path: &'static str,
    body: BodyFn,
    /// Upstream calls the request makes at once — each carries its own
    /// wire copy of the body, so the bound grows by one per call.
    requests: usize,
}

impl Family {
    const fn new(name: &'static str, path: &'static str, body: BodyFn) -> Self {
        Self {
            name,
            path,
            body,
            requests: 1,
        }
    }

    /// The parsed request the handler keeps, plus one wire body per
    /// upstream call in flight.
    fn limit(&self) -> f64 {
        1.0 + self.requests as f64 + MARGIN
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_body_is_held_at_most_twice_while_upstream_is_pending() {
    let mut upstream = start_upstream();
    let app = router(&upstream.base);

    // Bound per family: the parsed request the handler keeps for retry,
    // fail-over and Model Group routing, plus the wire body reqwest holds
    // until the response head — 2x the body for a single upstream call.
    let families = [
        Family::new("chat -> openai", "/v1/chat/completions", chat_body),
        Family::new(
            "chat -> anthropic",
            "/v1/chat/completions",
            chat_to_anthropic_body,
        ),
        // The effort mapping rewrites a copy of the request per attempt.
        Family::new(
            "chat, effort-mapped",
            "/v1/chat/completions",
            effort_mapped_chat_body,
        ),
        // Two panel members in flight at once, each with its own request.
        Family {
            requests: 2,
            ..Family::new("ensemble panel", "/v1/chat/completions", ensemble_body)
        },
        Family::new("embeddings", "/v1/embeddings", embeddings_body),
        // Relayed to an Anthropic upstream as-is: the route builds the
        // outbound body itself rather than through a provider bridge.
        Family::new("messages passthrough", "/v1/messages", messages_body),
        Family::new(
            "messages -> openai (bridged)",
            "/v1/messages",
            messages_to_openai_body,
        ),
        Family::new(
            "count_tokens passthrough",
            "/v1/messages/count_tokens",
            messages_body,
        ),
        // Served natively by the OpenAI upstream, likewise built by the route.
        Family::new("responses passthrough", "/v1/responses", responses_body),
        Family::new(
            "responses -> anthropic (bridged)",
            "/v1/responses",
            responses_to_anthropic_body,
        ),
        // Each attempt rewrites its own copy of the body for its target.
        Family::new("rerank", "/v1/rerank", rerank_body),
    ];

    // Warm lazily-initialised statics (TLS roots, regexes, metric
    // registries, the pooled upstream connections) so they are not charged
    // to a measurement.
    for family in &families {
        held_while_pending(&app, &mut upstream, family.path, family.requests, || {
            (family.body)(SMALL)
        })
        .await;
    }

    let mut report = Vec::new();
    for family in &families {
        let m = in_flight_multiplier(&app, &mut upstream, family).await;
        eprintln!(
            "{}: {m:.4}x request body held while upstream is pending (limit {:.1}x)",
            family.name,
            family.limit()
        );
        report.push((family, m));
    }
    for (family, m) in report {
        assert!(
            m <= family.limit(),
            "{} held {m:.2}x the request body while waiting on the upstream (limit {:.1}x)",
            family.name,
            family.limit()
        );
    }
}
