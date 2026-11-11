//! Semantic-routing runtime: embed the request, score it against each
//! route's cached example embeddings, and resolve a single direct-model
//! target (or the router's `default`).
//!
//! [`resolve`] produces a one-element `attempt_models` list that the
//! existing chat dispatch loop then drives exactly like a routing target —
//! so semantic routing reuses all of the streaming / failover / telemetry
//! machinery and only adds the "which target" decision on top.
//!
//! A router with a `classifier` block makes the same decision a different
//! way: one call to a hosted decision model ([`classify`]) picks a route by
//! its description. Everything after the pick — the member gates, the
//! failure policy, the served-route attribution — is shared.
//!
//! The scoring core ([`decide`], [`failure_target`]) and the
//! example-vector cache ([`SemanticVectorCache`]) are pure and unit-tested
//! in isolation; the async embedding call lives in [`resolve`].

use std::borrow::Cow;
use std::sync::Arc;

use dashmap::DashMap;

use aisix_core::models::{
    resolve_model_ref, EmbeddingFailureMode, OnEmbeddingFailure, Semantic, SemanticClassifier,
    SemanticRoute, TYPESAFE_PROVIDER,
};
use aisix_core::resource::ResourceEntry;
use aisix_core::{AisixSnapshot, Model};
use aisix_gateway::{EmbeddingRequest, EmbeddingUsageSource, EmbeddingVector};
use aisix_obs::{
    GatewayEmbeddingCall, GatewayEmbeddingOutcome, GatewayEmbeddingPurpose,
    GatewayEmbeddingUsageSource, SemanticAccessLog, SemanticFallback,
};

use crate::error::ProxyError;
use crate::routing::{AttemptModel, RoutingRequest};
use crate::session_affinity::Pin;
use crate::state::ProxyState;

// Scoring primitives are shared with the `kind: "semantic"` guardrail,
// which scores the same way against its example prototypes — one
// implementation in `aisix-core` so the two cannot drift on the
// degenerate cases (length mismatch, zero magnitude, empty prototype
// set).
pub(crate) use aisix_core::best_similarity;

/// Outcome of scoring a request against a semantic router.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RouteDecision {
    /// Index into `Semantic::routes` of the winning route, or `None` when
    /// no route cleared its threshold (the caller falls back to `default`).
    pub winner: Option<usize>,
    /// Per-route aggregated score, aligned with `Semantic::routes`.
    pub scores: Vec<f32>,
}

/// Score `request_vec` against each route's example vectors and pick the
/// highest-scoring route that clears its (per-route or router-level)
/// threshold. `max` aggregation: a route's score is its single
/// best-matching example. `route_example_vecs[i]` are the cached vectors
/// for `semantic.routes[i]`.
pub(crate) fn decide(
    semantic: &Semantic,
    request_vec: &[f32],
    route_example_vecs: &[Vec<Arc<Vec<f32>>>],
) -> RouteDecision {
    let mut scores = Vec::with_capacity(semantic.routes.len());
    for ex_vecs in route_example_vecs {
        // A route with no cached example vectors scores 0.0, not
        // `-inf`: it must lose to every real route without poisoning the
        // comparison.
        scores.push(
            best_similarity(request_vec, ex_vecs.iter().map(|v| v.as_slice())).unwrap_or(0.0),
        );
    }
    let mut winner: Option<usize> = None;
    let mut best = f32::NEG_INFINITY;
    for (i, route) in semantic.routes.iter().enumerate() {
        if scores[i] >= semantic.route_threshold(route) && scores[i] > best {
            best = scores[i];
            winner = Some(i);
        }
    }
    RouteDecision { winner, scores }
}

/// Why a semantic router could not make its decision: the closed vocabulary
/// [`ProxyError::SemanticDecisionFailed`] carries to the caller and the
/// access log. Free text about the failure goes only to the warn line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionFailure {
    ClassifierTimeout,
    /// The classifier answered a non-2xx status, or the call failed in
    /// transport.
    ClassifierUpstream,
    /// The classifier answered 2xx with no usable choice.
    ClassifierInvalidResponse,
    /// The classifier's provider key is missing or not a `typesafe` one.
    ClassifierUnavailable,
    EmbeddingTimeout,
    /// The embedding upstream answered an error, or the call failed in
    /// transport or decoding.
    EmbeddingUpstream,
    /// The embedding model is missing, misconfigured, or returned vectors
    /// that do not fit it.
    EmbeddingUnavailable,
}

impl DecisionFailure {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::ClassifierTimeout => "classifier_timeout",
            Self::ClassifierUpstream => "classifier_upstream",
            Self::ClassifierInvalidResponse => "classifier_invalid_response",
            Self::ClassifierUnavailable => "classifier_unavailable",
            Self::EmbeddingTimeout => "embedding_timeout",
            Self::EmbeddingUpstream => "embedding_upstream",
            Self::EmbeddingUnavailable => "embedding_unavailable",
        }
    }

    fn of_classifier_transport(err: &reqwest::Error) -> (Self, String) {
        let failure = if err.is_timeout() {
            Self::ClassifierTimeout
        } else {
            Self::ClassifierUpstream
        };
        (failure, aisix_gateway::transport_error_message(err))
    }

    fn of_embedding_error(err: &EmbedTextsError) -> Self {
        use aisix_gateway::BridgeError;
        match err {
            EmbedTextsError::Dispatch(ProxyError::Bridge(BridgeError::Timeout { .. })) => {
                Self::EmbeddingTimeout
            }
            EmbedTextsError::Dispatch(ProxyError::Bridge(
                BridgeError::UpstreamStatus { .. }
                | BridgeError::UpstreamInBand { .. }
                | BridgeError::UpstreamDecode(_)
                | BridgeError::UpstreamBodyUnread { .. }
                | BridgeError::TokenExchangeDecode(_)
                | BridgeError::Transport(_)
                | BridgeError::StreamAborted,
            )) => Self::EmbeddingUpstream,
            EmbedTextsError::Dispatch(_) | EmbedTextsError::UnusableResponse(_) => {
                Self::EmbeddingUnavailable
            }
        }
    }
}

impl std::fmt::Display for DecisionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Model alias (a direct model or a model group) to dispatch to when the router cannot decide — the
/// embedding call failed (`on_embedding_failure`), or the classifier call
/// did (`classifier.on_failure`). `None` means the policy is `fail` — the
/// caller returns `503`.
///
/// Each of the two aliases it can return is a model reference in its own
/// right, so both are resolved against the snapshot: an id spelling follows
/// a rename, and an id that resolves to nothing comes back as itself and
/// dispatches nowhere, exactly as a dangling alias does.
pub(crate) fn failure_target<'a>(
    snapshot: &AisixSnapshot,
    semantic: &'a Semantic,
) -> Option<Cow<'a, str>> {
    let policy = match &semantic.classifier {
        Some(classifier) => &classifier.on_failure,
        None => &semantic.on_embedding_failure,
    };
    match policy {
        OnEmbeddingFailure::Mode(EmbeddingFailureMode::Default) => {
            Some(semantic.default_ref(snapshot))
        }
        OnEmbeddingFailure::Mode(EmbeddingFailureMode::Fail) => None,
        OnEmbeddingFailure::Target { target, target_id } => {
            Some(resolve_model_ref(snapshot, target, target_id.as_deref()))
        }
    }
}

/// Per-instance cache of route example-utterance embeddings, populated
/// lazily on the first request that needs them and reused across requests
/// so a steady-state request's logical embedding batch contains only the
/// prompt. Keyed by `(embedding_model_id, dimensions, example_text)` —
/// changing the embedding model or its dimensions auto-invalidates, since
/// a stale vector of the wrong dimension must never be served.
#[derive(Debug, Default)]
pub struct SemanticVectorCache {
    vectors: DashMap<(String, u32, String), Arc<Vec<f32>>>,
}

impl SemanticVectorCache {
    /// Embeddings cached; nothing is ever evicted.
    pub fn entry_count(&self) -> u64 {
        self.vectors.len() as u64
    }

    pub(crate) fn get(
        &self,
        embedding_model_id: &str,
        dimensions: u32,
        text: &str,
    ) -> Option<Arc<Vec<f32>>> {
        self.vectors
            .get(&(embedding_model_id.to_string(), dimensions, text.to_string()))
            .map(|e| e.clone())
    }

    pub(crate) fn insert(
        &self,
        embedding_model_id: &str,
        dimensions: u32,
        text: &str,
        vec: Arc<Vec<f32>>,
    ) {
        self.vectors.insert(
            (embedding_model_id.to_string(), dimensions, text.to_string()),
            vec,
        );
    }
}

/// What a router's decision step produced, before the member gates run.
struct Pick {
    /// Index into `Semantic::routes` of the chosen route; `None` sends the
    /// request to `default`.
    winner: Option<usize>,
    /// The deciding score: the winning (or, with no winner, the best)
    /// route's cosine similarity, or the classifier's confidence.
    score: Option<f64>,
    /// Why there is no winner.
    miss: Option<SemanticFallback>,
}

/// What a semantic router decided to dispatch to.
pub(crate) struct SemanticDispatch {
    /// The attempts the dispatch loop drives: the one direct target, or
    /// the members a model-group target selected for this request.
    pub attempts: Vec<AttemptModel>,
    /// Name of the route that served the request; `None` when it fell
    /// through to `default` or to the failure target.
    pub route: Option<String>,
    /// The chosen target, when it is a model group: its routing knobs
    /// govern the attempts, exactly as when a caller addresses it directly.
    pub group: Option<Arc<ResourceEntry<Model>>>,
}

/// Resolve a semantic router to its dispatch: the attempts of the chosen
/// target (one direct model, or a model group's members) + the name of the
/// route that matched (`None` when the request fell through to `default`).
/// Every endpoint that honours a semantic router decides through here.
///
/// `prompt` is the request's classification text ([`classification_text`]).
/// `routing` carries what a model-group target needs to select its members.
/// How it decided is published for the request's access-log line
/// ([`crate::attribution::note_semantic_decision`]).
pub(crate) async fn resolve(
    state: &ProxyState,
    snapshot: &AisixSnapshot,
    router_entry: &ResourceEntry<Model>,
    prompt: &str,
    request_id: &str,
    routing: RoutingRequest<'_>,
    session: SessionRequest<'_>,
) -> Result<SemanticDispatch, ProxyError> {
    let semantic = router_entry
        .value
        .semantic
        .as_ref()
        .expect("resolve called on a non-semantic model");
    let router = &router_entry.value;

    // Every alias this function dispatches to is read through the model
    // reference helpers: a router that names its targets by id keeps
    // working across a rename of any of them, and an id that resolves to
    // nothing behaves as the dangling alias it stands in for.
    let default_target = semantic.default_ref(snapshot);

    let affinity = session_affinity_key(router_entry, semantic, routing, session);
    if let Some((key, ttl)) = &affinity {
        if let Some(hit) = state.session_pins.lookup(key, *ttl).await {
            match pinned_route(snapshot, semantic, &hit.pin) {
                Some(i) => {
                    let dispatch = serve_route(
                        state,
                        snapshot,
                        router,
                        routing,
                        semantic,
                        i,
                        &default_target,
                        None,
                        true,
                    )?;
                    state.session_pins.touch(key, hit, *ttl).await;
                    return Ok(dispatch);
                }
                None => state.session_pins.discard(key, &hit).await,
            }
        }
    }

    // No user text to classify (e.g. a system-only or tool-only request):
    // route to `default` without a decision call rather than deciding on
    // an empty string, which could spuriously match a route.
    if prompt.trim().is_empty() {
        note_decision(None, None, Some(SemanticFallback::EmptyPrompt));
        let eligible = select_eligible(state, snapshot, router, routing, &default_target, None)?;
        return Ok(eligible.into_dispatch(None));
    }

    let pick = match &semantic.classifier {
        Some(classifier) => match classify(snapshot, &semantic.routes, classifier, prompt).await {
            Ok((picked, confidence)) => {
                let confident = confidence >= classifier.min_confidence;
                Ok(Pick {
                    winner: confident.then_some(picked),
                    score: Some(confidence),
                    miss: (!confident).then_some(SemanticFallback::LowConfidence),
                })
            }
            Err((failure, error)) => {
                tracing::warn!(
                    router = %router.display_name,
                    reason = %failure,
                    error = %error,
                    "semantic classifier call failed; applying classifier.on_failure",
                );
                Err(failure)
            }
        },
        None => {
            decide_by_embedding(state, snapshot, router_entry, semantic, prompt, request_id).await
        }
    };
    let pick = match pick {
        Ok(pick) => pick,
        Err(failure) => {
            note_decision(None, None, Some(SemanticFallback::DecisionFailed));
            return fallback(state, snapshot, router, routing, semantic, failure);
        }
    };

    let dispatch = match pick.winner {
        Some(i) => {
            let dispatch = serve_route(
                state,
                snapshot,
                router,
                routing,
                semantic,
                i,
                &default_target,
                pick.score,
                false,
            )?;
            match (&affinity, &dispatch.route) {
                (Some((key, ttl)), Some(route)) if session.create_pins => {
                    let target_id = snapshot
                        .models
                        .get_by_name(&semantic.routes[i].target_ref(snapshot))
                        .map(|e| e.id.clone())
                        .expect("a served route's target resolves");
                    let own = Pin {
                        target_id,
                        route: route.clone(),
                    };
                    let pin = state
                        .session_pins
                        .create(key, own.clone(), *ttl, |p| {
                            pinned_route(snapshot, semantic, p).is_some()
                        })
                        .await;
                    if pin.target_id == own.target_id {
                        dispatch
                    } else {
                        // Another request of the session pinned it first:
                        // go where that one went.
                        let winner = pinned_route(snapshot, semantic, &pin)
                            .expect("create returns a usable winner");
                        serve_route(
                            state,
                            snapshot,
                            router,
                            routing,
                            semantic,
                            winner,
                            &default_target,
                            None,
                            true,
                        )?
                    }
                }
                _ => dispatch,
            }
        }
        None => {
            note_decision(None, pick.score, pick.miss);
            select_eligible(state, snapshot, router, routing, &default_target, None)?
                .into_dispatch(None)
        }
    };
    tracing::debug!(
        router = %router_entry.value.display_name,
        resolved_route = ?dispatch.route,
        group = ?dispatch.group.as_ref().map(|g| g.value.display_name.as_str()),
        first_target = ?dispatch.attempts.first().map(|a| a.model.display_name.as_str()),
        "semantic routing decision",
    );
    Ok(dispatch)
}

/// What session affinity reads off a request besides its headers.
#[derive(Clone, Copy, Default)]
pub(crate) struct SessionRequest<'a> {
    /// The body's `metadata` object, where a client may name its session.
    pub metadata: Option<&'a serde_json::Value>,
    /// Whether a matched route pins the session. Off for requests that only
    /// measure (`count_tokens`): they follow a pin but never set one.
    pub create_pins: bool,
}

/// The pin key and TTL for this request, when its router keeps session
/// affinity and the request names a session.
fn session_affinity_key(
    router_entry: &ResourceEntry<Model>,
    semantic: &Semantic,
    routing: RoutingRequest<'_>,
    session: SessionRequest<'_>,
) -> Option<(String, std::time::Duration)> {
    let affinity = semantic.session_affinity.as_ref().filter(|a| a.enabled)?;
    let session_id = crate::session_id::resolve(routing.headers?, session.metadata)?;
    Some((
        crate::session_affinity::pin_key(&router_entry.id, routing.api_key_id, &session_id),
        affinity.ttl(),
    ))
}

/// The route `pin` still holds: one whose target is the pinned model,
/// the pinned route's own name first. `None` once no route targets it.
fn pinned_route(snapshot: &AisixSnapshot, semantic: &Semantic, pin: &Pin) -> Option<usize> {
    let targets_pin = |route: &SemanticRoute| {
        snapshot
            .models
            .get_by_name(&route.target_ref(snapshot))
            .is_some_and(|e| e.id == pin.target_id)
    };
    semantic
        .routes
        .iter()
        .position(|r| r.name == pin.route && targets_pin(r))
        .or_else(|| semantic.routes.iter().position(targets_pin))
}

/// Dispatch as route `i` having won: its target when that is eligible,
/// else `default` as `target_unavailable`.
#[allow(clippy::too_many_arguments)]
fn serve_route(
    state: &ProxyState,
    snapshot: &AisixSnapshot,
    router: &Model,
    routing: RoutingRequest<'_>,
    semantic: &Semantic,
    i: usize,
    default_target: &str,
    score: Option<f64>,
    pinned: bool,
) -> Result<SemanticDispatch, ProxyError> {
    let eligible = select_eligible(
        state,
        snapshot,
        router,
        routing,
        &semantic.routes[i].target_ref(snapshot),
        Some(default_target),
    )?;
    // `x-aisix-route` reports the route that actually served the request:
    // a winner displaced by its target's gates is a fall-through to
    // `default`, not a served route.
    let name = (!eligible.fell_back).then(|| semantic.routes[i].name.clone());
    crate::attribution::note_semantic_decision(SemanticAccessLog {
        route: name.clone(),
        score,
        fallback: eligible
            .fell_back
            .then_some(SemanticFallback::TargetUnavailable),
        pinned,
    });
    Ok(eligible.into_dispatch(name))
}

const REMINDER_OPEN: &str = "<system-reminder>";
const REMINDER_CLOSE: &str = "</system-reminder>";

/// The text a semantic router classifies: the newest user turn that has
/// human text. A turn's human text is its text parts, with every complete
/// `<system-reminder>…</system-reminder>` block removed — agent harnesses
/// inject those as ordinary text beside the live ask, and a turn holding
/// only tool results, images or reminders has none, so the walk moves on to
/// the turn before it. Empty when no turn has any.
///
/// `user_turns` yields each user turn's text parts, newest turn first. The
/// per-protocol readers ([`chat_prompt`], [`anthropic_prompt`],
/// [`responses_prompt`]) decide what a user turn and a text part are.
pub(crate) fn classification_text<'a, T, P>(user_turns: T) -> String
where
    T: IntoIterator<Item = P>,
    P: IntoIterator<Item = &'a str>,
{
    for parts in user_turns {
        let joined = parts.into_iter().collect::<Vec<_>>().join(" ");
        let text = strip_reminders(&joined);
        if !text.is_empty() {
            return text;
        }
    }
    String::new()
}

/// `text` without its complete reminder blocks, the pieces around them
/// trimmed and joined by a space. The delimiters match case-insensitively;
/// an opening tag with no closing tag after it is not a block and stays.
/// A literal scan rather than a regex: this runs on caller-controlled
/// input before routing, and the cursor only moves forward.
fn strip_reminders(text: &str) -> String {
    // ASCII lowering keeps every byte offset, so positions found in it
    // index `text` directly.
    let lowered = text.to_ascii_lowercase();
    let mut kept: Vec<&str> = Vec::new();
    let mut keep_from = 0;
    while let Some(open) = lowered[keep_from..].find(REMINDER_OPEN) {
        let start = keep_from + open;
        let body = start + REMINDER_OPEN.len();
        let Some(close) = lowered[body..].find(REMINDER_CLOSE) else {
            break;
        };
        kept.push(text[keep_from..start].trim());
        keep_from = body + close + REMINDER_CLOSE.len();
    }
    kept.push(text[keep_from..].trim());
    kept.retain(|piece| !piece.is_empty());
    kept.join(" ")
}

/// [`classification_text`] of a Chat Completions request: its `user`
/// messages, their string content or the `text` parts of an array content.
pub(crate) fn chat_prompt(req: &aisix_gateway::ChatFormat) -> String {
    classification_text(
        req.messages
            .iter()
            .rev()
            .filter(|m| m.role == aisix_gateway::Role::User)
            .map(|m| -> Vec<&str> {
                match (&m.content, &m.content_blocks) {
                    (Some(text), _) => vec![text.as_str()],
                    (None, Some(blocks)) => text_parts(blocks, "text").collect(),
                    (None, None) => Vec::new(),
                }
            }),
    )
}

/// [`classification_text`] of an Anthropic Messages body: its `user`
/// messages, their string content or their `text` blocks (a `tool_result`
/// block is not text). The top-level `system` is never classified.
pub(crate) fn anthropic_prompt(body: &serde_json::Value) -> String {
    let messages = body
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    classification_text(
        messages
            .iter()
            .rev()
            .filter(|m| m.get("role").and_then(serde_json::Value::as_str) == Some("user"))
            .map(|m| content_text_parts(m.get("content"), "text")),
    )
}

/// [`classification_text`] of a Responses body: a string `input` is the one
/// user turn; in an array `input`, the `message` items with role `user` are,
/// read through their `input_text` parts. Every other item type — a
/// `function_call_output` included — is not a user turn, and `instructions`
/// is never classified.
pub(crate) fn responses_prompt(body: &serde_json::Value) -> String {
    match body.get("input") {
        Some(serde_json::Value::String(text)) => classification_text([[text.as_str()]]),
        Some(serde_json::Value::Array(items)) => classification_text(
            items
                .iter()
                .rev()
                .filter(|item| {
                    item.get("type")
                        .and_then(serde_json::Value::as_str)
                        .is_none_or(|t| t == "message")
                        && item.get("role").and_then(serde_json::Value::as_str) == Some("user")
                })
                .map(|item| content_text_parts(item.get("content"), "input_text")),
        ),
        _ => String::new(),
    }
}

fn content_text_parts<'a>(
    content: Option<&'a serde_json::Value>,
    part_type: &'static str,
) -> Vec<&'a str> {
    match content {
        Some(serde_json::Value::String(text)) => vec![text.as_str()],
        Some(serde_json::Value::Array(parts)) => text_parts(parts, part_type).collect(),
        _ => Vec::new(),
    }
}

fn text_parts<'a>(
    parts: &'a [serde_json::Value],
    part_type: &'static str,
) -> impl Iterator<Item = &'a str> + 'a {
    parts
        .iter()
        .filter(move |p| p.get("type").and_then(serde_json::Value::as_str) == Some(part_type))
        .filter_map(|p| p.get("text").and_then(serde_json::Value::as_str))
}

fn note_decision(route: Option<String>, score: Option<f64>, fallback: Option<SemanticFallback>) {
    crate::attribution::note_semantic_decision(SemanticAccessLog {
        route,
        score,
        fallback,
        pinned: false,
    });
}

/// The embedding decision: embed the prompt, score it against every
/// route's examples. `Err` when the embedding could not be computed, which
/// the caller answers with `on_embedding_failure`.
async fn decide_by_embedding(
    state: &ProxyState,
    snapshot: &AisixSnapshot,
    router_entry: &ResourceEntry<Model>,
    semantic: &Semantic,
    prompt: &str,
    request_id: &str,
) -> Result<Pick, DecisionFailure> {
    // Resolve the embedding model + its modality metadata. A dangling or
    // wrong-kind reference is a config error; degrade via the failure
    // policy rather than 500.
    let embedding_model = semantic.embedding_model_ref(snapshot);
    let embed_entry = match snapshot.models.get_by_name(&embedding_model) {
        Some(e) if e.value.is_embedding() => e,
        other => {
            tracing::warn!(
                router = %router_entry.value.display_name,
                embedding_model = %embedding_model,
                found = other.is_some(),
                "semantic router references a missing or non-embedding embedding_model; \
                 applying on_embedding_failure",
            );
            return Err(DecisionFailure::EmbeddingUnavailable);
        }
    };
    let dims = embed_entry
        .value
        .embedding
        .as_ref()
        .map(|e| e.dimensions)
        .unwrap_or(0);

    // Assemble the prompt (index 0) with every uncached example text as one
    // logical embedding batch. Providers without batch input may split it
    // into multiple upstream calls. Steady state: only the prompt is uncached.
    let mut pending: Vec<String> = Vec::new();
    for route in &semantic.routes {
        for ex in &route.examples {
            if state
                .semantic_cache
                .get(&embed_entry.id, dims, ex)
                .is_none()
                && !pending.iter().any(|p| p == ex)
            {
                pending.push(ex.clone());
            }
        }
    }
    let mut to_embed: Vec<String> = Vec::with_capacity(1 + pending.len());
    to_embed.push(prompt.to_string());
    to_embed.extend(pending.iter().cloned());

    // Deadline chain for the embed sub-call: the router-level knob wins,
    // then the embedding model's own `timeout`, then the deployment
    // default — previously only the router knob applied, so a hung
    // embedding upstream stalled every semantic request unbounded.
    let embed_deadline = semantic.embedding_timeout().or_else(|| {
        crate::routing::effective_timeouts(&embed_entry.value, None, state.default_timeouts).request
    });
    let vectors = match embed_texts(
        &state.hub,
        snapshot,
        &embed_entry,
        embed_deadline,
        request_id,
        GatewayEmbeddingPurpose::SemanticRoute,
        &to_embed,
    )
    .await
    {
        Ok(v) => v,
        Err(e) => {
            let failure = DecisionFailure::of_embedding_error(&e);
            tracing::warn!(
                router = %router_entry.value.display_name,
                reason = %failure,
                error = %embed_failure_text(&e),
                "semantic embedding call failed; applying on_embedding_failure",
            );
            return Err(failure);
        }
    };

    let mut iter = vectors.into_iter();
    let prompt_vec = iter
        .next()
        .expect("embed_texts returns at least the prompt vector");
    for text in &pending {
        if let Some(v) = iter.next() {
            state
                .semantic_cache
                .insert(&embed_entry.id, dims, text, Arc::new(v));
        }
    }

    let route_vecs: Vec<Vec<Arc<Vec<f32>>>> = semantic
        .routes
        .iter()
        .map(|route| {
            route
                .examples
                .iter()
                .filter_map(|ex| state.semantic_cache.get(&embed_entry.id, dims, ex))
                .collect()
        })
        .collect();

    let decision = decide(semantic, &prompt_vec, &route_vecs);
    // The winner's score, or with no winner the closest miss: the number an
    // operator tunes a threshold against.
    let score = match decision.winner {
        Some(i) => decision.scores.get(i).copied(),
        None => decision.scores.iter().copied().reduce(f32::max),
    };
    Ok(Pick {
        winner: decision.winner,
        score: score.map(log_score),
        miss: decision
            .winner
            .is_none()
            .then_some(SemanticFallback::NoMatch),
    })
}

/// A cosine score as the access log writes it. Widening an `f32` exposes
/// its binary representation (`0.8` becomes `0.800000011920929`), which
/// reads as precision the score does not have.
fn log_score(score: f32) -> f64 {
    (f64::from(score) * 1e6).round() / 1e6
}

/// Apply the router's failure policy: route to the fallback target, or
/// surface `503` when the policy is `fail`.
fn fallback(
    state: &ProxyState,
    snapshot: &AisixSnapshot,
    router: &Model,
    routing: RoutingRequest<'_>,
    semantic: &Semantic,
    failure: DecisionFailure,
) -> Result<SemanticDispatch, ProxyError> {
    match failure_target(snapshot, semantic) {
        Some(alias) => Ok(
            select_eligible(state, snapshot, router, routing, &alias, None)?.into_dispatch(None),
        ),
        None => Err(ProxyError::SemanticDecisionFailed {
            router: router.display_name.clone(),
            reason: failure,
        }),
    }
}

/// Endpoint a `typesafe` Provider Key reaches when it sets no `api_base`.
pub(crate) const TYPESAFE_DEFAULT_API_BASE: &str = "https://api.typesafe.ai";

/// The one question every classifier call asks. The wording is the one the
/// classifier mode was validated with; change it only against a fresh
/// evaluation.
const CLASSIFIER_QUESTION: &str = "route";
const CLASSIFIER_INSTRUCTIONS: &str =
    "Which route should handle this user request to an AI assistant?";

#[derive(serde::Serialize)]
struct ClassifierRequest<'a> {
    state: &'a str,
    model: &'a str,
    questions: ClassifierQuestions<'a>,
}

struct ClassifierQuestions<'a>(&'a [SemanticRoute]);

impl serde::Serialize for ClassifierQuestions<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        #[derive(serde::Serialize)]
        struct Choice<'a> {
            r#type: &'static str,
            instructions: &'static str,
            criteria: Criteria<'a>,
        }
        let mut map = s.serialize_map(Some(1))?;
        map.serialize_entry(
            CLASSIFIER_QUESTION,
            &Choice {
                r#type: "choice",
                instructions: CLASSIFIER_INSTRUCTIONS,
                criteria: Criteria(self.0),
            },
        )?;
        map.end()
    }
}

/// The options: exactly the configured routes, in configured order — which
/// a `serde_json::Map` (sorted by key) would not keep. The gateway adds no
/// option of its own; a catch-all is a route the operator configures.
struct Criteria<'a>(&'a [SemanticRoute]);

impl serde::Serialize for Criteria<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = s.serialize_map(Some(self.0.len()))?;
        for route in self.0 {
            map.serialize_entry(
                &route.name,
                route.description.as_deref().unwrap_or_default(),
            )?;
        }
        map.end()
    }
}

#[derive(serde::Deserialize)]
struct ClassifierResponse {
    #[serde(default)]
    answers: std::collections::HashMap<String, ClassifierAnswer>,
}

#[derive(serde::Deserialize)]
struct ClassifierAnswer {
    choice: String,
    confidence: f64,
}

/// Ask the classifier which route fits `prompt`. Returns the index of the
/// route it picked and its confidence.
/// Every way the call can fail comes back as `Err` with a reason for the
/// log, and the caller applies `classifier.on_failure`.
///
/// Detached, like the embedding sub-call: it is a call the GATEWAY decided
/// to make, not one the caller addressed, and it is not metered as one.
async fn classify(
    snapshot: &AisixSnapshot,
    routes: &[SemanticRoute],
    classifier: &SemanticClassifier,
    prompt: &str,
) -> Result<(usize, f64), (DecisionFailure, String)> {
    crate::attribution::detached(classify_inner(snapshot, routes, classifier, prompt)).await
}

async fn classify_inner(
    snapshot: &AisixSnapshot,
    routes: &[SemanticRoute],
    classifier: &SemanticClassifier,
    prompt: &str,
) -> Result<(usize, f64), (DecisionFailure, String)> {
    use DecisionFailure::{ClassifierInvalidResponse, ClassifierUnavailable, ClassifierUpstream};
    let pk = snapshot
        .provider_keys
        .get_by_id(&classifier.provider_key_id)
        .ok_or_else(|| {
            (
                ClassifierUnavailable,
                format!(
                    "classifier provider_key_id {:?} matches no provider key",
                    classifier.provider_key_id
                ),
            )
        })?;
    if pk.value.provider != TYPESAFE_PROVIDER {
        return Err((
            ClassifierUnavailable,
            format!(
                "classifier provider key {:?} has provider {:?}, not {TYPESAFE_PROVIDER:?}",
                pk.value.display_name, pk.value.provider
            ),
        ));
    }
    let base = pk
        .value
        .api_base
        .as_deref()
        .filter(|b| !b.is_empty())
        .unwrap_or(TYPESAFE_DEFAULT_API_BASE)
        .trim_end_matches('/');
    let body = serde_json::to_vec(&ClassifierRequest {
        state: prompt,
        model: &classifier.model,
        questions: ClassifierQuestions(routes),
    })
    .map_err(|e| {
        (
            ClassifierUnavailable,
            format!("encoding the classifier request: {e}"),
        )
    })?;
    let resp = crate::http_client::client_for(pk.value.upstream_connection().as_ref())
        .post(format!("{base}/v1/systemone"))
        .bearer_auth(&pk.value.api_key)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .timeout(classifier.timeout())
        .body(body)
        .send()
        .await
        .map_err(|e| DecisionFailure::of_classifier_transport(&e))?;
    let status = resp.status();
    if !status.is_success() {
        return Err((
            ClassifierUpstream,
            format!("classifier returned HTTP {}", status.as_u16()),
        ));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| DecisionFailure::of_classifier_transport(&e))?;
    // Neither error below repeats what the classifier answered: its
    // response can quote the prompt it was sent (`state`), and these
    // errors end up on the router's warn line.
    let mut parsed: ClassifierResponse = serde_json::from_slice(&bytes).map_err(|e| {
        (
            ClassifierInvalidResponse,
            format!(
                "classifier response is not the expected shape (line {} column {})",
                e.line(),
                e.column()
            ),
        )
    })?;
    let answer = parsed.answers.remove(CLASSIFIER_QUESTION).ok_or((
        ClassifierInvalidResponse,
        "classifier response carries no answer to the routing question".to_owned(),
    ))?;
    let picked = routes.iter().position(|r| r.name == answer.choice).ok_or((
        ClassifierInvalidResponse,
        "classifier picked a choice that is not one of the offered routes".to_owned(),
    ))?;
    Ok((picked, answer.confidence))
}

/// A route, default or failure target, resolved: a model group is
/// dispatched through its own members, anything else as the one attempt.
enum Candidate {
    Direct(Arc<ResourceEntry<Model>>),
    Group(Arc<ResourceEntry<Model>>),
}

impl Candidate {
    fn model(&self) -> &Model {
        match self {
            Self::Direct(entry) | Self::Group(entry) => &entry.value,
        }
    }
}

fn candidate_for_target(snapshot: &AisixSnapshot, alias: &str) -> Result<Candidate, ProxyError> {
    let entry = snapshot.models.get_by_name(alias).ok_or_else(|| {
        ProxyError::InvalidRequest(format!(
            "semantic router target {alias:?} does not resolve to a known model",
        ))
    })?;
    Ok(if entry.value.is_routing() {
        Candidate::Group(entry)
    } else {
        Candidate::Direct(entry)
    })
}

/// The target [`select_eligible`] picked.
struct Eligible {
    attempts: Vec<AttemptModel>,
    group: Option<Arc<ResourceEntry<Model>>>,
    /// Whether the fallback displaced the primary.
    fell_back: bool,
}

impl Eligible {
    fn into_dispatch(self, route: Option<String>) -> SemanticDispatch {
        SemanticDispatch {
            attempts: self.attempts,
            route,
            group: self.group,
        }
    }
}

/// Pick the dispatchable target among `{primary, fallback}` — the
/// semantic single-winner analogue of the group filter
/// (`routing::targets_allowed_for_ip` + `filter_attempt_models`):
///
/// - the target's client-IP allowlist is a HARD gate: an excluded
///   candidate is dropped, and with every candidate excluded the caller
///   gets the same 403 as the entry gate, naming only the alias they
///   addressed (`routing.rs` does the same for all-excluded groups, so
///   a router never becomes a probe for which members exist);
/// - a direct target's provider-key budget is a HARD gate too: with every
///   candidate excluded by it, the caller gets the budget's own answer;
/// - a direct target's health (request-path cooldown / background-unhealthy)
///   is a SOFT preference: prefer an available candidate, but with none
///   available dispatch the first direct candidate anyway — the router has
///   no `when_all_unavailable` knob, and the semantically-right target
///   beats failing the request on an advisory signal;
/// - a model-group target is available when it selects at least one member
///   for this request, through the same selection a caller addressing the
///   group gets (its strategy, member allowlists and budgets, health and
///   `when_all_unavailable`). A group that selects none is passed over, and
///   when nothing else can serve, its refusal is the answer. It is resolved
///   only once it is reached, so a group that does not serve keeps its
///   balancing state untouched.
fn select_eligible(
    state: &ProxyState,
    snapshot: &AisixSnapshot,
    router: &Model,
    routing: RoutingRequest<'_>,
    primary: &str,
    fallback_alias: Option<&str>,
) -> Result<Eligible, ProxyError> {
    let mut candidates: Vec<(Candidate, bool)> = Vec::with_capacity(2);
    candidates.push((candidate_for_target(snapshot, primary)?, false));
    if let Some(alias) = fallback_alias {
        // A dangling `default` only matters when it must actually serve;
        // as a mere fallback candidate it is skipped, not fatal.
        if alias != primary {
            if let Ok(candidate) = candidate_for_target(snapshot, alias) {
                candidates.push((candidate, true));
            }
        }
    }
    candidates.retain(|(candidate, fell_back)| {
        let allowed = candidate.model().ip_allowed(routing.source_ip);
        if !allowed {
            tracing::debug!(
                router = %router.display_name,
                target = %candidate.model().display_name,
                fallback = fell_back,
                "semantic target excluded by its allowed_cidrs",
            );
        }
        allowed
    });
    if candidates.is_empty() {
        return Err(ProxyError::ModelIpRestricted(router.display_name.clone()));
    }
    // A candidate on a provider key whose budget is exhausted is excluded
    // as hard as one outside its allowlist. A group carries no provider
    // key: its members are judged when it selects them.
    let owned_gate;
    let gate = match routing.budget {
        Some(gate) => gate,
        None => {
            owned_gate = crate::budget::gate(state, snapshot);
            &owned_gate
        }
    };
    let mut refusal: Option<ProxyError> = None;
    candidates.retain(|(candidate, fell_back)| {
        let Candidate::Direct(entry) = candidate else {
            return true;
        };
        match gate.check_model(&entry.value) {
            Ok(()) => true,
            Err(err) => {
                tracing::debug!(
                    router = %router.display_name,
                    target = %entry.value.display_name,
                    fallback = fell_back,
                    error = %err,
                    "semantic target excluded by its provider key's budget",
                );
                if !matches!(refusal, Some(ProxyError::BudgetExceeded(_))) {
                    refusal = Some(err);
                }
                false
            }
        }
    });
    if let (true, Some(err)) = (candidates.is_empty(), refusal) {
        return Err(err);
    }

    let mut group_refusal: Option<ProxyError> = None;
    let mut picked: Option<(usize, Vec<AttemptModel>)> = None;
    for (i, (candidate, _)) in candidates.iter().enumerate() {
        match candidate {
            Candidate::Direct(entry) => {
                let stale_after = entry
                    .value
                    .background_model_check
                    .as_ref()
                    .map(|cfg| std::time::Duration::from_secs(cfg.stale_after_seconds));
                if matches!(
                    state
                        .runtime_status
                        .status_with_stale(&entry.id, stale_after)
                        .status,
                    crate::RuntimeStatus::Healthy | crate::RuntimeStatus::NotApplicable
                ) {
                    picked = Some((i, vec![direct_attempt(entry)]));
                    break;
                }
            }
            Candidate::Group(group) => match crate::routing::resolve_attempt_models(
                &state.routing,
                &state.runtime_status,
                &state.pricing,
                snapshot,
                &group.value.display_name,
                &group.id,
                &group.value,
                routing,
            ) {
                Ok(members) => {
                    picked = Some((i, members));
                    break;
                }
                Err(err) => {
                    tracing::debug!(
                        router = %router.display_name,
                        group = %group.value.display_name,
                        error = %err,
                        "semantic group target selects no member",
                    );
                    group_refusal.get_or_insert(err);
                }
            },
        }
    }
    let picked = picked.or_else(|| {
        candidates
            .iter()
            .enumerate()
            .find_map(|(i, (c, _))| match c {
                Candidate::Direct(entry) => Some((i, vec![direct_attempt(entry)])),
                Candidate::Group(_) => None,
            })
    });
    let Some((i, attempts)) = picked else {
        // Every candidate is a group that selected nothing. Its refusal is
        // the answer, under the router's name where it names the group.
        let err = group_refusal.expect("a passed-over group recorded why");
        return Err(match err {
            ProxyError::ModelIpRestricted(_) => {
                ProxyError::ModelIpRestricted(router.display_name.clone())
            }
            other => other,
        });
    };
    let (candidate, fell_back) = candidates.swap_remove(i);
    if fell_back {
        tracing::debug!(
            router = %router.display_name,
            served = %candidate.model().display_name,
            "semantic route target unavailable; serving default",
        );
    }
    Ok(Eligible {
        attempts,
        group: match candidate {
            Candidate::Group(group) => Some(group),
            Candidate::Direct(_) => None,
        },
        fell_back,
    })
}

fn direct_attempt(entry: &ResourceEntry<Model>) -> AttemptModel {
    AttemptModel {
        id: entry.id.clone(),
        model: entry.value.clone(),
        priority: 0,
        weight: 1,
    }
}

/// Why [`embed_texts`] produced no vectors.
#[derive(Debug)]
pub(crate) enum EmbedTextsError {
    /// The call could not be made, or it failed: the embedding model's
    /// configuration, transport, a timeout, or an upstream error answer.
    Dispatch(ProxyError),
    /// The upstream answered 2xx with vectors that do not fit the request:
    /// the wrong dimension, base64 instead of floats, or the wrong count.
    /// The text names only lengths, never response content.
    UnusableResponse(String),
}

impl From<ProxyError> for EmbedTextsError {
    fn from(err: ProxyError) -> Self {
        Self::Dispatch(err)
    }
}

/// Embed `texts` through the embedding model's bridge in one batched call,
/// returning one float vector per input in input order. Shared by the
/// semantic router (route matching) and the cache gate's semantic layer
/// (`aisix-proxy::chat`), which is why the deadline arrives as a plain
/// `Option<Duration>` rather than either feature's config type.
pub(crate) async fn embed_texts(
    hub: &aisix_gateway::Hub,
    snapshot: &AisixSnapshot,
    embed_entry: &ResourceEntry<Model>,
    timeout: Option<std::time::Duration>,
    request_id: &str,
    purpose: GatewayEmbeddingPurpose,
    texts: &[String],
) -> Result<Vec<Vec<f32>>, EmbedTextsError> {
    // Detached: this is a dispatch the GATEWAY decided to make — for a
    // semantic guardrail, a semantic route, or the semantic cache — not one
    // the caller addressed. Two of its three callers run after the winning
    // attempt has already settled, so letting it commit the embedding model
    // to the request's attribution cell would put that model on the
    // request's own access-log line and usage event
    // (see `attribution::detached`).
    crate::attribution::detached(embed_texts_inner(
        hub,
        snapshot,
        embed_entry,
        timeout,
        request_id,
        purpose,
        texts,
    ))
    .await
}

async fn embed_texts_inner(
    hub: &aisix_gateway::Hub,
    snapshot: &AisixSnapshot,
    embed_entry: &ResourceEntry<Model>,
    timeout: Option<std::time::Duration>,
    request_id: &str,
    purpose: GatewayEmbeddingPurpose,
    texts: &[String],
) -> Result<Vec<Vec<f32>>, EmbedTextsError> {
    let model = &embed_entry.value;
    crate::dispatch::require_provider(model)?;
    let pk_entry = crate::dispatch::resolve_provider_key(snapshot, model)?;
    let bridge = crate::dispatch::resolve_bridge(hub, &pk_entry.value)
        .ok_or(ProxyError::ProviderUnavailable)?;
    let upstream_model = crate::dispatch::require_upstream_model(model)?.to_string();
    let audit_model_id = crate::attribution::capped_gateway_embedding_model_id(&embed_entry.id);
    let dimensions = model.embedding.as_ref().map(|e| e.dimensions);

    let req = EmbeddingRequest {
        extra: Default::default(),
        token_input: None,
        model: upstream_model,
        input: texts.to_vec(),
        input_was_single: texts.len() == 1,
        encoding_format: Some("float".to_string()),
        dimensions,
        input_type: None,
        embedding_types: None,
    };
    let ctx = {
        let base = crate::dispatch::bridge_ctx(
            request_id,
            &embed_entry.id,
            Arc::new(model.clone()),
            &pk_entry.id,
            Arc::new(pk_entry.value.clone()),
            None,
        );
        match timeout {
            Some(d) => base.with_deadline(d),
            None => base,
        }
    };

    // Keep this guard alive across the bridge await. If the downstream client
    // cancels this parent request while the bridge is pending, its Drop writes
    // a failed child audit directly to the shared parent ledger before the
    // cancel emitter takes that ledger into the terminal 499 event.
    let mut audit_call = crate::attribution::begin_gateway_embedding_call(purpose, &audit_model_id);
    let response = bridge.embed(&req, &ctx).await;
    let latency_ms = audit_call.settle();
    let resp = match response {
        Ok(resp) => {
            crate::attribution::note_gateway_embedding_call(GatewayEmbeddingCall {
                count: 1,
                purpose,
                embedding_model_id: audit_model_id.clone(),
                prompt_tokens: resp.usage.prompt_tokens,
                total_tokens: resp.usage.total_tokens,
                usage_source: match resp.usage.source {
                    EmbeddingUsageSource::Reported => GatewayEmbeddingUsageSource::Reported,
                    EmbeddingUsageSource::Unavailable => GatewayEmbeddingUsageSource::Unavailable,
                },
                latency_ms,
                outcome: GatewayEmbeddingOutcome::Succeeded,
            });
            resp
        }
        Err(err) => {
            crate::attribution::note_gateway_embedding_call(GatewayEmbeddingCall {
                count: 1,
                purpose,
                embedding_model_id: audit_model_id,
                prompt_tokens: 0,
                total_tokens: 0,
                usage_source: GatewayEmbeddingUsageSource::Unavailable,
                latency_ms,
                outcome: GatewayEmbeddingOutcome::Failed,
            });
            return Err(ProxyError::Bridge(err).into());
        }
    };
    let mut data = resp.data;
    data.sort_by_key(|d| d.index);
    let expected_dims = dimensions.map(|d| d as usize);
    let mut out = Vec::with_capacity(data.len());
    for obj in data {
        match obj.embedding {
            EmbeddingVector::Float(v) => {
                // Surface a wrong-dimension response explicitly instead of
                // letting cosine similarity fold the length mismatch into
                // 0.0 (which would silently route every request to default).
                if let Some(expected) = expected_dims {
                    if v.len() != expected {
                        return Err(EmbedTextsError::UnusableResponse(format!(
                            "embedding endpoint returned a vector of length {}; expected {expected}",
                            v.len(),
                        )));
                    }
                }
                out.push(v);
            }
            EmbeddingVector::Base64(_) => {
                return Err(EmbedTextsError::UnusableResponse(
                    "embedding endpoint returned base64 vectors; float vectors are required".into(),
                ));
            }
        }
    }
    if out.len() != texts.len() {
        return Err(EmbedTextsError::UnusableResponse(format!(
            "embedding endpoint returned {} vectors for {} inputs",
            out.len(),
            texts.len(),
        )));
    }
    Ok(out)
}

/// What a failed [`embed_texts`] call may say on a log line.
///
/// The text embedded is a caller's prompt or a guardrail's screened input,
/// and an upstream response can quote it back, so nothing a response said
/// is kept: an error envelope keeps only its status, and a decode failure
/// only the position serde reported (its message quotes the offending
/// value). Everything else — transport, timeout, configuration — keeps
/// its full error chain.
pub(crate) fn embed_failure_text(err: &EmbedTextsError) -> String {
    use aisix_gateway::BridgeError;
    let err = match err {
        EmbedTextsError::Dispatch(err) => err,
        EmbedTextsError::UnusableResponse(message) => return message.clone(),
    };
    match err {
        ProxyError::Bridge(BridgeError::UpstreamStatus { status, .. }) => {
            format!("upstream returned HTTP {status}")
        }
        ProxyError::Bridge(BridgeError::UpstreamInBand { status, .. }) => match status {
            Some(status) => format!("upstream reported an in-band error (status {status})"),
            None => "upstream reported an in-band error".to_owned(),
        },
        ProxyError::Bridge(BridgeError::UpstreamDecode(message)) => {
            match decode_position(message) {
                Some((line, column)) => {
                    format!("upstream response could not be decoded (line {line} column {column})")
                }
                None => "upstream response could not be decoded".to_owned(),
            }
        }
        _ => aisix_guardrails::error_chain(err),
    }
}

/// The `line N column M` suffix serde_json appends to its messages. Only
/// the two numbers are taken, so nothing of the message itself survives.
fn decode_position(message: &str) -> Option<(u64, u64)> {
    let (_, position) = message.rsplit_once(" at line ")?;
    let (line, column) = position.split_once(" column ")?;
    Some((line.parse().ok()?, column.parse().ok()?))
}

#[cfg(test)]
mod embed_failure_text_tests {
    use super::*;
    use aisix_gateway::BridgeError;

    fn logged(err: BridgeError) -> String {
        embed_failure_text(&EmbedTextsError::Dispatch(ProxyError::Bridge(err)))
    }

    #[test]
    fn an_error_envelope_keeps_only_its_status() {
        let err = BridgeError::upstream_status(400, "invalid input: screened text");
        assert_eq!(logged(err), "upstream returned HTTP 400");
    }

    #[test]
    fn a_decode_failure_keeps_only_its_position() {
        let serde = serde_json::from_str::<Vec<f32>>(r#""screened text""#).unwrap_err();
        let message = format!("upstream body: {serde}");
        assert!(message.contains("screened text"), "{message}");
        assert_eq!(
            logged(BridgeError::UpstreamDecode(message)),
            "upstream response could not be decoded (line 1 column 15)"
        );
        assert_eq!(
            logged(BridgeError::UpstreamDecode(
                "error decoding response body".into()
            )),
            "upstream response could not be decoded"
        );
    }

    #[test]
    fn transport_and_configuration_errors_keep_their_text() {
        assert!(logged(BridgeError::Transport("connection refused".into()))
            .contains("connection refused"));
        let config =
            BridgeError::Config("token mint upstream returned HTTP 400: invalid_grant".into());
        assert!(logged(config).contains("invalid_grant"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::resource::ResourceEntry;

    fn semantic(json: &str) -> Semantic {
        serde_json::from_str(json).unwrap()
    }

    fn arc(v: Vec<f32>) -> Arc<Vec<f32>> {
        Arc::new(v)
    }

    fn router() -> Semantic {
        semantic(
            r#"{
                "embedding_model": "bge-m3",
                "routes": [
                    {"name": "legal", "target": "opus", "examples": ["a"], "threshold": 0.8},
                    {"name": "code",  "target": "sonnet", "examples": ["b"]}
                ],
                "default": "gpt-4o",
                "match": {"threshold": 0.5}
            }"#,
        )
    }

    /// Embedding is a dispatch the GATEWAY decides to make, and two of its
    /// three callers — a semantic guardrail's OUTPUT hook, and the semantic
    /// cache's write — run after the winning attempt has already settled.
    ///
    /// The request's attribution cell records the last target
    /// `resolve_provider_key` committed to, and the access-log line and the
    /// cancelled-request usage event both read it. So without a cell of its
    /// own, an ordinary request that happens to run a semantic guardrail
    /// would report the EMBEDDING model as the upstream it dispatched to —
    /// on the very line an operator reads to find out which member of a
    /// routing group served them (AISIX-Cloud#1571). Wrong, not merely
    /// absent, which is the worse of the two.
    #[tokio::test]
    async fn embedding_does_not_overwrite_the_caller_s_target() {
        use aisix_core::snapshot::ResourceTable;

        let embed_model: Model = serde_json::from_value(serde_json::json!({
            "display_name": "bge-m3",
            "provider": "openai",
            "model_name": "text-embedding-3-small",
            "provider_key_id": "pk-embed",
        }))
        .unwrap();
        let pk: aisix_core::ProviderKey = serde_json::from_value(serde_json::json!({
            "display_name": "embed-key",
            "secret": "sk-embed",
            "api_base": "http://127.0.0.1:1",
            "provider": "openai",
            "adapter": "openai",
        }))
        .unwrap();
        let provider_keys = ResourceTable::default();
        provider_keys.insert(ResourceEntry::new("pk-embed", pk, 1));
        let snapshot = AisixSnapshot {
            provider_keys,
            ..Default::default()
        };
        let embed_entry = ResourceEntry::new("m-embed", embed_model, 1);

        let cell = std::sync::Arc::new(crate::attribution::RequestAttribution::default());
        crate::attribution::scope(cell.clone(), async {
            // What the CALLER addressed and the gateway dispatched to.
            let served: Model = serde_json::from_value(serde_json::json!({
                "display_name": "served-by",
                "provider": "anthropic",
                "model_name": "claude-sonnet-4",
                "provider_key_id": "pk-chat",
            }))
            .unwrap();
            crate::attribution::note_target(&served, "pk-chat");

            // An empty `Hub` means no bridge, so this fails — but only
            // AFTER `resolve_provider_key` has committed the embedding
            // target, which is the write under test.
            let err = embed_texts(
                &aisix_gateway::Hub::new(),
                &snapshot,
                &embed_entry,
                None,
                "req-embed",
                GatewayEmbeddingPurpose::SemanticRoute,
                &["scan me".to_string()],
            )
            .await
            .expect_err("premise: no bridge is registered, so this must fail");
            assert!(
                matches!(
                    err,
                    EmbedTextsError::Dispatch(ProxyError::ProviderUnavailable)
                ),
                "premise: it must fail at bridge resolution, i.e. after the \
                 provider key was resolved — got {err:?}",
            );

            let resolved = crate::attribution::current().expect("in scope");
            assert_eq!(resolved.upstream_model, "claude-sonnet-4");
            assert_eq!(resolved.provider, "anthropic");
            assert_eq!(resolved.provider_key_id, "pk-chat");
        })
        .await;
    }

    #[test]
    fn decide_picks_highest_route_clearing_its_threshold() {
        let s = router();
        let req = vec![1.0, 0.0];
        // legal example == request (cos 1.0 ≥ 0.8 ✓); code example orthogonal (0.0 < 0.5).
        let route_vecs = vec![vec![arc(vec![1.0, 0.0])], vec![arc(vec![0.0, 1.0])]];
        let d = decide(&s, &req, &route_vecs);
        assert_eq!(d.winner, Some(0));
        assert!((d.scores[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn decide_falls_through_to_default_when_none_clear() {
        let s = router();
        let req = vec![1.0, 0.0];
        // Both examples weakly aligned: legal 0.6 (< 0.8 own threshold),
        // code 0.4 (< 0.5 router threshold) → no winner.
        let route_vecs = vec![
            vec![arc(vec![0.6, 0.8])],   // cos ≈ 0.6
            vec![arc(vec![0.4, 0.917])], // cos ≈ 0.4
        ];
        let d = decide(&s, &req, &route_vecs);
        assert_eq!(d.winner, None);
    }

    #[test]
    fn decide_uses_router_threshold_when_route_has_no_override() {
        let s = router();
        let req = vec![1.0, 0.0];
        // code route (no override → 0.5): give it cos ≈ 0.71 (clears 0.5).
        // legal route (0.8): give it cos ≈ 0.71 (does NOT clear 0.8).
        let route_vecs = vec![vec![arc(vec![1.0, 1.0])], vec![arc(vec![1.0, 1.0])]];
        let d = decide(&s, &req, &route_vecs);
        assert_eq!(
            d.winner,
            Some(1),
            "only the code route clears its threshold"
        );
    }

    #[test]
    fn decide_max_aggregation_takes_best_example() {
        let s = router();
        let req = vec![1.0, 0.0];
        // legal has two examples: one orthogonal (0.0), one identical (1.0).
        // max → 1.0 clears 0.8.
        let route_vecs = vec![
            vec![arc(vec![0.0, 1.0]), arc(vec![1.0, 0.0])],
            vec![arc(vec![0.0, 1.0])],
        ];
        let d = decide(&s, &req, &route_vecs);
        assert_eq!(d.winner, Some(0));
        assert!((d.scores[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn failure_target_maps_each_policy() {
        let snap = AisixSnapshot::default();
        let default_policy = router();
        assert_eq!(
            failure_target(&snap, &default_policy).as_deref(),
            Some("gpt-4o")
        );

        let fail = semantic(
            r#"{"embedding_model":"e","routes":[{"name":"a","target":"m","examples":["x"]}],
                "default":"d","match":{"threshold":0.5},"on_embedding_failure":"fail"}"#,
        );
        assert!(failure_target(&snap, &fail).is_none());

        let target = semantic(
            r#"{"embedding_model":"e","routes":[{"name":"a","target":"m","examples":["x"]}],
                "default":"d","match":{"threshold":0.5},"on_embedding_failure":{"target":"safe"}}"#,
        );
        assert_eq!(failure_target(&snap, &target).as_deref(), Some("safe"));
    }

    /// The id spelling decides at both `on_embedding_failure` shapes, and
    /// resolves against the live table — so a rename of the fallback model
    /// needs no edit to the router.
    #[test]
    fn failure_target_follows_the_id_spelling() {
        let snap = AisixSnapshot::default();
        snap.models.insert(ResourceEntry::new(
            "m-safe",
            serde_json::from_str::<Model>(
                r#"{"display_name":"safe-v2","provider":"openai","model_name":"gpt-4o",
                    "provider_key_id":"11111111-1111-1111-1111-111111111111"}"#,
            )
            .unwrap(),
            1,
        ));

        let explicit = semantic(
            r#"{"embedding_model":"e","routes":[{"name":"a","target":"m","examples":["x"]}],
                "default":"d","match":{"threshold":0.5},
                "on_embedding_failure":{"target":"stale","target_id":"m-safe"}}"#,
        );
        assert_eq!(failure_target(&snap, &explicit).as_deref(), Some("safe-v2"));

        let by_default = semantic(
            r#"{"embedding_model":"e","routes":[{"name":"a","target":"m","examples":["x"]}],
                "default":"stale","default_id":"m-safe","match":{"threshold":0.5}}"#,
        );
        assert_eq!(
            failure_target(&snap, &by_default).as_deref(),
            Some("safe-v2")
        );

        // An id that resolves to nothing stands in as its own name, which
        // dispatches nowhere — what a dangling alias already does.
        let dangling = semantic(
            r#"{"embedding_model":"e","routes":[{"name":"a","target":"m","examples":["x"]}],
                "default":"d","default_id":"m-gone","match":{"threshold":0.5}}"#,
        );
        let resolved = failure_target(&snap, &dangling).unwrap();
        assert_eq!(resolved, "m-gone");
        assert!(snap.models.get_by_name(&resolved).is_none());
    }

    #[test]
    fn reminder_blocks_are_removed_and_the_text_around_them_kept() {
        assert_eq!(
            strip_reminders("<system-reminder>ctx</system-reminder>\nfix the bug"),
            "fix the bug"
        );
        assert_eq!(
            strip_reminders("before <SYSTEM-REMINDER>a</System-Reminder> mid <system-reminder>b</system-reminder> after"),
            "before mid after"
        );
        assert_eq!(
            strip_reminders("<system-reminder>only</system-reminder>"),
            ""
        );
        // An opening tag nothing closes is not a block.
        assert_eq!(
            strip_reminders("<system-reminder>a</system-reminder> x <system-reminder> y"),
            "x <system-reminder> y"
        );
    }

    #[test]
    fn classification_walks_back_past_turns_without_human_text() {
        let turns: Vec<Vec<&str>> = vec![
            vec![],
            vec!["<system-reminder>r</system-reminder>"],
            vec![
                "<system-reminder>r</system-reminder>",
                "review the",
                "contract",
            ],
            vec!["older ask"],
        ];
        assert_eq!(classification_text(turns), "review the contract");
        assert_eq!(classification_text(Vec::<Vec<&str>>::new()), "");
    }

    #[test]
    fn each_protocol_reads_its_user_turns() {
        let messages = serde_json::json!({
            "system": "system text is never classified",
            "messages": [
                {"role": "user", "content": "the real ask"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "x", "input": {}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": "tool output"},
                    {"type": "text", "text": "<system-reminder>r</system-reminder>"}
                ]}
            ]
        });
        assert_eq!(anthropic_prompt(&messages), "the real ask");

        let responses = serde_json::json!({
            "instructions": "never classified",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "first ask"}]},
                {"role": "user", "content": "second ask"},
                {"type": "function_call", "call_id": "c", "name": "x", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c", "output": "tool output"}
            ]
        });
        assert_eq!(responses_prompt(&responses), "second ask");
        assert_eq!(
            responses_prompt(&serde_json::json!({"input": "plain string"})),
            "plain string"
        );
    }

    #[test]
    fn cache_round_trips_and_dimension_change_invalidates() {
        let cache = SemanticVectorCache::default();
        assert!(cache.get("bge", 1024, "hello").is_none());
        cache.insert("bge", 1024, "hello", arc(vec![1.0, 2.0]));
        assert_eq!(
            cache.get("bge", 1024, "hello").unwrap().as_slice(),
            &[1.0, 2.0]
        );
        // Different dimensions → different key → miss (auto-invalidation).
        assert!(cache.get("bge", 512, "hello").is_none());
        // Different embedding model → miss.
        assert!(cache.get("other", 1024, "hello").is_none());
    }
}
