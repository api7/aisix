//! The upstream a usage event's attempt was addressed to, for the OTLP span
//! encoder (`gen_ai.provider.name`, `gen_ai.request.model`,
//! `server.address` / `server.port`).
//!
//! Read off the request's own snapshot at the emission chokepoint rather
//! than threaded through every dispatch path: each event already names the
//! concrete Model row its attempt used (`model_id`) and the ProviderKey it
//! authenticated with, and the snapshot is the one dispatch resolved them
//! from, so the two cannot disagree. Nothing here writes to the request's
//! attribution cell — an event for a failed attempt is emitted after later
//! attempts resolved theirs, and must not move the cell's "last target".

use aisix_core::{AisixSnapshot, ApiSurface, Model, ProviderKey};
use aisix_obs::{genai_provider_name, UpstreamFacts, UsageEvent};

/// The event's upstream, or `None` when it names none the snapshot holds:
/// a pre-dispatch failure on a Model Group, a deleted row, a route with no
/// upstream resource.
pub(crate) fn upstream_facts(
    snap: &AisixSnapshot,
    event: &UsageEvent,
    provider_key_id: &str,
) -> Option<UpstreamFacts> {
    match event.operation.as_str() {
        "mcp" => {
            let server = snap.mcp_servers.get_by_name(&event.mcp_server_name)?;
            return Some(UpstreamFacts::default().with_base_url(&server.value.url));
        }
        "a2a" => {
            // A remote agent has no GenAI provider the gateway can name, but
            // the attribute is required on an `invoke_agent` span as the
            // telemetry-flavour discriminator: the protocol is the flavour.
            let facts = UpstreamFacts {
                provider_name: "a2a".into(),
                ..UpstreamFacts::default()
            };
            return Some(match snap.a2a_agents.get_by_name(&event.a2a_agent_name) {
                Some(agent) => facts.with_base_url(&agent.value.url),
                None => facts,
            });
        }
        "passthrough" => return passthrough_facts(snap, event, provider_key_id),
        _ => {}
    }
    if let Some(row) = snap.models.get_by_id(&event.model_id) {
        let model = &row.value;
        if model.is_routing() || model.is_ensemble() || model.is_semantic() {
            return None;
        }
        let pk = model
            .provider_key_id
            .as_deref()
            .and_then(|id| snap.provider_keys.get_by_id(id));
        let pk = pk.as_ref().map(|e| &e.value);
        let vendor = pk
            .map(|p| p.provider.as_str())
            .filter(|v| !v.is_empty())
            .or(model.provider.as_deref())
            .unwrap_or_default();
        let facts = UpstreamFacts {
            provider_name: genai_provider_name(vendor, pk.and_then(|p| p.adapter)),
            request_model: request_model(model, event).unwrap_or_default(),
            request_model_caller_named: model.display_name.contains('*'),
            ..UpstreamFacts::default()
        };
        return Some(
            match pk.and_then(|p| upstream_base(model, p, &event.operation)) {
                Some(base) => facts.with_base_url(&base),
                None => facts,
            },
        );
    }
    None
}

/// A passthrough route relays to its own `target_url`; the ProviderKey it
/// may name only supplies the credential, so it says which provider but not
/// where. A `preserve_host` route relays to whatever host the caller
/// addressed, which no resource records — no address then.
fn passthrough_facts(
    snap: &AisixSnapshot,
    event: &UsageEvent,
    provider_key_id: &str,
) -> Option<UpstreamFacts> {
    let route = snap
        .passthrough_routes
        .get_by_name(&event.passthrough_route_name)?;
    let route = &route.value;
    let facts = UpstreamFacts {
        provider_name: snap
            .provider_keys
            .get_by_id(provider_key_id)
            .map(|pk| genai_provider_name(&pk.value.provider, pk.value.adapter))
            .unwrap_or_default(),
        ..UpstreamFacts::default()
    };
    Some(match route.target_url.as_deref() {
        Some(target) if !route.preserve_host => facts.with_base_url(target),
        _ => facts,
    })
}

/// The model name sent upstream. A wildcard row's own `model_name` is a
/// template; the concrete name is the one the caller's alias substitutes
/// into it, exactly as `model_resolve::resolve_model` built it for
/// dispatch.
fn request_model(model: &Model, event: &UsageEvent) -> Option<String> {
    if !model.display_name.contains('*') {
        return model.model_name.clone();
    }
    // A routing target names the row itself, not a concrete alias, so only
    // a direct request carries the name the template was filled from.
    if !event.attempt_model.is_empty() {
        return None;
    }
    crate::model_resolve::wildcard_upstream_model(model, &event.requested_model)
}

/// The base URL the attempt was sent to, by the same resolution the
/// dispatch paths use. `None` where the upstream URL is not a function of
/// the key alone (Bedrock's and Vertex's regional endpoints) or the key
/// cannot be resolved.
fn upstream_base(model: &Model, pk: &ProviderKey, operation: &str) -> Option<String> {
    use aisix_core::Adapter;
    if matches!(
        pk.adapter,
        Some(Adapter::Bedrock) | Some(Adapter::Vertex) | Some(Adapter::AzureOpenai)
    ) {
        return pk
            .api_base
            .as_deref()
            .map(str::trim)
            .filter(|b| !b.is_empty())
            .map(str::to_string);
    }
    let surface = match operation {
        "messages" | "count_tokens" => Some(ApiSurface::Messages),
        "responses" => Some(ApiSurface::Responses),
        _ => None,
    };
    let native = surface.filter(|s| crate::dispatch::serves_natively_with(model, Some(pk), *s));
    crate::dispatch::configured_base_url(pk, native)
}

/// Record the caller's GenAI request parameters on the request's trace
/// bundle, for the inference span. A body that is not a JSON object (the
/// extractor already refused anything else) records nothing.
pub(crate) fn note_request(
    client: &crate::client_ip::ClientContext,
    params: Option<aisix_obs::GenAiRequestParams>,
) {
    if let (Some(trace), Some(params)) = (&client.trace, params) {
        trace.set_genai_request(params);
    }
}

/// [`aisix_obs::GenAiRequestParams::from_openai_chat`] over the typed chat
/// body: its typed fields plus the extras that name a parameter, without
/// cloning the rest of the request.
pub(crate) fn chat_params(req: &aisix_gateway::ChatFormat) -> aisix_obs::GenAiRequestParams {
    use serde_json::Value;
    let mut body = serde_json::Map::new();
    for key in [
        "max_completion_tokens",
        "top_k",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "n",
        "service_tier",
    ] {
        if let Some(v) = req.extra.get(key) {
            body.insert(key.to_string(), v.clone());
        }
    }
    if let Some(kind) = req.extra.get("response_format").and_then(|f| f.get("type")) {
        body.insert(
            "response_format".into(),
            serde_json::json!({ "type": kind }),
        );
    }
    // Through the shortest decimal form, so `0.2` is reported as `0.2`
    // rather than the f32's widened `0.20000000298…`.
    let widen = |v: f32| v.to_string().parse::<f64>().map(Value::from).ok();
    if let Some(v) = req.temperature.and_then(widen) {
        body.insert("temperature".into(), v);
    }
    if let Some(v) = req.top_p.and_then(widen) {
        body.insert("top_p".into(), v);
    }
    if let Some(v) = req.max_tokens {
        body.insert("max_tokens".into(), Value::from(v));
    }
    if let Some(v) = req.stream {
        body.insert("stream".into(), Value::from(v));
    }
    aisix_obs::GenAiRequestParams::from_openai_chat(&body)
}
