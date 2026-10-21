//! Stored-response routes addressed by id: `GET` / `DELETE
//! /v1/responses/{id}`, `GET /v1/responses/{id}/input_items` and
//! `POST /v1/responses/{id}/cancel`.
//!
//! A response a Model Group served carries its target in the id the client
//! received ([`crate::responses_affinity`]), so these calls go to that
//! target's provider key with the upstream's own id. An untagged id names no
//! target, and the caller selects the model with `?model=` or
//! `x-aisix-model`, as on the jobs surface.
//!
//! The decoded target is only reachable if the caller could address it
//! anyway: directly, or through a Model Group it may use that routes to it.
//! A tag naming anything else, and a target that no longer exists or does
//! not serve `/v1/responses` natively (nothing was stored upstream), answers
//! the same 404, so a crafted id learns nothing about what exists.
//!
//! Telemetry is the jobs surface's management-call contract: a zero-token
//! UsageEvent and an access-log line. A retrieved Response object's `usage`
//! was billed when it was created and is not read here.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use aisix_core::models::model::Adapter;
use aisix_core::resource::ResourceEntry;
use aisix_core::Model;
use axum::body::{Body, Bytes};
use axum::extract::{Query, RawQuery, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode};
use axum::response::Response;
use futures::StreamExt as _;
use serde_json::Value;

use crate::auth::AuthenticatedKey;
use crate::client_ip::ClientContext;
use crate::error::ProxyError;
use crate::jobs::{JobTarget, UpstreamBody};
use crate::reject::AisixPath;
use crate::responses_affinity::ResponseMark;
use crate::state::ProxyState;

#[allow(clippy::too_many_arguments)]
pub(crate) async fn response(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    client: ClientContext,
    method: Method,
    AisixPath(id): AisixPath<String>,
    Query(params): Query<HashMap<String, String>>,
    RawQuery(raw_query): RawQuery,
    headers: HeaderMap,
) -> Response {
    forward(
        state, auth, client, method, id, "", params, raw_query, headers,
    )
    .await
}

pub(crate) async fn input_items(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    client: ClientContext,
    AisixPath(id): AisixPath<String>,
    Query(params): Query<HashMap<String, String>>,
    RawQuery(raw_query): RawQuery,
    headers: HeaderMap,
) -> Response {
    let method = Method::GET;
    forward(
        state,
        auth,
        client,
        method,
        id,
        "/input_items",
        params,
        raw_query,
        headers,
    )
    .await
}

pub(crate) async fn cancel(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    client: ClientContext,
    AisixPath(id): AisixPath<String>,
    Query(params): Query<HashMap<String, String>>,
    RawQuery(raw_query): RawQuery,
    headers: HeaderMap,
) -> Response {
    let method = Method::POST;
    forward(
        state, auth, client, method, id, "/cancel", params, raw_query, headers,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn forward(
    state: ProxyState,
    auth: AuthenticatedKey,
    client: ClientContext,
    method: Method,
    id: String,
    suffix: &'static str,
    params: HashMap<String, String>,
    raw_query: Option<String>,
    headers: HeaderMap,
) -> Response {
    let started = Instant::now();
    let request_id = client.request_id.clone();
    let snapshot = state.snapshot.load();

    let result = async {
        let (target, raw, mark) = resolve(&snapshot, &auth, &client, &id, &params, &headers)?;
        let _reservation = crate::quota::enforce(
            &state,
            &snapshot,
            &auth,
            Some(&crate::quota::ModelRateLimit::from_model(
                &target.model_entry.value.display_name,
                &target.model_entry.id,
                &target.model_entry.value,
            )),
        )
        .await?;

        let base = crate::dispatch::resolve_base_url_for(
            &target.pk_entry.value,
            aisix_core::ApiSurface::Responses,
        )?;
        let mut url =
            crate::dispatch::build_openai_url(&base, &format!("/responses/{raw}{suffix}"));
        let query = crate::jobs::forwardable_query(raw_query.as_deref());
        if !query.is_empty() {
            url = format!("{url}?{query}");
        }
        let streamed = params.get("stream").is_some_and(|v| v == "true");
        let upstream = crate::jobs::send_upstream_head(
            &state,
            &target,
            method.clone(),
            &url,
            UpstreamBody::Empty,
            &request_id,
            !streamed,
        )
        .await?;
        let resp = relay(&state, &target, upstream, mark).await?;
        Ok((resp, target))
    }
    .await;

    crate::jobs::finish(
        &state,
        &snapshot,
        crate::operation::RESPONSES,
        method,
        format!("/v1/responses/{id}{suffix}"),
        &auth,
        &client,
        started,
        request_id,
        result,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        String::new(),
        Vec::new(),
    )
}

/// The target, the upstream's own id, and how the reply is tagged.
fn resolve(
    snapshot: &aisix_core::AisixSnapshot,
    auth: &AuthenticatedKey,
    client: &ClientContext,
    id: &str,
    params: &HashMap<String, String>,
    headers: &HeaderMap,
) -> Result<(JobTarget, String, ResponseMark), ProxyError> {
    let not_found = || ProxyError::ResponseNotFound(id.to_string());
    let (entry, raw, mark) = match crate::responses_affinity::decode_response_id(id) {
        Some((origin, raw)) => {
            let entry = snapshot.models.get_by_id(&origin).ok_or_else(not_found)?;
            if !reachable(snapshot, auth, &entry.value) {
                return Err(not_found());
            }
            let mark = ResponseMark::for_target(&entry.id);
            (entry, raw, mark)
        }
        None => {
            let name = crate::jobs::explicit_model(params, headers).ok_or_else(|| {
                ProxyError::InvalidRequest(
                    "this response id names no model; pass `model` as a query parameter \
                     or the `x-aisix-model` header"
                        .into(),
                )
            })?;
            let entry = crate::model_resolve::resolve_model(snapshot, &name)
                .ok_or_else(|| ProxyError::ModelNotFound(name.clone()))?;
            if !auth.key().can_access(snapshot, &name) {
                return Err(ProxyError::ModelForbidden(name));
            }
            let m = &entry.value;
            if m.is_routing() || m.is_ensemble() || m.is_semantic() {
                return Err(ProxyError::InvalidRequest(format!(
                    "model {name:?} is a virtual model; name the model that served the response"
                )));
            }
            (entry, id.to_string(), ResponseMark::default())
        }
    };
    crate::jobs::require_safe_upstream_id(&raw)?;
    crate::dispatch::check_ip_access(&entry.value, &client.source_ip)?;
    let target = native_target(snapshot, entry, client).ok_or_else(not_found)?;
    Ok((target, raw, mark))
}

/// Whether the caller may address `target` itself or a Model Group that
/// routes to it.
fn reachable(
    snapshot: &aisix_core::AisixSnapshot,
    auth: &AuthenticatedKey,
    target: &Model,
) -> bool {
    let key = auth.key();
    key.can_access(snapshot, &target.display_name)
        || snapshot.models.entries().iter().any(|group| {
            group.value.routing.as_ref().is_some_and(|r| {
                r.targets
                    .iter()
                    .any(|t| t.model_ref(snapshot) == target.display_name)
            }) && key.can_access(snapshot, &group.value.display_name)
        })
}

/// `None` when `entry` cannot hold a stored response: a virtual model, or a
/// target whose `/v1/responses` traffic is bridged.
fn native_target(
    snapshot: &aisix_core::AisixSnapshot,
    entry: Arc<ResourceEntry<Model>>,
    client: &ClientContext,
) -> Option<JobTarget> {
    let model = &entry.value;
    if model.is_routing() || model.is_ensemble() || model.is_semantic() {
        return None;
    }
    if !crate::dispatch::serves_natively(snapshot, model, aisix_core::ApiSurface::Responses) {
        return None;
    }
    let pk = crate::dispatch::resolve_provider_key(snapshot, model).ok()?;
    let secret = crate::dispatch::require_api_key(&pk.value, model)
        .ok()?
        .to_string();
    Some(JobTarget::new(entry, pk, secret, Adapter::Openai, client))
}

/// The upstream reply with the target's tags applied: per frame for an
/// event stream, whole-body for JSON, verbatim otherwise.
async fn relay(
    state: &ProxyState,
    target: &JobTarget,
    upstream: reqwest::Response,
    mark: ResponseMark,
) -> Result<Response, ProxyError> {
    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = upstream.headers().get(header::CONTENT_TYPE).cloned();
    let is_sse = content_type
        .as_ref()
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/event-stream"));
    let builder = Response::builder().status(status);
    let builder = match content_type {
        Some(ct) => builder.header(header::CONTENT_TYPE, ct),
        None => builder.header(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        ),
    };
    if is_sse {
        crate::attribution::note_upstream_wait_end(true);
        let mut buf: Vec<u8> = Vec::new();
        let mut upstream = upstream.bytes_stream();
        let tagged = async_stream::stream! {
            while let Some(chunk) = upstream.next().await {
                match chunk {
                    Ok(chunk) => {
                        buf.extend_from_slice(&chunk);
                        let mut out = Vec::new();
                        while let Some(end) = aisix_gateway::sse::find_frame_end(&buf) {
                            let frame: Vec<u8> = buf.drain(..end).collect();
                            match mark.apply_to_sse_frame(&frame) {
                                Some(t) => out.extend_from_slice(&t),
                                None => out.extend_from_slice(&frame),
                            }
                        }
                        if !out.is_empty() {
                            yield Ok::<_, std::io::Error>(Bytes::from(out));
                        }
                    }
                    Err(e) => {
                        yield Err(std::io::Error::other(e));
                        return;
                    }
                }
            }
            if !buf.is_empty() {
                yield Ok(Bytes::from(std::mem::take(&mut buf)));
            }
        };
        return Ok(builder.body(Body::from_stream(tagged)).unwrap());
    }
    let bytes = upstream.bytes().await;
    crate::attribution::note_upstream_wait_end(bytes.is_ok());
    let bytes = bytes
        .map_err(|e| {
            crate::cooldown::note_failure(
                &state.runtime_status,
                &target.model_entry.id,
                target.model_entry.value.cooldown.as_ref(),
                aisix_gateway::BridgeError::UpstreamDecode(e.to_string()),
            )
        })
        .map_err(ProxyError::Bridge)?;
    let body = match serde_json::from_slice::<Value>(&bytes) {
        Ok(mut v) if status.is_success() => {
            mark.apply_to_body(&mut v);
            Bytes::from(serde_json::to_vec(&v).unwrap_or_else(|_| bytes.to_vec()))
        }
        _ => bytes,
    };
    Ok(builder.body(Body::from(body)).unwrap())
}
