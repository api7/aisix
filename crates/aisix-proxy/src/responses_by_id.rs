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
//! These calls act on a response its creation already metered and screened,
//! as a video poll does: they write the access log and no usage row, skip
//! model-level rate limits (key-level ones still apply), and run no output
//! guardrails over what they return.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use aisix_core::models::model::Adapter;
use aisix_core::resource::ResourceEntry;
use aisix_core::Model;
use axum::body::{Body, Bytes};
use axum::extract::{Query, RawQuery, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
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

    let result: Result<(Response, JobTarget), ProxyError> = async {
        let (target, raw, mark) = resolve(&snapshot, &auth, &client, &id, &params, &headers)?;
        // Model-level limits do not apply, as on a video poll: the response
        // was admitted when it was created. Key-level layers still do.
        let reservation = crate::quota::enforce(&state, &snapshot, &auth, None).await?;

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
        let resp = if params.get("stream").is_some_and(|v| v == "true") {
            let upstream = crate::jobs::send_upstream_head(
                &state,
                &target,
                method.clone(),
                &url,
                UpstreamBody::Empty,
                &request_id,
                false,
            )
            .await?;
            relay_stream(&state, &target, upstream, mark)
        } else {
            let (status, resp_headers, bytes) = crate::jobs::send_upstream(
                &state,
                &target,
                method.clone(),
                &url,
                UpstreamBody::Empty,
                &request_id,
            )
            .await?;
            let body = match serde_json::from_slice::<Value>(&bytes) {
                Ok(mut v) if status.is_success() => {
                    mark.apply_to_body(&mut v);
                    if suffix == "/input_items" {
                        mark.apply_to_item_list(&mut v);
                    }
                    Bytes::from(serde_json::to_vec(&v).unwrap_or_else(|_| bytes.to_vec()))
                }
                _ => bytes,
            };
            with_content_type(
                Response::builder().status(status),
                resp_headers.get(header::CONTENT_TYPE),
            )
            .body(Body::from(body))
            .unwrap()
        };
        reservation.commit_tokens(0).await;
        Ok((resp, target))
    }
    .await;

    // No usage row, whatever the outcome: these calls act on a response its
    // creation already metered.
    crate::attribution::note_unmetered_route();
    let path = format!("/v1/responses/{id}{suffix}");
    let elapsed = started.elapsed();
    let (target, status, error) = match &result {
        Ok((resp, target)) => (Some(target), resp.status().as_u16(), None),
        Err(err) => (None, err.status().as_u16(), Some(err)),
    };
    crate::jobs::emit_access_log(
        &method,
        &path,
        target,
        &auth.entry.id,
        status,
        elapsed,
        &request_id,
        error,
    );
    crate::request_metrics::record(
        &state,
        crate::normalize_endpoint_label(&path),
        crate::request_metrics::Caller::new(&auth),
        crate::request_metrics::Upstream {
            provider: target
                .and_then(|t| t.model_entry.value.provider.as_deref())
                .unwrap_or(""),
            model: target.map_or(crate::usage_attr::UNRESOLVED_MODEL_LABEL, |t| {
                t.model_entry.value.display_name.as_str()
            }),
            ..Default::default()
        },
        status,
        elapsed,
    );
    let mut resp = match result {
        Ok((resp, _)) => resp,
        Err(err) => err.into_response(),
    };
    if let Ok(hv) = HeaderValue::from_str(&request_id) {
        resp.headers_mut().insert("x-aisix-request-id", hv);
    }
    resp
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
            if !reachable(snapshot, auth, &entry.value, &client.source_ip) {
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
            let entry =
                crate::jobs::resolve_named_direct(snapshot, auth, &name, "stored-response")?;
            (entry, id.to_string(), ResponseMark::default())
        }
    };
    crate::jobs::require_safe_upstream_id(&raw)?;
    crate::dispatch::check_ip_access(&entry.value, &client.source_ip)?;
    let target = native_target(snapshot, entry, client).ok_or_else(not_found)?;
    Ok((target, raw, mark))
}

/// Whether the caller may address `target` itself, or a Model Group that
/// routes to it and admits the client IP — the group's own `allowed_cidrs`
/// gate its targets just as they gate a request naming the group.
fn reachable(
    snapshot: &aisix_core::AisixSnapshot,
    auth: &AuthenticatedKey,
    target: &Model,
    source_ip: &str,
) -> bool {
    let key = auth.key();
    key.can_access(snapshot, &target.display_name)
        || snapshot.models.entries().iter().any(|group| {
            group.value.routing.as_ref().is_some_and(|r| {
                r.targets
                    .iter()
                    .any(|t| t.model_ref(snapshot) == target.display_name)
            }) && key.can_access(snapshot, &group.value.display_name)
                && group.value.ip_allowed(source_ip)
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

fn with_content_type(
    builder: axum::http::response::Builder,
    content_type: Option<&HeaderValue>,
) -> axum::http::response::Builder {
    builder.header(
        header::CONTENT_TYPE,
        content_type
            .cloned()
            .unwrap_or_else(|| HeaderValue::from_static("application/json")),
    )
}

/// A streamed retrieve relayed frame by frame with the target's tags, under
/// the model's streaming budget between chunks.
fn relay_stream(
    state: &ProxyState,
    target: &JobTarget,
    upstream: reqwest::Response,
    mark: ResponseMark,
) -> Response {
    crate::attribution::note_upstream_wait_end(true);
    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let builder = with_content_type(
        Response::builder().status(status),
        upstream.headers().get(header::CONTENT_TYPE),
    );
    let idle =
        crate::routing::effective_timeouts(&target.model_entry.value, None, state.default_timeouts)
            .stream;
    let runtime_status = state.runtime_status.clone();
    let model_id = target.model_entry.id.clone();
    let cooldown = target.model_entry.value.cooldown.clone();
    let mut upstream = upstream.bytes_stream();
    let tagged = async_stream::stream! {
        let mut buf: Vec<u8> = Vec::new();
        loop {
            let next: Result<Option<Bytes>, String> = match idle {
                Some(d) => match tokio::time::timeout(d, upstream.next()).await {
                    Ok(next) => next.transpose().map_err(|e| aisix_gateway::transport_error_message(&e)),
                    Err(_) => Err("stream idle timeout".to_string()),
                },
                None => upstream
                    .next()
                    .await
                    .transpose()
                    .map_err(|e| aisix_gateway::transport_error_message(&e)),
            };
            let chunk = match next {
                Ok(None) => break,
                Ok(Some(chunk)) => chunk,
                Err(reason) => {
                    tracing::warn!(model_id = %model_id, %reason, "stored-response stream ended early");
                    let _ = crate::cooldown::note_failure(
                        &runtime_status,
                        &model_id,
                        cooldown.as_ref(),
                        aisix_gateway::BridgeError::Transport(reason.clone()),
                    );
                    yield Err(std::io::Error::other(reason));
                    return;
                }
            };
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
        if !buf.is_empty() {
            yield Ok(Bytes::from(buf));
        }
    };
    builder.body(Body::from_stream(tagged)).unwrap()
}
