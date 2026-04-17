//! `POST /v1/chat/completions` handler.
//!
//! Flow:
//! 1. [`AuthenticatedKey`] extractor runs first — rejects unauthenticated
//!    requests with a 401 envelope.
//! 2. Parse [`ChatFormat`] from the JSON body.
//! 3. Resolve `req.model` against the snapshot's Model table → 404 if
//!    absent.
//! 4. Check the ApiKey's `allowed_models` whitelist → 403 if disallowed.
//! 5. Look up the matching `Bridge` on the Hub by `Model::provider()` →
//!    503 if no bridge registered.
//! 6. Build a [`BridgeContext`] and dispatch:
//!    - `stream == true`  → `chat_stream` + Sse response
//!    - otherwise          → `chat` + JSON response rendered as OpenAI
//! 7. Any `BridgeError` surfaces through [`ProxyError::Bridge`] which
//!    supplies the right HTTP status and OpenAI-style error type.

use aisix_gateway::{BridgeContext, ChatFormat};
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::{Stream, StreamExt};
use std::convert::Infallible;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::render::{render_chunk, render_response};
use crate::state::ProxyState;

pub async fn chat_completions(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    Json(req): Json<ChatFormat>,
) -> Result<Response, ProxyError> {
    if req.messages.is_empty() {
        return Err(ProxyError::InvalidRequest(
            "messages array must not be empty".into(),
        ));
    }

    let snapshot = state.snapshot.load();
    let model_entry = snapshot
        .models
        .get_by_name(&req.model)
        .ok_or_else(|| ProxyError::ModelNotFound(req.model.clone()))?;

    if !auth.key().can_access(&req.model) {
        return Err(ProxyError::ModelForbidden(req.model.clone()));
    }

    let provider = model_entry
        .value
        .provider()
        .ok_or_else(|| ProxyError::InvalidRequest("model has no provider prefix".into()))?;
    let bridge = state
        .hub
        .get(provider)
        .ok_or(ProxyError::ProviderUnavailable)?;

    let request_id = format!("req-{}", Uuid::new_v4());
    let model_arc = std::sync::Arc::new(model_entry.value.clone());
    let ctx = BridgeContext::new(&request_id, model_arc);

    let now = created_ts();

    if req.is_streaming() {
        let upstream = bridge.chat_stream(&req, &ctx).await?;
        let model_name = req.model.clone();
        let sse_stream = build_sse_stream(upstream, model_name, now);
        let response =
            Sse::new(sse_stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)));
        return Ok(response.into_response());
    }

    let upstream = bridge.chat(&req, &ctx).await?;
    let rendered = render_response(now, upstream);
    Ok(Json(rendered).into_response())
}

fn created_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn build_sse_stream(
    upstream: aisix_gateway::ChatChunkStream,
    _model: String,
    created: i64,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        futures::pin_mut!(upstream);
        while let Some(item) = upstream.next().await {
            let ev = match item {
                Ok(chunk) => {
                    let rendered = render_chunk(created, chunk);
                    match serde_json::to_string(&rendered) {
                        Ok(json) => Event::default().data(json),
                        Err(err) => Event::default()
                            .event("error")
                            .data(err.to_string()),
                    }
                }
                Err(err) => Event::default()
                    .event("error")
                    .data(err.to_string()),
            };
            yield Ok::<_, Infallible>(ev);
        }
        // Emit the OpenAI-style [DONE] sentinel so clients that terminate
        // on it behave correctly.
        yield Ok::<_, Infallible>(Event::default().data("[DONE]"));
    }
}
