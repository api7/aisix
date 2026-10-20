//! `/v1/messages` and `/v1/messages/count_tokens` for Claude on Vertex
//! AI, served on its own Anthropic Messages routes: `:rawPredict`,
//! `:streamRawPredict` and the `count-tokens:rawPredict` model. The
//! caller's body reaches Claude as it was sent, apart from what
//! [`aisix_provider_anthropic::platform::shape_request`] adapts, and the
//! response — JSON or Anthropic SSE — comes back as Vertex AI sent it.

use std::time::Instant;

use aisix_gateway::media_fetch::{fetch_urls, Encoding};
use aisix_gateway::{BridgeContext, BridgeError, NativeMessages};
use aisix_provider_anthropic::platform::{self, Platform};
use futures::StreamExt;
use http::header::{HeaderName, HeaderValue};
use serde_json::Value;

use super::{
    build_request_headers, map_http_error, outbound_bytes, upstream_model, validate_url_token,
    with_deadline, VertexBridge, VertexPublisher, VertexSecret,
};

pub(super) fn serves_anthropic_messages(upstream_id: &str) -> bool {
    VertexPublisher::from_upstream_id(upstream_id) == Some(VertexPublisher::Anthropic)
}

impl VertexBridge {
    fn anthropic_url(
        &self,
        ctx: &BridgeContext,
        creds: &VertexSecret,
        model: &str,
        action: &'static str,
    ) -> Result<aisix_gateway::url_cache::EndpointUrl, BridgeError> {
        validate_url_token("project", &creds.project)?;
        validate_url_token("region", &creds.region)?;
        validate_url_token("upstream_id", model)?;
        aisix_gateway::url_cache::cached_model_endpoint_url(
            self.url_cache_key(ctx),
            action,
            model,
            &[
                ctx.provider_key.api_base.as_deref().unwrap_or(""),
                &creds.project,
                &creds.region,
            ],
            || {
                let base =
                    self.resolve_api_base(&creds.region, ctx.provider_key.api_base.as_deref())?;
                let method = match action {
                    "vertex/anthropic-native-stream" => "streamRawPredict",
                    _ => "rawPredict",
                };
                Ok::<_, BridgeError>(format!(
                    "{base}/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{model}:{method}",
                    project = creds.project,
                    region = creds.region,
                ))
            },
        )
    }

    pub(super) async fn native_messages(
        &self,
        mut body: Value,
        ctx: &BridgeContext,
    ) -> Result<NativeMessages, BridgeError> {
        let creds = VertexSecret::parse(&ctx.provider_key.api_key)?;
        let upstream_id = upstream_model(ctx)?;
        let stream = body.get("stream").and_then(Value::as_bool) == Some(true);
        let url = self.anthropic_url(
            ctx,
            &creds,
            upstream_id,
            if stream {
                "vertex/anthropic-native-stream"
            } else {
                "vertex/anthropic-native"
            },
        )?;
        let fetched = fetch_urls(platform::media_urls(&body), ctx, Encoding::Base64).await?;
        let charged = fetched.charge(ctx);
        let ctx = &*charged;
        platform::inline_media(&mut body, Platform::Vertex, &fetched)
            .map_err(BridgeError::InvalidRequest)?;
        drop(fetched);

        let access_token = creds.resolve_access_token(&self.token_minter).await?;
        let mut headers = build_request_headers(&access_token, &ctx.request_id, &ctx.header_ctx())?;
        // Every `anthropic-beta` value reaching Vertex AI — the caller's
        // own, and any an operator header adds — is filtered to the ones it
        // takes; an unknown one is a 400 there.
        let mut betas = platform::header_betas(ctx.client_headers.as_deref());
        betas.extend(platform::header_betas(Some(&headers)));
        let betas = platform::shape_request(&mut body, Platform::Vertex, upstream_id, betas);
        let beta_header = HeaderName::from_static("anthropic-beta");
        headers.remove(&beta_header);
        if !betas.is_empty() {
            let value = HeaderValue::from_str(&betas.join(","))
                .map_err(|e| BridgeError::Config(format!("anthropic-beta header: {e}")))?;
            headers.insert(beta_header, value);
        }
        let bytes = outbound_bytes(&body)?;
        drop(body);

        let client = self.client_for(ctx);
        let started = Instant::now();
        let resp = with_deadline(ctx.deadline, started, async {
            url.post_on(&client)
                .headers(headers)
                .body(bytes)
                .send()
                .await
                .map_err(aisix_gateway::send_error)
        })
        .await?;
        aisix_gateway::note_upstream_response(resp.headers());
        let status = resp.status();
        if !status.is_success() {
            return Err(map_http_error(status, resp).await);
        }
        if !stream || aisix_gateway::response_is_json(&resp) {
            let json = with_deadline(ctx.deadline, started, async {
                resp.json::<Value>()
                    .await
                    .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))
            })
            .await?;
            return Ok(NativeMessages::Json(json));
        }
        let frames = resp.bytes_stream().map(|item| {
            item.map_err(|e| BridgeError::Transport(aisix_gateway::transport_error_message(&e)))
        });
        Ok(NativeMessages::Stream(Box::pin(frames)))
    }

    pub(super) async fn native_count_tokens(
        &self,
        body: Value,
        ctx: &BridgeContext,
    ) -> Result<Value, BridgeError> {
        let creds = VertexSecret::parse(&ctx.provider_key.api_key)?;
        // The token-counting model refuses a versioned name
        // (`claude-sonnet-4-6@default`).
        let upstream_id = upstream_model(ctx)?;
        let model = upstream_id.split('@').next().unwrap_or(upstream_id);
        let body = platform::vertex_count_tokens_body(&body, model);
        let url =
            self.anthropic_url(ctx, &creds, "count-tokens", "vertex/anthropic-count-tokens")?;
        let access_token = creds.resolve_access_token(&self.token_minter).await?;
        let mut headers = build_request_headers(&access_token, &ctx.request_id, &ctx.header_ctx())?;
        // Token counting takes no beta, and an unknown one is a 400.
        headers.remove("anthropic-beta");
        let bytes = outbound_bytes(&body)?;
        let client = self.client_for(ctx);
        let started = Instant::now();
        with_deadline(ctx.deadline, started, async {
            let resp = url
                .post_on(&client)
                .headers(headers)
                .body(bytes)
                .send()
                .await
                .map_err(aisix_gateway::send_error)?;
            aisix_gateway::note_upstream_response(resp.headers());
            let status = resp.status();
            if !status.is_success() {
                return Err(map_http_error(status, resp).await);
            }
            let json: Value = resp
                .json()
                .await
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
            let tokens = json
                .get("input_tokens")
                .and_then(Value::as_u64)
                .ok_or_else(|| {
                    BridgeError::UpstreamDecode(
                        "count-tokens answer carries no input_tokens".into(),
                    )
                })?;
            Ok(serde_json::json!({"input_tokens": tokens}))
        })
        .await
    }
}
