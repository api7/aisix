use std::sync::Arc;
use std::time::Duration;

use aisix_core::models::BackgroundModelCheck;
use aisix_core::{AisixSnapshot, Model};
use aisix_gateway::{BridgeContext, BridgeError, ChatFormat, ChatMessage, Hub};
use tokio::sync::Semaphore;

use crate::dispatch;
use crate::health::ModelRuntimeStatusTracker;

/// Cap on the number of background model checks that may run
/// concurrently across all configured direct models. Each check
/// issues a real chat completion against the upstream provider —
/// burning the operator's quota and dollars — so we serialize them
/// to keep the cost bounded regardless of how many direct models
/// the operator has registered.
///
/// Rationale: a deployment with 100 direct models all configured
/// with the same `interval_seconds` would otherwise fan out 100
/// concurrent requests to upstream providers every interval. The
/// semaphore turns that into a slow trickle of ≤4 in-flight checks
/// at any time. The total cost-per-interval is unchanged; the
/// burstiness (and the chance of self-induced 429 on small
/// accounts) is dampened.
const MAX_CONCURRENT_BACKGROUND_CHECKS: usize = 4;

pub async fn run_background_model_check_once(
    snapshot: Arc<AisixSnapshot>,
    hub: Arc<Hub>,
    tracker: Arc<ModelRuntimeStatusTracker>,
    request_id: &str,
) {
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_BACKGROUND_CHECKS));
    let mut tasks = Vec::new();
    for entry in snapshot.models.entries() {
        let model = &entry.value;
        if model.is_routing() {
            continue;
        }
        let Some(cfg) = model.background_model_check.as_ref() else {
            continue;
        };
        if !cfg.enabled {
            continue;
        }
        let id = entry.id.clone();
        let model = model.clone();
        let cfg = cfg.clone();
        let snapshot = Arc::clone(&snapshot);
        let hub = Arc::clone(&hub);
        let tracker = Arc::clone(&tracker);
        let request_id = request_id.to_string();
        let permits = Arc::clone(&semaphore);
        tasks.push(tokio::spawn(async move {
            // Acquire concurrency permit. If the semaphore is closed
            // (only happens during shutdown), the check is skipped
            // and the tracker stays at last-known state — fine.
            let _permit = match permits.acquire_owned().await {
                Ok(p) => p,
                Err(_) => return,
            };
            run_one(snapshot, hub, tracker, &id, &model, &cfg, &request_id).await;
        }));
    }
    for task in tasks {
        let _ = task.await;
    }
}

async fn run_one(
    snapshot: Arc<AisixSnapshot>,
    hub: Arc<Hub>,
    tracker: Arc<ModelRuntimeStatusTracker>,
    id: &str,
    model: &Model,
    cfg: &BackgroundModelCheck,
    request_id: &str,
) {
    let outcome = check_direct_model(&snapshot, &hub, id, model, cfg, request_id).await;
    match outcome {
        Ok(()) => tracker.clear_unhealthy(id),
        Err(BridgeError::UpstreamStatus { status, .. })
            if cfg.ignore_statuses.contains(&status) =>
        {
            tracker.record_ignored_check(id, status, "ignored_transient_error")
        }
        Err(BridgeError::Timeout { .. }) if cfg.ignore_statuses.contains(&408) => {
            tracker.record_ignored_check(id, 408, "ignored_transient_error")
        }
        Err(err) => {
            tracker.mark_unhealthy(id, background_status_code(&err), "background_check_failed")
        }
    }
}

async fn check_direct_model(
    snapshot: &AisixSnapshot,
    hub: &Hub,
    model_id: &str,
    model: &Model,
    cfg: &BackgroundModelCheck,
    request_id: &str,
) -> Result<(), BridgeError> {
    let _provider =
        dispatch::require_provider(model).map_err(|e| BridgeError::Config(e.to_string()))?;
    let pk_entry = dispatch::resolve_provider_key(snapshot, model)
        .map_err(|e| BridgeError::Config(e.to_string()))?;
    let bridge = dispatch::resolve_bridge(hub, &pk_entry.value).ok_or_else(|| {
        BridgeError::Config(format!(
            "no bridge registered for provider_key provider={:?} adapter={:?}",
            pk_entry.value.provider, pk_entry.value.adapter
        ))
    })?;

    let req = ChatFormat {
        model: model.display_name.clone(),
        messages: vec![ChatMessage::user(cfg.prompt.clone())],
        max_tokens: Some(cfg.max_tokens),
        ..ChatFormat::new(model.display_name.clone(), vec![])
    };
    let model_arc = Arc::new(model.clone());
    let pk_arc = Arc::new(pk_entry.value.clone());
    let ctx = BridgeContext::new(request_id, model_arc, pk_arc).with_deadline(timeout(cfg));

    let _ = bridge.chat(&req, &ctx).await?;
    let _ = model_id;
    Ok(())
}

fn timeout(cfg: &BackgroundModelCheck) -> Duration {
    Duration::from_secs(cfg.timeout_seconds)
}

fn background_status_code(err: &BridgeError) -> Option<u16> {
    match err {
        BridgeError::UpstreamStatus { status, .. } => Some(*status),
        BridgeError::Timeout { .. } => Some(408),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use aisix_core::resource::ResourceEntry;
    use aisix_provider_openai::OpenAiBridge;
    use reqwest::Client;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn openai_test_bridge() -> OpenAiBridge {
        let client = Client::builder()
            .user_agent("aisix-test/0.1")
            .no_proxy()
            .build()
            .unwrap();
        OpenAiBridge::with_client(client)
    }

    fn provider_key_entry(id: &str, api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let cfg = format!(
            r#"{{"display_name":"pk-{id}","secret":"sk-upstream","api_base":"{api_base}","provider":"openai","adapter":"openai"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new(id, pk, 1)
    }

    fn direct_model_entry(
        id: &str,
        name: &str,
        pk_id: &str,
        enabled: bool,
        ignore: &[u16],
    ) -> ResourceEntry<Model> {
        let cfg = serde_json::json!({
            "display_name": name,
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "provider_key_id": pk_id,
            "background_model_check": {
                "enabled": enabled,
                "interval_seconds": 30,
                "timeout_seconds": 10,
                "prompt": "Respond with OK",
                "max_tokens": 8,
                "ignore_statuses": ignore,
                "stale_after_seconds": 90
            }
        });
        let model: Model = serde_json::from_value(cfg).unwrap();
        ResourceEntry::new(id, model, 1)
    }

    #[tokio::test]
    async fn background_check_marks_unhealthy_on_failure() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(503).set_body_string("down"))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snapshot = Arc::new(AisixSnapshot::new());
        snapshot
            .provider_keys
            .insert(provider_key_entry("pk-1", &upstream.uri()));
        snapshot.models.insert(direct_model_entry(
            "m-1",
            "bg-model",
            "pk-1",
            true,
            &[408, 429],
        ));
        let tracker = Arc::new(ModelRuntimeStatusTracker::new());

        run_background_model_check_once(snapshot, hub, tracker.clone(), "bg-check-1").await;

        let status = tracker.status("m-1");
        assert_eq!(status.status, crate::RuntimeStatus::Unhealthy);
        assert_eq!(status.last_check_status, Some(503));
    }

    #[tokio::test]
    async fn background_check_ignores_configured_transient_status() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("slow down"))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snapshot = Arc::new(AisixSnapshot::new());
        snapshot
            .provider_keys
            .insert(provider_key_entry("pk-1", &upstream.uri()));
        snapshot.models.insert(direct_model_entry(
            "m-1",
            "bg-model",
            "pk-1",
            true,
            &[408, 429],
        ));
        let tracker = Arc::new(ModelRuntimeStatusTracker::new());

        run_background_model_check_once(snapshot, hub, tracker.clone(), "bg-check-1").await;

        let status = tracker.status("m-1");
        assert_eq!(status.status, crate::RuntimeStatus::Healthy);
        assert_eq!(status.last_check_status, Some(429));
        assert_eq!(
            status.status_reason.as_deref(),
            Some("ignored_transient_error")
        );
    }
}
