//! Runtime per-model status: the render shared by
//! `GET /admin/v1/models/status` (admin listener) and
//! `GET /status/models` (metrics/status listener).

use axum::extract::State;
use axum::Json;
use serde::Serialize;
use std::time::Duration;

use aisix_core::resource::ResourceEntry;
use aisix_core::Model;
use aisix_proxy::{ModelRuntimeStatusTracker, RuntimeStatus, RuntimeStatusSnapshot};

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Direct,
    Routing,
    Ensemble,
    Semantic,
}

#[derive(Debug, Serialize)]
pub struct ModelStatusView {
    pub id: String,
    pub display_name: String,
    pub kind: ModelKind,
    #[serde(flatten)]
    pub details: RuntimeStatusSnapshot,
}

pub async fn get_models_status(
    _auth: AdminAuth,
    State(state): State<AdminState>,
) -> Result<Json<Vec<ModelStatusView>>, AdminError> {
    let all_models = state.store.list_models().await?;
    Ok(Json(render_models_status(
        all_models,
        state.runtime_status_tracker.as_deref(),
    )))
}

/// Render the per-model runtime status view from a resource listing plus
/// the proxy's runtime tracker. The single render behind both serving
/// endpoints — admin listener and metrics/status listener — so the two
/// responses cannot drift while both exist.
pub(crate) fn render_models_status(
    all_models: Vec<ResourceEntry<Model>>,
    tracker: Option<&ModelRuntimeStatusTracker>,
) -> Vec<ModelStatusView> {
    all_models
        .into_iter()
        .map(|entry| {
            // Virtual routers (routing / ensemble / semantic) have no
            // upstream of their own, so runtime health is not applicable —
            // it lives on the direct Models they dispatch to.
            let virtual_kind = if entry.value.is_routing() {
                Some(ModelKind::Routing)
            } else if entry.value.is_ensemble() {
                Some(ModelKind::Ensemble)
            } else if entry.value.is_semantic() {
                Some(ModelKind::Semantic)
            } else {
                None
            };
            if let Some(kind) = virtual_kind {
                ModelStatusView {
                    id: entry.id,
                    display_name: entry.value.display_name,
                    kind,
                    details: RuntimeStatusSnapshot {
                        status: RuntimeStatus::NotApplicable,
                        ..RuntimeStatusSnapshot::default()
                    },
                }
            } else {
                let details = tracker
                    .map(|t| {
                        let stale_after = entry
                            .value
                            .background_model_check
                            .as_ref()
                            .map(|cfg| Duration::from_secs(cfg.stale_after_seconds));
                        t.status_with_stale(&entry.id, stale_after)
                    })
                    .unwrap_or_default();
                ModelStatusView {
                    id: entry.id,
                    display_name: entry.value.display_name,
                    kind: ModelKind::Direct,
                    details,
                }
            }
        })
        .collect()
}
