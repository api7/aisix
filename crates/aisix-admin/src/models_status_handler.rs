//! `GET /admin/v1/models/status` — runtime per-model status.

use axum::extract::State;
use axum::Json;
use serde::Serialize;
use std::time::Duration;

use aisix_proxy::{RuntimeStatus, RuntimeStatusSnapshot};

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
    let tracker = state.runtime_status_tracker.as_ref();

    let models = all_models
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
        .collect();

    Ok(Json(models))
}
