//! `GET /admin/v1/health` — per-model health status.
//!
//! Returns the health level for every Model currently in the snapshot,
//! enriched with live failure counters from the in-process
//! [`aisix_proxy::HealthTracker`]. If no tracker is wired the endpoint
//! still returns all models with level 0 (Healthy).
//!
//! Response shape:
//! ```json
//! {
//!   "status": "ok",
//!   "models": [
//!     {"id": "m-uuid", "name": "my-gpt4", "health": 0},
//!     {"id": "m-uuid-2", "name": "claude", "health": 1}
//!   ]
//! }
//! ```
//!
//! **Health levels**:
//! - `0` — Healthy (no recent failures)
//! - `1` — Degraded (4–7 consecutive upstream failures)
//! - `2` — Down (8+ consecutive upstream failures)

use axum::extract::State;
use axum::Json;
use serde::Serialize;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

#[derive(Debug, Serialize)]
pub struct ModelHealth {
    pub id: String,
    pub name: String,
    /// Numeric health level: 0 = Healthy, 1 = Degraded, 2 = Down.
    pub health: u8,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Overall gateway status — always "ok" at the protocol level; operators
    /// should look at individual model health levels for actionable signal.
    pub status: &'static str,
    pub models: Vec<ModelHealth>,
}

pub async fn get_health(
    _auth: AdminAuth,
    State(state): State<AdminState>,
) -> Result<Json<HealthResponse>, AdminError> {
    // Read from the store so the list is always consistent with what
    // operators have written — the snapshot is updated asynchronously by
    // the etcd watch supervisor and may lag by up to 500 ms.
    let all_models = state.store.list_models().await?;

    let models: Vec<ModelHealth> = all_models
        .into_iter()
        .map(|entry| {
            let health_level = state
                .health_tracker
                .as_ref()
                .map(|t| {
                    let level = t.level(&entry.value.display_name);
                    u8::from(level)
                })
                .unwrap_or(0); // no tracker → assume Healthy

            ModelHealth {
                id: entry.id.clone(),
                name: entry.value.display_name.clone(),
                health: health_level,
            }
        })
        .collect();

    Ok(Json(HealthResponse {
        status: "ok",
        models,
    }))
}

#[cfg(test)]
mod tests {
    use aisix_proxy::HealthTracker;
    use std::sync::Arc;

    fn make_tracker() -> Arc<HealthTracker> {
        Arc::new(HealthTracker::new())
    }

    #[test]
    fn health_level_serialises_to_u8() {
        use aisix_proxy::health::HealthLevel;
        let h: u8 = u8::from(HealthLevel::Healthy);
        assert_eq!(h, 0);
        let d: u8 = u8::from(HealthLevel::Degraded);
        assert_eq!(d, 1);
        let down: u8 = u8::from(HealthLevel::Down);
        assert_eq!(down, 2);
    }

    #[test]
    fn tracker_level_reflects_failures() {
        let t = make_tracker();
        assert_eq!(t.level("m"), aisix_proxy::health::HealthLevel::Healthy);
        // 4 failures → degraded
        for _ in 0..4 {
            t.record_failure("m");
        }
        assert_eq!(t.level("m"), aisix_proxy::health::HealthLevel::Degraded);
        // 8+ → down
        for _ in 0..4 {
            t.record_failure("m");
        }
        assert_eq!(t.level("m"), aisix_proxy::health::HealthLevel::Down);
        // success resets
        t.record_success("m");
        assert_eq!(t.level("m"), aisix_proxy::health::HealthLevel::Healthy);
    }
}
