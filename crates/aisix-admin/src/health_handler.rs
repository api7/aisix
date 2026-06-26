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
//!   ],
//!   "config": {
//!     "snapshot_revision": 1234567,
//!     "snapshot_age_seconds": 5
//!   }
//! }
//! ```
//!
//! **Health levels**:
//! - `0` — Healthy (no recent failures)
//! - `1` — Degraded (4–7 consecutive upstream failures)
//! - `2` — Down (8+ consecutive upstream failures)
//!
//! The `config` block surfaces the etcd watch supervisor's freshness
//! state — without it a wedged watch can let the gateway serve a
//! frozen snapshot for hours while still reporting healthy. See
//! issue #114. The block is omitted when the supervisor isn't wired
//! (legacy / test deployments).

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

/// Etcd watch supervisor freshness, surfaced so operators can detect
/// a wedged config stream. `snapshot_age_seconds` is `None` before the
/// supervisor's first apply (boot) — the JSON serialises that as
/// `null`, distinct from `0` (just-applied).
#[derive(Debug, Serialize)]
pub struct ConfigStatus {
    /// Highest etcd revision currently reflected in the snapshot. Zero
    /// before first apply.
    pub snapshot_revision: i64,
    /// Wall-clock seconds since the supervisor last applied an event.
    /// `null` when the supervisor has not yet completed its first
    /// cycle — a fresh boot before etcd is reachable. A large value
    /// (e.g. > 300) suggests a stalled watch.
    pub snapshot_age_seconds: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Aggregate health summary — `"ok"`, `"degraded"`, or `"unhealthy"`,
    /// derived from per-model health (`models[].health`) and config
    /// freshness (`config`). See [`aggregate_status`]. `models[].health`
    /// and `config` remain the per-dimension detail.
    pub status: &'static str,
    pub models: Vec<ModelHealth>,
    /// Etcd watch supervisor freshness. Omitted when the supervisor
    /// isn't wired into AdminState.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<ConfigStatus>,
}

/// A snapshot older than this (or never applied) means the etcd watch is
/// not delivering fresh config, so the gateway may be serving a frozen
/// snapshot — reflected as `degraded` (#114, #618).
const SNAPSHOT_STALE_SECS: u64 = 300;

/// Aggregate the top-level `status` from per-model health and config
/// freshness so a single glance reflects real state instead of a fixed
/// `"ok"` marker (#618):
///
/// - `"unhealthy"` — at least one model is Down (health level 2).
/// - `"degraded"` — at least one model is Degraded (level 1), or the
///   config watch is stale / not yet applied.
/// - `"ok"` — every model is Healthy and config is fresh (or the watch
///   supervisor isn't wired, so freshness can't be assessed).
fn aggregate_status(models: &[ModelHealth], config: Option<&ConfigStatus>) -> &'static str {
    if models.iter().any(|m| m.health >= 2) {
        return "unhealthy";
    }
    let degraded_model = models.iter().any(|m| m.health == 1);
    let stale_config = match config {
        Some(c) => match c.snapshot_age_seconds {
            None => true,
            Some(age) => age > SNAPSHOT_STALE_SECS,
        },
        None => false,
    };
    if degraded_model || stale_config {
        return "degraded";
    }
    "ok"
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

    let config = state.watch_status.as_ref().map(|ws| {
        let snap = ws.snapshot();
        ConfigStatus {
            snapshot_revision: snap.revision,
            snapshot_age_seconds: snap.last_apply_age.map(|d| d.as_secs()),
        }
    });

    let status = aggregate_status(&models, config.as_ref());

    Ok(Json(HealthResponse {
        status,
        models,
        config,
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

    /// Regression for issue #114: HealthResponse with no watch_status
    /// wired must not include a "config" key at all (older clients
    /// that use the legacy schema must keep parsing). With watch_status,
    /// "config.snapshot_revision" + "config.snapshot_age_seconds" must
    /// appear with the right values.
    #[test]
    fn response_serialisation_omits_config_when_watch_status_unwired() {
        use super::HealthResponse;
        let r = HealthResponse {
            status: "ok",
            models: vec![],
            config: None,
        };
        let json = serde_json::to_value(&r).unwrap();
        assert!(json.get("config").is_none());
    }

    #[test]
    fn response_serialisation_carries_config_when_watch_status_wired() {
        use super::{ConfigStatus, HealthResponse};
        let r = HealthResponse {
            status: "ok",
            models: vec![],
            config: Some(ConfigStatus {
                snapshot_revision: 1234567,
                snapshot_age_seconds: Some(5),
            }),
        };
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["config"]["snapshot_revision"], 1234567);
        assert_eq!(json["config"]["snapshot_age_seconds"], 5);
    }

    #[test]
    fn response_serialises_age_as_null_pre_first_apply() {
        // The supervisor hasn't applied anything yet → age is None →
        // JSON null, distinct from `0` (just-applied). Operators can
        // distinguish "boot, etcd unreached" from "fresh, healthy".
        use super::{ConfigStatus, HealthResponse};
        let r = HealthResponse {
            status: "ok",
            models: vec![],
            config: Some(ConfigStatus {
                snapshot_revision: 0,
                snapshot_age_seconds: None,
            }),
        };
        let json = serde_json::to_value(&r).unwrap();
        assert!(
            json["config"]["snapshot_age_seconds"].is_null(),
            "pre-first-apply age should serialise as null, got {}",
            json["config"]["snapshot_age_seconds"]
        );
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

    fn mh(health: u8) -> super::ModelHealth {
        super::ModelHealth {
            id: "id".into(),
            name: "m".into(),
            health,
        }
    }

    fn cfg(age: Option<u64>) -> super::ConfigStatus {
        super::ConfigStatus {
            snapshot_revision: 1,
            snapshot_age_seconds: age,
        }
    }

    #[test]
    fn aggregate_ok_when_all_healthy_and_fresh() {
        let models = vec![mh(0), mh(0)];
        assert_eq!(super::aggregate_status(&models, Some(&cfg(Some(5)))), "ok");
    }

    #[test]
    fn aggregate_ok_when_watch_status_unwired() {
        // No supervisor → freshness can't be assessed, base on models only.
        let models = vec![mh(0)];
        assert_eq!(super::aggregate_status(&models, None), "ok");
    }

    #[test]
    fn aggregate_degraded_when_a_model_degraded() {
        let models = vec![mh(0), mh(1)];
        assert_eq!(
            super::aggregate_status(&models, Some(&cfg(Some(5)))),
            "degraded"
        );
    }

    #[test]
    fn aggregate_unhealthy_when_a_model_down_even_if_config_stale() {
        let models = vec![mh(0), mh(2)];
        assert_eq!(
            super::aggregate_status(&models, Some(&cfg(Some(9999)))),
            "unhealthy"
        );
    }

    #[test]
    fn aggregate_degraded_when_config_stale() {
        let models = vec![mh(0)];
        assert_eq!(
            super::aggregate_status(&models, Some(&cfg(Some(super::SNAPSHOT_STALE_SECS + 1)))),
            "degraded"
        );
    }

    #[test]
    fn aggregate_degraded_when_config_never_applied() {
        let models = vec![mh(0)];
        assert_eq!(
            super::aggregate_status(&models, Some(&cfg(None))),
            "degraded"
        );
    }
}
