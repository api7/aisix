//! `GET /admin/v1/spend` — aggregate spend reporting.
//!
//! Returns current-month USD spend per ApiKey, sourced from the in-process
//! [`aisix_proxy::budget::BudgetTracker`]. If no tracker is wired (e.g. the
//! admin server runs standalone), the endpoint returns an empty list rather
//! than an error.
//!
//! Response shape:
//! ```json
//! {
//!   "period": "2026-04",
//!   "total_usd": 12.34,
//!   "entries": [
//!     {"api_key_id": "k-uuid-1", "api_key": "sk-...", "spend_usd": 12.34}
//!   ]
//! }
//! ```

use axum::extract::State;
use axum::Json;
use chrono::Utc;
use serde::Serialize;

use crate::auth::AdminAuth;
use crate::state::AdminState;

#[derive(Debug, Serialize)]
pub struct SpendEntry {
    pub api_key_id: String,
    /// The `key` field of the matching ApiKey (masked for safety in logs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_hint: Option<String>,
    pub spend_usd: f64,
}

#[derive(Debug, Serialize)]
pub struct SpendResponse {
    /// ISO 8601 year-month (e.g. "2026-04").
    pub period: String,
    pub total_usd: f64,
    pub entries: Vec<SpendEntry>,
}

pub async fn get_spend(_auth: AdminAuth, State(state): State<AdminState>) -> Json<SpendResponse> {
    let period = Utc::now().format("%Y-%m").to_string();

    let tracker = match &state.budget_tracker {
        Some(t) => t.clone(),
        None => {
            return Json(SpendResponse {
                period,
                total_usd: 0.0,
                entries: vec![],
            });
        }
    };

    let raw_entries = tracker.all_entries();
    let total_usd: f64 = raw_entries.iter().map(|(_, v)| v).sum();

    // Enrich entries with the ApiKey's masked key value so operators can
    // correlate spend to a credential without the handler needing access
    // to the full key. We do a best-effort lookup from the snapshot.
    let snapshot = state.snapshot.load();
    let entries: Vec<SpendEntry> = raw_entries
        .into_iter()
        .map(|(api_key_id, spend_usd)| {
            // Find the ApiKey whose runtime_id matches api_key_id. The
            // snapshot secondary index is by `key` (the bearer string),
            // not by uuid, so we do a linear scan here. This is called
            // only for human-facing reporting — not on the hot path.
            // v3 (§9A.7B.4): plaintext is unavailable post-creation;
            // the hash is what we have. The hint is informational, so
            // showing the first 8 chars of the SHA-256 hex is fine for
            // operator-facing reporting (and stable across restarts).
            let api_key_hint = snapshot
                .apikeys
                .entries()
                .into_iter()
                .find(|e| e.id == api_key_id)
                .map(|e| mask_key(&e.value.key_hash));
            SpendEntry {
                api_key_id,
                api_key_hint,
                spend_usd,
            }
        })
        .collect();

    Json(SpendResponse {
        period,
        total_usd,
        entries,
    })
}

/// Return the first 7 characters of a key followed by `…` so logs are
/// useful for identification without leaking the full secret.
fn mask_key(key: &str) -> String {
    if key.len() <= 7 {
        key.to_string()
    } else {
        format!("{}…", &key[..7])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_short_key_unchanged() {
        assert_eq!(mask_key("sk-abc"), "sk-abc");
    }

    #[test]
    fn mask_long_key_truncates() {
        assert_eq!(mask_key("sk-verylongkey"), "sk-very…");
    }
}
