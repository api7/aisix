//! What the request has committed to upstream, for the terminal emitters
//! that cannot see it.
//!
//! Two of them run without the values the rest of the request already
//! resolved:
//!
//! - the client-cancel guard in `record_request_telemetry` is armed in
//!   middleware — before auth, body parsing and model resolution — and
//!   fires from `Drop` once the handler future has been dropped, so it
//!   could only ever label `endpoint` (AISIX-Cloud#1317);
//! - every handler's failure branch holds a `ProxyError`, which carries
//!   no upstream identity. A request that reached a real provider and
//!   came back 502 therefore reported `provider="unknown"` on the same
//!   series its successes report the real provider, which puts successes
//!   and failures of one ProviderKey in different time series and makes
//!   a per-provider failure rate meaningless (AISIX-Cloud#1325).
//!
//! Both read the cell the telemetry middleware installs for the request.
//! Writes ride the two resolution chokepoints every endpoint already goes
//! through — [`crate::model_resolve::resolve_model`] for the model the
//! caller addressed and [`crate::dispatch::resolve_provider_key`] for the
//! target about to be dispatched to — so a new endpoint is attributed
//! without opting in, and a retry / fallback loop leaves the LAST target
//! it selected behind, which is the attempt the client's error came from.
//!
//! It is a task-local rather than a threaded parameter because the cancel
//! guard reads it from outside the handler entirely. A write with no
//! scope installed (background health checks, unit tests) is dropped.

use std::future::Future;
use std::sync::{Arc, Mutex};

use aisix_core::Model;

/// What the request resolved, as it resolved it. An empty field means
/// "never got there", which readers render as the `unknown` label value
/// the request families already use for the same condition.
#[derive(Clone, Default)]
pub(crate) struct Resolved {
    /// The model name the CALLER addressed — raw, so it must be bounded
    /// through `usage_attr::metric_model_label` before it becomes a
    /// label (#451). Kept raw here because the bounding needs a snapshot
    /// and only the readers have one.
    pub requested_model: String,
    /// Vendor id of the last target the request selected.
    pub provider: String,
    /// That target's upstream model id.
    pub upstream_model: String,
    /// That target's ProviderKey id. The readable name is resolved from
    /// it at read time, so the pair is byte-identical to the one the
    /// success path emits.
    pub provider_key_id: String,
}

/// The per-request cell. Attempts within a request are sequential, so the
/// lock is uncontended; it exists because the cancel guard may read the
/// cell from a different point in the stack than the writer.
#[derive(Default)]
pub(crate) struct RequestAttribution(Mutex<Resolved>);

impl RequestAttribution {
    pub(crate) fn get(&self) -> Resolved {
        self.lock().clone()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Resolved> {
        self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

tokio::task_local! {
    static CURRENT: Arc<RequestAttribution>;
}

/// Install `attribution` as the cell for everything `fut` does.
pub(crate) fn scope<F: Future>(
    attribution: Arc<RequestAttribution>,
    fut: F,
) -> impl Future<Output = F::Output> {
    CURRENT.scope(attribution, fut)
}

/// Note the model name the caller addressed. Called once per request from
/// the resolution chokepoint, and only when the name resolved — an
/// unresolvable name never reaches a target and its request fails before
/// anything can read the cell.
pub(crate) fn note_requested_model(requested: &str) {
    with(|r| {
        if r.requested_model.is_empty() {
            r.requested_model = requested.to_string();
        }
    });
}

/// Note the target this request is about to dispatch to. Called for every
/// attempt, so the cell holds the last one — the target whose failure the
/// caller was ultimately served.
pub(crate) fn note_target(model: &Model, provider_key_id: &str) {
    with(|r| {
        r.provider = model.provider.clone().unwrap_or_default();
        r.upstream_model = model.upstream_model().unwrap_or_default().to_string();
        r.provider_key_id = provider_key_id.to_string();
    });
}

/// What the current request has resolved, or `None` outside a request.
pub(crate) fn current() -> Option<Resolved> {
    CURRENT.try_with(|a| a.get()).ok()
}

fn with(f: impl FnOnce(&mut Resolved)) {
    let _ = CURRENT.try_with(|a| f(&mut a.lock()));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(provider: &str, upstream: &str) -> Model {
        serde_json::from_value(serde_json::json!({
            "display_name": "m",
            "provider": provider,
            "model_name": upstream,
            "provider_key_id": "pk-1",
        }))
        .unwrap()
    }

    /// A write with no request scope must not panic — `resolve_provider_key`
    /// also runs from the background health checker.
    #[test]
    fn writes_outside_a_request_are_dropped() {
        note_requested_model("gpt-4o");
        note_target(&model("openai", "gpt-4o"), "pk-1");
        assert!(current().is_none());
    }

    #[tokio::test]
    async fn the_last_attempted_target_is_what_a_failure_reads() {
        scope(Arc::new(RequestAttribution::default()), async {
            note_requested_model("my-group");
            note_target(&model("openai", "gpt-4o"), "pk-openai");
            // The group fell back; the caller's error came from the second
            // target, so that is the one the terminal emit must name.
            note_target(&model("anthropic", "claude-3-5-sonnet"), "pk-anthropic");
            let r = current().expect("in scope");
            assert_eq!(r.requested_model, "my-group");
            assert_eq!(r.provider, "anthropic");
            assert_eq!(r.upstream_model, "claude-3-5-sonnet");
            assert_eq!(r.provider_key_id, "pk-anthropic");
        })
        .await;
    }

    /// The caller-addressed name is the FIRST one noted: a routed request
    /// resolves its group, then each target, and the `model` label belongs
    /// to what the client asked for.
    #[tokio::test]
    async fn the_requested_model_is_not_overwritten_by_a_target() {
        scope(Arc::new(RequestAttribution::default()), async {
            note_requested_model("my-group");
            note_requested_model("gpt-4o");
            assert_eq!(current().unwrap().requested_model, "my-group");
        })
        .await;
    }
}
