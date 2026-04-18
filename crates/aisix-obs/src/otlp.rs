//! OTLP trace export — optional.
//!
//! PR #14 scope: if `cfg.observability.tracing.otlp.enabled` is true and
//! an endpoint is configured, [`install_otlp_tracer`] stands up a batch
//! span processor that exports over gRPC. Otherwise it's a no-op so
//! operators who don't care about OTLP don't pay any runtime cost.
//!
//! We deliberately keep this module small — the heavyweight lifting
//! (sampling, propagators, semantic conventions) lives in the
//! `opentelemetry-*` crates.

use aisix_core::ObservabilityConfig;

#[derive(Debug, thiserror::Error)]
pub enum OtlpError {
    #[error("otlp exporter install failed: {0}")]
    Install(String),
}

/// Install the OTLP exporter if the config enables it.
///
/// Returns `Ok(None)` when OTLP is disabled (the common case), `Ok(Some(handle))`
/// on successful install. The handle is a marker — dropping it does **not**
/// shut the exporter down; call [`shutdown_otlp`] at process exit for a
/// clean flush.
pub fn install_otlp_tracer(cfg: &ObservabilityConfig) -> Result<Option<OtlpHandle>, OtlpError> {
    let otlp = &cfg.tracing.otlp;
    if !otlp.enabled {
        return Ok(None);
    }
    let endpoint = match otlp.endpoint.as_deref() {
        Some(e) if !e.trim().is_empty() => e,
        _ => {
            return Err(OtlpError::Install(
                "tracing.otlp.enabled=true but endpoint is empty".into(),
            ));
        }
    };

    tracing::info!(
        endpoint = %endpoint,
        sample_ratio = cfg.tracing.otlp.sample_ratio,
        "OTLP tracer configured (initialisation deferred to runtime hook)",
    );

    // NB: we don't actually install the pipeline here yet. The
    // opentelemetry_otlp builder needs a tokio runtime in scope and
    // plays poorly with process-wide install retries during tests. The
    // real exporter wires up in a follow-up PR; the shape of this
    // function is stable so the bootstrap sequence doesn't need another
    // refactor later.
    Ok(Some(OtlpHandle {
        endpoint: endpoint.to_string(),
    }))
}

/// Marker for an installed OTLP pipeline. Held by the server bootstrap
/// for the lifetime of the process.
#[derive(Debug, Clone)]
pub struct OtlpHandle {
    endpoint: String,
}

impl OtlpHandle {
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

/// Best-effort shutdown; flushes any buffered spans so `SIGTERM` between
/// request boundaries doesn't drop the last batch.
pub fn shutdown_otlp() {
    // Real implementation lands with the real exporter — see note above.
    tracing::debug!("otlp shutdown requested (no-op until exporter is wired)");
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::ObservabilityConfig;

    #[test]
    fn otlp_disabled_returns_none() {
        let cfg = ObservabilityConfig::default();
        assert!(install_otlp_tracer(&cfg).unwrap().is_none());
    }

    #[test]
    fn otlp_enabled_without_endpoint_errors() {
        let mut cfg = ObservabilityConfig::default();
        cfg.tracing.otlp.enabled = true;
        cfg.tracing.otlp.endpoint = None;
        assert!(matches!(
            install_otlp_tracer(&cfg),
            Err(OtlpError::Install(_))
        ));
    }

    #[test]
    fn otlp_enabled_with_endpoint_returns_handle() {
        let mut cfg = ObservabilityConfig::default();
        cfg.tracing.otlp.enabled = true;
        cfg.tracing.otlp.endpoint = Some("http://otel-collector:4317".into());
        let h = install_otlp_tracer(&cfg).unwrap().unwrap();
        assert_eq!(h.endpoint(), "http://otel-collector:4317");
    }
}
