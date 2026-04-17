//! aisix-obs — observability bootstrap.
//!
//! This crate will hold the OpenTelemetry + Prometheus wiring and the
//! access-log middleware. For PR #5 we expose just [`init_tracing`], the
//! startup-sequence call that turns the config's `log_level` into an
//! `EnvFilter` and installs a formatted subscriber.
//!
//! OTLP export, Langfuse, and Prometheus registration arrive in their own
//! PRs so this crate stays focused.

#![deny(rust_2018_idioms)]

use aisix_core::ObservabilityConfig;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[derive(Debug, thiserror::Error)]
pub enum ObsError {
    #[error("invalid log filter directive {directive:?}: {source}")]
    Filter {
        directive: String,
        #[source]
        source: tracing_subscriber::filter::ParseError,
    },
    #[error("tracing subscriber already initialised")]
    AlreadyInitialised,
}

/// Install a process-wide tracing subscriber.
///
/// - Log level comes from `cfg.log_level` (a `tracing_subscriber::EnvFilter`
///   directive string, e.g. `"info"`, `"aisix=debug,tower=warn"`).
/// - The `RUST_LOG` env var, if set, overrides the config's directive —
///   standard Rust convention so operators can debug without a restart
///   of the rollout pipeline.
/// - Output format is plain text to stderr for now; structured JSON will
///   be a config knob in a later PR.
pub fn init_tracing(cfg: &ObservabilityConfig) -> Result<(), ObsError> {
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(&cfg.log_level))
        .map_err(|source| ObsError::Filter {
            directive: cfg.log_level.clone(),
            source,
        })?;

    let fmt_layer = fmt::layer().with_target(true).with_writer(std::io::stderr);

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .try_init()
        .map_err(|_| ObsError::AlreadyInitialised)?;

    tracing::info!(
        service = %cfg.service_name,
        level = %cfg.log_level,
        "tracing initialised",
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn already_initialised_variant_is_displayable() {
        let err = ObsError::AlreadyInitialised;
        assert_eq!(err.to_string(), "tracing subscriber already initialised");
    }

    #[test]
    fn filter_error_carries_the_bad_directive() {
        // Directly exercise the EnvFilter parse error path — fabricating
        // the error avoids touching the global tracing subscriber, which
        // is a process-wide singleton that can't be re-initialised safely
        // across tests.
        let bad = "BAD=@notalevel";
        let err = tracing_subscriber::EnvFilter::try_new(bad).unwrap_err();
        let wrapped = ObsError::Filter {
            directive: bad.into(),
            source: err,
        };
        assert!(wrapped.to_string().contains(bad));
    }
}
