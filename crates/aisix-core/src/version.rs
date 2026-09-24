//! Build-time version identity.
//!
//! Release builds are stamped by CI: the `docker-image` workflow derives
//! `AISIX_BUILD_VERSION` from the release tag (`v0.4.0` → `0.4.0`) and
//! passes it as a Docker build-arg, so a released binary always
//! self-reports the version it was tagged with — the `Server` response
//! header, `aisix --version`, and the heartbeat `dp_version` all come
//! from here, and no manual `Cargo.toml` bump is required at release
//! time. Builds without a stamp fall back to the workspace crate
//! version — and so do builds where the stamp is set but empty, which is
//! what every non-tag image gets: the Dockerfile always exports
//! `AISIX_BUILD_VERSION`, and CI fills it only for release tags. (QA v0.3.0 finding: the 0.3.0 image self-reported 0.1.0
//! because the crate version was the only source and was never bumped.)

/// Version the binary reports about itself.
pub const BUILD_VERSION: &str = resolve_build_version(
    option_env!("AISIX_BUILD_VERSION"),
    env!("CARGO_PKG_VERSION"),
);

const fn resolve_build_version(
    stamp: Option<&'static str>,
    fallback: &'static str,
) -> &'static str {
    match stamp {
        Some(v) if !v.is_empty() => v,
        _ => fallback,
    }
}

#[cfg(test)]
mod tests {
    use super::{resolve_build_version, BUILD_VERSION};

    #[test]
    fn empty_stamp_falls_back_to_crate_version() {
        assert_eq!(resolve_build_version(Some(""), "0.3.0"), "0.3.0");
        assert_eq!(resolve_build_version(None, "0.3.0"), "0.3.0");
        assert_eq!(
            resolve_build_version(Some("1.4.0-rc.2"), "0.3.0"),
            "1.4.0-rc.2"
        );
    }

    #[test]
    fn build_version_is_nonempty_semverish() {
        assert!(!BUILD_VERSION.is_empty());
        // Both sources (stamp or crate version) must look like a
        // dotted version, not a placeholder.
        assert!(
            BUILD_VERSION.chars().next().unwrap().is_ascii_digit(),
            "BUILD_VERSION must start with a digit, got {BUILD_VERSION:?}"
        );
    }
}
