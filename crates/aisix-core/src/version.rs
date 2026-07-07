//! Build-time version identity.
//!
//! Release builds are stamped by CI: the `docker-image` workflow derives
//! `AISIX_BUILD_VERSION` from the release tag (`v0.4.0` → `0.4.0`) and
//! passes it as a Docker build-arg, so a released binary always
//! self-reports the version it was tagged with — the `Server` response
//! header, `aisix --version`, and the heartbeat `dp_version` all come
//! from here, and no manual `Cargo.toml` bump is required at release
//! time. Local builds (no stamp) fall back to the workspace crate
//! version. (QA v0.3.0 finding: the 0.3.0 image self-reported 0.1.0
//! because the crate version was the only source and was never bumped.)

/// Version the binary reports about itself.
pub const BUILD_VERSION: &str = match option_env!("AISIX_BUILD_VERSION") {
    Some(v) => v,
    None => env!("CARGO_PKG_VERSION"),
};

#[cfg(test)]
mod tests {
    use super::BUILD_VERSION;

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
