//! Build-time version identity.
//!
//! The version comes only from two compile-time stamps, and an empty stamp
//! counts as unset (the Dockerfile always exports both, possibly empty):
//!
//! - `AISIX_BUILD_VERSION` — set by the `docker-image` workflow on release
//!   tags only (`v1.4.0` → `1.4.0`). Reported unchanged.
//! - `AISIX_BUILD_SHA` — the short commit sha every CI image build passes.
//!   Without a release version the binary reports `dev+sha-<sha>`.
//! - Neither → `dev` (plain `cargo build`, `docker build` without args).
//!
//! The workspace crate version is a placeholder and is never reported.
//! [`BUILD_VERSION`] is what `aisix --version`, the `Server` header and the
//! outbound User-Agents carry; [`HEARTBEAT_VERSION`] is what the heartbeat
//! reports to the control plane.

use std::sync::LazyLock;

const BUILD_VERSION_STAMP: Option<&str> = option_env!("AISIX_BUILD_VERSION");
const BUILD_SHA_STAMP: Option<&str> = option_env!("AISIX_BUILD_SHA");

/// Version the binary reports about itself.
pub static BUILD_VERSION: LazyLock<String> =
    LazyLock::new(|| resolve_version(BUILD_VERSION_STAMP, BUILD_SHA_STAMP));

/// Version reported in the heartbeat. Release builds append the image sha
/// (`1.4.0+sha-103d3ec`) so a node can be matched to its image; every other
/// build reports [`BUILD_VERSION`] as is, which already carries the sha.
pub static HEARTBEAT_VERSION: LazyLock<String> =
    LazyLock::new(|| heartbeat_version(BUILD_VERSION_STAMP, BUILD_SHA_STAMP));

fn stamped(value: Option<&str>) -> Option<&str> {
    value.filter(|v| !v.is_empty())
}

fn resolve_version(version: Option<&str>, sha: Option<&str>) -> String {
    match (stamped(version), stamped(sha)) {
        (Some(v), _) => v.to_string(),
        (None, Some(sha)) => format!("dev+sha-{sha}"),
        (None, None) => "dev".to_string(),
    }
}

fn heartbeat_version(version: Option<&str>, sha: Option<&str>) -> String {
    match (stamped(version), stamped(sha)) {
        (Some(v), Some(sha)) => format!("{v}+sha-{sha}"),
        _ => resolve_version(version, sha),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_stamp_wins_and_heartbeat_appends_sha() {
        assert_eq!(resolve_version(Some("1.4.0"), Some("103d3ec")), "1.4.0");
        assert_eq!(
            heartbeat_version(Some("1.5.0-rc.1"), Some("103d3ec")),
            "1.5.0-rc.1+sha-103d3ec"
        );
    }

    #[test]
    fn sha_only_reports_dev_with_a_single_sha_suffix() {
        assert_eq!(resolve_version(None, Some("103d3ec")), "dev+sha-103d3ec");
        assert_eq!(heartbeat_version(None, Some("103d3ec")), "dev+sha-103d3ec");
    }

    #[test]
    fn unstamped_reports_dev() {
        assert_eq!(resolve_version(None, None), "dev");
        assert_eq!(heartbeat_version(None, None), "dev");
    }

    #[test]
    fn empty_stamps_count_as_unset() {
        // What every non-tag image gets: BUILD_VERSION="" plus a real sha.
        assert_eq!(
            resolve_version(Some(""), Some("103d3ec")),
            "dev+sha-103d3ec"
        );
        assert_eq!(
            heartbeat_version(Some(""), Some("103d3ec")),
            "dev+sha-103d3ec"
        );
        // `docker build` without args exports both, empty.
        assert_eq!(resolve_version(Some(""), Some("")), "dev");
        assert_eq!(heartbeat_version(Some(""), Some("")), "dev");
        assert_eq!(resolve_version(Some("1.4.0"), Some("")), "1.4.0");
        assert_eq!(heartbeat_version(Some("1.4.0"), Some("")), "1.4.0");
    }

    #[test]
    fn statics_follow_the_compile_time_stamps() {
        assert_eq!(
            *BUILD_VERSION,
            resolve_version(BUILD_VERSION_STAMP, BUILD_SHA_STAMP)
        );
        assert_eq!(
            *HEARTBEAT_VERSION,
            heartbeat_version(BUILD_VERSION_STAMP, BUILD_SHA_STAMP)
        );
    }
}
