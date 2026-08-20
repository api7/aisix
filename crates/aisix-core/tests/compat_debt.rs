//! Release-debt gate: time-boxed compatibility code comes due mechanically.
//!
//! This repository deliberately ships compatibility shims that are meant to
//! live for exactly one release — a tombstone the next generation consumes
//! and ignores, a `410` migration pointer, a lenient reader for a document
//! shape the control plane no longer writes. Until this gate existed those
//! deadlines lived only as prose in code comments, so nothing ever made them
//! come due and they were found by grep, one release too late.
//!
//! # The marker
//!
//! Put this on the compat code, in whatever comment syntax the file uses:
//!
//! ```text
//! COMPAT-SINCE: 0.10.0 #1009 — what this tolerates and why it can go
//! ```
//!
//! - `0.10.0` — the **anchor**: the release this compatibility is scoped to.
//!   It must be a version that has ALREADY shipped (it is checked against
//!   `git tag`), never a guess at the next one. "Due at 0.11.0" is a
//!   prediction that fails silently when the next release turns out to be
//!   `1.0.0`; "anchored at 0.10.0" is a fact, and the question the gate asks
//!   — *has a release shipped since the one named?* — is answerable from the
//!   tag list instead of from anyone's roadmap.
//! - `#1009` — the tracking issue (`#N`, or `owner/repo#N` when it lives in
//!   the other plane's repo), so whoever trips the gate has somewhere to go.
//! - the rest — one sentence a stranger can act on. It continues to the end
//!   of the marker's comment paragraph, so a wrapped reason reaches the
//!   failure report whole; a blank comment line ends it.
//!
//! # When it comes due
//!
//! The anchor licenses the code to ride the releases of the anchor's own
//! `MAJOR.MINOR` line and no further. The gate fails once a stable release
//! tag with a **higher `MAJOR.MINOR` than the anchor** exists — 0.11.0 or
//! 1.0.0 for an anchor of 0.10.0, whichever the next release turns out to
//! be. A patch on the anchor's own line (0.10.1) is the same release line,
//! not a release since, so it does not trip the gate.
//!
//! A release CANDIDATE (`v0.11.0-rc.1`) never counts as shipped. A candidate
//! can be superseded or abandoned, so it is not a fact; and the release
//! runbook promotes the final tag onto the *same commit* as the QA'd
//! candidate, so making an rc come due would force a code change that
//! invalidates the very build QA signed off on.
//!
//! # What to do when it fires
//!
//! Remove the compat code and the marker, and close the tracking issue — or,
//! if the debt is being deferred on purpose, re-anchor the marker to the
//! newest shipped release in the same commit and say why in the issue.
//! Re-anchoring is a visible, reviewable decision; letting a marker rot is
//! not.
//!
//! # Deliberately NOT derived from the manifest
//!
//! The workspace version in `Cargo.toml` is frozen (real versions are
//! stamped by CI via `BUILD_VERSION`), so it says nothing about what has
//! shipped. Git tags are the only source of truth here, and the gate reads
//! nothing else.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The one spelling the gate recognises. Anything that merely *looks* like
/// it (`COMPAT_SINCE`, a missing colon, lowercase) is reported as malformed
/// rather than skipped — an unrecognised marker is invisible debt, which is
/// the exact failure this gate exists to remove.
const MARKER: &str = "COMPAT-SINCE:";

/// Paths that talk ABOUT the convention and must not be scanned as if they
/// carried real debt: this file, and the instruction files whose examples
/// would otherwise register as live markers.
const EXEMPT: &[&str] = &[
    "crates/aisix-core/tests/compat_debt.rs",
    "CLAUDE.md",
    "AGENTS.md",
];

/// Directory names never worth walking: build output, vendored deps, VCS.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    ".next",
    "dist",
    "coverage",
    ".venv",
    "__pycache__",
    "deps",
];

/// Cap on a scanned file: a marker lives in a comment, and no comment lives
/// past this. Keeps a stray large fixture out of the walk.
const MAX_FILE_BYTES: u64 = 1_048_576;

// ---------------------------------------------------------------------------
// Versions
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Version {
    major: u64,
    minor: u64,
    patch: u64,
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl Version {
    /// The release LINE a version belongs to. Two versions share a line when
    /// they differ only in patch level.
    fn line(&self) -> (u64, u64) {
        (self.major, self.minor)
    }
}

/// Strict `MAJOR.MINOR.PATCH`. No `v` prefix, no pre-release suffix, no
/// build metadata — one spelling, so a marker cannot drift.
fn parse_version(s: &str) -> Option<Version> {
    let mut parts = s.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some(Version {
        major,
        minor,
        patch,
    })
}

/// A tag naming a SHIPPED release. `v0.10.0` yes; `v0.10.0-rc.2` no (see the
/// module docs on why a candidate is not a release).
fn parse_release_tag(tag: &str) -> Option<Version> {
    parse_version(tag.strip_prefix('v').unwrap_or(tag))
}

/// The shipped releases in a tag list, deduplicated and ordered.
fn releases(tags: &[String]) -> BTreeSet<Version> {
    tags.iter().filter_map(|t| parse_release_tag(t)).collect()
}

/// Whether a marker anchored at `anchor` has come due, given the releases
/// that have shipped and — when building a maintenance line — which line
/// this checkout serves.
///
/// `line_scope` is `Some((major, minor))` on a `release/X.Y` branch. A
/// maintenance line only ever ships from its own history, and removing a
/// compat shim there would be a behaviour change in a patch release, so
/// releases newer than the line are out of scope for it.
fn is_due(anchor: Version, shipped: &BTreeSet<Version>, line_scope: Option<(u64, u64)>) -> bool {
    shipped
        .iter()
        .filter(|r| line_scope.is_none_or(|scope| r.line() <= scope))
        .any(|r| r.line() > anchor.line())
}

// ---------------------------------------------------------------------------
// Marker parsing
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
struct Marker {
    anchor: Version,
    issue: String,
    reason: String,
}

/// Shortest reason that can plausibly tell a stranger what the shim
/// tolerates. Guards against `COMPAT-SINCE: 0.10.0 #1 — tbd`.
const MIN_REASON_CHARS: usize = 12;

/// Cap on a reason gathered across a comment paragraph, so a marker placed
/// mid-doc-comment cannot drag a page of prose into the failure report.
const MAX_REASON_CHARS: usize = 400;

/// The continuation of a marker's reason: the rest of its comment PARAGRAPH.
///
/// A reason that stopped at the marker's own line would quote half a
/// sentence in the report, which is the one thing the report cannot afford —
/// so the lines that follow, sharing the marker's comment lead-in (`    ///`,
/// `\t//`, `#`), are joined onto it. The paragraph ends at a blank comment
/// line, the next marker, the end of the comment, or the length cap. A marker
/// with no comment lead-in at all (column zero) has no continuation.
fn reason_continuation(rest: &[&str], lead_in: &str) -> Option<String> {
    let lead_in = lead_in.trim_end();
    if lead_in.is_empty() {
        return None;
    }
    let mut parts: Vec<&str> = Vec::new();
    let mut len = 0;
    for line in rest {
        let Some(tail) = line.trim_end().strip_prefix(lead_in) else {
            break;
        };
        let tail = tail.trim();
        if tail.is_empty() || tail.contains(MARKER) {
            break;
        }
        len += tail.chars().count() + 1;
        parts.push(tail);
        if len >= MAX_REASON_CHARS {
            break;
        }
    }
    (!parts.is_empty()).then(|| parts.join(" "))
}

/// Does this line at least *intend* to be a marker? Used so a typo becomes a
/// hard failure instead of silently unmarked debt.
fn looks_like_marker(line: &str) -> bool {
    let lowered = line.to_ascii_lowercase();
    ["compat-since", "compat_since", "compat since"]
        .iter()
        .any(|needle| lowered.contains(needle))
}

/// Parse one source line. `None` — not a marker and not trying to be.
/// `Some(Err)` — meant to be a marker but is not usable.
fn parse_marker_line(line: &str) -> Option<Result<Marker, String>> {
    let Some(idx) = line.find(MARKER) else {
        return looks_like_marker(line).then(|| {
            Err(format!(
                "looks like a marker but is not spelled `{MARKER}` \
                 (exactly, uppercase, with the colon)"
            ))
        });
    };
    Some(parse_marker_body(line[idx + MARKER.len()..].trim_start()))
}

fn parse_marker_body(body: &str) -> Result<Marker, String> {
    let Some(version_token) = body.split_whitespace().next() else {
        return Err(format!(
            "empty marker — expected `{MARKER} <MAJOR.MINOR.PATCH> <issue> — <reason>`"
        ));
    };
    let anchor = parse_version(version_token).ok_or_else(|| {
        if version_token.starts_with('v') {
            format!("anchor `{version_token}` has a leading `v` — write it as `MAJOR.MINOR.PATCH`")
        } else if version_token.contains('-') || version_token.contains('+') {
            format!(
                "anchor `{version_token}` carries a pre-release or build suffix — \
                 anchor on a shipped release, e.g. `0.10.0`"
            )
        } else {
            format!("anchor `{version_token}` is not a `MAJOR.MINOR.PATCH` version")
        }
    })?;

    let after_version = body[version_token.len()..].trim_start();
    let Some(issue_token) = after_version.split_whitespace().next() else {
        return Err("missing the tracking issue reference after the anchor (`#1234`)".into());
    };
    if !is_issue_ref(issue_token) {
        return Err(format!(
            "`{issue_token}` is not a tracking issue reference — write `#1234`, \
             or `owner/repo#1234` when the issue lives in the other plane's repo"
        ));
    }

    let after_issue = after_version[issue_token.len()..].trim_start();
    let reason = after_issue
        .strip_prefix('\u{2014}')
        .or_else(|| after_issue.strip_prefix("--"))
        .or_else(|| after_issue.strip_prefix('-'))
        .ok_or("missing the ` — ` separator between the issue reference and the reason")?
        .trim();
    if reason.chars().count() < MIN_REASON_CHARS {
        return Err(format!(
            "reason is too short to act on — say what `{anchor}` compatibility this \
             tolerates and why it can go"
        ));
    }

    Ok(Marker {
        anchor,
        issue: issue_token.to_string(),
        reason: reason.to_string(),
    })
}

/// `#1234` or `owner/repo#1234`.
fn is_issue_ref(token: &str) -> bool {
    let Some((repo, number)) = token.split_once('#') else {
        return false;
    };
    let repo_ok = repo.is_empty()
        || (repo.matches('/').count() == 1
            && repo.split('/').all(|part| {
                !part.is_empty()
                    && part
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
            }));
    repo_ok && !number.is_empty() && number.chars().all(|c| c.is_ascii_digit())
}

// ---------------------------------------------------------------------------
// Scanning
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Found {
    path: String,
    line_no: usize,
    parsed: Result<Marker, String>,
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn scan(root: &Path) -> Vec<Found> {
    let mut found = Vec::new();
    walk(root, root, &mut found);
    found.sort_by(|a, b| (&a.path, a.line_no).cmp(&(&b.path, b.line_no)));
    found
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<Found>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            // `target*` covers the cargo output dir and the ad-hoc
            // `target-<something>` dirs a second build tree leaves behind.
            if SKIP_DIRS.contains(&name.as_ref()) || name.starts_with("target") {
                continue;
            }
            walk(root, &path, out);
        } else if file_type.is_file() {
            scan_file(root, &path, out);
        }
    }
}

fn scan_file(root: &Path, path: &Path, out: &mut Vec<Found>) {
    let rel = path
        .strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");
    if EXEMPT.contains(&rel.as_str()) {
        return;
    }
    if std::fs::metadata(path).is_ok_and(|m| m.len() > MAX_FILE_BYTES) {
        return;
    }
    // Non-UTF-8 is binary; a marker cannot live there.
    let Ok(text) = std::fs::read_to_string(path) else {
        return;
    };
    let lines: Vec<&str> = text.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let Some(parsed) = parse_marker_line(line) else {
            continue;
        };
        let parsed = parsed.map(|mut marker| {
            let lead_in = &line[..line.find(MARKER).expect("matched above")];
            if let Some(tail) = reason_continuation(&lines[i + 1..], lead_in) {
                marker.reason.push(' ');
                marker.reason.push_str(&tail);
            }
            marker
        });
        out.push(Found {
            path: rel.clone(),
            line_no: i + 1,
            parsed,
        });
    }
}

// ---------------------------------------------------------------------------
// Git
// ---------------------------------------------------------------------------

fn git(root: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .map_err(|e| format!("running `git {}`: {e}", args.join(" ")))?;
    if !output.status.success() {
        return Err(format!(
            "`git {}` failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn git_tags(root: &Path) -> Result<Vec<String>, String> {
    Ok(git(root, &["tag", "--list"])?
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect())
}

/// The maintenance line this checkout serves, if any. In CI the PR's base
/// branch is the honest answer (the merge ref is not a branch name); locally
/// it is the checked-out branch.
fn maintenance_line(root: &Path) -> Option<(u64, u64)> {
    let branch = ["GITHUB_BASE_REF", "GITHUB_REF_NAME"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .find(|v| !v.is_empty())
        .or_else(|| git(root, &["rev-parse", "--abbrev-ref", "HEAD"]).ok())
        .unwrap_or_default();
    let (major, minor) = branch.trim().strip_prefix("release/")?.split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

/// Every `COMPAT-SINCE:` marker in the tree parses.
///
/// Split from the due check because well-formedness needs no git: a typo'd
/// marker is unmarked debt, and that must fail even where tags are absent.
#[test]
fn compat_since_markers_are_well_formed() {
    let root = workspace_root();
    let mut report = String::new();
    for found in scan(&root) {
        if let Err(why) = &found.parsed {
            let _ = writeln!(report, "  {}:{} — {why}", found.path, found.line_no);
        }
    }
    assert!(
        report.is_empty(),
        "malformed release-debt marker(s):\n{report}\n\
         The one accepted spelling is:\n\
         \x20 {MARKER} <MAJOR.MINOR.PATCH> <#issue> — <what this tolerates and why it can go>\n\
         e.g. {MARKER} 0.10.0 #1009 — consumes the pre-0.10.0 `mode` tombstone the CP no longer emits\n\
         See the module docs in crates/aisix-core/tests/compat_debt.rs."
    );
}

/// No marker in the tree has come due, and every anchor names a release that
/// actually shipped.
///
/// This is the gate. It runs in the `rust unit + coverage` job, which is a
/// required check, so a due marker blocks every PR until the debt is paid or
/// deliberately re-anchored.
#[test]
fn release_debt_markers_are_not_due() {
    let root = workspace_root();
    // The ONLY path that skips: no repository at all (a vendored source
    // tree). CI always has one, so the gate can never silently vanish there.
    if !root.join(".git").exists() {
        eprintln!(
            "no git repository at {} — release-debt gate skipped",
            root.display()
        );
        return;
    }

    let tags = git_tags(&root).expect("listing git tags");
    let shipped = releases(&tags);
    assert!(
        !shipped.is_empty(),
        "no release tags found in this checkout, so the release-debt gate cannot tell \
         what has shipped.\nCI must fetch tags (`fetch-tags: true` on actions/checkout); \
         locally run `git fetch --tags`.\nFailing rather than passing: a tagless checkout \
         would otherwise disable this gate silently."
    );
    let newest = *shipped.iter().next_back().expect("non-empty");
    let scope = maintenance_line(&root);

    let mut report = String::new();
    for found in scan(&root) {
        let Ok(marker) = &found.parsed else {
            continue; // reported by compat_since_markers_are_well_formed
        };
        if !shipped.contains(&marker.anchor) {
            let _ = writeln!(
                report,
                "  {}:{}\n    anchor {} has never been released — anchor on a version that \
                 has ALREADY shipped (newest is {}), never on a guess at the next one.",
                found.path, found.line_no, marker.anchor, newest
            );
            continue;
        }
        if is_due(marker.anchor, &shipped, scope) {
            let superseding = shipped
                .iter()
                .filter(|r| scope.is_none_or(|s| r.line() <= s))
                .find(|r| r.line() > marker.anchor.line())
                .expect("is_due found one");
            let _ = writeln!(
                report,
                "  {}:{}\n    {MARKER} {} {}\n    reason: {}\n    \
                 due: {} has shipped since the {}.{} line this is anchored to.",
                found.path,
                found.line_no,
                marker.anchor,
                marker.issue,
                marker.reason,
                superseding,
                marker.anchor.major,
                marker.anchor.minor,
            );
        }
    }

    assert!(
        report.is_empty(),
        "release debt is due:\n{report}\n\
         A `{MARKER} <version>` marker licenses compatibility code to ride the releases of \
         that version's line and no further.\n\n\
         Do one of these, in the same PR:\n\
         \x20 1. Remove the compatibility code and its marker, and close the tracking issue.\n\
         \x20 2. If the debt is being deferred ON PURPOSE, re-anchor the marker to {newest} and \
         say why in the tracking issue.\n\
         \x20    Re-anchoring is a visible, reviewable decision; letting a marker rot is not.\n\n\
         Release candidates never count as shipped, and a patch on the anchor's own line does \
         not come due.\nSee the module docs in crates/aisix-core/tests/compat_debt.rs."
    );
}

// ---------------------------------------------------------------------------
// Unit tests for the pure logic — synthetic tags, so they pin the decision
// itself rather than whatever the repository happens to be tagged at today.
// ---------------------------------------------------------------------------

mod logic {
    use super::*;

    fn tags(list: &[&str]) -> BTreeSet<Version> {
        releases(&list.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn release_candidates_are_not_releases() {
        let shipped = tags(&["v0.10.0", "v0.11.0-rc.1", "v0.11.0-rc.2"]);
        assert_eq!(shipped.len(), 1);
        assert!(!is_due(parse_version("0.10.0").unwrap(), &shipped, None));
    }

    #[test]
    fn a_newer_minor_line_makes_the_marker_due() {
        let anchor = parse_version("0.10.0").unwrap();
        assert!(is_due(anchor, &tags(&["v0.10.0", "v0.11.0"]), None));
    }

    #[test]
    fn the_next_release_need_not_be_the_predicted_number() {
        // The whole point of anchoring: 1.0.0 comes due exactly like 0.11.0
        // would have, with nothing in the marker to update.
        let anchor = parse_version("0.10.0").unwrap();
        assert!(is_due(anchor, &tags(&["v0.10.0", "v1.0.0"]), None));
    }

    #[test]
    fn a_patch_on_the_anchors_own_line_is_not_a_release_since() {
        let anchor = parse_version("0.10.0").unwrap();
        assert!(!is_due(anchor, &tags(&["v0.10.0", "v0.10.1"]), None));
    }

    #[test]
    fn nothing_newer_than_the_anchor_is_not_due() {
        let anchor = parse_version("0.10.0").unwrap();
        assert!(!is_due(anchor, &tags(&["v0.9.0", "v0.10.0"]), None));
    }

    #[test]
    fn a_maintenance_line_ignores_releases_that_moved_past_it() {
        // On release/0.10, v0.11.0 exists globally but the 0.10 line only
        // ever ships from its own history — pulling the shim out there would
        // be a behaviour change in a patch release.
        let anchor = parse_version("0.10.0").unwrap();
        let shipped = tags(&["v0.10.0", "v0.10.1", "v0.11.0"]);
        assert!(is_due(anchor, &shipped, None));
        assert!(!is_due(anchor, &shipped, Some((0, 10))));
        // ...but the 0.11 line is past it, so there the debt is due.
        assert!(is_due(anchor, &shipped, Some((0, 11))));
    }

    #[test]
    fn an_older_anchor_is_due_on_a_maintenance_line_too() {
        let anchor = parse_version("0.8.0").unwrap();
        let shipped = tags(&["v0.8.0", "v0.9.0", "v0.10.0"]);
        assert!(is_due(anchor, &shipped, Some((0, 9))));
    }

    #[test]
    fn the_canonical_marker_parses() {
        let parsed = parse_marker_line(
            "/// COMPAT-SINCE: 0.10.0 #1009 — consumes the pre-0.10.0 `mode` tombstone",
        )
        .expect("recognised")
        .expect("well formed");
        assert_eq!(parsed.anchor, parse_version("0.10.0").unwrap());
        assert_eq!(parsed.issue, "#1009");
        assert!(parsed.reason.starts_with("consumes the pre-0.10.0"));
    }

    #[test]
    fn ascii_separators_and_cross_repo_issue_refs_parse() {
        let parsed = parse_marker_line(
            "// COMPAT-SINCE: 0.9.0 api7/AISIX-Cloud#1355 -- tolerates the old projection",
        )
        .expect("recognised")
        .expect("well formed");
        assert_eq!(parsed.issue, "api7/AISIX-Cloud#1355");
    }

    #[test]
    fn a_line_with_no_marker_is_not_reported() {
        assert!(parse_marker_line("// nothing to see here").is_none());
    }

    #[test]
    fn near_misses_are_failures_not_silence() {
        // Each of these would otherwise be debt the gate never sees.
        for line in [
            "// COMPAT_SINCE: 0.10.0 #1009 — underscore",
            "// COMPAT-SINCE 0.10.0 #1009 — no colon",
            "// compat-since: 0.10.0 #1009 — lowercase",
        ] {
            let err = parse_marker_line(line)
                .expect("recognised as an attempt")
                .expect_err("must not be accepted");
            assert!(err.contains("COMPAT-SINCE:"), "unhelpful error: {err}");
        }
    }

    #[test]
    fn a_predicted_anchor_spelling_is_rejected() {
        for (line, needle) in [
            ("// COMPAT-SINCE: v0.10.0 #1 — leading v", "leading `v`"),
            (
                "// COMPAT-SINCE: 0.11.0-rc.1 #1 — a candidate is not a release",
                "pre-release",
            ),
            ("// COMPAT-SINCE: next #1 — not a version", "not a"),
        ] {
            let err = parse_marker_line(line).unwrap().expect_err("rejected");
            assert!(err.contains(needle), "error {err:?} lacks {needle:?}");
        }
    }

    fn write(root: &Path, rel: &str, body: &str) {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }

    #[test]
    fn a_reason_runs_to_the_end_of_its_comment_paragraph() {
        // A reason cut off at the marker's own line would quote half a
        // sentence in the failure report.
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "src/x.rs",
            "/// Unrelated doc above.\n\
             ///\n\
             /// COMPAT-SINCE: 0.9.0 #1 — tolerates the old shape written by a\n\
             /// control plane that has since stopped emitting it.\n\
             ///\n\
             /// A separate paragraph, not part of the reason.\n\
             pub const X: u8 = 0;\n",
        );
        let found = scan(dir.path());
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].line_no, 3);
        assert_eq!(
            found[0].parsed.as_ref().unwrap().reason,
            "tolerates the old shape written by a control plane that has since \
             stopped emitting it."
        );
    }

    #[test]
    fn the_scan_skips_what_cannot_hold_a_marker() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write(
            root,
            "crates/x/src/lib.rs",
            "// COMPAT-SINCE: 0.9.0 #1 — the only real one here\n",
        );
        write(
            root,
            "node_modules/pkg/index.js",
            "// COMPAT-SINCE: 0.1.0 #2 — vendored, must be ignored\n",
        );
        write(
            root,
            "target/debug/build.rs",
            "// COMPAT-SINCE: 0.1.0 #3 — build output\n",
        );
        std::fs::write(
            root.join("blob.bin"),
            b"\xff\xfe\x00COMPAT-SINCE: 0.1.0 #4 -- binary",
        )
        .unwrap();
        let found = scan(root);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].path, "crates/x/src/lib.rs");
    }

    #[test]
    fn a_marker_without_a_usable_issue_or_reason_is_rejected() {
        for line in [
            "// COMPAT-SINCE: 0.10.0 — no issue reference at all",
            "// COMPAT-SINCE: 0.10.0 1009 — issue is missing its hash",
            "// COMPAT-SINCE: 0.10.0 #1009 tolerates old rows",
            "// COMPAT-SINCE: 0.10.0 #1009 — tbd",
        ] {
            assert!(
                parse_marker_line(line).unwrap().is_err(),
                "wrongly accepted: {line}"
            );
        }
    }
}
