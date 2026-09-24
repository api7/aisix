//! The bounded name of the work a request asked for, and the handler family
//! that served it.
//!
//! A usage event already says which protocol addressed the gateway
//! (`inbound_protocol`) and which model answered — but not what was asked
//! for. Every OpenAI-shaped route collapses onto `inbound_protocol="openai"`,
//! so a text chat, an image generation and a video submission are one
//! undifferentiated stream to an exporter consumer. The only way to tell them
//! apart was to regex a captured prompt, which `content_mode = metadata_only`
//! deployments do not have and which a video submission (zero tokens, no
//! captured content) does not answer at all (AISIX-Cloud#1461).
//!
//! [`Surface`] pairs the two labels a terminal emitter needs and hands them
//! out as constants, because they must not be able to disagree: `handler` is
//! the Prometheus label a family has always reported, and `operation` is the
//! finer name that goes on the usage event. Most families map one to one.
//! Three do not, and those are exactly the families where the coarse label
//! answers "what kind of call was this" wrongly:
//!
//! - `/v1/images/generations` and `/v1/images/edits` share `handler="images"`;
//! - the three `/v1/audio/*` routes share `handler="audio"`, and one of them
//!   consumes audio while another produces it;
//! - `handler="videos"` covers a submission that generates a video and two
//!   routes that only poll and download it, so the label alone cannot say
//!   how much video was actually generated.
//!
//! Both values are `&'static str` chosen by the route, never by caller text,
//! so neither can mint an unbounded label (#451).

/// A request family's telemetry identity.
///
/// Handed to the emission chokepoint as one value so the metric label and the
/// usage event's `operation` are chosen together, at one place, from this
/// module's constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Surface {
    /// The `handler` label on `aisix_usage_events_emitted_total`. One value
    /// per handler family; unchanged from what each family already reported.
    pub(crate) handler: &'static str,
    /// `UsageEvent::operation` — what the caller asked the gateway to do.
    pub(crate) operation: &'static str,
}

impl Surface {
    /// A family whose handler serves exactly one kind of work, so the two
    /// labels are the same word.
    const fn uniform(name: &'static str) -> Self {
        Self {
            handler: name,
            operation: name,
        }
    }

    /// A family whose handler serves several kinds of work.
    const fn split(handler: &'static str, operation: &'static str) -> Self {
        Self { handler, operation }
    }
}

pub(crate) const CHAT: Surface = Surface::uniform("chat");
pub(crate) const COMPLETIONS: Surface = Surface::uniform("completions");
pub(crate) const MESSAGES: Surface = Surface::uniform("messages");
pub(crate) const COUNT_TOKENS: Surface = Surface::uniform("count_tokens");
pub(crate) const RESPONSES: Surface = Surface::uniform("responses");
pub(crate) const EMBEDDINGS: Surface = Surface::uniform("embeddings");
pub(crate) const RERANK: Surface = Surface::uniform("rerank");
pub(crate) const REALTIME: Surface = Surface::uniform("realtime");
pub(crate) const FILES: Surface = Surface::uniform("files");
pub(crate) const BATCHES: Surface = Surface::uniform("batches");
pub(crate) const FINE_TUNING: Surface = Surface::uniform("fine_tuning");
pub(crate) const MCP: Surface = Surface::uniform("mcp");
pub(crate) const A2A: Surface = Surface::uniform("a2a");

pub(crate) const IMAGE_GENERATION: Surface = Surface::split("images", "image_generation");
pub(crate) const IMAGE_EDIT: Surface = Surface::split("images", "image_edit");

pub(crate) const TRANSCRIPTION: Surface = Surface::split("audio", "transcription");
pub(crate) const TRANSLATION: Surface = Surface::split("audio", "translation");
pub(crate) const SPEECH: Surface = Surface::split("audio", "speech");

/// Only the POST meters: the two GET routes under `/v1/videos` poll and
/// download a job the submission already accounted for, and neither emits a
/// usage event (see the census below). Naming the submission
/// `video_generation` rather than `videos` is what keeps "how much video
/// traffic is there" from being answerable only as a route count.
pub(crate) const VIDEO_GENERATION: Surface = Surface::split("videos", "video_generation");

/// The handler label predates the route name; the operation follows the
/// event-family vocabulary the rest of the values use.
pub(crate) const PASSTHROUGH: Surface = Surface::split("passthrough_route", "passthrough");

/// The background poll that attributes a finished batch job's tokens and
/// cost, long after the caller's `/v1/batches` request returned.
///
/// The only surface here with no route of its own, so the census below does
/// not cover it. Held apart from [`BATCHES`] on purpose: those rows are
/// zero-token management calls, this one carries the batch's real tokens and
/// spend, and folding the two would make batch cost unseparable from the
/// traffic that submitted it.
pub(crate) const BATCH_COMPLETION: Surface = Surface::split("batch", "batch_completion");

/// The surface a request on this route reports when the caller walks away,
/// keyed by [`crate::normalize_endpoint_label`] output.
///
/// Exists for the emitter that runs with no handler: `ClientCancelGuard`
/// serves such a request entirely from `Drop`, so it has the route but none
/// of the handler's own values (AISIX-Cloud#1571).
///
/// EVERY route that meters maps here — including the ones the caller never
/// names a model on. The row is not a model's row, it is the request's: the
/// guard writes a `499` access-log line for any route it wraps, and a line
/// an operator cannot find in the usage log by the same `request_id` is the
/// gap this exists to close. So `/mcp`, `/a2a`, the passthrough namespace
/// and the jobs surface report their own attribution (route, server/tool,
/// agent) with the model fields empty, and `/v1/realtime` reports the
/// pre-upgrade phase — after the upgrade its session runs detached from
/// this guard and writes its own terminal event.
///
/// `None` is only for a route that emits no usage event AT ALL, whatever
/// the outcome: the health probes, the two OAuth/agent-card discovery
/// routes, `/v1/models`, and the two `/v1/videos/:id` polls, whose work was
/// already metered by the submission. They have no `operation` to report
/// and would have to mint one to say anything here.
pub(crate) fn surface_for_endpoint(endpoint: &str) -> Option<Surface> {
    Some(match endpoint {
        "/v1/chat/completions" => CHAT,
        "/v1/completions" => COMPLETIONS,
        "/v1/embeddings" => EMBEDDINGS,
        "/v1/images/generations" => IMAGE_GENERATION,
        "/v1/images/edits" => IMAGE_EDIT,
        "/v1/messages" => MESSAGES,
        "/v1/messages/count_tokens" => COUNT_TOKENS,
        "/v1/rerank" => RERANK,
        "/v1/responses" => RESPONSES,
        "/v1/audio/transcriptions" => TRANSCRIPTION,
        "/v1/audio/translations" => TRANSLATION,
        "/v1/audio/speech" => SPEECH,
        "/v1/videos" => VIDEO_GENERATION,
        "/v1/realtime" => REALTIME,
        "/v1/files" | "/v1/files/:id" => FILES,
        "/v1/batches" | "/v1/batches/:id" => BATCHES,
        "/v1/fine_tuning/jobs" | "/v1/fine_tuning/jobs/:id" => FINE_TUNING,
        "/mcp" | "/mcp/{server}" => MCP,
        "/a2a" => A2A,
        // Both labels the passthrough family can carry. A route an operator
        // claimed under `/passthrough/` normalizes to the first; a
        // host-matched route, or a claimed prefix anywhere else, is not a
        // path this gateway knows before routing and lands in `other` — and
        // `other` IS the passthrough family, because every path no typed
        // route claims reaches `passthrough_route::entry` through the
        // router's fallback. An unmatched path answers 404 without ever
        // awaiting, so it has no window to be cancelled in.
        "/passthrough_route" | "other" => PASSTHROUGH,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The source of the routing table. Parsed rather than duplicated so a
    /// new `.route(...)` cannot slip past this census — the same move
    /// `guardrail_coverage` makes, for the same reason: a hand-written list
    /// of endpoints agrees with itself forever.
    const ROUTER_SRC: &str = include_str!("lib.rs");

    /// The router's `.fallback(...)` seat, which has no path literal.
    const FALLBACK_SURFACE: &str = "<fallback>";

    /// What a mounted route contributes to the usage stream.
    #[derive(Debug, Clone, Copy)]
    enum Emits {
        /// Usage events from this route carry this surface.
        Usage(Surface),
        /// This route emits no usage event at all. Carries why.
        Nothing(&'static str),
    }

    /// Every surface `build_router` mounts, and the operation its usage
    /// events carry. Checked against the parsed routing table below, so a
    /// route added without deciding what it reports fails the build.
    const ROUTE_OPERATIONS: &[(&str, Emits)] = &[
        ("/livez", Emits::Nothing("liveness probe")),
        ("/readyz", Emits::Nothing("readiness probe")),
        (
            "/v1/models",
            Emits::Nothing("catalog listing; reaches no upstream and meters nothing"),
        ),
        ("/v1/chat/completions", Emits::Usage(CHAT)),
        ("/v1/completions", Emits::Usage(COMPLETIONS)),
        ("/v1/embeddings", Emits::Usage(EMBEDDINGS)),
        ("/v1/images/generations", Emits::Usage(IMAGE_GENERATION)),
        ("/v1/images/edits", Emits::Usage(IMAGE_EDIT)),
        ("/v1/messages", Emits::Usage(MESSAGES)),
        ("/v1/messages/count_tokens", Emits::Usage(COUNT_TOKENS)),
        ("/v1/rerank", Emits::Usage(RERANK)),
        ("/v1/responses", Emits::Usage(RESPONSES)),
        ("/v1/audio/transcriptions", Emits::Usage(TRANSCRIPTION)),
        ("/v1/audio/translations", Emits::Usage(TRANSLATION)),
        ("/v1/audio/speech", Emits::Usage(SPEECH)),
        ("/v1/videos", Emits::Usage(VIDEO_GENERATION)),
        (
            "/v1/videos/:id",
            Emits::Nothing("polls a job the submission already metered"),
        ),
        (
            "/v1/videos/:id/content",
            Emits::Nothing("downloads a job the submission already metered"),
        ),
        ("/v1/realtime", Emits::Usage(REALTIME)),
        ("/v1/files", Emits::Usage(FILES)),
        ("/v1/files/:id", Emits::Usage(FILES)),
        ("/v1/files/:id/content", Emits::Usage(FILES)),
        ("/v1/batches", Emits::Usage(BATCHES)),
        ("/v1/batches/:id", Emits::Usage(BATCHES)),
        ("/v1/batches/:id/cancel", Emits::Usage(BATCHES)),
        ("/v1/fine_tuning/jobs", Emits::Usage(FINE_TUNING)),
        ("/v1/fine_tuning/jobs/:id", Emits::Usage(FINE_TUNING)),
        ("/v1/fine_tuning/jobs/:id/cancel", Emits::Usage(FINE_TUNING)),
        (
            "/.well-known/oauth-protected-resource",
            Emits::Nothing("RFC 9728 discovery; unauthenticated, meters nothing"),
        ),
        (
            "/.well-known/oauth-protected-resource/mcp",
            Emits::Nothing("RFC 9728 discovery; unauthenticated, meters nothing"),
        ),
        ("/mcp", Emits::Usage(MCP)),
        ("/mcp/", Emits::Usage(MCP)),
        ("/mcp/:server", Emits::Usage(MCP)),
        ("/a2a/:agent", Emits::Usage(A2A)),
        (
            "/a2a/:agent/.well-known/agent-card.json",
            Emits::Nothing("agent-card discovery; reaches no upstream on the caller's behalf"),
        ),
        (FALLBACK_SURFACE, Emits::Usage(PASSTHROUGH)),
    ];

    /// Pull every surface `build_router` mounts out of its own source.
    fn mounted_surfaces() -> BTreeSet<String> {
        let start = ROUTER_SRC
            .find("pub fn build_router(")
            .expect("build_router must exist in lib.rs");
        let body = &ROUTER_SRC[start..];
        let end = body
            .find("\n}\n")
            .expect("build_router must be brace-balanced at column 0");
        let body = &body[..end];

        let mut found = BTreeSet::new();
        for (idx, _) in body.match_indices(".route(") {
            let rest = &body[idx + ".route(".len()..];
            let open = rest.find('"').expect(".route( must take a path literal");
            let close = open
                + 1
                + rest[open + 1..]
                    .find('"')
                    .expect("unterminated route path literal");
            found.insert(rest[open + 1..close].to_string());
        }
        if body.contains(".fallback(") {
            found.insert(FALLBACK_SURFACE.to_string());
        }
        found
    }

    #[test]
    fn every_mounted_route_declares_what_it_reports() {
        let mounted = mounted_surfaces();
        let declared: BTreeSet<String> = ROUTE_OPERATIONS
            .iter()
            .map(|(p, _)| (*p).to_string())
            .collect();

        // Sanity: the parse must find the router, not silently yield an
        // empty set that makes both assertions vacuous.
        assert!(
            mounted.len() > 20,
            "the routing-table parse found only {} surfaces — it has stopped tracking \
             build_router",
            mounted.len()
        );

        let unclassified: Vec<_> = mounted.difference(&declared).collect();
        assert!(
            unclassified.is_empty(),
            "these surfaces are mounted in build_router but declare no operation: \
             {unclassified:?}\n\
             Say what a usage event from each one reports. If the route meters a call, give it \
             a Surface; if it emits no usage event at all, say Emits::Nothing and why.",
        );

        let stale: Vec<_> = declared.difference(&mounted).collect();
        assert!(
            stale.is_empty(),
            "these surfaces declare an operation but are no longer mounted: {stale:?}",
        );
    }

    /// `operation` is an exporter index key and a control-plane filter value,
    /// so a duplicate spelling silently merges two kinds of traffic. Every
    /// metering route must name its own operation, and the split families
    /// must keep the handler label their dashboards already query.
    #[test]
    fn operations_are_distinct_and_handlers_are_stable() {
        let mut owner: std::collections::BTreeMap<&str, &str> = std::collections::BTreeMap::new();
        for (route, emits) in ROUTE_OPERATIONS {
            let surface = match emits {
                Emits::Usage(surface) => surface,
                // A route that meters nothing must say why, so the next
                // person can tell "deliberately silent" from "nobody got
                // round to it".
                Emits::Nothing(reason) => {
                    assert!(!reason.is_empty(), "{route} is silent for no stated reason");
                    continue;
                }
            };
            assert!(
                !surface.operation.is_empty() && !surface.handler.is_empty(),
                "{route} declares an empty label",
            );
            // Routes of one family legitimately repeat an operation
            // (`/v1/files/:id` is still `files`); what must not happen is two
            // DIFFERENT handler families landing on one word, which would
            // merge two kinds of traffic under one index key. So the map is
            // keyed operation -> owning handler: a repeat is only allowed
            // from the family that already owns it.
            if let Some(prev_handler) = owner.get(surface.operation) {
                assert_eq!(
                    *prev_handler, surface.handler,
                    "{route} takes operation {:?}, which handler {:?} already owns",
                    surface.operation, prev_handler,
                );
            }
            owner.insert(surface.operation, surface.handler);
        }

        // The handler label is a shipped Prometheus dimension: renaming one
        // silently empties every query built on it. Pinned absolutely and in
        // full — a relative equality (`IMAGE_GENERATION.handler ==
        // IMAGE_EDIT.handler`) survives renaming both, and an unlisted
        // constant survives renaming outright. `BATCH_COMPLETION` is the one
        // most likely to be "tidied" to match its operation; its handler is
        // `batch`, and a customer's dashboard queries that word.
        for (surface, handler) in [
            (CHAT, "chat"),
            (COMPLETIONS, "completions"),
            (MESSAGES, "messages"),
            (COUNT_TOKENS, "count_tokens"),
            (RESPONSES, "responses"),
            (EMBEDDINGS, "embeddings"),
            (RERANK, "rerank"),
            (REALTIME, "realtime"),
            (FILES, "files"),
            (BATCHES, "batches"),
            (FINE_TUNING, "fine_tuning"),
            (MCP, "mcp"),
            (A2A, "a2a"),
            (IMAGE_GENERATION, "images"),
            (IMAGE_EDIT, "images"),
            (TRANSCRIPTION, "audio"),
            (TRANSLATION, "audio"),
            (SPEECH, "audio"),
            (VIDEO_GENERATION, "videos"),
            (PASSTHROUGH, "passthrough_route"),
            (BATCH_COMPLETION, "batch"),
        ] {
            assert_eq!(
                surface.handler, handler,
                "the {:?} surface renamed a shipped Prometheus handler label",
                surface.operation,
            );
        }
    }

    /// Every route the cancel guard wraps must be able to NAME what it
    /// reports when the caller walks away, and the answer is forced by the
    /// route table: a label any metering route normalizes to reports that
    /// route's surface, and a label no metering route reaches at all
    /// reports nothing.
    ///
    /// Derived rather than listed, because the guard sees only the
    /// NORMALIZED label (`crate::normalize_endpoint_label`) and several
    /// routes share one — the A2A agent card normalizes to `/a2a`, and the
    /// two OAuth discovery routes land in `other` beside the passthrough
    /// fallback. A hand-written exemption list would let a route be quietly
    /// dropped from the emitting side while its own sibling still meters,
    /// which is the AISIX-Cloud#1571 bug arriving one route at a time.
    /// Routes that file no usage row at any outcome yet NORMALIZE to a label
    /// a metering route also uses. The guard sees only the label, so each of
    /// these must call `attribution::note_unmetered_route()` itself — there
    /// is nothing else that can tell it apart from its metering sibling.
    ///
    /// Listed here so the pairing is visible: adding a route to a shared
    /// label without that call means a cancelled request on it is filed as
    /// its sibling's traffic.
    const UNMETERED_ON_A_SHARED_LABEL: &[&str] = &[
        "/a2a/:agent/.well-known/agent-card.json",
        // Both normalize to `other`, which IS the passthrough family (every
        // path no typed route claims reaches its fallback). Unauthenticated
        // today, so the api_key gate would also hold — but that is a
        // property of the handler, not of the label, and this list is where
        // the label question is answered.
        "/.well-known/oauth-protected-resource",
        "/.well-known/oauth-protected-resource/mcp",
    ];

    #[test]
    fn every_route_the_guard_wraps_reports_a_cancel_it_can_name() {
        // label -> the surface its metering routes agree on, if any.
        let mut by_label: std::collections::BTreeMap<&str, Option<Surface>> =
            std::collections::BTreeMap::new();
        let mut silent_by_label: std::collections::BTreeMap<&str, Vec<&str>> =
            std::collections::BTreeMap::new();
        for (route, emits) in ROUTE_OPERATIONS {
            let label = crate::normalize_endpoint_label(route);
            let entry = by_label.entry(label).or_insert(None);
            let Emits::Usage(surface) = emits else {
                silent_by_label.entry(label).or_default().push(route);
                continue;
            };
            if let Some(prev) = entry {
                assert_eq!(
                    prev, surface,
                    "{route} normalizes to {label:?}, which another metering route already \
                     reports as {:?} — the cancel guard has only the label, so the two cannot \
                     report different operations",
                    prev.operation,
                );
            }
            *entry = Some(*surface);
        }

        // A silent route that shares a label with a metering one can only be
        // silent by declaring itself so.
        for (label, silent) in &silent_by_label {
            if by_label.get(label).copied().flatten().is_none() {
                continue;
            }
            for route in silent {
                assert!(
                    UNMETERED_ON_A_SHARED_LABEL.contains(route),
                    "{route} files no usage row, but normalizes to {label:?}, which a metering \
                     route reports — so a cancelled request on it is filed as that route's \
                     traffic. Call attribution::note_unmetered_route() in its handler and name \
                     it in UNMETERED_ON_A_SHARED_LABEL.",
                );
            }
        }

        // A label NO metering route reaches must report nothing — the half
        // that keeps `/livez`, `/readyz`, `/v1/models` and the video polls
        // from being given a surface by accident.
        for (label, silent) in &silent_by_label {
            if by_label.get(label).copied().flatten().is_some() {
                continue;
            }
            assert!(
                surface_for_endpoint(label).is_none(),
                "{silent:?} meter nothing at any outcome, yet {label:?} reports {:?} on a \
                 cancel",
                surface_for_endpoint(label),
            );
        }

        for (label, expected) in by_label {
            assert_eq!(
                surface_for_endpoint(label),
                expected,
                "the cancel guard reports {label:?} as {:?}, but the routes that normalize to \
                 it meter as {expected:?}. Every request the guard writes a 499 access-log \
                 line for has to be findable in the usage log by the same request_id: map the \
                 label in surface_for_endpoint, or make the route stop metering.",
                surface_for_endpoint(label),
            );
        }
    }

    /// Surfaces with no route of their own, which the route census above
    /// therefore cannot see.
    ///
    /// This is an input to `emitted_operations`, so forgetting an entry
    /// REMOVES an operation from what this file believes it emits — the doc
    /// census would then agree with a `usage.rs` that is also missing it, and
    /// both would be wrong together. `every_surface_is_accounted_for` is what
    /// closes that: it reads the constants out of this file's own source and
    /// requires each to appear here or in the route table.
    const NON_ROUTE: &[Surface] = &[BATCH_COMPLETION];

    /// Every operation this crate can emit, from the two places they come
    /// from: the route table (itself pinned to `build_router`) and the
    /// routeless surfaces above.
    fn emitted_operations() -> BTreeSet<&'static str> {
        ROUTE_OPERATIONS
            .iter()
            .filter_map(|(_, emits)| match emits {
                Emits::Usage(surface) => Some(surface.operation),
                Emits::Nothing(_) => None,
            })
            .chain(NON_ROUTE.iter().map(|s| s.operation))
            .collect()
    }

    /// This module's own source, for the two censuses that read the constant
    /// list out of it. A hand-maintained list of the constants would be one
    /// more thing to forget — which is the failure both of them exist to
    /// catch.
    const SELF_SRC: &str = include_str!("operation.rs");

    /// The names of every `Surface` constant this module defines.
    fn declared_surfaces() -> BTreeSet<&'static str> {
        const DECL: &str = "pub(crate) const ";
        let mut names = BTreeSet::new();
        for (idx, _) in SELF_SRC.match_indices(DECL) {
            let rest = &SELF_SRC[idx + DECL.len()..];
            let Some(colon) = rest.find(':') else {
                continue;
            };
            let name = rest[..colon].trim();
            if rest[colon..]
                .trim_start_matches([':', ' '])
                .starts_with("Surface")
            {
                names.insert(name);
            }
        }
        names
    }

    /// The text between `open` and the next line that closes at `close`.
    fn block_after(open: &str, close: &str) -> &'static str {
        let start = SELF_SRC
            .find(open)
            .unwrap_or_else(|| panic!("this file must still contain {open:?}"));
        let rest = &SELF_SRC[start + open.len()..];
        let end = rest
            .find(close)
            .unwrap_or_else(|| panic!("{open:?} must be closed by {close:?}"));
        &rest[..end]
    }

    /// Every constant must reach BOTH censuses, and neither can say so on its
    /// own.
    ///
    /// `every_mounted_route_declares_what_it_reports` forces a new ROUTED
    /// surface into the route table, and the doc census then forces it into
    /// `usage.rs`. Neither notices a surface with no route of its own — the
    /// `BATCH_COMPLETION` shape — and neither notices a constant missing from
    /// the handler-label table, which is the one thing that table exists for.
    /// So the constant list comes out of the source and both memberships are
    /// checked against it.
    #[test]
    fn every_surface_is_accounted_for() {
        let declared = declared_surfaces();
        assert!(
            declared.len() > 15,
            "the constant parse found only {} surfaces — it has stopped \
             tracking this module",
            declared.len(),
        );

        let routed = block_after("const ROUTE_OPERATIONS: &[(&str, Emits)] = &[", "\n    ];");
        let routeless = block_after("const NON_ROUTE: &[Surface] = &[", "];");
        let handlers = block_after("for (surface, handler) in [", "\n        ] {");

        let mut unemitted = Vec::new();
        let mut unpinned = Vec::new();
        for name in &declared {
            // `Emits::Usage(CHAT)` / `&[BATCH_COMPLETION]`.
            let emitted = routed.contains(&format!("({name})"))
                || routeless.split([',', ' ', ']']).any(|t| t == *name);
            if !emitted {
                unemitted.push(*name);
            }
            // `(CHAT, "chat"),`
            if !handlers.contains(&format!("({name},")) {
                unpinned.push(*name);
            }
        }

        assert!(
            unemitted.is_empty(),
            "these surfaces are defined but reach no emission census, so \
             `emitted_operations` does not know they exist and the doc census \
             will happily agree with a `usage.rs` that also omits them: \
             {unemitted:?}\n\
             Put each on a route in ROUTE_OPERATIONS, or in NON_ROUTE if it \
             has no route of its own.",
        );
        assert!(
            unpinned.is_empty(),
            "these surfaces are missing from the handler-label table, so their \
             shipped Prometheus label can be renamed with every test still \
             green — which is the only thing that table is for: {unpinned:?}",
        );
    }

    /// `UsageEvent::operation`'s doc comment restates the value set, and says
    /// a consumer should take it verbatim — the control plane builds its Logs
    /// filter as a picker over exactly such a fixed set. So it is a second
    /// definition of this module's vocabulary, in another crate, and it has
    /// already drifted once: it listed two values nothing emits and omitted
    /// one that is emitted.
    ///
    /// Parsed rather than re-copied, the same move `ROUTE_OPERATIONS` makes
    /// against `build_router`.
    #[test]
    fn the_wire_field_documents_exactly_what_this_crate_emits() {
        const USAGE_SRC: &str = include_str!("../../aisix-obs/src/usage.rs");
        const OPENER: &str = "What the caller asked the gateway to DO, from a fixed set:";

        let start = USAGE_SRC
            .find(OPENER)
            .expect("UsageEvent::operation must still introduce its value set with that sentence");
        let rest = &USAGE_SRC[start + OPENER.len()..];
        // The list runs to the end of that doc paragraph — the first line
        // carrying no value, i.e. the bare `///` separator.
        let end = rest
            .find("\n    ///\n")
            .expect("the value list must be its own doc paragraph");
        let listed = &rest[..end];

        let mut documented = BTreeSet::new();
        let mut chars = listed.split('`');
        // Odd-indexed pieces are the backtick-delimited values.
        chars.next();
        while let Some(value) = chars.next() {
            documented.insert(value);
            chars.next();
        }

        let emitted = emitted_operations();
        assert_eq!(
            documented,
            emitted,
            "UsageEvent::operation's doc comment and this module disagree.\n               documented but never emitted: {:?}\n               emitted but undocumented: {:?}",
            documented.difference(&emitted).collect::<Vec<_>>(),
            emitted.difference(&documented).collect::<Vec<_>>(),
        );
    }

    /// The declared operation is what a driven request actually reports.
    ///
    /// The table above only states an intention; nothing in it reaches a
    /// running handler, and a family that passes the wrong constant to the
    /// emission chokepoint would satisfy every other test in this file. So
    /// this drives the real router — reusing `guardrail_coverage`'s census
    /// fixtures and usage-event sink, which already exist and already derive
    /// their surface set from `build_router` — and reads the operation off
    /// the events that come out.
    ///
    /// The requests are guardrail-REFUSED, which makes the coverage better
    /// rather than worse: the failure emitters are the ones that
    /// historically drop a field the success path carries, and this is where
    /// `operation` has to be present, because a refused request is exactly
    /// the row whose endpoint nothing else on the event can name.
    #[tokio::test]
    async fn a_driven_surface_reports_the_operation_it_declares() {
        use crate::guardrail_coverage::{census_router_with_usage, fixture, Posture, POSTURE};
        use tower::ServiceExt;

        let declared: std::collections::BTreeMap<&str, &Emits> =
            ROUTE_OPERATIONS.iter().map(|(p, e)| (*p, e)).collect();

        let mut wrong = Vec::new();
        let mut checked = 0usize;
        for (surface, posture) in POSTURE {
            if !matches!(posture, Posture::Enforced) {
                continue;
            }
            let Some(request) = fixture(surface) else {
                // The missing-fixture complaint belongs to the census that
                // owns the fixtures; reporting it twice buries this one.
                continue;
            };
            let Some(Emits::Usage(expected)) = declared.get(surface).copied() else {
                wrong.push(format!(
                    "{surface} is driveable and reaches an upstream, but this file says it \
                     emits no usage event"
                ));
                continue;
            };

            let (router, mut rx) = census_router_with_usage();
            let response = router.oneshot(request).await.expect("router must answer");
            let _ = axum::body::to_bytes(response.into_body(), 1 << 20).await;

            let mut events = Vec::new();
            while let Ok(Some(event)) =
                tokio::time::timeout(std::time::Duration::from_millis(300), rx.recv()).await
            {
                events.push(event);
            }
            if events.is_empty() {
                // Likewise: "refused but emitted nothing" is the sibling
                // census's finding.
                continue;
            }
            checked += 1;
            for event in &events {
                if event.operation != expected.operation {
                    wrong.push(format!(
                        "{surface}: emitted operation={:?}, declared {:?}",
                        event.operation, expected.operation,
                    ));
                }
            }
        }

        assert!(
            wrong.is_empty(),
            "the operation on a usage event must be the one its route declares:\n  {}",
            wrong.join("\n  "),
        );
        // Derived, not a hand-picked floor: every driveable enforced surface
        // must have contributed an event, or this check silently covers
        // fewer routes than it appears to. WHICH surface stopped emitting is
        // `an_enforced_surface_reports_the_refusal_it_makes`'s finding; the
        // count is here only so this census cannot go vacuous.
        let driveable = POSTURE
            .iter()
            .filter(|(surface, posture)| {
                matches!(posture, Posture::Enforced) && fixture(surface).is_some()
            })
            .count();
        assert_eq!(
            checked, driveable,
            "{checked} of {driveable} driveable surfaces produced a usage event — the \
             operation census is covering fewer routes than it appears to",
        );
    }

    /// The three families where the coarse handler label cannot answer the
    /// question the operation exists for.
    #[test]
    fn split_families_report_distinct_operations() {
        assert_ne!(IMAGE_GENERATION.operation, IMAGE_EDIT.operation);
        for pair in [
            (TRANSCRIPTION, TRANSLATION),
            (TRANSCRIPTION, SPEECH),
            (TRANSLATION, SPEECH),
        ] {
            assert_ne!(
                pair.0.operation, pair.1.operation,
                "{} and {} must not share an operation",
                pair.0.handler, pair.1.handler,
            );
        }
    }
}
