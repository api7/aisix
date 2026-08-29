//! Canonical etcd snapshot → resources-file document (the inverse of
//! `aisix_core::filesource`).
//!
//! For each resource kind the exporter re-emits every entry through the
//! same typed model the loader decodes, then rewrites it into idiomatic
//! file form:
//!
//! - **id-stripping** — canonical documents carry no `id` (the typed
//!   models `#[serde(skip)]` their runtime id); the file derives every id
//!   from the entry's name, so nothing to strip beyond what serialization
//!   already omits.
//! - **reference resugaring** — a model's `provider_key_id` becomes
//!   `provider_key: <that key's name>`; a rate-limit policy's `scope_ref`
//!   for `api_key` / `model` scopes becomes the referenced entry's name
//!   (team / member / team_member scopes pass through). A reference that
//!   resolves to no entry in the export set is kept verbatim and a
//!   warning is raised — a dangling reference is a real data issue, not
//!   something to hide.
//! - **api-key identity** — canonical api-key documents have no name, but
//!   the file keys every entry by one; a deterministic `apikey-<hash…>`
//!   display_name is synthesized from the already-safe `key_hash`.
//! - **secret redaction** — see [`super::secrets`].
//!
//! Two entries of one kind that would collapse to the same file identity
//! raise a warning: the file cannot represent both (identities are unique
//! per kind), so surfacing the collision beats silently dropping one.

use std::collections::{BTreeMap, BTreeSet};

use aisix_core::models::GuardrailScopeType;
use aisix_core::AisixSnapshot;
use serde::Serialize;
use serde_json::Value;

use super::secrets::{
    redact_by_key, redact_headers, redact_string_map, redact_top_level, RedactionCtx,
    SecretPlaceholder,
};

/// Guardrail credential field names, at any nesting depth, that are
/// always secrets in the guardrail schema.
const GUARDRAIL_SECRET_KEYS: &[&str] = &["api_key", "access_key_secret", "secret_access_key"];

/// Issues surfaced during a build, split by whether they leave the
/// exported file loadable. `blocking` issues mean the file cannot be
/// loaded back as-is (an identity collision or a dangling reference the
/// loader rejects); the command exits non-zero so a scripted migration
/// can't mistake a broken file for a finished one. `warnings` are
/// everything the operator should see but that still yields a loadable
/// file (a scope-changing guardrail omitted, an inert cache scope, a
/// placeholder-variable collision).
#[derive(Default)]
struct Diagnostics {
    warnings: Vec<String>,
    blocking: Vec<String>,
}

/// The assembled export, ready to emit as YAML plus the side-channel
/// information the command reports on stderr.
pub struct ExportDocument {
    /// `(kind, entries)` in the file's fixed collection order; only
    /// non-empty kinds are included.
    pub collections: Vec<(&'static str, Vec<Value>)>,
    /// Placeholders substituted for live credentials (empty when
    /// `reveal_secrets` is set).
    pub secret_placeholders: Vec<SecretPlaceholder>,
    /// Issues that still yield a loadable file (omitted scope-changing
    /// guardrails, inert cache scopes, placeholder collisions).
    pub warnings: Vec<String>,
    /// Issues that leave the file non-loadable (identity collisions,
    /// dangling references the loader rejects). Non-empty → the command
    /// writes the file for inspection but exits non-zero.
    pub blocking: Vec<String>,
}

/// Build the resources-file document from a decoded etcd snapshot.
pub fn build_export_document(snapshot: &AisixSnapshot, reveal_secrets: bool) -> ExportDocument {
    let mut diag = Diagnostics::default();
    let mut placeholders = Vec::new();

    // Reference resolution maps: etcd id → the name the file keys the
    // entry by. Built once so every resugared reference resolves against
    // the same identities the entries are emitted under.
    let provider_key_names = id_to_name(&snapshot.provider_keys, |pk| pk.display_name.clone());
    let model_names = id_to_name(&snapshot.models, |m| m.display_name.clone());
    let api_key_names = id_to_name(&snapshot.apikeys, |k| synthetic_api_key_name(&k.key_hash));

    let mut collections: Vec<(&'static str, Vec<Value>)> = Vec::new();

    // provider_keys — identity: display_name; secret: api_key.
    push_kind(
        &mut collections,
        "provider_keys",
        emit_entries(
            &snapshot.provider_keys,
            |pk| pk.display_name.clone(),
            "provider_keys",
            &mut diag,
            |_, _, _| {},
            |doc, identity| {
                let mut ctx = RedactionCtx {
                    kind_token: "PROVIDER_KEY",
                    kind: "provider_keys",
                    identity,
                    reveal: reveal_secrets,
                    out: &mut placeholders,
                };
                redact_top_level(doc, "api_key", &mut ctx);
                // `request.default_headers` are outbound auth headers to the
                // upstream and `request.default_body_fields` can carry a
                // secondary credential — neither is covered by `api_key`, and
                // the OTLP-exporter `headers` map (same shape) is redacted, so
                // these must be too. String values only (a `safe_prompt: true`
                // flag stays intact).
                if let Some(request) = doc.get_mut("request") {
                    redact_string_map(request, "default_headers", "default header", &mut ctx);
                    redact_string_map(
                        request,
                        "default_body_fields",
                        "default body field",
                        &mut ctx,
                    );
                }
            },
        ),
    );

    // models — identity: display_name; resugar provider_key_id → provider_key.
    push_kind(
        &mut collections,
        "models",
        emit_entries(
            &snapshot.models,
            |m| m.display_name.clone(),
            "models",
            &mut diag,
            |doc, identity, diag| resugar_provider_key(doc, identity, &provider_key_names, diag),
            |_, _| {},
        ),
    );

    // api_keys — identity: synthesized display_name; key_hash emitted verbatim.
    push_kind(
        &mut collections,
        "api_keys",
        emit_entries(
            &snapshot.apikeys,
            |k| synthetic_api_key_name(&k.key_hash),
            "api_keys",
            &mut diag,
            |doc, identity, _warnings| {
                if let Value::Object(map) = doc {
                    map.insert("display_name".into(), Value::String(identity.to_string()));
                }
            },
            |_, _| {},
        ),
    );

    // guardrails — identity: name; recursive credential redaction.
    //
    // Every guardrail is exported, attached or not: the file format now
    // carries `guardrail_attachments`, and both sources agree that a
    // guardrail's scope is its attachments and nothing else, so an
    // unattached guardrail is inert on either side (AISIX-Cloud#1450). It
    // used to be the opposite — the file had no attachment collection and a
    // file-defined guardrail applied gateway-wide — so anything not already
    // env-scoped had to be dropped or it would WIDEN on import.
    let mut guardrails = emit_entries(
        &snapshot.guardrails,
        |g| g.name.clone(),
        "guardrails",
        &mut diag,
        |_, _, _| {},
        |doc, identity| {
            let mut ctx = RedactionCtx {
                kind_token: "GUARDRAIL",
                kind: "guardrails",
                identity,
                reveal: reveal_secrets,
                out: &mut placeholders,
            };
            redact_by_key(doc, GUARDRAIL_SECRET_KEYS, &mut ctx);
        },
    );
    guardrails.sort_by(|a, b| {
        a.get("name")
            .and_then(Value::as_str)
            .cmp(&b.get("name").and_then(Value::as_str))
    });
    push_kind(&mut collections, "guardrails", guardrails);

    // guardrail_attachments — the scope that makes each guardrail apply.
    // References are emitted as the file identities the other collections
    // use, since the loader resolves them back to derived ids.
    push_kind(
        &mut collections,
        "guardrail_attachments",
        emit_guardrail_attachments(snapshot, &model_names, &api_key_names, &mut diag),
    );

    // mcp_servers — identity: name; secret: secret.
    push_kind(
        &mut collections,
        "mcp_servers",
        emit_entries(
            &snapshot.mcp_servers,
            |s| s.name.clone(),
            "mcp_servers",
            &mut diag,
            |_, _, _| {},
            |doc, identity| {
                let mut ctx = RedactionCtx {
                    kind_token: "MCP_SERVER",
                    kind: "mcp_servers",
                    identity,
                    reveal: reveal_secrets,
                    out: &mut placeholders,
                };
                redact_top_level(doc, "secret", &mut ctx);
            },
        ),
    );

    // a2a_agents — identity: name; secret: secret.
    push_kind(
        &mut collections,
        "a2a_agents",
        emit_entries(
            &snapshot.a2a_agents,
            |a| a.name.clone(),
            "a2a_agents",
            &mut diag,
            |_, _, _| {},
            |doc, identity| {
                let mut ctx = RedactionCtx {
                    kind_token: "A2A_AGENT",
                    kind: "a2a_agents",
                    identity,
                    reveal: reveal_secrets,
                    out: &mut placeholders,
                };
                redact_top_level(doc, "secret", &mut ctx);
            },
        ),
    );

    // cache_policies — identity: name; no secrets.
    push_kind(
        &mut collections,
        "cache_policies",
        emit_entries(
            &snapshot.cache_policies,
            |c| c.name.clone(),
            "cache_policies",
            &mut diag,
            |doc, identity, diag| resugar_cache_applies_to(doc, identity, &api_key_names, diag),
            |_, _| {},
        ),
    );

    // observability_exporters — identity: name; redact OTLP headers.
    push_kind(
        &mut collections,
        "observability_exporters",
        emit_entries(
            &snapshot.observability_exporters,
            |e| e.name.clone(),
            "observability_exporters",
            &mut diag,
            |_, _, _| {},
            |doc, identity| {
                let mut ctx = RedactionCtx {
                    kind_token: "OBSERVABILITY_EXPORTER",
                    kind: "observability_exporters",
                    identity,
                    reveal: reveal_secrets,
                    out: &mut placeholders,
                };
                redact_headers(doc, &mut ctx);
            },
        ),
    );

    // rate_limit_policies — identity: name; resugar scope_ref for
    // api_key / model scopes to the referenced entry's name.
    push_kind(
        &mut collections,
        "rate_limit_policies",
        emit_entries(
            &snapshot.rate_limit_policies,
            |p| p.name.clone(),
            "rate_limit_policies",
            &mut diag,
            |doc, identity, diag| {
                resugar_scope_ref(doc, identity, &model_names, &api_key_names, diag)
            },
            |_, _| {},
        ),
    );

    // oidc_providers — identity: name; no secrets (issuer / audiences /
    // JWKS endpoint are all public trust configuration).
    push_kind(
        &mut collections,
        "oidc_providers",
        emit_entries(
            &snapshot.oidc_providers,
            |p| p.name.clone(),
            "oidc_providers",
            &mut diag,
            |_, _, _| {},
            |_, _| {},
        ),
    );

    // claim_mappings — identity: name; `resolve.api_key_id` resugars to
    // the referenced key's file name so the reference survives reload.
    push_kind(
        &mut collections,
        "claim_mappings",
        emit_entries(
            &snapshot.claim_mappings,
            |m| m.name.clone(),
            "claim_mappings",
            &mut diag,
            |doc, identity, diag| resugar_claim_resolve(doc, identity, &api_key_names, diag),
            |_, _| {},
        ),
    );

    // mcp_auth_settings — singleton per environment with a fixed
    // identity; no secrets (the resource URL is public discovery data).
    push_kind(
        &mut collections,
        "mcp_auth_settings",
        emit_entries(
            &snapshot.mcp_auth_settings,
            |_| "mcp_auth_settings".to_string(),
            "mcp_auth_settings",
            &mut diag,
            |_, _, _| {},
            |_, _| {},
        ),
    );

    // guardrail_attachments are emitted above as their own collection, with
    // every id reference rewritten to the identity its collection is keyed by.

    // Two entries whose identities differ only in characters `sanitize`
    // folds to `_` (e.g. `openai-prod` vs `openai.prod`) derive the SAME
    // placeholder variable, so the operator can only supply one value for
    // both secrets — silently feeding one credential to the wrong entry.
    // Surface it like the duplicate-identity check rather than hide it.
    let mut var_owner: BTreeMap<&str, &str> = BTreeMap::new();
    for placeholder in &placeholders {
        match var_owner.get(placeholder.env_var.as_str()) {
            Some(&owner) if owner != placeholder.identity => diag.warnings.push(format!(
                "secret placeholder ${{{}}} is derived for both {:?} and {:?}; their names \
                 collapse to the same environment variable, so one real credential cannot be \
                 supplied to each — rename one entry to disambiguate",
                placeholder.env_var, owner, placeholder.identity
            )),
            Some(_) => {}
            None => {
                var_owner.insert(&placeholder.env_var, &placeholder.identity);
            }
        }
    }

    ExportDocument {
        collections,
        secret_placeholders: placeholders,
        warnings: diag.warnings,
        blocking: diag.blocking,
    }
}

/// Build the file's `guardrail_attachments` entries, rewriting every id
/// reference into the identity its own collection is keyed by in the file.
///
/// A `team` scope carries its id through verbatim, the way
/// `resugar_scope_ref` already does for a team-scoped rate-limit policy:
/// there is no teams collection to name, but `api_keys[].team_id` IS a file
/// field and the runtime compares the two ids as bare strings, so the scope
/// really does resolve standalone. Dropping it would narrow a guardrail while
/// the rate limit beside it survived.
///
/// An attachment the file genuinely cannot express is dropped with a warning
/// rather than emitted dangling: a `passthrough_route` scope (the export
/// carries no routes collection to point at) and a `scope_id` naming a
/// resource missing from the snapshot. Dropping is the safe direction — the
/// guardrail loses that scope and governs less, never more.
fn emit_guardrail_attachments(
    snapshot: &AisixSnapshot,
    model_names: &BTreeMap<String, String>,
    api_key_names: &BTreeMap<String, String>,
    diag: &mut Diagnostics,
) -> Vec<Value> {
    let guardrail_names = id_to_name(&snapshot.guardrails, |g| g.name.clone());
    let mcp_server_names = id_to_name(&snapshot.mcp_servers, |s| s.name.clone());

    let mut out: Vec<Value> = Vec::new();
    for entry in snapshot.guardrail_attachments.entries() {
        let a = &entry.value;
        let Some(guardrail) = guardrail_names.get(&a.guardrail_id) else {
            diag.warnings.push(format!(
                "guardrail attachment {:?} references a guardrail that is not in the snapshot; omitted",
                entry.id
            ));
            continue;
        };
        let scope_name = match a.scope_type {
            GuardrailScopeType::Env => None,
            GuardrailScopeType::Model => Some(("model", model_names.get(scope_ref(a)))),
            GuardrailScopeType::McpServer => {
                Some(("MCP server", mcp_server_names.get(scope_ref(a))))
            }
            GuardrailScopeType::ApiKey => Some(("api key", api_key_names.get(scope_ref(a)))),
            // The export carries no `passthrough_routes` collection, so a
            // route-scoped attachment has nothing to point at in the file —
            // emitting the name anyway makes the whole file fail to load.
            GuardrailScopeType::PassthroughRoute => {
                diag.warnings.push(format!(
                    "guardrail {guardrail:?} has a passthrough-route-scoped attachment; the \
                     export does not carry passthrough routes, so that scope is omitted",
                ));
                continue;
            }
            // Verbatim: `api_keys[].team_id` is a file field and
            // `IndexEntry::applies_to` compares the two ids as bare strings.
            GuardrailScopeType::Team => Some(("team", a.scope_id.as_ref())),
        };
        let resolved = match scope_name {
            None => None,
            Some((label, Some(name))) => {
                let _ = label;
                Some(name.clone())
            }
            Some((label, None)) => {
                diag.warnings.push(format!(
                    "guardrail {guardrail:?} is scoped to a {label} that is not in the snapshot; \
                     that scope is omitted",
                ));
                continue;
            }
        };

        let mut doc = serde_json::Map::new();
        doc.insert("guardrail_id".into(), Value::String(guardrail.clone()));
        doc.insert(
            "scope_type".into(),
            serde_json::to_value(&a.scope_type).unwrap_or(Value::Null),
        );
        if let Some(name) = resolved {
            doc.insert("scope_id".into(), Value::String(name));
        }
        doc.insert("priority".into(), Value::from(a.priority));
        if !a.enabled {
            doc.insert("enabled".into(), Value::Bool(false));
        }
        // Same `$` escaping every other collection gets from `emit_entries`:
        // the loader unescapes `$$` before interpolating, so a name carrying
        // a `$` must be written escaped on BOTH sides of the reference or the
        // two stop matching.
        let mut doc = Value::Object(doc);
        escape_dollars(&mut doc);
        out.push(doc);
    }
    // Deterministic file output: same snapshot must emit the same bytes.
    out.sort_by(|a, b| {
        (
            a.get("guardrail_id").and_then(Value::as_str),
            a.get("scope_type").and_then(Value::as_str),
            a.get("scope_id").and_then(Value::as_str),
        )
            .cmp(&(
                b.get("guardrail_id").and_then(Value::as_str),
                b.get("scope_type").and_then(Value::as_str),
                b.get("scope_id").and_then(Value::as_str),
            ))
    });
    out
}

/// An attachment's `scope_id`, or the empty string — which never matches a
/// real resource id, so an absent scope on a narrow scope_type resolves to
/// "not in the snapshot" and is reported as such.
fn scope_ref(a: &aisix_core::models::GuardrailAttachment) -> &str {
    a.scope_id.as_deref().unwrap_or("")
}

/// Deterministic file identity for a canonical api-key document, which
/// carries no name of its own. `key_hash` is already a SHA-256 hash
/// (safe to surface) and unique per credential, so a hash prefix keys the
/// entry stably without exposing anything sensitive. 16 hex chars (64
/// bits) keeps the label short while making a cross-key prefix collision
/// (which would surface as a duplicate-identity warning anyway) vanishing.
fn synthetic_api_key_name(key_hash: &str) -> String {
    let short: String = key_hash.chars().take(16).collect();
    format!("apikey-{short}")
}

fn push_kind(
    collections: &mut Vec<(&'static str, Vec<Value>)>,
    kind: &'static str,
    entries: Vec<Value>,
) {
    if !entries.is_empty() {
        collections.push((kind, entries));
    }
}

/// Build an `etcd id → file identity` map for one table.
fn id_to_name<T, F>(
    table: &aisix_core::snapshot::ResourceTable<T>,
    identity: F,
) -> BTreeMap<String, String>
where
    T: aisix_core::resource::Resource,
    F: Fn(&T) -> String,
{
    table
        .entries()
        .into_iter()
        .map(|entry| (entry.id.clone(), identity(&entry.value)))
        .collect()
}

/// Serialize every entry of a table (sorted by identity for stable
/// output) and shape it into file form in three ordered steps:
/// `resugar` (rewrite id references to names, synthesize identities),
/// then `$`-escaping of every literal string, then `redact` (swap
/// secrets for `${VAR}` placeholders).
///
/// The order matters. The file loader interpolates `${VAR}` and unescapes
/// `$$` on every string scalar it reads, so a stored value that literally
/// contains `$` (e.g. a guardrail regex matching `${jndi:`) has to be
/// escaped to survive the round-trip — but the placeholders `redact`
/// inserts are the one thing that *should* interpolate, so they are added
/// after escaping and left intact. `resugar` runs before escaping so the
/// reference names it inserts are escaped identically to the identities
/// they point at.
///
/// `resugar` is handed the [`Diagnostics`] sink (threaded through rather
/// than captured, so it and this function share one sink without a double
/// borrow); `redact` collects its placeholders through captures.
fn emit_entries<T, I, Pre, Post>(
    table: &aisix_core::snapshot::ResourceTable<T>,
    identity: I,
    kind: &'static str,
    diag: &mut Diagnostics,
    mut resugar: Pre,
    mut redact: Post,
) -> Vec<Value>
where
    T: aisix_core::resource::Resource + Serialize,
    I: Fn(&T) -> String,
    Pre: FnMut(&mut Value, &str, &mut Diagnostics),
    Post: FnMut(&mut Value, &str),
{
    let mut entries: Vec<_> = table.entries();
    // Stable, human-diffable order independent of DashMap shard layout.
    entries.sort_by(|a, b| identity(&a.value).cmp(&identity(&b.value)));

    let mut out = Vec::with_capacity(entries.len());
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for entry in entries {
        let id = identity(&entry.value);
        let mut doc = match serde_json::to_value(&entry.value) {
            Ok(Value::Object(map)) => Value::Object(map),
            Ok(other) => {
                diag.warnings.push(format!(
                    "{kind} entry {id:?} did not serialize to a document ({other}); skipped"
                ));
                continue;
            }
            Err(e) => {
                diag.warnings.push(format!(
                    "{kind} entry {id:?} could not be serialized ({e}); skipped"
                ));
                continue;
            }
        };
        if !seen.insert(id.clone()) {
            diag.blocking.push(format!(
                "two {kind} entries share the identity {id:?}; the resources file keys entries by \
                 name and rejects duplicates, so the exported file will fail to load until the \
                 source collision is resolved"
            ));
        }
        resugar(&mut doc, &id, diag);
        escape_dollars(&mut doc);
        redact(&mut doc, &id);
        out.push(doc);
    }
    out
}

/// Escape every `$` as `$$` in every string *value* (not object keys —
/// the file loader never interpolates keys). This inverts the loader's
/// `$$` → `$` unescaping so a stored value containing `$` — or a literal
/// `${…}` — round-trips unchanged instead of being read as an
/// interpolation directive. Runs before secret redaction so the
/// `${VAR}` placeholders inserted afterward are the only strings meant
/// to interpolate.
fn escape_dollars(value: &mut Value) {
    match value {
        Value::String(s) => {
            if s.contains('$') {
                *s = s.replace('$', "$$");
            }
        }
        Value::Array(items) => items.iter_mut().for_each(escape_dollars),
        Value::Object(map) => map.values_mut().for_each(escape_dollars),
        _ => {}
    }
}

/// `provider_key_id` (canonical) → `provider_key: <name>` (file sugar).
fn resugar_provider_key(
    doc: &mut Value,
    model: &str,
    provider_key_names: &BTreeMap<String, String>,
    diag: &mut Diagnostics,
) {
    let Some(map) = doc.as_object_mut() else {
        return;
    };
    let Some(id) = map
        .get("provider_key_id")
        .and_then(Value::as_str)
        .map(str::to_string)
    else {
        return;
    };
    match provider_key_names.get(&id) {
        Some(name) => {
            map.remove("provider_key_id");
            map.insert("provider_key".into(), Value::String(name.clone()));
        }
        // A raw provider_key_id the file can't resolve is rejected by the
        // loader (an explicit id must match a file-defined key) — blocking.
        None => diag.blocking.push(format!(
            "model {model:?} references provider_key_id {id:?}, which is not among the exported \
             provider keys — kept as a raw id (dangling reference in the source data; the file \
             will not load until it is resolved)"
        )),
    }
}

/// `claim_mapping.resolve.api_key_id` (etcd id) → `resolve.api_key`
/// (name), the file's sugar form, which the loader resolves back to the
/// key's derived id. Emitting the raw etcd UUID would resolve to nothing
/// at runtime (a silently dead mapping), so a dangling id is blocking.
fn resugar_claim_resolve(
    doc: &mut Value,
    mapping: &str,
    api_key_names: &BTreeMap<String, String>,
    diag: &mut Diagnostics,
) {
    let Some(resolve) = doc.get_mut("resolve").and_then(Value::as_object_mut) else {
        return;
    };
    let Some(id) = resolve
        .get("api_key_id")
        .and_then(Value::as_str)
        .map(str::to_string)
    else {
        return;
    };
    match api_key_names.get(&id) {
        Some(name) => {
            resolve.remove("api_key_id");
            resolve.insert("api_key".into(), Value::String(name.clone()));
        }
        None => diag.blocking.push(format!(
            "claim_mapping {mapping:?} resolve.api_key_id references api key id {id:?}, which is \
             not among the exported api keys — kept as a raw id (dangling reference in the source \
             data; the file will not load until it is resolved)"
        )),
    }
}

/// `cache_policy.applies_to = "api_key:<etcd-id>"` → the id the file will
/// derive for that api_key, so the policy still matches after reload.
///
/// Unlike the other references, the file source has no desugar for
/// `applies_to`: the proxy matches an `api_key` scope against the api
/// key's runtime id, which is name-derived in file mode. Emitting the raw
/// etcd UUID would match nothing (a silent no-op cache policy), so the
/// exporter substitutes the id the loader will assign. `"model:<name>"`
/// matches by the stable model alias and `"all"` needs nothing, so both
/// pass through. A dangling id is kept raw and warned.
fn resugar_cache_applies_to(
    doc: &mut Value,
    policy: &str,
    api_key_names: &BTreeMap<String, String>,
    diag: &mut Diagnostics,
) {
    let Some(map) = doc.as_object_mut() else {
        return;
    };
    // Clone to release the immutable borrow before the insert below.
    let Some(applies_to) = map
        .get("applies_to")
        .and_then(Value::as_str)
        .map(str::to_string)
    else {
        return;
    };
    let Some(etcd_id) = applies_to.trim().strip_prefix("api_key:").map(str::trim) else {
        return;
    };
    match api_key_names.get(etcd_id) {
        Some(name) => {
            let file_id = aisix_core::filesource::derive_id("api_keys", name);
            map.insert(
                "applies_to".into(),
                Value::String(format!("api_key:{file_id}")),
            );
        }
        // `applies_to` is a free-form string the loader accepts as-is (an
        // unknown api_key scope parses to a no-op), so the file still
        // loads — the policy just silently matches nothing. Warning, not
        // blocking.
        None => diag.warnings.push(format!(
            "cache_policy {policy:?} applies_to references api_key id {etcd_id:?}, which is not \
             among the exported api keys — kept as a raw id; the policy will match nothing after \
             reload (dangling reference in the source data)"
        )),
    }
}

/// `scope_ref` (canonical id) → the referenced entry's name for
/// `api_key` / `model` scopes. Team-family scopes pass through verbatim.
fn resugar_scope_ref(
    doc: &mut Value,
    policy: &str,
    model_names: &BTreeMap<String, String>,
    api_key_names: &BTreeMap<String, String>,
    diag: &mut Diagnostics,
) {
    let Some(map) = doc.as_object_mut() else {
        return;
    };
    let (lookup, label) = match map.get("scope").and_then(Value::as_str) {
        Some("model") => (model_names, "model"),
        Some("api_key") => (api_key_names, "api key"),
        // team / member / team_member reference external ids the file
        // carries verbatim; anything else is left for schema validation.
        _ => return,
    };
    let Some(id) = map
        .get("scope_ref")
        .and_then(Value::as_str)
        .map(str::to_string)
    else {
        return;
    };
    match lookup.get(&id) {
        Some(name) => {
            map.insert("scope_ref".into(), Value::String(name.clone()));
        }
        // The file loader resolves an api_key/model scope_ref by name; a
        // raw id left here resolves to nothing and fails the load — blocking.
        None => diag.blocking.push(format!(
            "rate_limit_policy {policy:?} scope_ref references {label} id {id:?}, which is not \
             among the exported {label}s — kept as a raw id (dangling reference in the source \
             data; the file will not load until it is resolved)"
        )),
    }
}

#[cfg(test)]
#[path = "document_tests.rs"]
mod tests;
