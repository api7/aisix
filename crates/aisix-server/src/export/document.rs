//! Decoded etcd snapshot → resources-file document.
//!
//! A resources file is the set of etcd documents written down (see
//! `aisix_etcd::file_source`), so every entry is emitted as its etcd id
//! plus the document exactly as etcd stores it — never re-serialized from
//! the typed model, which writes defaults and nulls the read schema does
//! not accept back. Only rows the loader accepted are emitted. Two things
//! are done to each document on the way:
//!
//! - **`$`-escaping** — the file interpolates `${VAR}` and unescapes `$$`
//!   in every string it reads, so a stored `$` is written as `$$`;
//! - **secret redaction** — see [`super::secrets`].
//!
//! A reference naming an id that is not among the exported entries is
//! reported as a warning: the file carries it as stored, and a gateway
//! loading it resolves it to nothing, exactly as the source gateway does.

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
const GUARDRAIL_SECRET_KEYS: &[&str] = &[
    "api_key",
    "access_key_secret",
    "secret_access_key",
    "session_token",
];

/// The assembled export, ready to emit as YAML plus the side-channel
/// information the command reports on stderr.
pub struct ExportDocument {
    /// `(kind, entries)` in a fixed collection order; only non-empty kinds
    /// are included. Every entry carries its `id`.
    pub collections: Vec<(&'static str, Vec<Value>)>,
    /// Placeholders substituted for live credentials (empty when
    /// `reveal_secrets` is set).
    pub secret_placeholders: Vec<SecretPlaceholder>,
    /// Things the operator should see: kinds left out, dangling
    /// references, placeholder-variable collisions.
    pub warnings: Vec<String>,
}

/// Which credentials of one kind's documents are redacted.
type Redact = fn(&mut Value, &mut RedactionCtx<'_>);

/// The stored documents, keyed by `(kind, id)`.
pub type RawDocs = BTreeMap<(String, String), serde_json::Map<String, Value>>;

/// Build the resources-file document from the stored documents and the
/// snapshot the loader decoded from them (which decides what was accepted).
pub fn build_export_document(
    snapshot: &AisixSnapshot,
    raw: &RawDocs,
    reveal_secrets: bool,
) -> ExportDocument {
    let mut placeholders = Vec::new();
    let mut collections: Vec<(&'static str, Vec<Value>)> = Vec::new();

    macro_rules! export {
        ($table:ident, $kind:literal, $token:literal, $name:expr, $redact:expr) => {
            push_kind(
                &mut collections,
                $kind,
                emit_entries(
                    &snapshot.$table,
                    raw,
                    $kind,
                    $token,
                    $name,
                    $redact,
                    reveal_secrets,
                    &mut placeholders,
                ),
            )
        };
    }

    export!(
        provider_keys,
        "provider_keys",
        "PROVIDER_KEY",
        |pk| pk.display_name.clone(),
        |doc, ctx| {
            // `secret` is the former spelling of `api_key`, still loaded.
            redact_top_level(doc, "api_key", ctx);
            redact_top_level(doc, "secret", ctx);
            // `request.default_headers` are outbound auth headers to the
            // upstream and `request.default_body_fields` can carry a
            // secondary credential — neither is covered by `api_key`.
            // String values only (a `safe_prompt: true` flag stays intact).
            if let Some(request) = doc.get_mut("request") {
                redact_string_map(request, "default_headers", "default header", ctx);
                redact_string_map(request, "default_body_fields", "default body field", ctx);
            }
        }
    );
    export!(
        models,
        "models",
        "MODEL",
        |m| m.display_name.clone(),
        |_, _| {}
    );
    export!(apikeys, "api_keys", "API_KEY", |_| String::new(), |_, _| {});
    export!(
        guardrails,
        "guardrails",
        "GUARDRAIL",
        |g| g.name.clone(),
        |doc, ctx| {
            // A `kind: custom` script's credentials, keyed by the names it
            // reads, are redacted as a map — kept out of the by-name pass,
            // which would give a `secrets.api_key` a second placeholder.
            let secrets = doc.as_object_mut().and_then(|d| d.remove("secrets"));
            redact_by_key(doc, GUARDRAIL_SECRET_KEYS, ctx);
            if let (Some(secrets), Some(d)) = (secrets, doc.as_object_mut()) {
                d.insert("secrets".into(), secrets);
                redact_string_map(doc, "secrets", "secret", ctx);
            }
        }
    );
    export!(
        guardrail_attachments,
        "guardrail_attachments",
        "GUARDRAIL_ATTACHMENT",
        |_| String::new(),
        |_, _| {}
    );
    export!(
        mcp_servers,
        "mcp_servers",
        "MCP_SERVER",
        |s| s.name.clone(),
        |doc, ctx| { redact_top_level(doc, "secret", ctx) }
    );
    export!(
        mcp_policies,
        "mcp_policies",
        "MCP_POLICY",
        |_| String::new(),
        |_, _| {}
    );
    export!(
        a2a_agents,
        "a2a_agents",
        "A2A_AGENT",
        |a| a.name.clone(),
        |doc, ctx| { redact_top_level(doc, "secret", ctx) }
    );
    export!(
        cache_policies,
        "cache_policies",
        "CACHE_POLICY",
        |c| c.name.clone(),
        |_, _| {}
    );
    export!(
        observability_exporters,
        "observability_exporters",
        "OBSERVABILITY_EXPORTER",
        |e| e.name.clone(),
        redact_headers
    );
    export!(
        rate_limit_policies,
        "rate_limit_policies",
        "RATE_LIMIT_POLICY",
        |p| p.name.clone(),
        |_, _| {}
    );
    // Issuer, audiences and the JWKS endpoint are public trust
    // configuration; a shared-secret provider's `hmac_secret` is the
    // credential every one of its callers authenticates with.
    export!(
        oidc_providers,
        "oidc_providers",
        "OIDC_PROVIDER",
        |p| p.name.clone(),
        |doc, ctx| {
            redact_top_level(doc, "hmac_secret", ctx);
            if let Some(login) = doc.get_mut("login") {
                redact_by_key(login, &["client_secret"], ctx);
            }
        }
    );
    export!(
        claim_mappings,
        "claim_mappings",
        "CLAIM_MAPPING",
        |m| m.name.clone(),
        |_, _| {}
    );
    export!(
        passthrough_routes,
        "passthrough_routes",
        "PASSTHROUGH_ROUTE",
        |r| r.name.clone(),
        |_, _| {}
    );
    export!(
        mcp_auth_settings,
        "mcp_auth_settings",
        "MCP_AUTH_SETTINGS",
        |_| String::new(),
        |_, _| {}
    );

    let mut warnings = dangling_references(snapshot);

    // Pricing documents are not exported, so a model priced by one changes
    // price on a gateway loading this file.
    for entry in snapshot.models.entries() {
        let model = &entry.value;
        let Some(key) = model.pricing_key.as_deref() else {
            continue;
        };
        // Read off the stored document: a model's price is otherwise only
        // ever read through the pricing resolver.
        let has_cost = raw
            .get(&("models".to_string(), entry.id.clone()))
            .is_some_and(|doc| doc.contains_key("cost"));
        let effect = if has_cost {
            "falls back to its inline `cost`, a different price whenever the two disagree"
        } else {
            "carries no price and ranks last under `least_cost`; set `cost` if the price matters"
        };
        warnings.push(format!(
            "model {:?} ({}) is priced by the pricing document {key:?}, which is not exported \
             — on a gateway loading this file it {effect}",
            model.display_name, entry.id
        ));
    }

    // Pricing, teams, users, budgets and usage accounting are not exported: a gateway loading
    // this file runs without them, and says so here.
    for (kind, count, effect) in [
        (
            "teams",
            snapshot.teams.len(),
            "the `team_name` metric label reads `unknown`",
        ),
        (
            "users",
            snapshot.users.len(),
            "the `user_name` metric label falls back to each api key's own `user_name`",
        ),
        ("budgets", snapshot.budgets.len(), "no budget is enforced"),
        (
            "budget_policy",
            snapshot.budget_policy.len(),
            "budgeted traffic behaves as `sticky`",
        ),
        (
            "usage_accounting",
            snapshot.usage_accounting.len(),
            "estimated usage counts toward token rate limits",
        ),
        (
            "pricing",
            snapshot.pricing.len() + snapshot.global_pricing.len(),
            "a model with a `pricing_key` falls back to its inline `cost`",
        ),
    ] {
        if count > 0 {
            warnings.push(format!(
                "{count} {kind} document(s) not exported — on a gateway loading this file {effect}"
            ));
        }
    }

    // Two entries whose names differ only in characters `sanitize` folds
    // to `_` (or share a name outright) derive the SAME placeholder
    // variable, so the operator can only supply one value for both secrets
    // — silently feeding one credential to the wrong entry.
    let mut var_owner: BTreeMap<&str, &SecretPlaceholder> = BTreeMap::new();
    for placeholder in &placeholders {
        match var_owner.get(placeholder.env_var.as_str()) {
            Some(owner) if (&owner.id, &owner.field) != (&placeholder.id, &placeholder.field) => {
                warnings.push(format!(
                    "secret placeholder ${{{}}} is derived for both {} {:?} {} and {} {:?} {}; \
                     they collapse to the same environment variable, so one real credential \
                     cannot be supplied to each — rename one or edit one placeholder",
                    placeholder.env_var,
                    owner.kind,
                    owner.id,
                    owner.field,
                    placeholder.kind,
                    placeholder.id,
                    placeholder.field
                ))
            }
            Some(_) => {}
            None => {
                var_owner.insert(&placeholder.env_var, placeholder);
            }
        }
    }

    ExportDocument {
        collections,
        secret_placeholders: placeholders,
        warnings,
    }
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

/// Emit every accepted entry of a table (sorted by id for stable output) as
/// `id` + its stored document, then `$`-escape every literal string, then
/// swap secrets for `${VAR}` placeholders.
///
/// The order matters: the file unescapes `$$` on every string scalar it
/// reads, so a stored value that literally contains `$` (e.g. a guardrail
/// regex matching `${jndi:`) is escaped to survive the round trip — but
/// the placeholders redaction inserts are the one thing that *should*
/// interpolate, so they are added after escaping.
#[allow(clippy::too_many_arguments)]
fn emit_entries<T, N>(
    table: &aisix_core::snapshot::ResourceTable<T>,
    raw: &RawDocs,
    kind: &'static str,
    kind_token: &'static str,
    name: N,
    redact: Redact,
    reveal: bool,
    placeholders: &mut Vec<SecretPlaceholder>,
) -> Vec<Value>
where
    T: aisix_core::resource::Resource,
    N: Fn(&T) -> String,
{
    let mut entries = table.entries();
    entries.sort_by(|a, b| a.id.cmp(&b.id));

    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let Some(stored) = raw.get(&(kind.to_string(), entry.id.clone())) else {
            continue;
        };
        // `id` from the key, never from a stray `id` field in the document.
        let mut doc = serde_json::Map::new();
        doc.insert("id".into(), Value::String(entry.id.clone()));
        doc.extend(
            stored
                .iter()
                .filter(|(k, _)| k.as_str() != "id")
                .map(|(k, v)| (k.clone(), v.clone())),
        );
        let mut doc = Value::Object(doc);
        escape_dollars(&mut doc);
        let label = name(&entry.value);
        let mut ctx = RedactionCtx {
            kind_token,
            kind,
            identity: if label.is_empty() { &entry.id } else { &label },
            id: &entry.id,
            reveal,
            out: placeholders,
        };
        redact(&mut doc, &mut ctx);
        out.push(doc);
    }
    out
}

/// Escape every `$` as `$$` in every string *value* (not object keys —
/// the file never interpolates keys).
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

/// Every reference whose id is not among the exported entries of the kind
/// it names, as warnings.
fn dangling_references(snapshot: &AisixSnapshot) -> Vec<String> {
    fn ids<T>(table: &aisix_core::snapshot::ResourceTable<T>) -> BTreeSet<String>
    where
        T: aisix_core::resource::Resource,
    {
        table.entries().into_iter().map(|e| e.id.clone()).collect()
    }
    let known: BTreeMap<&str, BTreeSet<String>> = BTreeMap::from([
        ("provider_keys", ids(&snapshot.provider_keys)),
        ("models", ids(&snapshot.models)),
        ("api_keys", ids(&snapshot.apikeys)),
        ("guardrails", ids(&snapshot.guardrails)),
        ("mcp_servers", ids(&snapshot.mcp_servers)),
        ("passthrough_routes", ids(&snapshot.passthrough_routes)),
    ]);

    let mut warnings = Vec::new();
    let mut check = |owner_kind: &str, owner: &str, field: &str, target: &str, id: &str| {
        if !known.get(target).is_some_and(|ids| ids.contains(id)) {
            warnings.push(format!(
                "{owner_kind} {owner:?} `{field}` references {target} id {id:?}, which is not \
                 among the exported {target} (dangling reference in the source data)"
            ));
        }
    };

    for (kind, table) in [
        ("models", to_docs(&snapshot.models)),
        ("api_keys", to_docs(&snapshot.apikeys)),
        ("guardrails", to_docs(&snapshot.guardrails)),
        ("cache_policies", to_docs(&snapshot.cache_policies)),
        ("claim_mappings", to_docs(&snapshot.claim_mappings)),
        ("mcp_auth_settings", to_docs(&snapshot.mcp_auth_settings)),
        ("mcp_policies", to_docs(&snapshot.mcp_policies)),
        ("passthrough_routes", to_docs(&snapshot.passthrough_routes)),
        (
            "rate_limit_policies",
            to_docs(&snapshot.rate_limit_policies),
        ),
    ] {
        for (id, doc) in &table {
            for (path, target) in reference_paths(kind) {
                for (field, value) in values_at(doc, path) {
                    check(kind, id, &field, target, &value);
                }
            }
            match kind {
                "cache_policies" => {
                    if let Some(key) = doc
                        .get("applies_to")
                        .and_then(Value::as_str)
                        .and_then(|s| s.trim().strip_prefix("api_key:"))
                    {
                        check(kind, id, "applies_to", "api_keys", key.trim());
                    }
                }
                "rate_limit_policies" => {
                    let target = match doc.get("scope").and_then(Value::as_str) {
                        Some("api_key") => Some("api_keys"),
                        Some("model") => Some("models"),
                        _ => None,
                    };
                    if let (Some(target), Some(scope_ref)) =
                        (target, doc.get("scope_ref").and_then(Value::as_str))
                    {
                        check(kind, id, "scope_ref", target, scope_ref);
                    }
                    if let Some(conditions) = doc.get("conditions") {
                        condition_refs(conditions, &mut |target, value| {
                            check(kind, id, "conditions", target, value)
                        });
                    }
                }
                _ => {}
            }
        }
    }

    for entry in snapshot.guardrail_attachments.entries() {
        let a = &entry.value;
        check(
            "guardrail_attachments",
            &entry.id,
            "guardrail_id",
            "guardrails",
            &a.guardrail_id,
        );
        let target = match a.scope_type {
            GuardrailScopeType::Model => "models",
            GuardrailScopeType::McpServer => "mcp_servers",
            GuardrailScopeType::ApiKey => "api_keys",
            GuardrailScopeType::PassthroughRoute => "passthrough_routes",
            // `env` carries no scope_id; a team is not a resource here.
            GuardrailScopeType::Env | GuardrailScopeType::Team => continue,
        };
        check(
            "guardrail_attachments",
            &entry.id,
            "scope_id",
            target,
            a.scope_id.as_deref().unwrap_or(""),
        );
    }
    warnings
}

fn to_docs<T>(table: &aisix_core::snapshot::ResourceTable<T>) -> Vec<(String, Value)>
where
    T: aisix_core::resource::Resource + Serialize,
{
    let mut out: Vec<(String, Value)> = table
        .entries()
        .into_iter()
        .filter_map(|e| Some((e.id.clone(), aisix_core::json::to_value(&e.value).ok()?)))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// The id-form references a document of `kind` carries, as paths: `[]`
/// steps into every array element, `{}` takes every key of a map.
fn reference_paths(kind: &str) -> &'static [(&'static str, &'static str)] {
    match kind {
        "models" => &[
            ("provider_key_id", "provider_keys"),
            ("routing.targets.[].model_id", "models"),
            ("ensemble.panel.[].model_id", "models"),
            ("ensemble.judge.model_id", "models"),
            ("semantic.routes.[].target_id", "models"),
            ("semantic.default_id", "models"),
            ("semantic.embedding_model_id", "models"),
            ("semantic.on_embedding_failure.target_id", "models"),
            ("semantic.classifier.model_id", "models"),
            ("semantic.classifier.provider_key_id", "provider_keys"),
            ("semantic.classifier.on_failure.target_id", "models"),
        ],
        "api_keys" => &[
            ("allowed_model_ids.[]", "models"),
            ("mcp_rate_limits_by_id.{}", "mcp_servers"),
            ("mcp_access.allow_ids.[].server_id", "mcp_servers"),
            ("mcp_access.deny_ids.[].server_id", "mcp_servers"),
        ],
        "guardrails" => &[("embedding_model_id", "models")],
        "cache_policies" => &[
            ("applies_to_model_id", "models"),
            ("semantic.embedding_model_id", "models"),
        ],
        "claim_mappings" => &[("resolve.api_key_id", "api_keys")],
        "mcp_auth_settings" => &[
            ("anonymous.api_key_id", "api_keys"),
            ("anonymous.server_ids.[]", "mcp_servers"),
        ],
        "mcp_policies" => &[
            ("allow_ids.[].server_id", "mcp_servers"),
            ("deny_ids.[].server_id", "mcp_servers"),
        ],
        "passthrough_routes" => &[
            ("anonymous_key_id", "api_keys"),
            ("provider_key_id", "provider_keys"),
        ],
        _ => &[],
    }
}

/// The string values at `path` in `doc`, each with the field it was
/// found under. A trailing `{}` takes the keys of the map instead.
fn values_at(doc: &Value, path: &str) -> Vec<(String, String)> {
    let (body, keys) = match path.strip_suffix(".{}") {
        Some(body) => (body, true),
        None => (path, false),
    };
    let mut nodes = vec![doc];
    for segment in body.split('.') {
        nodes = nodes
            .into_iter()
            .flat_map(|node| -> Vec<&Value> {
                if segment == "[]" {
                    node.as_array().into_iter().flatten().collect()
                } else {
                    node.get(segment).into_iter().collect()
                }
            })
            .collect();
    }
    let field = body.replace(".[]", "[]");
    if keys {
        nodes
            .into_iter()
            .filter_map(Value::as_object)
            .flat_map(|map| map.keys())
            .map(|key| (field.clone(), key.clone()))
            .collect()
    } else {
        nodes
            .into_iter()
            .filter_map(Value::as_str)
            .map(|id| (field.clone(), id.to_string()))
            .collect()
    }
}

/// The `api_key` / `model` leaf values of a condition tree.
fn condition_refs(nodes: &Value, found: &mut dyn FnMut(&'static str, &str)) {
    for node in nodes.as_array().into_iter().flatten() {
        if let Some(children) = node.get("children") {
            condition_refs(children, found);
            continue;
        }
        let target = match node.get("dimension").and_then(Value::as_str) {
            Some("api_key") => "api_keys",
            Some("model") => "models",
            _ => continue,
        };
        match node.get("value") {
            Some(Value::String(v)) => found(target, v),
            Some(Value::Array(values)) => values
                .iter()
                .filter_map(Value::as_str)
                .for_each(|v| found(target, v)),
            _ => {}
        }
    }
}

#[cfg(test)]
#[path = "document_tests.rs"]
mod tests;
