use super::*;
use aisix_core::resource::ResourceEntry;
use serde_json::json;

fn provider_key(display_name: &str, api_key: &str) -> aisix_core::models::ProviderKey {
    serde_json::from_value(json!({"display_name": display_name, "api_key": api_key})).unwrap()
}

fn model_value(json: Value) -> aisix_core::models::Model {
    serde_json::from_value(json).unwrap()
}

fn find<'a>(doc: &'a ExportDocument, kind: &str) -> &'a [Value] {
    doc.collections
        .iter()
        .find(|(k, _)| *k == kind)
        .map(|(_, v)| v.as_slice())
        .unwrap_or(&[])
}

#[test]
fn provider_key_ref_resugars_to_name() {
    let snap = AisixSnapshot::new();
    snap.provider_keys.insert(ResourceEntry::new(
        "pk-uuid-1",
        provider_key("openai-prod", "sk-live"),
        1,
    ));
    snap.models.insert(ResourceEntry::new(
        "m-uuid-1",
        model_value(json!({
            "display_name": "gpt-4o",
            "provider": "openai",
            "model_name": "gpt-4o-2024-11-20",
            "provider_key_id": "pk-uuid-1"
        })),
        1,
    ));

    let doc = build_export_document(&snap, false);
    let models = find(&doc, "models");
    assert_eq!(models.len(), 1);
    // Canonical id reference gone; file name sugar in its place.
    assert!(models[0].get("provider_key_id").is_none());
    assert_eq!(models[0]["provider_key"], json!("openai-prod"));
}

#[test]
fn dangling_provider_key_ref_is_kept_and_warned() {
    let snap = AisixSnapshot::new();
    snap.models.insert(ResourceEntry::new(
        "m-uuid-1",
        model_value(json!({
            "display_name": "orphan",
            "provider": "openai",
            "model_name": "gpt-4o",
            "provider_key_id": "pk-does-not-exist"
        })),
        1,
    ));
    let doc = build_export_document(&snap, false);
    let models = find(&doc, "models");
    assert_eq!(models[0]["provider_key_id"], json!("pk-does-not-exist"));
    assert!(models[0].get("provider_key").is_none());
    // A dangling provider_key_id makes the file non-loadable → blocking.
    assert!(
        doc.blocking
            .iter()
            .any(|w| w.contains("dangling") && w.contains("orphan")),
        "{:?}",
        doc.blocking
    );
}

#[test]
fn api_key_gets_synthetic_name_and_keeps_key_hash() {
    let snap = AisixSnapshot::new();
    let key_hash = "91ed2dbc407561556f3e7be98ba0bd2a57986d6a868c482d867d19c6d40d201c";
    snap.apikeys.insert(ResourceEntry::new(
        "k-uuid-1",
        serde_json::from_value(json!({"key_hash": key_hash, "allowed_models": ["*"]})).unwrap(),
        1,
    ));
    let doc = build_export_document(&snap, false);
    let keys = find(&doc, "api_keys");
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0]["display_name"], json!("apikey-91ed2dbc40756155"));
    // key_hash is already hashed — emitted verbatim, no placeholder.
    assert_eq!(keys[0]["key_hash"], json!(key_hash));
    assert!(doc.secret_placeholders.is_empty());
}

#[test]
fn scope_ref_resolves_for_model_and_api_key_scopes() {
    let snap = AisixSnapshot::new();
    let key_hash = "aa".repeat(32);
    snap.models.insert(ResourceEntry::new(
            "m-uuid-1",
            model_value(json!({"display_name": "gpt-4o", "provider": "openai", "model_name": "x", "provider_key_id": "pk"})),
            1,
        ));
    snap.apikeys.insert(ResourceEntry::new(
        "k-uuid-1",
        serde_json::from_value(json!({"key_hash": key_hash, "allowed_models": ["*"]})).unwrap(),
        1,
    ));
    snap.provider_keys
        .insert(ResourceEntry::new("pk", provider_key("pk", "sk"), 1));
    for (name, scope, scope_ref) in [
        ("cap-model", "model", "m-uuid-1"),
        ("cap-key", "api_key", "k-uuid-1"),
        ("cap-team", "team", "team-uuid-9"),
    ] {
        snap.rate_limit_policies.insert(ResourceEntry::new(
            format!("rlp-{name}"),
            serde_json::from_value(json!({
                "name": name, "scope": scope, "scope_ref": scope_ref,
                "window": "minute", "max_requests": 10
            }))
            .unwrap(),
            1,
        ));
    }

    let doc = build_export_document(&snap, false);
    let policies = find(&doc, "rate_limit_policies");
    let by_name = |n: &str| policies.iter().find(|p| p["name"] == json!(n)).unwrap();
    assert_eq!(by_name("cap-model")["scope_ref"], json!("gpt-4o"));
    assert_eq!(
        by_name("cap-key")["scope_ref"],
        json!(synthetic_api_key_name(&"aa".repeat(32)))
    );
    // team scope passes through verbatim.
    assert_eq!(by_name("cap-team")["scope_ref"], json!("team-uuid-9"));
}

#[test]
fn duplicate_identity_within_a_kind_warns() {
    let snap = AisixSnapshot::new();
    // Two provider keys with the same display_name but distinct ids —
    // possible in raw etcd, impossible in the file.
    snap.provider_keys
        .insert(ResourceEntry::new("pk-a", provider_key("dup", "sk-a"), 1));
    snap.provider_keys
        .insert(ResourceEntry::new("pk-b", provider_key("dup", "sk-b"), 1));
    let doc = build_export_document(&snap, false);
    // Duplicate identity makes the file non-loadable → blocking.
    assert!(
        doc.blocking
            .iter()
            .any(|w| w.contains("share the identity") && w.contains("dup")),
        "{:?}",
        doc.blocking
    );
}

#[test]
fn provider_key_request_default_headers_and_body_fields_are_redacted() {
    let snap = AisixSnapshot::new();
    let pk: aisix_core::models::ProviderKey = serde_json::from_value(json!({
        "display_name": "pk",
        "api_key": "sk-main-SECRET",
        "request": {
            "default_headers": { "x-tenant-token": "hdr-SECRET" },
            "default_body_fields": { "api_key": "body-SECRET", "safe_prompt": true }
        }
    }))
    .unwrap();
    snap.provider_keys.insert(ResourceEntry::new("pk-1", pk, 1));
    let doc = build_export_document(&snap, false);
    let rendered = serde_json::to_string(&find(&doc, "provider_keys")).unwrap();
    for secret in ["sk-main-SECRET", "hdr-SECRET", "body-SECRET"] {
        assert!(!rendered.contains(secret), "leaked {secret}: {rendered}");
    }
    // Non-string body field preserved.
    let pk_out = &find(&doc, "provider_keys")[0];
    assert_eq!(
        pk_out["request"]["default_body_fields"]["safe_prompt"],
        json!(true)
    );
}

#[test]
fn cache_policy_api_key_applies_to_resugars_to_derived_id() {
    use aisix_core::filesource::derive_id;
    let snap = AisixSnapshot::new();
    let key_hash = "cd".repeat(32);
    snap.apikeys.insert(ResourceEntry::new(
        "k-uuid-1",
        serde_json::from_value(json!({"key_hash": key_hash, "allowed_models": ["*"]})).unwrap(),
        1,
    ));
    snap.cache_policies.insert(ResourceEntry::new(
        "cp-1",
        serde_json::from_value(json!({"name": "cap-key", "applies_to": "api_key:k-uuid-1"}))
            .unwrap(),
        1,
    ));
    snap.cache_policies.insert(ResourceEntry::new(
        "cp-2",
        serde_json::from_value(json!({"name": "cap-model", "applies_to": "model:gpt-4o"})).unwrap(),
        1,
    ));
    let doc = build_export_document(&snap, false);
    let policies = find(&doc, "cache_policies");
    let by_name = |n: &str| policies.iter().find(|p| p["name"] == json!(n)).unwrap();
    // api_key id → the id the file loader will derive from the api key's
    // synthesized name, so the policy still matches after reload.
    let expected = format!(
        "api_key:{}",
        derive_id("api_keys", &synthetic_api_key_name(&"cd".repeat(32)))
    );
    assert_eq!(by_name("cap-key")["applies_to"], json!(expected));
    // model scope matches by alias — unchanged.
    assert_eq!(by_name("cap-model")["applies_to"], json!("model:gpt-4o"));
}

#[test]
fn cache_policy_dangling_api_key_applies_to_is_kept_and_warned() {
    let snap = AisixSnapshot::new();
    snap.cache_policies.insert(ResourceEntry::new(
        "cp-1",
        serde_json::from_value(json!({"name": "orphan", "applies_to": "api_key:missing-uuid"}))
            .unwrap(),
        1,
    ));
    let doc = build_export_document(&snap, false);
    assert_eq!(
        find(&doc, "cache_policies")[0]["applies_to"],
        json!("api_key:missing-uuid")
    );
    assert!(
        doc.warnings
            .iter()
            .any(|w| w.contains("dangling") && w.contains("orphan")),
        "{:?}",
        doc.warnings
    );
}

#[test]
fn placeholder_env_var_collision_across_identities_warns() {
    let snap = AisixSnapshot::new();
    // Two provider keys whose display_names differ only in a character
    // `sanitize` folds to `_` → the same derived env var.
    snap.provider_keys.insert(ResourceEntry::new(
        "pk-a",
        provider_key("openai-prod", "sk-a"),
        1,
    ));
    snap.provider_keys.insert(ResourceEntry::new(
        "pk-b",
        provider_key("openai.prod", "sk-b"),
        1,
    ));
    let doc = build_export_document(&snap, false);
    assert!(
        doc.warnings
            .iter()
            .any(|w| w.contains("same environment variable")),
        "{:?}",
        doc.warnings
    );
}

#[test]
fn default_export_emits_no_live_provider_secret() {
    let snap = AisixSnapshot::new();
    snap.provider_keys.insert(ResourceEntry::new(
        "pk-1",
        provider_key("openai-prod", "sk-super-secret-do-not-leak"),
        1,
    ));
    let doc = build_export_document(&snap, false);
    let pks = find(&doc, "provider_keys");
    assert_eq!(
        pks[0]["api_key"],
        json!("${AISIXSECRET_PROVIDER_KEY_OPENAI_PROD_API_KEY}")
    );
    // Secret must appear nowhere in the assembled collections.
    let rendered =
        serde_json::to_string(&doc.collections.iter().map(|(_, v)| v).collect::<Vec<_>>()).unwrap();
    assert!(
        !rendered.contains("sk-super-secret-do-not-leak"),
        "{rendered}"
    );
    assert_eq!(doc.secret_placeholders.len(), 1);
}

#[test]
fn reveal_secrets_emits_the_real_value_inline() {
    let snap = AisixSnapshot::new();
    snap.provider_keys.insert(ResourceEntry::new(
        "pk-1",
        provider_key("openai-prod", "sk-real-value"),
        1,
    ));
    let doc = build_export_document(&snap, true);
    let pks = find(&doc, "provider_keys");
    assert_eq!(pks[0]["api_key"], json!("sk-real-value"));
    assert!(doc.secret_placeholders.is_empty());
}

fn keyword_guardrail(name: &str) -> aisix_core::models::Guardrail {
    serde_json::from_value(json!({
        "name": name, "kind": "keyword",
        "patterns": [{ "kind": "literal", "value": "blocked-phrase" }]
    }))
    .unwrap()
}

fn attachment(guardrail_id: &str, scope_type: &str, scope_id: Option<&str>) -> Value {
    let mut a = json!({"guardrail_id": guardrail_id, "scope_type": scope_type, "priority": 1});
    if let Some(id) = scope_id {
        a["scope_id"] = json!(id);
    }
    a
}

#[test]
fn env_scoped_guardrail_exports_with_its_attachment() {
    // The file carries the scope now, so the guardrail and the attachment
    // that puts it in force are exported together.
    let snap = AisixSnapshot::new();
    snap.guardrails.insert(ResourceEntry::new(
        "g-1",
        keyword_guardrail("global-guard"),
        1,
    ));
    snap.guardrail_attachments.insert(ResourceEntry::new(
        "att-1",
        serde_json::from_value(attachment("g-1", "env", None)).unwrap(),
        1,
    ));
    let doc = build_export_document(&snap, false);
    let guardrails = find(&doc, "guardrails");
    assert_eq!(guardrails.len(), 1);
    assert_eq!(guardrails[0]["name"], json!("global-guard"));

    let attachments = find(&doc, "guardrail_attachments");
    assert_eq!(attachments.len(), 1);
    assert_eq!(attachments[0]["guardrail_id"], json!("global-guard"));
    assert_eq!(attachments[0]["scope_type"], json!("env"));
    assert!(attachments[0].get("scope_id").is_none());
}

#[test]
fn narrow_scope_survives_the_export_as_a_reference_by_name() {
    // AISIX-Cloud#1450. This used to assert the opposite: a model-scoped
    // guardrail was DROPPED, because the file had no attachment collection
    // and anything it did carry applied gateway-wide — exporting a narrow
    // rule would have widened it to all traffic. The file expresses scope
    // now, so the scope round-trips instead of being discarded, and the
    // reference is emitted as the model's file identity.
    let snap = AisixSnapshot::new();
    snap.models.insert(ResourceEntry::new(
        "m-1",
        serde_json::from_value(json!({
            "display_name": "gpt-4o",
            "provider": "openai",
            "model_name": "gpt-4o-2024-11-20",
            "provider_key_id": "pk-1"
        }))
        .unwrap(),
        1,
    ));
    snap.guardrails.insert(ResourceEntry::new(
        "g-scoped",
        keyword_guardrail("model-guard"),
        1,
    ));
    snap.guardrail_attachments.insert(ResourceEntry::new(
        "att-1",
        serde_json::from_value(attachment("g-scoped", "model", Some("m-1"))).unwrap(),
        1,
    ));

    let doc = build_export_document(&snap, false);
    let attachments = find(&doc, "guardrail_attachments");
    assert_eq!(attachments.len(), 1);
    assert_eq!(attachments[0]["guardrail_id"], json!("model-guard"));
    assert_eq!(attachments[0]["scope_type"], json!("model"));
    assert_eq!(
        attachments[0]["scope_id"],
        json!("gpt-4o"),
        "scope must be emitted as the model's file identity, not its etcd id",
    );
}

#[test]
fn an_unattached_guardrail_exports_with_no_attachment() {
    // Inert on both sides now, so exporting it is faithful rather than
    // dangerous: it governs nothing in etcd and nothing in the file.
    let snap = AisixSnapshot::new();
    snap.guardrails.insert(ResourceEntry::new(
        "g-inert",
        keyword_guardrail("inert-guard"),
        1,
    ));
    let doc = build_export_document(&snap, false);
    let guardrails = find(&doc, "guardrails");
    assert_eq!(guardrails.len(), 1);
    assert_eq!(guardrails[0]["name"], json!("inert-guard"));
    assert!(
        doc.collections
            .iter()
            .all(|(k, _)| *k != "guardrail_attachments"),
        "nothing may be synthesized on the guardrail's behalf",
    );
}

#[test]
fn an_attachment_the_file_cannot_name_is_dropped_with_a_warning() {
    // Dropping loses scope, which makes the guardrail govern LESS — the
    // safe direction. Emitting a dangling reference would fail the import
    // outright (the loader rejects the whole file), and inventing one would
    // widen it.
    let snap = AisixSnapshot::new();
    snap.guardrails.insert(ResourceEntry::new(
        "g-1",
        keyword_guardrail("team-guard"),
        1,
    ));
    snap.guardrail_attachments.insert(ResourceEntry::new(
        "att-missing-model",
        serde_json::from_value(attachment("g-1", "model", Some("m-gone"))).unwrap(),
        1,
    ));
    // The export carries no passthrough_routes collection at all, so a
    // route-scoped attachment has nothing to point at. Emitting the name
    // anyway produced a file the loader rejected WHOLE, with export still
    // exiting 0.
    snap.guardrail_attachments.insert(ResourceEntry::new(
        "att-route",
        serde_json::from_value(attachment("g-1", "passthrough_route", Some("r-1"))).unwrap(),
        1,
    ));

    let doc = build_export_document(&snap, false);
    assert!(
        doc.collections
            .iter()
            .all(|(k, _)| *k != "guardrail_attachments"),
        "neither attachment can be named in the file: {:?}",
        doc.collections
    );
    assert!(
        doc.warnings
            .iter()
            .any(|w| w.contains("not in the snapshot")),
        "dangling model scope must be reported: {:?}",
        doc.warnings
    );
    assert!(
        doc.warnings
            .iter()
            .any(|w| w.contains("does not carry passthrough routes")),
        "route scope must be reported: {:?}",
        doc.warnings
    );
}

#[test]
fn empty_snapshot_yields_only_a_header_later() {
    let snap = AisixSnapshot::new();
    let doc = build_export_document(&snap, false);
    assert!(doc.collections.is_empty());
    assert!(doc.secret_placeholders.is_empty());
    assert!(doc.warnings.is_empty());
}

#[test]
fn escape_dollars_doubles_dollars_in_string_values_only() {
    let mut v = json!({
        "plain": "no dollars",
        "regex": "price=\\$5 and ${jndi:x}",
        "nested": { "list": ["a$b", 3, true] }
    });
    escape_dollars(&mut v);
    assert_eq!(v["plain"], json!("no dollars"));
    assert_eq!(v["regex"], json!("price=\\$$5 and $${jndi:x}"));
    assert_eq!(v["nested"]["list"][0], json!("a$$b"));
    // Non-strings untouched.
    assert_eq!(v["nested"]["list"][1], json!(3));
    assert_eq!(v["nested"]["list"][2], json!(true));
}

#[test]
fn export_output_reloads_through_the_real_file_loader() {
    use aisix_core::filesource::{derive_id, load_from_str};
    use std::collections::HashMap;

    let snap = AisixSnapshot::new();
    snap.provider_keys.insert(ResourceEntry::new(
        "pk-1",
        provider_key("openai-prod", "sk-live-value"),
        1,
    ));
    snap.models.insert(ResourceEntry::new(
        "m-1",
        model_value(json!({
            "display_name": "gpt-4o",
            "provider": "openai",
            "model_name": "gpt-4o-2024-11-20",
            "provider_key_id": "pk-1"
        })),
        1,
    ));
    // A guardrail whose literal contains a real `${...}` — the exact
    // shape a Log4Shell/template-injection blocklist rule takes. If it
    // were emitted unescaped the loader would try to interpolate it and
    // the whole file would fail to load; escaping is what lets it
    // survive.
    snap.guardrails.insert(ResourceEntry::new(
        "g-1",
        serde_json::from_value(json!({
            "name": "log4shell",
            "kind": "keyword",
            "patterns": [{ "kind": "literal", "value": "${jndi:ldap}" }]
        }))
        .unwrap(),
        1,
    ));
    // env-scoped attachment → the guardrail is gateway-wide, so it is
    // exported (and its `${jndi:ldap}` literal must round-trip).
    snap.guardrail_attachments.insert(ResourceEntry::new(
        "att-1",
        serde_json::from_value(attachment("g-1", "env", None)).unwrap(),
        1,
    ));
    // A claim mapping whose `resolve.api_key_id` must resugar to the
    // key's (synthetic) file name and re-resolve on load — plus the key
    // and trust provider it references, so the loader's cross-checks
    // hold.
    snap.apikeys.insert(ResourceEntry::new(
        "ak-1",
        serde_json::from_value(json!({
            "key_hash": "91ed2dbc407561556f3e7be98ba0bd2a57986d6a868c482d867d19c6d40d201c",
            "allowed_models": ["gpt-4o"]
        }))
        .unwrap(),
        1,
    ));
    snap.oidc_providers.insert(ResourceEntry::new(
        "op-1",
        serde_json::from_value(json!({
            "name": "corp",
            "issuer": "https://sso.example.com/realms/agents",
            "audiences": ["aisix"]
        }))
        .unwrap(),
        1,
    ));
    snap.claim_mappings.insert(ResourceEntry::new(
        "cm-1",
        serde_json::from_value(json!({
            "name": "finance-dept",
            "jwt_provider": "corp",
            "match": [{"claim": "department", "op": "exact", "values": ["finance"]}],
            "resolve": {"api_key_id": "ak-1"}
        }))
        .unwrap(),
        1,
    ));

    let doc = build_export_document(&snap, false);
    let yaml = crate::export::yaml_emit::emit_yaml(&doc).expect("emit");

    // Feed the placeholders the file loader will interpolate.
    let env: HashMap<String, String> = doc
        .secret_placeholders
        .iter()
        .map(|p| (p.env_var.clone(), "sk-real".to_string()))
        .collect();
    let loaded = load_from_str(&yaml, "exported.yaml", 1, &|n| env.get(n).cloned())
        .expect("the exported file must re-load through the file source");

    // Same resource set, and the reference resugared then re-resolved to
    // the same derived id the loader assigns.
    assert_eq!(loaded.provider_keys.len(), 1);
    assert_eq!(loaded.models.len(), 1);
    assert_eq!(loaded.guardrails.len(), 1);
    let model = loaded.models.get_by_name("gpt-4o").unwrap();
    assert_eq!(
        model.value.provider_key_id.as_deref(),
        Some(derive_id("provider_keys", "openai-prod").as_str())
    );
    // The `${jndi:ldap}` literal came back byte-for-byte — not
    // interpolated, not corrupted.
    let guardrail = loaded.guardrails.get_by_name("log4shell").unwrap();
    let value = serde_json::to_value(&guardrail.value).unwrap();
    assert_eq!(value["patterns"][0]["value"], json!("${jndi:ldap}"));

    // The claim mapping's key reference resugared to the synthetic file
    // name and re-resolved to the id the loader derives for that key —
    // the whole reason the exporter cannot emit the raw etcd uuid.
    assert_eq!(loaded.oidc_providers.len(), 1);
    assert_eq!(loaded.claim_mappings.len(), 1);
    let cm = loaded.claim_mappings.get_by_name("finance-dept").unwrap();
    assert_eq!(cm.value.jwt_provider, "corp");
    assert_eq!(
        cm.value.resolve.api_key_id,
        derive_id("api_keys", "apikey-91ed2dbc40756155")
    );
}

#[test]
fn dangling_claim_mapping_target_is_kept_and_blocking() {
    let snap = AisixSnapshot::new();
    snap.claim_mappings.insert(ResourceEntry::new(
        "cm-1",
        serde_json::from_value(json!({
            "name": "finance-dept",
            "jwt_provider": "corp",
            "match": [{"claim": "department", "op": "exact", "values": ["finance"]}],
            "resolve": {"api_key_id": "ak-does-not-exist"}
        }))
        .unwrap(),
        1,
    ));
    let doc = build_export_document(&snap, false);
    let mappings = find(&doc, "claim_mappings");
    assert_eq!(mappings.len(), 1);
    // Raw id kept, no name sugar minted for a key that isn't there.
    assert_eq!(
        mappings[0]["resolve"]["api_key_id"],
        json!("ak-does-not-exist")
    );
    assert!(mappings[0]["resolve"].get("api_key").is_none());
    // A dangling target makes the file non-loadable → blocking.
    assert!(
        doc.blocking
            .iter()
            .any(|w| w.contains("dangling") && w.contains("finance-dept")),
        "{:?}",
        doc.blocking
    );
}

/// A team scope has no collection to name, but it still round-trips: the id
/// goes through verbatim, `api_keys[].team_id` is a file field, and the
/// runtime compares the two as bare strings. Dropping it would narrow a
/// guardrail while the team-scoped rate limit beside it survived.
#[test]
fn a_team_scope_is_carried_through_verbatim() {
    let snap = AisixSnapshot::new();
    snap.guardrails.insert(ResourceEntry::new(
        "g-1",
        keyword_guardrail("team-guard"),
        1,
    ));
    snap.guardrail_attachments.insert(ResourceEntry::new(
        "att-team",
        serde_json::from_value(attachment("g-1", "team", Some("team-alpha"))).unwrap(),
        1,
    ));

    let doc = build_export_document(&snap, false);
    let attachments = find(&doc, "guardrail_attachments");
    assert_eq!(attachments.len(), 1);
    assert_eq!(attachments[0]["scope_type"], json!("team"));
    assert_eq!(
        attachments[0]["scope_id"],
        json!("team-alpha"),
        "a team id has no name to resolve to and must survive unchanged",
    );
    assert!(
        doc.warnings.is_empty(),
        "a team scope is expressible, so nothing should be reported: {:?}",
        doc.warnings
    );
}

#[test]
fn api_key_allowed_model_ids_resugar_to_names() {
    let snap = AisixSnapshot::new();
    snap.provider_keys
        .insert(ResourceEntry::new("pk", provider_key("pk", "sk"), 1));
    for (id, display_name) in [("m-uuid-1", "gpt-4o"), ("m-uuid-2", "claude")] {
        snap.models.insert(ResourceEntry::new(
            id,
            model_value(json!({
                "display_name": display_name,
                "provider": "openai",
                "model_name": "x",
                "provider_key_id": "pk"
            })),
            1,
        ));
    }
    snap.apikeys.insert(ResourceEntry::new(
        "k-uuid-1",
        serde_json::from_value(json!({
            "key_hash": "aa".repeat(32),
            "allowed_models": ["stale-name"],
            "allowed_model_ids": ["m-uuid-2"]
        }))
        .unwrap(),
        1,
    ));

    let doc = build_export_document(&snap, false);
    let keys = find(&doc, "api_keys");
    // The file source grants by name only: the id form never reaches it,
    // and the name it resolves to replaces whatever `allowed_models` held.
    assert!(keys[0].get("allowed_model_ids").is_none());
    assert_eq!(keys[0]["allowed_models"], json!(["claude"]));
    assert!(doc.warnings.is_empty(), "{:?}", doc.warnings);
    assert!(doc.blocking.is_empty(), "{:?}", doc.blocking);
}

#[test]
fn api_key_unresolvable_model_id_is_dropped_and_warned() {
    let snap = AisixSnapshot::new();
    snap.provider_keys
        .insert(ResourceEntry::new("pk", provider_key("pk", "sk"), 1));
    snap.models.insert(ResourceEntry::new(
        "m-uuid-1",
        model_value(
            json!({"display_name": "gpt-4o", "provider": "openai", "model_name": "x", "provider_key_id": "pk"}),
        ),
        1,
    ));
    snap.apikeys.insert(ResourceEntry::new(
        "k-uuid-1",
        serde_json::from_value(json!({
            "key_hash": "aa".repeat(32),
            "allowed_model_ids": ["m-uuid-1", "m-gone"]
        }))
        .unwrap(),
        1,
    ));

    let doc = build_export_document(&snap, false);
    let keys = find(&doc, "api_keys");
    // Emitting "m-gone" as a name would fail the loader's model
    // cross-reference and take the whole file down; dropping it only
    // narrows the key, which is what the gateway already does.
    assert_eq!(keys[0]["allowed_models"], json!(["gpt-4o"]));
    assert!(
        doc.warnings.iter().any(|w| w.contains("m-gone")),
        "{:?}",
        doc.warnings
    );
    assert!(doc.blocking.is_empty(), "{:?}", doc.blocking);
}

#[test]
fn api_key_empty_allowed_model_ids_export_as_no_grant() {
    let snap = AisixSnapshot::new();
    snap.apikeys.insert(ResourceEntry::new(
        "k-uuid-1",
        serde_json::from_value(json!({
            "key_hash": "aa".repeat(32),
            "allowed_models": ["*"],
            "allowed_model_ids": []
        }))
        .unwrap(),
        1,
    ));
    let doc = build_export_document(&snap, false);
    let keys = find(&doc, "api_keys");
    // An empty id list is authoritative at runtime, so the exported file
    // must not resurrect the ignored `allowed_models: ["*"]`.
    assert_eq!(keys[0]["allowed_models"], json!([]));
}

/// Every id-form model reference in a model document leaves the export as
/// the name form the resources file accepts — the file refuses the id
/// form outright, so an export that kept it would not reload.
#[test]
fn model_reference_ids_resugar_to_names() {
    let snap = AisixSnapshot::new();
    snap.provider_keys
        .insert(ResourceEntry::new("pk-1", provider_key("pk", "sk-x"), 1));
    for (id, name) in [("m-a", "alpha"), ("m-b", "beta"), ("m-c", "gamma")] {
        snap.models.insert(ResourceEntry::new(
            id,
            model_value(json!({
                "display_name": name,
                "provider": "openai",
                "model_name": "gpt-4o",
                "provider_key_id": "pk-1"
            })),
            1,
        ));
    }
    snap.models.insert(ResourceEntry::new(
        "m-embed",
        model_value(json!({
            "display_name": "embedder",
            "provider": "openai",
            "model_name": "text-embedding-3-small",
            "provider_key_id": "pk-1",
            "embedding": {"dimensions": 4}
        })),
        1,
    ));
    snap.models.insert(ResourceEntry::new(
        "m-group",
        model_value(json!({
            "display_name": "group",
            "routing": {"targets": [{"model_id": "m-a"}, {"model": "beta"}]}
        })),
        1,
    ));
    snap.models.insert(ResourceEntry::new(
        "m-panel",
        model_value(json!({
            "display_name": "panel",
            "ensemble": {
                "panel": [{"model_id": "m-a"}, {"model_id": "m-b"}],
                "judge": {"model_id": "m-c"}
            }
        })),
        1,
    ));
    snap.models.insert(ResourceEntry::new(
        "m-router",
        model_value(json!({
            "display_name": "router",
            "semantic": {
                "embedding_model_id": "m-embed",
                "routes": [{"name": "r", "target_id": "m-a", "examples": ["hi"]}],
                "default_id": "m-b",
                "match": {"threshold": 0.5},
                "on_embedding_failure": {"target_id": "m-c"}
            }
        })),
        1,
    ));

    let doc = build_export_document(&snap, false);
    let by_name = |name: &str| -> Value {
        find(&doc, "models")
            .iter()
            .find(|m| m["display_name"] == json!(name))
            .cloned()
            .unwrap_or_else(|| panic!("{name} exported"))
    };

    let group = by_name("group");
    assert_eq!(group["routing"]["targets"][0]["model"], json!("alpha"));
    assert!(group["routing"]["targets"][0].get("model_id").is_none());
    assert_eq!(group["routing"]["targets"][1]["model"], json!("beta"));

    let panel = by_name("panel");
    assert_eq!(panel["ensemble"]["panel"][0]["model"], json!("alpha"));
    assert_eq!(panel["ensemble"]["panel"][1]["model"], json!("beta"));
    assert_eq!(panel["ensemble"]["judge"]["model"], json!("gamma"));
    assert!(panel["ensemble"]["judge"].get("model_id").is_none());

    let router = by_name("router");
    assert_eq!(router["semantic"]["embedding_model"], json!("embedder"));
    assert_eq!(router["semantic"]["routes"][0]["target"], json!("alpha"));
    assert_eq!(router["semantic"]["default"], json!("beta"));
    assert_eq!(
        router["semantic"]["on_embedding_failure"]["target"],
        json!("gamma")
    );
    assert!(router["semantic"].get("embedding_model_id").is_none());
    assert!(router["semantic"].get("default_id").is_none());

    assert!(doc.blocking.is_empty(), "{:?}", doc.blocking);
}

/// An id no exported model answers to is emitted under the name field as
/// itself — the same dangling reference the gateway already sees. For a
/// model the loader cross-checks it, so it is blocking.
#[test]
fn dangling_model_reference_id_is_emitted_as_a_name_and_blocking() {
    let snap = AisixSnapshot::new();
    snap.models.insert(ResourceEntry::new(
        "m-group",
        model_value(json!({
            "display_name": "group",
            "routing": {"targets": [{"model_id": "m-gone"}]}
        })),
        1,
    ));
    let doc = build_export_document(&snap, false);
    let group = &find(&doc, "models")[0];
    assert_eq!(group["routing"]["targets"][0]["model"], json!("m-gone"));
    assert!(group["routing"]["targets"][0].get("model_id").is_none());
    assert!(
        doc.blocking
            .iter()
            .any(|b| b.contains("dangling") && b.contains("m-gone")),
        "{:?}",
        doc.blocking
    );
}

/// A cache policy's model scope collapses into the `applies_to` string the
/// file understands, overriding whatever that string held — the same
/// precedence the gateway applies.
#[test]
fn cache_policy_model_scope_id_resugars_into_applies_to() {
    let snap = AisixSnapshot::new();
    snap.provider_keys
        .insert(ResourceEntry::new("pk-1", provider_key("pk", "sk-x"), 1));
    snap.models.insert(ResourceEntry::new(
        "m-embed",
        model_value(json!({
            "display_name": "embedder",
            "provider": "openai",
            "model_name": "text-embedding-3-small",
            "provider_key_id": "pk-1",
            "embedding": {"dimensions": 4}
        })),
        1,
    ));
    let policy: aisix_core::models::CachePolicy = serde_json::from_value(json!({
        "name": "faq",
        "applies_to": "all",
        "applies_to_model_id": "m-embed",
        "semantic": {"embedding_model_id": "m-embed", "threshold": 0.9}
    }))
    .unwrap();
    snap.cache_policies
        .insert(ResourceEntry::new("cp-1", policy, 1));

    let doc = build_export_document(&snap, false);
    let exported = &find(&doc, "cache_policies")[0];
    assert_eq!(exported["applies_to"], json!("model:embedder"));
    assert!(exported.get("applies_to_model_id").is_none());
    assert_eq!(exported["semantic"]["embedding_model"], json!("embedder"));
    assert!(exported["semantic"].get("embedding_model_id").is_none());
}

/// A semantic guardrail's embedder id becomes the name form. The loader
/// does not cross-check it, so a dangling one is a warning and the file
/// still loads (screening then refuses, fail-closed, as it already did).
#[test]
fn guardrail_embedder_id_resugars_to_a_name() {
    let snap = AisixSnapshot::new();
    snap.provider_keys
        .insert(ResourceEntry::new("pk-1", provider_key("pk", "sk-x"), 1));
    snap.models.insert(ResourceEntry::new(
        "m-embed",
        model_value(json!({
            "display_name": "embedder",
            "provider": "openai",
            "model_name": "text-embedding-3-small",
            "provider_key_id": "pk-1",
            "embedding": {"dimensions": 4}
        })),
        1,
    ));
    let row = |name: &str, id: &str| -> aisix_core::models::Guardrail {
        serde_json::from_value(json!({
            "name": name,
            "kind": "semantic",
            "embedding_model_id": id,
            "deny_examples": ["x"],
            "deny_threshold": 0.8
        }))
        .unwrap()
    };
    snap.guardrails
        .insert(ResourceEntry::new("g-1", row("resolved", "m-embed"), 1));
    snap.guardrails
        .insert(ResourceEntry::new("g-2", row("dangling", "m-gone"), 1));

    let doc = build_export_document(&snap, false);
    let by_name = |name: &str| -> Value {
        find(&doc, "guardrails")
            .iter()
            .find(|g| g["name"] == json!(name))
            .cloned()
            .unwrap_or_else(|| panic!("{name} exported"))
    };
    assert_eq!(by_name("resolved")["embedding_model"], json!("embedder"));
    assert!(by_name("resolved").get("embedding_model_id").is_none());
    assert_eq!(by_name("dangling")["embedding_model"], json!("m-gone"));
    assert!(doc.blocking.is_empty(), "{:?}", doc.blocking);
    assert!(
        doc.warnings.iter().any(|w| w.contains("m-gone")),
        "{:?}",
        doc.warnings
    );
}
