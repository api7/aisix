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
    // outright, and inventing one would widen it.
    let snap = AisixSnapshot::new();
    snap.guardrails.insert(ResourceEntry::new(
        "g-1",
        keyword_guardrail("team-guard"),
        1,
    ));
    snap.guardrail_attachments.insert(ResourceEntry::new(
        "att-team",
        serde_json::from_value(attachment("g-1", "team", Some("t-1"))).unwrap(),
        1,
    ));
    snap.guardrail_attachments.insert(ResourceEntry::new(
        "att-missing-model",
        serde_json::from_value(attachment("g-1", "model", Some("m-gone"))).unwrap(),
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
            .any(|w| w.contains("no teams collection")),
        "team scope must be reported: {:?}",
        doc.warnings
    );
    assert!(
        doc.warnings
            .iter()
            .any(|w| w.contains("not in the snapshot")),
        "dangling model scope must be reported: {:?}",
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
