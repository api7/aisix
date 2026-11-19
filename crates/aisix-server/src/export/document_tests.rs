use super::*;
use aisix_etcd::{build_snapshot, PrefixSet, RawEntry, WatchedPrefix};
use serde_json::json;

const PK: &str = "0b6f3c1e-2f0a-4c5e-9d55-7d1a8b0c2e11";
const MODEL: &str = "6efd592d-a73b-573d-8e2c-e9e103692f92";
const KEY: &str = "4a1c9e2b-7d3f-4b6a-8c5e-1f2a3b4c5d6e";
const POLICY: &str = "9e8d7c6b-5a4f-4e3d-8c2b-1a0f9e8d7c6b";
const GONE: &str = "00000000-0000-4000-8000-000000000000";

/// A snapshot decoded from etcd rows, the way `aisix export` builds it.
fn snapshot(rows: &[(&str, &str, Value)]) -> AisixSnapshot {
    let entries: Vec<RawEntry> = rows
        .iter()
        .enumerate()
        .map(|(i, (kind, id, value))| RawEntry {
            key: format!("/aisix/{kind}/{id}"),
            value: serde_json::to_vec(value).unwrap(),
            revision: i as i64 + 1,
        })
        .collect();
    let prefixes = PrefixSet::new(vec![WatchedPrefix::environment("/aisix")]);
    let (snap, stats) = build_snapshot(&prefixes, &entries);
    assert!(stats.rejections.is_empty(), "{:?}", stats.rejections);
    snap
}

/// Export `rows` as `aisix export` does: the stored documents, filtered by
/// what the loader accepted.
fn export(rows: &[(&str, &str, Value)], reveal: bool) -> ExportDocument {
    let raw: RawDocs = rows
        .iter()
        .map(|(kind, id, value)| {
            let Value::Object(doc) = value.clone() else {
                unreachable!()
            };
            ((kind.to_string(), id.to_string()), doc)
        })
        .collect();
    build_export_document(&snapshot(rows), &raw, reveal)
}

fn find<'a>(doc: &'a ExportDocument, kind: &str) -> &'a [Value] {
    doc.collections
        .iter()
        .find(|(k, _)| *k == kind)
        .map(|(_, entries)| entries.as_slice())
        .unwrap_or_else(|| panic!("no {kind} collection"))
}

fn rows() -> Vec<(&'static str, &'static str, Value)> {
    vec![
        (
            "provider_keys",
            PK,
            json!({"display_name": "openai-prod", "provider": "openai", "api_key": "sk-live"}),
        ),
        (
            "models",
            MODEL,
            json!({
                "display_name": "gpt-4o",
                "provider": "openai",
                "model_name": "gpt-4o-2024-11-20",
                "provider_key_id": PK
            }),
        ),
        (
            "api_keys",
            KEY,
            json!({"key_hash": "ab".repeat(32), "allowed_model_ids": [MODEL]}),
        ),
        (
            "cache_policies",
            POLICY,
            json!({"name": "per-key", "applies_to": format!("api_key:{KEY}"), "ttl_seconds": 60}),
        ),
    ]
}

#[test]
fn every_entry_carries_its_etcd_id_and_its_references_as_stored() {
    let doc = export(&rows(), false);
    assert_eq!(find(&doc, "provider_keys")[0]["id"], json!(PK));
    let model = &find(&doc, "models")[0];
    assert_eq!(model["id"], json!(MODEL));
    assert_eq!(model["provider_key_id"], json!(PK));
    assert!(model.get("provider_key").is_none());
    let key = &find(&doc, "api_keys")[0];
    assert_eq!(key["id"], json!(KEY));
    assert_eq!(key["allowed_model_ids"], json!([MODEL]));
    assert!(key.get("display_name").is_none(), "no name is synthesized");
    assert_eq!(
        find(&doc, "cache_policies")[0]["applies_to"],
        json!(format!("api_key:{KEY}"))
    );
    assert!(doc.warnings.is_empty(), "{:?}", doc.warnings);
}

#[test]
fn the_exported_file_loads_back_as_the_same_rows() {
    let doc = export(&rows(), false);
    let yaml = super::super::yaml_emit::emit_yaml(&doc).unwrap();
    let env = |name: &str| {
        (name == "AISIXSECRET_PROVIDER_KEY_OPENAI_PROD_API_KEY").then(|| "sk-live".to_string())
    };
    let entries = aisix_etcd::file_source::read_str(&yaml, "exported.yaml", 1, &env)
        .unwrap_or_else(|e| panic!("{e}\n{yaml}"));
    let (loaded, stats) = build_snapshot(&aisix_etcd::file_source::prefixes(), &entries);
    assert!(stats.rejections.is_empty(), "{:?}", stats.rejections);

    let model = loaded.models.get_by_id(MODEL).expect("model keeps its id");
    assert_eq!(model.value.provider_key_id.as_deref(), Some(PK));
    assert_eq!(
        loaded.provider_keys.get_by_id(PK).unwrap().value.api_key,
        "sk-live"
    );
    let key = loaded.apikeys.get_by_id(KEY).expect("api key keeps its id");
    assert_eq!(
        key.value.allowed_model_ids.as_deref(),
        Some([MODEL.to_string()].as_slice())
    );
    assert!(loaded.cache_policies.get_by_id(POLICY).is_some());
}

#[test]
fn a_reference_to_an_id_not_exported_is_kept_and_warned() {
    let mut rows = rows();
    rows[1].2["provider_key_id"] = json!(GONE);
    rows[2].2["allowed_model_ids"] = json!([MODEL, GONE]);
    rows[3].2["applies_to"] = json!(format!("api_key:{GONE}"));
    rows.push((
        "rate_limit_policies",
        "1b2c3d4e-5f60-4718-9a0b-1c2d3e4f5a6b",
        json!({
            "name": "cap",
            "scope": "api_key",
            "scope_ref": GONE,
            "window": "minute",
            "max_requests": 10
        }),
    ));
    let doc = export(&rows, false);
    assert_eq!(find(&doc, "models")[0]["provider_key_id"], json!(GONE));
    let warnings = doc.warnings.join("\n");
    for field in [
        "`provider_key_id` references provider_keys",
        "`allowed_model_ids[]` references models",
        "`applies_to` references api_keys",
        "`scope_ref` references api_keys",
    ] {
        assert!(
            warnings.contains(field),
            "missing {field:?} in:\n{warnings}"
        );
    }
    assert_eq!(doc.warnings.len(), 4, "{warnings}");
}

#[test]
fn a_dangling_attachment_scope_is_warned() {
    let doc = export(
        &[(
            "guardrail_attachments",
            POLICY,
            json!({"guardrail_id": GONE, "scope_type": "model", "scope_id": MODEL, "priority": 1}),
        )],
        false,
    );
    let warnings = doc.warnings.join("\n");
    assert!(
        warnings.contains("`guardrail_id` references guardrails"),
        "{warnings}"
    );
    assert!(
        warnings.contains("`scope_id` references models"),
        "{warnings}"
    );
}

#[test]
fn default_export_emits_no_live_secret() {
    let doc = export(&rows(), false);
    let pk = &find(&doc, "provider_keys")[0];
    assert_eq!(
        pk["api_key"],
        json!("${AISIXSECRET_PROVIDER_KEY_OPENAI_PROD_API_KEY}")
    );
    assert_eq!(doc.secret_placeholders.len(), 1);
    assert_eq!(doc.secret_placeholders[0].id, PK);

    let revealed = export(&rows(), true);
    assert_eq!(
        find(&revealed, "provider_keys")[0]["api_key"],
        json!("sk-live")
    );
    assert!(revealed.secret_placeholders.is_empty());
}

#[test]
fn credentials_beside_the_api_key_are_redacted() {
    let doc = export(
        &[
            (
                "provider_keys",
                PK,
                json!({
                    "display_name": "mistral",
                    "api_key": "sk-1",
                    "request": {
                        "default_headers": {"x-secondary": "tok-2"},
                        "default_body_fields": {"token": "tok-3", "safe_prompt": true}
                    }
                }),
            ),
            (
                "oidc_providers",
                MODEL,
                json!({"name": "corp", "hmac_secret": "s".repeat(40)}),
            ),
        ],
        false,
    );
    let text = serde_json::to_string(&doc.collections).unwrap();
    for secret in ["sk-1", "tok-2", "tok-3", &"s".repeat(40)] {
        assert!(!text.contains(secret), "{secret} leaked: {text}");
    }
    assert!(text.contains("\"safe_prompt\":true"), "{text}");
    // Replaced by a placeholder at each field, not dropped.
    let placeholder = |v: &Value| v.as_str().is_some_and(|s| s.starts_with("${AISIXSECRET_"));
    let pk = &find(&doc, "provider_keys")[0];
    assert!(placeholder(&pk["api_key"]), "{pk}");
    assert!(
        placeholder(&pk["request"]["default_headers"]["x-secondary"]),
        "{pk}"
    );
    assert!(
        placeholder(&pk["request"]["default_body_fields"]["token"]),
        "{pk}"
    );
    let oidc = &find(&doc, "oidc_providers")[0];
    assert!(placeholder(&oidc["hmac_secret"]), "{oidc}");
}

#[test]
fn two_entries_sharing_a_name_warn_about_their_placeholder() {
    let doc = export(
        &[
            (
                "provider_keys",
                PK,
                json!({"display_name": "same", "api_key": "a"}),
            ),
            (
                "provider_keys",
                MODEL,
                json!({"display_name": "same", "api_key": "b"}),
            ),
        ],
        false,
    );
    assert_eq!(doc.warnings.len(), 1, "{:?}", doc.warnings);
    assert!(doc.warnings[0].contains("AISIXSECRET_PROVIDER_KEY_SAME_API_KEY"));
}

#[test]
fn dollars_in_stored_strings_are_escaped() {
    let doc = export(
        &[(
            "provider_keys",
            PK,
            json!({"display_name": "costs $5", "api_key": "x"}),
        )],
        true,
    );
    assert_eq!(
        find(&doc, "provider_keys")[0]["display_name"],
        json!("costs $$5")
    );
}

const TEAM: &str = "1b2c3d4e-5f60-4718-8a9b-0c1d2e3f4a5b";
const USER: &str = "2c3d4e5f-6071-4829-9bac-1d2e3f4a5b6c";
const ENV: &str = "3d4e5f60-7182-493a-acbd-2e3f4a5b6c7d";
const PRICE: &str = "4e5f6071-8293-4a4b-bdce-3f4a5b6c7d8e";
const CATALOG_PRICE: &str = "5f607182-93a4-4b5c-8edf-4a5b6c7d8e9f";
const BUDGET: &str = "60718293-a4b5-4c6d-9fe0-5b6c7d8e9fa0";

/// One document of every kind beside the ones `rows()` carries that the
/// gateway reads, the catalog price under the global prefix.
fn every_other_kind() -> Vec<RawEntry> {
    [
        ("/aisix/teams", TEAM, json!({"name": "eng"})),
        ("/aisix/users", USER, json!({"name": "ana"})),
        (
            "/aisix/budget_policy",
            ENV,
            json!({"fail_mode": "closed", "closed_grace_seconds": 600}),
        ),
        (
            "/aisix/usage_accounting",
            ENV,
            json!({"exclude_estimated_usage": true}),
        ),
        (
            "/aisix/pricing",
            PRICE,
            json!({"key": "openai/gpt-4o", "input_per_1k": 5, "output_per_1k": 15}),
        ),
        (
            "/aisix/global/pricing",
            CATALOG_PRICE,
            json!({"key": "openai/gpt-4o", "input_per_1k": 2.5, "output_per_1k": 10}),
        ),
        (
            "/aisix/budgets",
            BUDGET,
            json!({"scope": "api_key", "scope_ref": KEY, "period": "month", "exhausted": {"reason": {}}}),
        ),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (prefix, id, value))| RawEntry {
        key: format!("{prefix}/{id}"),
        value: serde_json::to_vec(&value).unwrap(),
        revision: i as i64 + 100,
    })
    .collect()
}

/// `aisix export` over raw etcd rows, environment and global prefix both.
fn export_entries(entries: &[RawEntry]) -> ExportDocument {
    let prefixes = PrefixSet::new(vec![
        WatchedPrefix::environment("/aisix"),
        WatchedPrefix::global("/aisix/global/"),
    ]);
    let (snap, stats) = build_snapshot(&prefixes, entries);
    assert!(stats.rejections.is_empty(), "{:?}", stats.rejections);
    let (raw, notes) = stored_documents(&prefixes, entries, &stats);
    assert!(notes.is_empty(), "{notes:?}");
    build_export_document(&snap, &raw, false)
}

#[test]
fn every_kind_the_gateway_reads_is_exported_except_budgets() {
    let doc = export_entries(&every_other_kind());
    for (kind, id) in [
        ("teams", TEAM),
        ("users", USER),
        ("budget_policy", ENV),
        ("usage_accounting", ENV),
        ("pricing", PRICE),
    ] {
        assert_eq!(find(&doc, kind)[0]["id"], json!(id), "{kind}");
    }
    assert_eq!(
        find(&doc, "usage_accounting")[0]["exclude_estimated_usage"],
        json!(true)
    );
    assert_eq!(
        doc.global_collections,
        vec![(
            "pricing",
            vec![json!({
                "id": CATALOG_PRICE,
                "key": "openai/gpt-4o",
                "input_per_1k": 2.5,
                "output_per_1k": 10
            })]
        )]
    );
    assert!(doc.collections.iter().all(|(k, _)| *k != "budgets"));
    assert_eq!(doc.warnings.len(), 1, "{:?}", doc.warnings);
    assert!(
        doc.warnings[0].contains("1 budgets document(s) not exported"),
        "{}",
        doc.warnings[0]
    );
}

#[test]
fn every_exported_kind_loads_back_from_the_file() {
    let doc = export_entries(&every_other_kind());
    let yaml = super::super::yaml_emit::emit_yaml(&doc).unwrap();
    let entries = aisix_etcd::file_source::read_str(&yaml, "exported.yaml", 1, &|_| None)
        .unwrap_or_else(|e| panic!("{e}\n{yaml}"));
    let (loaded, stats) = build_snapshot(&aisix_etcd::file_source::prefixes(), &entries);
    assert!(stats.rejections.is_empty(), "{:?}", stats.rejections);
    assert!(loaded.teams.get_by_id(TEAM).is_some());
    assert!(loaded.users.get_by_id(USER).is_some());
    assert!(loaded.budget_policy.get_by_id(ENV).is_some());
    assert!(loaded.excludes_estimated_usage());
    assert!(loaded.pricing.get_by_id(PRICE).is_some());
    assert!(loaded.global_pricing.get_by_id(CATALOG_PRICE).is_some());
    assert!(loaded.pricing.get_by_id(CATALOG_PRICE).is_none());
    assert!(loaded.budgets.is_empty());
}

#[test]
fn documents_are_written_as_stored_not_re_serialized() {
    // The typed model would add every default (`endpoint: null` among them,
    // which the read schema refuses); the file carries what etcd carries.
    let doc = export(
        &[(
            "guardrails",
            POLICY,
            json!({"name": "mod", "kind": "openai_moderation", "api_key": "sk-x"}),
        )],
        true,
    );
    assert_eq!(
        find(&doc, "guardrails")[0],
        json!({"id": POLICY, "name": "mod", "kind": "openai_moderation", "api_key": "sk-x"})
    );
}

#[test]
fn the_id_comes_from_the_key_not_from_a_stored_id_field() {
    let doc = export(
        &[(
            "provider_keys",
            PK,
            json!({"display_name": "pk", "api_key": "x", "id": GONE}),
        )],
        true,
    );
    assert_eq!(find(&doc, "provider_keys")[0]["id"], json!(PK));
}

#[test]
fn a_custom_guardrail_secret_named_like_a_credential_gets_one_placeholder() {
    let doc = export(
        &[(
            "guardrails",
            POLICY,
            json!({
                "name": "screen",
                "kind": "custom",
                "script": "export async function checkInput() { return { action: \"none\" }; }",
                "secrets": {"api_key": "tok-live"}
            }),
        )],
        false,
    );
    assert_eq!(
        doc.secret_placeholders.len(),
        1,
        "{:?}",
        doc.secret_placeholders
    );
    assert_eq!(
        find(&doc, "guardrails")[0]["secrets"]["api_key"],
        json!(format!("${{{}}}", doc.secret_placeholders[0].env_var))
    );
}

#[test]
fn a_custom_guardrail_s_script_secrets_are_redacted() {
    let doc = export(
        &[(
            "guardrails",
            POLICY,
            json!({
                "name": "screen",
                "kind": "custom",
                "script": "export async function checkInput() { return { action: \"none\" }; }",
                "secrets": {"SCREENING_TOKEN": "tok-live-secret"}
            }),
        )],
        false,
    );
    let gr = &find(&doc, "guardrails")[0];
    assert!(!gr.to_string().contains("tok-live-secret"), "{gr}");
    assert!(gr["secrets"]["SCREENING_TOKEN"]
        .as_str()
        .unwrap()
        .starts_with("${AISIXSECRET_"));
}

#[test]
fn a_model_priced_by_a_key_no_exported_pricing_document_carries_is_reported() {
    let mut rows = rows();
    rows[1].2["pricing_key"] = json!("openai/gpt-4o");
    let doc = export(&rows, false);
    assert_eq!(doc.warnings.len(), 1, "{:?}", doc.warnings);
    assert!(
        doc.warnings[0].contains("carries no price"),
        "{}",
        doc.warnings[0]
    );
    rows[1].2["cost"] = json!({"input_per_1k": 1, "output_per_1k": 2});
    let with_cost = export(&rows, false);
    assert!(
        with_cost.warnings[0].contains("falls back to its inline `cost`"),
        "{}",
        with_cost.warnings[0]
    );
    rows.push((
        "pricing",
        PRICE,
        json!({"key": "openai/gpt-4o", "input_per_1k": 5, "output_per_1k": 15}),
    ));
    let priced = export(&rows, false);
    assert!(priced.warnings.is_empty(), "{:?}", priced.warnings);
}

#[test]
fn of_two_prices_sharing_a_key_only_the_one_the_gateway_applies_is_exported() {
    // The newer row has the smaller id, so a file that stamps both with one
    // revision would apply the older price.
    let older = "ffffffff-0000-4000-8000-000000000000";
    let newer = "00000000-ffff-4000-8000-000000000000";
    let entries: Vec<RawEntry> = [(older, 1, 1), (newer, 2, 9)]
        .into_iter()
        .map(|(id, revision, price)| RawEntry {
            key: format!("/aisix/global/pricing/{id}"),
            value: serde_json::to_vec(
                &json!({"key": "openai/gpt-4o", "input_per_1k": price, "output_per_1k": price}),
            )
            .unwrap(),
            revision,
        })
        .collect();
    let doc = export_entries(&entries);
    let rows = &doc.global_collections[0].1;
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0]["id"], json!(newer));
    assert_eq!(rows[0]["input_per_1k"], json!(9));
    assert_eq!(doc.warnings.len(), 1, "{:?}", doc.warnings);
    assert!(doc.warnings[0].contains(older), "{}", doc.warnings[0]);
}

#[test]
fn two_fields_of_one_entry_sharing_a_placeholder_are_reported() {
    let doc = export(
        &[(
            "provider_keys",
            PK,
            json!({
                "display_name": "pk",
                "api_key": "a",
                "request": {"default_headers": {"x-tenant-token": "t1", "x_tenant_token": "t2"}}
            }),
        )],
        false,
    );
    assert_eq!(doc.warnings.len(), 1, "{:?}", doc.warnings);
    assert!(
        doc.warnings[0].contains("same environment variable"),
        "{}",
        doc.warnings[0]
    );
}

#[test]
fn fields_the_gateway_ignores_are_left_out_of_the_stored_documents() {
    let entries = vec![RawEntry {
        key: format!("/aisix/provider_keys/{PK}"),
        value: serde_json::to_vec(&json!({
            "display_name": "pk",
            "api_key": "x",
            "access_token": "tok-left-over",
            "request": {"timeout_hint": "tok-nested"}
        }))
        .unwrap(),
        revision: 1,
    }];
    let prefixes = PrefixSet::new(vec![WatchedPrefix::environment("/aisix")]);
    let (_, stats) = build_snapshot(&prefixes, &entries);
    let (docs, notes) = stored_documents(&prefixes, &entries, &stats);
    let doc = &docs[&("provider_keys".to_string(), PK.to_string())];
    let text = serde_json::to_string(doc).unwrap();
    assert!(
        !text.contains("tok-left-over") && !text.contains("tok-nested"),
        "{text}"
    );
    assert_eq!(doc["api_key"], json!("x"));
    assert_eq!(notes.len(), 1, "{notes:?}");
}
