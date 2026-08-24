//! The etcd read path, end to end, for a guardrail written by a control
//! plane one release ahead of this binary.
//!
//! The supported upgrade order puts the control plane in front of its data
//! planes, so for the length of that window a released — and therefore
//! unchangeable — data plane reads documents that carry fields it has never
//! heard of. An optional field is supposed to cost an entry in the
//! partial-compat report and nothing else. For anything nested inside a
//! guardrail's config it used to cost the whole row instead, which for a
//! content policy means it stops enforcing: PII flows through unmasked
//! until the data plane is upgraded. This pins the loader and the guardrail
//! build together, because either one alone can still drop the row.

use aisix_etcd::loader::{build_snapshot, PartialCompatEntry};
use aisix_etcd::provider::RawEntry;
use aisix_guardrails::{
    build_chain_from_snapshot, Guardrail, GuardrailEmbedderSlot, LocalModelRuntimeSlot,
};

/// A `kind: "pii"` guardrail whose custom pattern carries one field this
/// build does not know — the shape `custom_patterns[].replacement` had
/// against a 0.10.0 data plane.
const GUARDRAIL_FROM_A_NEWER_CP: &[u8] = br#"{
    "name": "redact-versions",
    "kind": "pii",
    "custom_patterns": [{
        "name": "eda_version",
        "regex": "version\\s*:\\s*(\\d+(?:\\.\\d+)+)",
        "action": "mask",
        "replacement": "***",
        "future_knob": "written by a newer control plane"
    }]
}"#;

#[test]
fn guardrail_with_an_unknown_nested_field_loads_and_still_masks() {
    let entries = vec![RawEntry {
        key: "/aisix/guardrails/g-1".to_string(),
        value: GUARDRAIL_FROM_A_NEWER_CP.to_vec(),
        revision: 1,
    }];
    let (snapshot, stats) = build_snapshot("/aisix", &entries);

    assert_eq!(stats.accepted, 1, "rejections: {:?}", stats.rejections);
    assert_eq!(snapshot.guardrails.len(), 1);
    assert_eq!(
        stats.partially_compatible,
        vec![PartialCompatEntry {
            kind: "guardrails".into(),
            field: "custom_patterns.[].future_knob".into(),
            count: 1,
        }],
        "the row loads, and the operator is told which field was ignored"
    );

    let chain = build_chain_from_snapshot(&snapshot.guardrails, None, &LocalModelRuntimeSlot::none(), &GuardrailEmbedderSlot::none());
    assert_eq!(chain.len(), 1, "the loaded row must reach the chain");
    let redacted = chain
        .redact_input_text("tool version: 12.1 done")
        .expect("the guardrail still masks");
    assert_eq!(redacted.text, "tool version: *** done");
}
