//! Context affinity for `/v1/responses`.
//!
//! A follow-up turn can carry context only the upstream that produced it can
//! read: a reasoning item's `encrypted_content` (stateless clients echo it
//! back) and `previous_response_id` (the upstream stored the response). Sent
//! to another deployment or account, the first is rejected with
//! `invalid_encrypted_content` and the second is not found.
//!
//! So the gateway tags what a model produced, in the values the client
//! echoes back unchanged:
//!
//! - `encrypted_content` → `aisix_enc_v1:<base64url("<target id>;<provider key id>")>;<original>`,
//!   the key id empty when the target has none — on every response, a direct
//!   model's included. Ciphertext is bound to the account that issued it.
//!   The gateway cannot see accounts, only provider keys, so it treats the
//!   origin model and the keys it can tie to that account as able to read
//!   it: the key that served it, and the origin model's current key. The
//!   origin model itself always reads it, even after its key changed — a
//!   key rotated to a new key on the same account must not cost an ongoing
//!   conversation its context, so a model moved to another account's key is
//!   left to the upstream's 400, which a group falls through only when its
//!   `fallback_on_statuses` lists 400. The client-addressed model plays no
//!   part: it can be a group one turn and a direct model the next.
//! - response `id` → `resp_` + base64url(`aisix:v1;<target id>;<original>`),
//!   for a group's target only: a direct model has nothing to pin.
//!
//! Gateways before the key was recorded read the first ciphertext segment as
//! a bare model id. Keeping the prefix and the segment boundary makes them
//! see an id no model has, so they strip the ciphertext rather than forward
//! it or a tag. Their own tags (a bare model id) still decode here, and are
//! judged by the origin model and its current key. The response id cannot degrade
//! that way — those gateways forward its original unconditionally and would
//! lose the pin — so it keeps recording the model only.
//!
//! On every request the tags are removed before anything else reads the
//! body, so a tagged value never reaches an upstream; every attempt whose
//! target — a group's or a direct model — cannot read an item's ciphertext
//! gets the item without it ([`ContextAffinity::body_for_target`]), and for
//! a group the decoded origin also pins the attempt order
//! ([`crate::routing`]).
//!
//! Who can read one origin's context is its [`Boundary`].

use base64::Engine as _;
use serde_json::Value;

use crate::json_splice::PathSeg;

const ENC_PREFIX: &str = "aisix_enc_v1:";
const RESP_ID_PREFIX: &str = "resp_";
const RESP_ID_TAG: &str = "aisix:v1;";

fn b64() -> base64::engine::GeneralPurpose {
    base64::engine::general_purpose::URL_SAFE_NO_PAD
}

pub(crate) fn mark_encrypted_content(
    target_id: &str,
    provider_key_id: Option<&str>,
    original: &str,
) -> String {
    let origin = format!("{target_id};{}", provider_key_id.unwrap_or_default());
    format!("{ENC_PREFIX}{};{original}", b64().encode(origin))
}

pub(crate) fn mark_response_id(target_id: &str, original: &str) -> String {
    format!(
        "{RESP_ID_PREFIX}{}",
        b64().encode(format!("{RESP_ID_TAG}{target_id};{original}"))
    )
}

/// The target that produced a piece of context, as its tag recorded it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Origin {
    pub(crate) model_id: String,
    /// The provider key that served it; `None` when the tag recorded only
    /// the model, or it was served without a key.
    pub(crate) key: Option<String>,
}

impl Origin {
    pub(crate) fn model(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            key: None,
        }
    }
}

enum EncryptedContent {
    /// Not a gateway tag: forwarded as the client sent it.
    Plain,
    Marked {
        origin: Origin,
        original: String,
    },
    /// Carries the tag prefix but does not decode.
    Undecodable,
}

fn decode_encrypted_content(value: &str) -> EncryptedContent {
    let Some(rest) = value.strip_prefix(ENC_PREFIX) else {
        return EncryptedContent::Plain;
    };
    let decoded = rest.split_once(';').and_then(|(origin, original)| {
        let origin = String::from_utf8(b64().decode(origin).ok()?).ok()?;
        let origin = match origin.split_once(';') {
            Some((model_id, key)) => Origin {
                model_id: model_id.to_string(),
                key: (!key.is_empty()).then(|| key.to_string()),
            },
            None => Origin::model(origin),
        };
        (!origin.model_id.is_empty()).then(|| (origin, original.to_string()))
    });
    match decoded {
        Some((origin, original)) => EncryptedContent::Marked { origin, original },
        None => EncryptedContent::Undecodable,
    }
}

/// `(origin target id, original id)` of a tagged response id; `None` for any
/// other id, which is forwarded unchanged.
fn decode_response_id(value: &str) -> Option<(String, String)> {
    let raw = b64().decode(value.strip_prefix(RESP_ID_PREFIX)?).ok()?;
    let text = String::from_utf8(raw).ok()?;
    let (origin, original) = text.strip_prefix(RESP_ID_TAG)?.split_once(';')?;
    (!origin.is_empty()).then(|| (origin.to_string(), original.to_string()))
}

/// The targets that can read context produced by one origin: the origin
/// model itself, whatever key it is on now; every target on the provider key
/// the tag recorded, whether or not the origin still exists; and every
/// target on the origin's current key, read from its own model document so
/// it resolves even when the origin is not a member of the group the
/// request addressed. A tag without a recorded key (older gateways, keyless
/// targets, and every response id) has only the first and the last.
#[derive(Debug, Clone)]
pub(crate) struct Boundary {
    origin_id: String,
    keys: Vec<String>,
}

impl Boundary {
    pub(crate) fn of(snapshot: &aisix_core::AisixSnapshot, origin: &Origin) -> Self {
        let current = snapshot
            .models
            .get_by_id(&origin.model_id)
            .and_then(|m| m.value.provider_key_id.clone());
        Self {
            origin_id: origin.model_id.clone(),
            keys: origin.key.iter().cloned().chain(current).collect(),
        }
    }

    pub(crate) fn is_origin(&self, target_id: &str) -> bool {
        self.origin_id == target_id
    }

    pub(crate) fn contains(&self, target_id: &str, target_provider_key_id: Option<&str>) -> bool {
        self.is_origin(target_id)
            || target_provider_key_id.is_some_and(|k| self.keys.iter().any(|key| key == k))
    }
}

/// What a request's tags said, recorded while removing them.
#[derive(Debug, Default)]
pub(crate) struct ContextAffinity {
    previous_origin: Option<Origin>,
    /// `(original, as the client sent it)` for a tagged `previous_response_id`,
    /// so the Response object can echo back the value the client sent.
    previous_echo: Option<(String, String)>,
    /// Index into `input[]` of each item whose ciphertext was tagged, with
    /// its origin. Items whose tag did not decode were already stripped.
    encrypted: Vec<(usize, Origin)>,
}

impl ContextAffinity {
    /// Restore every tagged value in a `/v1/responses` request body to the
    /// upstream's original bytes, and strip ciphertext whose tag does not
    /// decode (its origin is unknowable, so no upstream can be trusted to
    /// read it).
    pub(crate) fn unwrap_request(body: &mut Value) -> Self {
        let mut affinity = Self::default();
        if let Some(slot) = body.get_mut("previous_response_id") {
            if let Some((origin, original)) = slot.as_str().and_then(decode_response_id) {
                let sent = std::mem::replace(slot, Value::String(original.clone()));
                if let Value::String(sent) = sent {
                    affinity.previous_echo = Some((original, sent));
                }
                affinity.previous_origin = Some(Origin::model(origin));
            }
        }
        let Some(input) = body.get_mut("input").and_then(Value::as_array_mut) else {
            return affinity;
        };
        if !input.iter().any(|item| {
            item.get("encrypted_content")
                .and_then(Value::as_str)
                .is_some_and(|s| s.starts_with(ENC_PREFIX))
        }) {
            return affinity;
        }
        let items = std::mem::take(input);
        for mut item in items {
            let decoded = item
                .get("encrypted_content")
                .and_then(Value::as_str)
                .map(decode_encrypted_content);
            match decoded {
                Some(EncryptedContent::Marked { origin, original }) => {
                    item["encrypted_content"] = Value::String(original);
                    affinity.encrypted.push((input.len(), origin));
                }
                Some(EncryptedContent::Undecodable) => match strip_ciphertext(item) {
                    Some(kept) => item = kept,
                    None => continue,
                },
                Some(EncryptedContent::Plain) | None => {}
            }
            input.push(item);
        }
        affinity
    }

    /// The target that produced this conversation's context, if a tag named
    /// one: the stored `previous_response_id` first, else the latest tagged
    /// ciphertext in `input[]`.
    pub(crate) fn origin(&self) -> Option<&Origin> {
        self.previous_origin
            .as_ref()
            .or_else(|| self.encrypted.last().map(|(_, o)| o))
    }

    /// The body one attempt sends: `None` when every tagged
    /// ciphertext is readable by `target`, else a copy without the items
    /// it cannot read.
    pub(crate) fn body_for_target(
        &self,
        snapshot: &aisix_core::AisixSnapshot,
        body: &Value,
        target_id: &str,
        target_provider_key_id: Option<&str>,
    ) -> Option<Value> {
        let unreadable: Vec<usize> = self
            .encrypted
            .iter()
            .filter(|(_, origin)| {
                !Boundary::of(snapshot, origin).contains(target_id, target_provider_key_id)
            })
            .map(|(idx, _)| *idx)
            .collect();
        if unreadable.is_empty() {
            return None;
        }
        let mut body = body.clone();
        if let Some(input) = body.get_mut("input").and_then(Value::as_array_mut) {
            let items = std::mem::take(input);
            for (idx, item) in items.into_iter().enumerate() {
                if !unreadable.contains(&idx) {
                    input.push(item);
                } else if let Some(kept) = strip_ciphertext(item) {
                    input.push(kept);
                }
            }
        }
        Some(body)
    }

    /// How a response to this request is tagged: its ciphertext with the
    /// model that served it and that model's provider key, and — only when
    /// it is a routing group's target (`routed`) — its response id.
    pub(crate) fn response_mark(
        &self,
        target_id: &str,
        target_provider_key_id: Option<&str>,
        routed: bool,
    ) -> ResponseMark {
        ResponseMark {
            target_id: Some(target_id.to_string()),
            target_provider_key_id: target_provider_key_id.map(str::to_string),
            tags_response_id: routed,
            previous_echo: self.previous_echo.clone(),
        }
    }
}

/// An item without its ciphertext and the item id the ciphertext belongs to;
/// `None` when nothing readable is left to send.
fn strip_ciphertext(mut item: Value) -> Option<Value> {
    let obj = item.as_object_mut()?;
    obj.remove("encrypted_content");
    obj.remove("id");
    let readable = ["summary", "content"].iter().any(|key| {
        obj.get(*key)
            .and_then(Value::as_array)
            .is_some_and(|parts| {
                parts.iter().any(|p| {
                    p.get("text")
                        .and_then(Value::as_str)
                        .is_some_and(|t| !t.is_empty())
                })
            })
    });
    readable.then_some(item)
}

/// The tags one response carries back to the client.
#[derive(Debug, Clone, Default)]
pub(crate) struct ResponseMark {
    target_id: Option<String>,
    target_provider_key_id: Option<String>,
    tags_response_id: bool,
    previous_echo: Option<(String, String)>,
}

#[derive(Clone, Copy)]
enum Slot {
    ResponseId,
    PreviousResponseId,
    EncryptedContent,
}

impl ResponseMark {
    fn is_noop(&self) -> bool {
        self.target_id.is_none() && self.previous_echo.is_none()
    }

    /// The `previous_response_id` as the client sent it, when it was tagged:
    /// what a Response object the gateway builds itself echoes.
    pub(crate) fn previous_response_id_sent(&self) -> Option<&str> {
        self.previous_echo.as_ref().map(|(_, sent)| sent.as_str())
    }

    /// A response id the gateway minted itself (the bridged path).
    pub(crate) fn response_id(&self, id: String) -> String {
        match self.response_id_target() {
            Some(target) => mark_response_id(target, &id),
            None => id,
        }
    }

    fn response_id_target(&self) -> Option<&str> {
        self.target_id.as_deref().filter(|_| self.tags_response_id)
    }

    fn rewrite(&self, slot: Slot, value: &str) -> Option<String> {
        match slot {
            Slot::ResponseId => self
                .response_id_target()
                .filter(|_| !value.is_empty())
                .map(|t| mark_response_id(t, value)),
            Slot::EncryptedContent => self
                .target_id
                .as_deref()
                .filter(|_| !value.is_empty())
                .map(|t| mark_encrypted_content(t, self.target_provider_key_id.as_deref(), value)),
            Slot::PreviousResponseId => self
                .previous_echo
                .as_ref()
                .filter(|(original, _)| original == value)
                .map(|(_, sent)| sent.clone()),
        }
    }

    /// Tag a parsed (non-streaming) Response object.
    pub(crate) fn apply_to_body(&self, body: &mut Value) {
        if self.is_noop() {
            return;
        }
        for (key, slot) in [
            ("id", Slot::ResponseId),
            ("previous_response_id", Slot::PreviousResponseId),
        ] {
            if let Some(v) = body.get_mut(key) {
                if let Some(new) = v.as_str().and_then(|s| self.rewrite(slot, s)) {
                    *v = Value::String(new);
                }
            }
        }
        if let Some(items) = body.get_mut("output").and_then(Value::as_array_mut) {
            for item in items {
                if let Some(v) = item.get_mut("encrypted_content") {
                    if let Some(new) = v
                        .as_str()
                        .and_then(|s| self.rewrite(Slot::EncryptedContent, s))
                    {
                        *v = Value::String(new);
                    }
                }
            }
        }
    }

    /// Tag one Responses SSE event payload, byte-preserving: the Response
    /// object of the snapshot events (`response.created` / `.in_progress` /
    /// `.queued` / `.completed` / `.incomplete` / `.failed`) and the item of
    /// `response.output_item.added` / `.done`.
    fn apply_to_event_json(&self, payload: &[u8]) -> Option<Vec<u8>> {
        let slot = std::cell::Cell::new(Slot::ResponseId);
        let select = |path: &[PathSeg]| {
            let picked = match path {
                [PathSeg::Key(r), PathSeg::Key(k)] if r == "response" => match k.as_str() {
                    "id" => Some(Slot::ResponseId),
                    "previous_response_id" => Some(Slot::PreviousResponseId),
                    _ => None,
                },
                [PathSeg::Key(r), PathSeg::Key(o), PathSeg::Index(_), PathSeg::Key(e)]
                    if r == "response" && o == "output" && e == "encrypted_content" =>
                {
                    Some(Slot::EncryptedContent)
                }
                [PathSeg::Key(i), PathSeg::Key(e)] if i == "item" && e == "encrypted_content" => {
                    Some(Slot::EncryptedContent)
                }
                _ => None,
            };
            if let Some(s) = picked {
                slot.set(s);
            }
            picked.is_some()
        };
        match crate::json_splice::rewrite_string_values(payload, select, |v| {
            self.rewrite(slot.get(), v)
        }) {
            Ok(rewritten) => rewritten,
            Err(error) => {
                tracing::debug!(%error, "responses context tag skipped; forwarding verbatim");
                None
            }
        }
    }

    /// Tag one SSE frame; `None` forwards it unchanged.
    pub(crate) fn apply_to_sse_frame(&self, frame: &[u8]) -> Option<Vec<u8>> {
        if self.is_noop() {
            return None;
        }
        crate::model_echo::splice_sse_frame(frame, |payload| self.apply_to_event_json(payload))
    }

    /// Tag every frame of a complete, buffered SSE document.
    pub(crate) fn apply_to_sse_buffer(&self, buf: Vec<u8>) -> Vec<u8> {
        if self.is_noop() {
            return buf;
        }
        crate::model_echo::splice_sse_buffer(&buf, |payload| self.apply_to_event_json(payload))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tags_round_trip_and_foreign_values_are_plain() {
        let enc = mark_encrypted_content("m-1", Some("pk-1"), "gAAAA;x");
        assert!(enc.starts_with("aisix_enc_v1:"));
        assert!(matches!(
            decode_encrypted_content(&enc),
            EncryptedContent::Marked { origin, original }
                if origin == Origin { model_id: "m-1".into(), key: Some("pk-1".into()) }
                    && original == "gAAAA;x"
        ));
        assert!(matches!(
            decode_encrypted_content(&mark_encrypted_content("m-1", None, "CT")),
            EncryptedContent::Marked { origin, .. }
                if origin == Origin { model_id: "m-1".into(), key: None }
        ));
        // A tag written before the key was recorded: the bare model id.
        let v1 = format!("aisix_enc_v1:{};CT", b64().encode("m-1"));
        assert!(matches!(
            decode_encrypted_content(&v1),
            EncryptedContent::Marked { origin, .. } if origin == Origin::model("m-1")
        ));
        assert!(matches!(
            decode_encrypted_content("gAAAA"),
            EncryptedContent::Plain
        ));
        assert!(matches!(
            decode_encrypted_content("aisix_enc_v1:!!;x"),
            EncryptedContent::Undecodable
        ));

        let id = mark_response_id("m-1", "resp_abc");
        assert!(id.starts_with("resp_"));
        assert_eq!(
            decode_response_id(&id),
            Some(("m-1".into(), "resp_abc".into()))
        );
        assert_eq!(decode_response_id("resp_abc"), None);
        assert_eq!(decode_response_id("resp_68af0c1e2b"), None);
    }

    /// Gateways from before the key was recorded (1.7.0 through 1.10.0)
    /// decode the first segment as a bare model id, verbatim below. A new tag
    /// must still decode there, and to an id no model has, so a routing group
    /// strips the ciphertext instead of forwarding the whole tag upstream.
    #[test]
    fn a_new_tag_decodes_to_an_unknown_model_on_older_gateways() {
        fn released_decode(value: &str) -> Option<(String, String)> {
            let rest = value.strip_prefix("aisix_enc_v1:")?;
            rest.split_once(';').and_then(|(origin, original)| {
                let origin = String::from_utf8(b64().decode(origin).ok()?).ok()?;
                (!origin.is_empty()).then(|| (origin, original.to_string()))
            })
        }
        for key in [Some("pk-1"), None] {
            let (origin, original) =
                released_decode(&mark_encrypted_content("m-1", key, "gAAAA;x")).unwrap();
            assert_ne!(origin, "m-1");
            assert!(origin.starts_with("m-1;"));
            assert_eq!(original, "gAAAA;x");
        }
    }

    fn origin_on(id: &str, provider_key_id: &str) -> aisix_core::AisixSnapshot {
        let model: aisix_core::models::Model = serde_json::from_value(json!({
            "display_name": id,
            "provider": "openai",
            "model_name": "up",
            "provider_key_id": provider_key_id,
        }))
        .unwrap();
        let table = aisix_core::snapshot::ResourceTable::default();
        table.insert(aisix_core::ResourceEntry::new(id.to_string(), model, 1));
        aisix_core::AisixSnapshot {
            models: table,
            ..Default::default()
        }
    }

    #[test]
    fn a_boundary_is_the_origin_its_recorded_key_and_its_current_key() {
        let recorded = Origin {
            model_id: "m-1".into(),
            key: Some("pk-1".into()),
        };
        // The origin moved to pk-2 after it served the context.
        let moved = origin_on("m-1", "pk-2");
        let b = Boundary::of(&moved, &recorded);
        assert!(b.contains("m-1", Some("pk-2")));
        assert!(b.contains("m-2", Some("pk-1")));
        assert!(b.contains("m-3", Some("pk-2")));
        assert!(!b.contains("m-4", Some("pk-3")));
        assert!(!b.contains("m-4", None));
        // The origin is gone: its recorded key still reads.
        let gone = Boundary::of(&aisix_core::AisixSnapshot::default(), &recorded);
        assert!(gone.contains("m-2", Some("pk-1")));
        assert!(!gone.contains("m-3", Some("pk-2")));
        // No key recorded (older tags, keyless targets): the origin and its
        // current key.
        let bare = Boundary::of(&moved, &Origin::model("m-1"));
        assert!(bare.contains("m-1", Some("pk-3")));
        assert!(bare.contains("m-3", Some("pk-2")));
        assert!(!bare.contains("m-2", Some("pk-1")));
        assert!(!bare.contains("m-2", None));
    }

    #[test]
    fn unwrap_restores_tags_and_strips_undecodable_ciphertext() {
        let mut body = json!({
            "previous_response_id": mark_response_id("m-1", "resp_up"),
            "input": [
                {"type": "reasoning", "id": "rs_1", "summary": [],
                 "encrypted_content": "aisix_enc_v1:%%;junk"},
                {"type": "message", "role": "user", "content": "hi"},
                {"type": "reasoning", "id": "rs_2", "summary": [],
                 "encrypted_content": mark_encrypted_content("m-2", Some("pk-2"), "CT")},
            ]
        });
        let a = ContextAffinity::unwrap_request(&mut body);
        assert_eq!(body["previous_response_id"], "resp_up");
        assert_eq!(body["input"].as_array().unwrap().len(), 2);
        assert_eq!(body["input"][1]["encrypted_content"], "CT");
        assert_eq!(a.origin(), Some(&Origin::model("m-1")));
        assert_eq!(
            a.encrypted,
            vec![(
                1,
                Origin {
                    model_id: "m-2".into(),
                    key: Some("pk-2".into())
                }
            )]
        );
    }

    #[test]
    fn strip_keeps_readable_reasoning_without_its_ciphertext() {
        let kept = strip_ciphertext(json!({
            "type": "reasoning", "id": "rs_1", "encrypted_content": "CT",
            "summary": [{"type": "summary_text", "text": "why"}]
        }))
        .unwrap();
        assert_eq!(
            kept,
            json!({"type": "reasoning", "summary": [{"type": "summary_text", "text": "why"}]})
        );
        assert!(strip_ciphertext(
            json!({"type": "reasoning", "summary": [], "encrypted_content": "CT"})
        )
        .is_none());
    }

    #[test]
    fn sse_frame_tags_snapshot_and_item_events() {
        let mark = ResponseMark {
            target_id: Some("m-1".into()),
            target_provider_key_id: Some("pk-1".into()),
            tags_response_id: true,
            previous_echo: Some(("resp_prev".into(), "resp_SENT".into())),
        };
        let frame = b"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_up\",\"previous_response_id\":\"resp_prev\",\"output\":[{\"type\":\"reasoning\",\"encrypted_content\":\"CT\"}]}}\n\n";
        let out = mark.apply_to_sse_frame(frame).unwrap();
        let payload: Value = serde_json::from_slice(
            crate::redact::frame_payload(&out)
                .unwrap()
                .trim()
                .as_bytes(),
        )
        .unwrap();
        assert_eq!(
            payload["response"]["id"],
            mark_response_id("m-1", "resp_up")
        );
        assert_eq!(payload["response"]["previous_response_id"], "resp_SENT");
        assert_eq!(
            payload["response"]["output"][0]["encrypted_content"],
            mark_encrypted_content("m-1", Some("pk-1"), "CT")
        );
        let item = b"data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"encrypted_content\":\"CT\"}}\n\n";
        let out = mark.apply_to_sse_frame(item).unwrap();
        assert!(String::from_utf8(out).unwrap().contains("aisix_enc_v1:"));
        let delta = b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"x\"}\n\n";
        assert!(mark.apply_to_sse_frame(delta).is_none());
        let direct = ContextAffinity::default().response_mark("m-1", Some("pk-1"), false);
        assert!(direct.apply_to_sse_frame(delta).is_none());
        let out = String::from_utf8(direct.apply_to_sse_frame(frame).unwrap()).unwrap();
        assert!(out.contains("\"id\":\"resp_up\""));
        assert!(!out.contains("\"encrypted_content\":\"CT\""));
    }
}
