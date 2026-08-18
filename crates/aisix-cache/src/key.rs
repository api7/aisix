//! Canonical cache-key fingerprint.
//!
//! The key is a stable hash of the *request fingerprint* — the fields
//! that materially affect the upstream response. Anything else (request
//! id, deadlines, the caller's ApiKey, custom headers) is excluded so
//! two callers asking the same question hit the same entry.
//!
//! Hash function is `std::hash::DefaultHasher` (SipHash-1-3, u64). For an
//! in-memory exact-match cache that's fine: collisions over the bounded
//! request space are exceedingly rare, and a stronger hash would be a
//! single-line drop-in if we ever need it.

use aisix_gateway::{ChatFormat, ChatMessage, Role};
use std::hash::{Hash, Hasher};

/// Stable fingerprint of a chat request — the inputs to the upstream call.
/// We hash this struct (not the whole `ChatFormat`) so caching policy is
/// explicit about what counts as "the same request".
///
/// `extras` carries the OpenAI-shape fields that arrive through
/// `ChatFormat::extra` (`tools`, `tool_choice`, `response_format`, `seed`,
/// `stop`, `presence_penalty`, `frequency_penalty`, …). They materially
/// change the upstream response — a tool-calling request and a non-tool
/// request with the same prompt **must not** share a cache entry — so they
/// must be part of the fingerprint. We hash a sorted snapshot of the map
/// so the result is independent of JSON insertion order.
#[derive(Debug, Clone)]
pub struct CacheKey {
    pub model: String,
    pub messages: Vec<(String, String)>, // (role, content)
    pub temperature_milli: Option<u32>,  // f32 isn't Hash; quantise to milli
    pub top_p_milli: Option<u32>,
    pub max_tokens: Option<u32>,
    /// Sorted (key, canonical-json-value) pairs from `ChatFormat::extra`.
    /// We pre-sort + pre-stringify here so `Hash` stays trivially
    /// deterministic and so two requests that differ only in JSON key
    /// order collapse to the same fingerprint.
    pub extras: Vec<(String, String)>,
    /// The matched `CachePolicy`'s resource id. Scopes entries per
    /// policy so two policies never share entries even on the same
    /// backend instance.
    pub policy_id: String,
    /// The policy's `purge_generation` at request time. A purge bumps
    /// the generation, so every earlier entry's key becomes
    /// unreachable at once (entries are then reclaimed by TTL).
    pub purge_generation: u32,
    /// The caller's api_key id when the policy's `scope` is `api_key`;
    /// `None` under `scope: env`. Part of BOTH fingerprints, so scoped
    /// entries are invisible across callers.
    pub scope_api_key: Option<String>,
}

impl CacheKey {
    /// Build a key from the proxy's normalised `ChatFormat`. Streaming
    /// requests are *not* cached at this layer — callers should skip the
    /// cache when `req.is_streaming()`.
    ///
    /// The scope fields (`policy_id`, `purge_generation`,
    /// `scope_api_key`) start empty; the proxy fills them via
    /// [`CacheKey::with_scope`] once it knows the matched policy.
    pub fn from_request(req: &ChatFormat) -> Self {
        Self {
            model: req.model.clone(),
            messages: req.messages.iter().map(message_pair).collect(),
            temperature_milli: req.temperature.map(quantise_milli),
            top_p_milli: req.top_p.map(quantise_milli),
            max_tokens: req.max_tokens,
            extras: canonical_extras(&req.extra),
            policy_id: String::new(),
            purge_generation: 0,
            scope_api_key: None,
        }
    }

    /// Attach the matched policy's scope: its resource id, its current
    /// purge generation, and — when the policy isolates per caller —
    /// the caller's api_key id.
    pub fn with_scope(
        mut self,
        policy_id: &str,
        purge_generation: u32,
        scope_api_key: Option<&str>,
    ) -> Self {
        self.policy_id = policy_id.to_string();
        self.purge_generation = purge_generation;
        self.scope_api_key = scope_api_key.map(str::to_string);
        self
    }

    /// Hex-encoded u64 hash, used as the cache backend's lookup key.
    pub fn fingerprint(&self) -> String {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut h);
        format!("{:016x}", h.finish())
    }

    /// Fingerprint of everything EXCEPT the messages — the semantic
    /// layer's candidate filter. Two requests share a scope fingerprint
    /// iff they agree on model, sampling params, extras, policy,
    /// generation, and caller scope; only then may they match by
    /// embedding similarity. This is what keeps a tool-calling request
    /// from ever being answered by a similar-but-toolless entry.
    pub fn scope_fingerprint(&self) -> String {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.hash_scope(&mut h);
        format!("{:016x}", h.finish())
    }

    /// Hash the non-message fields. Shared by [`Hash`] and
    /// [`CacheKey::scope_fingerprint`] so the two can never disagree on
    /// which fields scope an entry.
    fn hash_scope<H: Hasher>(&self, state: &mut H) {
        self.model.hash(state);
        self.temperature_milli.hash(state);
        self.top_p_milli.hash(state);
        self.max_tokens.hash(state);
        for (k, v) in &self.extras {
            k.hash(state);
            v.hash(state);
        }
        self.policy_id.hash(state);
        self.purge_generation.hash(state);
        self.scope_api_key.hash(state);
    }
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_scope(state);
        for (role, content) in &self.messages {
            role.hash(state);
            content.hash(state);
        }
    }
}

/// Canonical text the semantic layer embeds for a request: one
/// `role: content` line per message, preserving roles and message
/// boundaries so `["ab"]` and `["a","b"]` embed differently. Backslash
/// and newline inside message text are escaped (`\\`, `\n`), so content
/// that *contains* a `\nrole: ` sequence cannot forge a message
/// boundary and collide with a genuinely multi-message conversation.
///
/// Returns `None` — semantic matching is skipped, exact matching still
/// applies — when a text embedding cannot faithfully represent the
/// request:
/// - any non-`text` content block (image, audio, …): similar prompts
///   about *different* images must never match;
/// - any message carrying `name`, `tool_call_id`, or forward-compat
///   `extra` fields (`tool_calls`, …): tool traffic differing only in
///   those fields would embed identically and replay the wrong answer;
/// - no non-whitespace text at all (embedding an empty string could
///   match almost anything).
pub fn semantic_prompt_text(req: &ChatFormat) -> Option<String> {
    let mut out = String::new();
    let mut has_text = false;
    for m in &req.messages {
        if m.name.is_some() || m.tool_call_id.is_some() || !m.extra.is_empty() {
            return None;
        }
        let text = match m.content_blocks.as_ref() {
            Some(blocks) => {
                let mut t = String::new();
                for b in blocks {
                    let obj = b.as_object()?;
                    match obj.get("type").and_then(|v| v.as_str()) {
                        Some("text") => {
                            if let Some(s) = obj.get("text").and_then(|v| v.as_str()) {
                                t.push_str(s);
                            }
                        }
                        _ => return None,
                    }
                }
                t
            }
            None => m.content_str().to_string(),
        };
        if !text.trim().is_empty() {
            has_text = true;
        }
        out.push_str(role_str(m.role));
        out.push_str(": ");
        out.push_str(&text.replace('\\', "\\\\").replace('\n', "\\n"));
        out.push('\n');
    }
    has_text.then_some(out)
}

/// Sort `extra` by key (recursively, into nested objects too) and emit
/// a stable canonical-JSON string per value. The recursion matters: two
/// callers can send byte-different JSON for `tools=[{...}]` if they
/// serialise the inner `parameters` object's keys in different order;
/// `serde_json::to_string` preserves whatever insertion order the parser
/// saw, so without recursive sorting two semantically-equal requests
/// would land in different cache slots.
fn canonical_extras(extra: &serde_json::Map<String, serde_json::Value>) -> Vec<(String, String)> {
    let mut pairs: Vec<(String, String)> = extra
        .iter()
        .map(|(k, v)| (k.clone(), canonical_json_string(v)))
        .collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    pairs
}

fn canonical_json_string(value: &serde_json::Value) -> String {
    canonicalise(value).to_string()
}

/// Return a clone of `value` with every nested object's keys reordered
/// alphabetically. `serde_json::Map` preserves insertion order on
/// serialise, so reordering here is what makes the eventual string form
/// deterministic.
fn canonicalise(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let mut out = serde_json::Map::with_capacity(entries.len());
            for (k, v) in entries {
                out.insert(k.clone(), canonicalise(v));
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(items) => {
            // Arrays are positional — `tools=[a, b]` ≠ `tools=[b, a]` for
            // models that respect declaration order. Preserve order;
            // canonicalise children only.
            serde_json::Value::Array(items.iter().map(canonicalise).collect())
        }
        other => other.clone(),
    }
}

fn message_pair(m: &ChatMessage) -> (String, String) {
    // For text-only messages, fingerprint on the role + content string.
    // For vision/multimodal messages (typed-block array form), the
    // raw `content_blocks` value is what distinguishes the request —
    // two messages with the same query text but different image URLs
    // MUST produce distinct fingerprints. Canonicalise the blocks
    // (sorted keys at every nesting level) so JSON-key-order
    // differences don't cause spurious cache misses.
    let content_repr = match m.content_blocks.as_ref() {
        Some(blocks) => canonical_json_string(&serde_json::Value::Array(blocks.clone())),
        None => m.content_str().to_string(),
    };
    // Message-level identity beyond the content changes what the
    // upstream sees just as much as the content does: `name`,
    // `tool_call_id`, and the forward-compat `extra` bag (`tool_calls`,
    // `refusal`, …). Two histories that differ only in an assistant's
    // `tool_calls` MUST NOT share a fingerprint.
    //
    // Each representation is type-prefixed (`t:` plain text, `b:`
    // canonical content blocks, `j:` decorated) so a plain message
    // whose CONTENT happens to equal another form's serialization can
    // never collide with it.
    if m.name.is_none() && m.tool_call_id.is_none() && m.extra.is_empty() {
        let prefix = if m.content_blocks.is_some() {
            "b:"
        } else {
            "t:"
        };
        return (
            role_str(m.role).to_string(),
            format!("{prefix}{content_repr}"),
        );
    }
    let decorated = serde_json::json!({
        "content": content_repr,
        "name": m.name,
        "tool_call_id": m.tool_call_id,
        "extra": canonicalise(&serde_json::Value::Object(m.extra.clone())),
    });
    (role_str(m.role).to_string(), format!("j:{decorated}"))
}

fn role_str(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// Convert an f32 in [0.0, 1.0]-ish range to a u32 in milli units.
/// Saturates negatives at 0 and >65 at u32::MAX-ish; collisions on weird
/// values are fine — the cache just doesn't help that request.
fn quantise_milli(v: f32) -> u32 {
    if v.is_nan() || v.is_sign_negative() {
        return 0;
    }
    let scaled = v * 1_000.0;
    if scaled > u32::MAX as f32 {
        u32::MAX
    } else {
        scaled as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(model: &str, messages: Vec<ChatMessage>, temp: Option<f32>) -> ChatFormat {
        let mut f = ChatFormat::new(model, messages);
        f.temperature = temp;
        f
    }

    #[test]
    fn identical_requests_share_a_fingerprint() {
        let a = req("m", vec![ChatMessage::user("hi")], Some(0.2));
        let b = req("m", vec![ChatMessage::user("hi")], Some(0.2));
        assert_eq!(
            CacheKey::from_request(&a).fingerprint(),
            CacheKey::from_request(&b).fingerprint(),
        );
    }

    #[test]
    fn changing_message_content_changes_the_fingerprint() {
        let a = req("m", vec![ChatMessage::user("hi")], None);
        let b = req("m", vec![ChatMessage::user("yo")], None);
        assert_ne!(
            CacheKey::from_request(&a).fingerprint(),
            CacheKey::from_request(&b).fingerprint(),
        );
    }

    #[test]
    fn vision_messages_with_different_image_urls_have_distinct_fingerprints() {
        // Regression test for the cache-key collision found in PR #184
        // audit (C1): with the typed-block array form of `content`,
        // `m.content` only carries the concatenated TEXT (e.g.
        // "What's in this image?"); the image URL lives in
        // `content_blocks`. Two requests asking the same question
        // about different images would produce the same `(role,
        // content)` pair if `message_pair` didn't include the blocks.
        // The cache would then return the cat-photo response when a
        // user asks about a dog photo. message_pair must canonicalise
        // and include the raw blocks.
        let mk = |url: &str| {
            let mut msg = ChatMessage::user("What's in this image?");
            msg.content_blocks = Some(vec![
                serde_json::json!({"type": "text", "text": "What's in this image?"}),
                serde_json::json!({"type": "image_url", "image_url": {"url": url}}),
            ]);
            req("m", vec![msg], None)
        };
        let a = mk("https://example.com/cat.jpg");
        let b = mk("https://example.com/dog.jpg");
        assert_ne!(
            CacheKey::from_request(&a).fingerprint(),
            CacheKey::from_request(&b).fingerprint(),
            "vision requests with different images must NOT share a cache slot",
        );
    }

    #[test]
    fn vision_messages_with_identical_blocks_share_a_fingerprint() {
        // Sibling to the test above: same image, same question, same
        // model — must hit the same cache slot. Sanity-check that the
        // canonicalisation isn't introducing spurious cache misses.
        let mk = || {
            let mut msg = ChatMessage::user("describe");
            msg.content_blocks = Some(vec![
                serde_json::json!({"type": "text", "text": "describe"}),
                serde_json::json!({"type": "image_url", "image_url": {"url": "https://example.com/x.jpg"}}),
            ]);
            req("m", vec![msg], None)
        };
        assert_eq!(
            CacheKey::from_request(&mk()).fingerprint(),
            CacheKey::from_request(&mk()).fingerprint(),
        );
    }

    #[test]
    fn changing_temperature_changes_the_fingerprint() {
        let a = req("m", vec![ChatMessage::user("hi")], Some(0.2));
        let b = req("m", vec![ChatMessage::user("hi")], Some(0.7));
        assert_ne!(
            CacheKey::from_request(&a).fingerprint(),
            CacheKey::from_request(&b).fingerprint(),
        );
    }

    #[test]
    fn near_identical_temperatures_within_milli_collapse_to_same_fingerprint() {
        // 0.2000001 quantises to 200 just like 0.2; intentional — float
        // noise from JSON parsing shouldn't shatter the cache.
        let a = req("m", vec![ChatMessage::user("hi")], Some(0.2));
        let b = req("m", vec![ChatMessage::user("hi")], Some(0.200_000_1));
        assert_eq!(
            CacheKey::from_request(&a).fingerprint(),
            CacheKey::from_request(&b).fingerprint(),
        );
    }

    /// Pre-#87 this test asserted that `extra` was excluded from the
    /// fingerprint — that was the bug. Post-fix, `extra` is part of the
    /// fingerprint, so the still-valid invariant is "no `extra` and an
    /// empty `extra` produce the same hash" (i.e. the empty-extra
    /// canonical form is stable).
    #[test]
    fn empty_extras_match_no_extras() {
        let a = req("m", vec![ChatMessage::user("hi")], None);
        let mut b = req("m", vec![ChatMessage::user("hi")], None);
        // Touch `extra` (no-op insert + remove leaves it empty but
        // exercises the map machinery).
        b.extra.insert("k".into(), serde_json::json!(1));
        b.extra.remove("k");
        assert_eq!(
            CacheKey::from_request(&a).fingerprint(),
            CacheKey::from_request(&b).fingerprint(),
        );
    }

    #[test]
    fn fingerprint_is_16_hex_chars() {
        let f = req("m", vec![ChatMessage::user("hi")], None);
        let fp = CacheKey::from_request(&f).fingerprint();
        assert_eq!(fp.len(), 16);
        assert!(fp.chars().all(|c| c.is_ascii_hexdigit()));
    }

    /// Tools / response_format / seed all arrive through `ChatFormat::extra`.
    /// Two requests that differ only on one of those fields **must not**
    /// share a cache entry — see issue #87 (silent correctness bug:
    /// tool-calling requests cross-pollinating with non-tool requests).
    #[test]
    fn changing_tools_changes_the_fingerprint() {
        let mut a = req("m", vec![ChatMessage::user("hi")], None);
        let mut b = req("m", vec![ChatMessage::user("hi")], None);
        a.extra.insert(
            "tools".into(),
            serde_json::json!([{"type": "function", "function": {"name": "get_weather"}}]),
        );
        // b has no tools at all — distinct fingerprint required.
        b.extra.remove("tools");
        assert_ne!(
            CacheKey::from_request(&a).fingerprint(),
            CacheKey::from_request(&b).fingerprint(),
        );
    }

    #[test]
    fn changing_response_format_changes_the_fingerprint() {
        let mut a = req("m", vec![ChatMessage::user("hi")], None);
        let mut b = req("m", vec![ChatMessage::user("hi")], None);
        a.extra.insert(
            "response_format".into(),
            serde_json::json!({"type": "json_object"}),
        );
        b.extra.insert(
            "response_format".into(),
            serde_json::json!({"type": "text"}),
        );
        assert_ne!(
            CacheKey::from_request(&a).fingerprint(),
            CacheKey::from_request(&b).fingerprint(),
        );
    }

    #[test]
    fn changing_seed_changes_the_fingerprint() {
        let mut a = req("m", vec![ChatMessage::user("hi")], None);
        let mut b = req("m", vec![ChatMessage::user("hi")], None);
        a.extra.insert("seed".into(), serde_json::json!(42));
        b.extra.insert("seed".into(), serde_json::json!(43));
        assert_ne!(
            CacheKey::from_request(&a).fingerprint(),
            CacheKey::from_request(&b).fingerprint(),
        );
    }

    /// JSON insertion order must not affect the fingerprint, otherwise
    /// callers using different SDKs get different cache slots for
    /// equivalent requests. The canonicaliser sorts top-level + nested
    /// object keys.
    #[test]
    fn extras_with_same_keys_in_different_order_share_a_fingerprint() {
        let mut a = req("m", vec![ChatMessage::user("hi")], None);
        let mut b = req("m", vec![ChatMessage::user("hi")], None);
        // Top-level: insertion order seed-then-tools vs tools-then-seed.
        a.extra.insert("seed".into(), serde_json::json!(7));
        a.extra.insert(
            "tools".into(),
            serde_json::json!([{"type": "function", "function": {"name": "f", "parameters": {"a": 1, "b": 2}}}]),
        );
        b.extra.insert(
            "tools".into(),
            // Nested-object keys also reversed (`parameters` keys b before a).
            serde_json::json!([{"function": {"parameters": {"b": 2, "a": 1}, "name": "f"}, "type": "function"}]),
        );
        b.extra.insert("seed".into(), serde_json::json!(7));
        assert_eq!(
            CacheKey::from_request(&a).fingerprint(),
            CacheKey::from_request(&b).fingerprint(),
        );
    }

    /// `tools=[a, b]` and `tools=[b, a]` are different declarations to
    /// the model — preserve array order even while sorting object keys.
    #[test]
    fn tool_array_order_changes_the_fingerprint() {
        let mut a = req("m", vec![ChatMessage::user("hi")], None);
        let mut b = req("m", vec![ChatMessage::user("hi")], None);
        a.extra.insert(
            "tools".into(),
            serde_json::json!([{"name": "x"}, {"name": "y"}]),
        );
        b.extra.insert(
            "tools".into(),
            serde_json::json!([{"name": "y"}, {"name": "x"}]),
        );
        assert_ne!(
            CacheKey::from_request(&a).fingerprint(),
            CacheKey::from_request(&b).fingerprint(),
        );
    }

    #[test]
    fn quantise_handles_pathological_floats() {
        assert_eq!(quantise_milli(f32::NAN), 0);
        assert_eq!(quantise_milli(-1.0), 0);
        assert_eq!(quantise_milli(0.0), 0);
        assert_eq!(quantise_milli(0.5), 500);
        assert_eq!(quantise_milli(f32::INFINITY), u32::MAX);
    }

    #[test]
    fn different_policies_never_share_a_fingerprint() {
        let r = req("m", vec![ChatMessage::user("hi")], None);
        let a = CacheKey::from_request(&r).with_scope("policy-a", 0, None);
        let b = CacheKey::from_request(&r).with_scope("policy-b", 0, None);
        assert_ne!(a.fingerprint(), b.fingerprint());
        assert_ne!(a.scope_fingerprint(), b.scope_fingerprint());
    }

    #[test]
    fn purge_generation_bump_invalidates_both_fingerprints() {
        let r = req("m", vec![ChatMessage::user("hi")], None);
        let a = CacheKey::from_request(&r).with_scope("p", 0, None);
        let b = CacheKey::from_request(&r).with_scope("p", 1, None);
        assert_ne!(a.fingerprint(), b.fingerprint());
        assert_ne!(a.scope_fingerprint(), b.scope_fingerprint());
    }

    #[test]
    fn api_key_scope_isolates_callers_env_scope_shares() {
        let r = req("m", vec![ChatMessage::user("hi")], None);
        let a = CacheKey::from_request(&r).with_scope("p", 0, Some("key-a"));
        let b = CacheKey::from_request(&r).with_scope("p", 0, Some("key-b"));
        let shared_a = CacheKey::from_request(&r).with_scope("p", 0, None);
        let shared_b = CacheKey::from_request(&r).with_scope("p", 0, None);
        assert_ne!(a.fingerprint(), b.fingerprint());
        assert_ne!(a.scope_fingerprint(), b.scope_fingerprint());
        assert_eq!(shared_a.fingerprint(), shared_b.fingerprint());
    }

    #[test]
    fn scope_fingerprint_ignores_messages_but_tracks_params() {
        let a = req("m", vec![ChatMessage::user("hello there")], Some(0.2));
        let b = req("m", vec![ChatMessage::user("совершенно другой")], Some(0.2));
        let c = req("m", vec![ChatMessage::user("hello there")], Some(0.7));
        let (ka, kb, kc) = (
            CacheKey::from_request(&a).with_scope("p", 0, None),
            CacheKey::from_request(&b).with_scope("p", 0, None),
            CacheKey::from_request(&c).with_scope("p", 0, None),
        );
        assert_eq!(ka.scope_fingerprint(), kb.scope_fingerprint());
        assert_ne!(ka.fingerprint(), kb.fingerprint());
        assert_ne!(ka.scope_fingerprint(), kc.scope_fingerprint());
    }

    #[test]
    fn scope_fingerprint_tracks_extras() {
        let mut a = req("m", vec![ChatMessage::user("hi")], None);
        let b = req("m", vec![ChatMessage::user("hi")], None);
        a.extra.insert(
            "tools".into(),
            serde_json::json!([{"type": "function", "function": {"name": "f"}}]),
        );
        assert_ne!(
            CacheKey::from_request(&a).scope_fingerprint(),
            CacheKey::from_request(&b).scope_fingerprint(),
        );
    }

    #[test]
    fn semantic_text_preserves_roles_and_message_boundaries() {
        let a = req(
            "m",
            vec![ChatMessage::system("be brief"), ChatMessage::user("hi")],
            None,
        );
        let text = semantic_prompt_text(&a).unwrap();
        assert_eq!(text, "system: be brief\nuser: hi\n");
        // One message "ab" vs two messages "a"/"b" must differ.
        let one = req("m", vec![ChatMessage::user("ab")], None);
        let two = req(
            "m",
            vec![ChatMessage::user("a"), ChatMessage::user("b")],
            None,
        );
        assert_ne!(
            semantic_prompt_text(&one).unwrap(),
            semantic_prompt_text(&two).unwrap(),
        );
    }

    #[test]
    fn semantic_text_none_for_non_text_blocks() {
        let mut msg = ChatMessage::user("what's in this image?");
        msg.content_blocks = Some(vec![
            serde_json::json!({"type": "text", "text": "what's in this image?"}),
            serde_json::json!({"type": "image_url", "image_url": {"url": "https://x/cat.jpg"}}),
        ]);
        let r = req("m", vec![msg], None);
        assert_eq!(semantic_prompt_text(&r), None);
    }

    #[test]
    fn semantic_text_accepts_text_only_blocks() {
        let mut msg = ChatMessage::user("hello");
        msg.content_blocks = Some(vec![
            serde_json::json!({"type": "text", "text": "hel"}),
            serde_json::json!({"type": "text", "text": "lo"}),
        ]);
        let r = req("m", vec![msg], None);
        assert_eq!(semantic_prompt_text(&r).unwrap(), "user: hello\n");
    }

    #[test]
    fn semantic_text_none_for_empty_content() {
        let r = req("m", vec![ChatMessage::user("   ")], None);
        assert_eq!(semantic_prompt_text(&r), None);
    }

    #[test]
    fn histories_differing_only_in_tool_calls_never_share_a_fingerprint() {
        // Assistant `tool_calls` ride the message-level `extra` bag; the
        // upstream sees them, so the fingerprint must too. Pre-fix, two
        // agent turns whose only difference was the tool call shared an
        // exact key and replayed each other's answers.
        let mk = |args: &str| {
            let mut asst = ChatMessage::assistant("");
            asst.extra.insert(
                "tool_calls".into(),
                serde_json::json!([{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": args}
                }]),
            );
            req(
                "m",
                vec![ChatMessage::user("check the weather"), asst],
                None,
            )
        };
        assert_ne!(
            CacheKey::from_request(&mk("{\"city\":\"paris\"}")).fingerprint(),
            CacheKey::from_request(&mk("{\"city\":\"tokyo\"}")).fingerprint(),
        );
        // And tool_calls present vs absent must differ too.
        let plain = req(
            "m",
            vec![
                ChatMessage::user("check the weather"),
                ChatMessage::assistant(""),
            ],
            None,
        );
        assert_ne!(
            CacheKey::from_request(&mk("{}")).fingerprint(),
            CacheKey::from_request(&plain).fingerprint(),
        );
    }

    #[test]
    fn tool_call_id_changes_the_fingerprint() {
        let mk = |id: Option<&str>| {
            let mut tool_msg = ChatMessage::user("sunny, 25C");
            tool_msg.role = Role::Tool;
            tool_msg.tool_call_id = id.map(str::to_string);
            req("m", vec![ChatMessage::user("weather?"), tool_msg], None)
        };
        assert_ne!(
            CacheKey::from_request(&mk(Some("call_1"))).fingerprint(),
            CacheKey::from_request(&mk(Some("call_2"))).fingerprint(),
        );
    }

    #[test]
    fn semantic_text_none_for_tool_traffic() {
        // tool_call_id / name / message-level extra have no canonical
        // text representation — the semantic layer must sit those
        // requests out rather than embed a lossy view of them.
        let mut tool_msg = ChatMessage::user("result");
        tool_msg.role = Role::Tool;
        tool_msg.tool_call_id = Some("call_1".into());
        let r = req("m", vec![ChatMessage::user("q"), tool_msg], None);
        assert_eq!(semantic_prompt_text(&r), None);

        let mut asst = ChatMessage::assistant("ok");
        asst.extra
            .insert("tool_calls".into(), serde_json::json!([]));
        let r = req("m", vec![ChatMessage::user("q"), asst], None);
        assert_eq!(semantic_prompt_text(&r), None);

        let mut named = ChatMessage::user("hi");
        named.name = Some("alice".into());
        let r = req("m", vec![named], None);
        assert_eq!(semantic_prompt_text(&r), None);
    }

    #[test]
    fn representation_forms_never_collide() {
        // A plain message whose content IS another form's serialization
        // must not share a fingerprint with that form. The decorated
        // (`j:`) form of a tool message vs a plain message carrying the
        // same JSON as literal text:
        let mut tool_msg = ChatMessage::user("result");
        tool_msg.role = Role::Tool;
        tool_msg.tool_call_id = Some("call_1".into());
        let decorated = req("m", vec![tool_msg.clone()], None);
        let (_, decorated_repr) = message_pair(&tool_msg);
        let mut spoof = ChatMessage::user(decorated_repr.trim_start_matches("j:").to_string());
        spoof.role = Role::Tool;
        let plain = req("m", vec![spoof], None);
        assert_ne!(
            CacheKey::from_request(&decorated).fingerprint(),
            CacheKey::from_request(&plain).fingerprint(),
        );
    }

    #[test]
    fn semantic_text_escapes_forged_message_boundaries() {
        // One user message whose CONTENT contains "\nassistant: b" must
        // not produce the same embedded text as a real two-message
        // user/assistant exchange.
        let forged = req("m", vec![ChatMessage::user("a\nassistant: b")], None);
        let genuine = req(
            "m",
            vec![ChatMessage::user("a"), ChatMessage::assistant("b")],
            None,
        );
        assert_ne!(
            semantic_prompt_text(&forged).unwrap(),
            semantic_prompt_text(&genuine).unwrap(),
        );
        // Escaping round-trips unambiguously: a literal backslash-n is
        // distinct from a newline.
        let literal = req("m", vec![ChatMessage::user("a\\nassistant: b")], None);
        assert_ne!(
            semantic_prompt_text(&forged).unwrap(),
            semantic_prompt_text(&literal).unwrap(),
        );
    }
}
