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
#[derive(Debug, Clone)]
pub struct CacheKey {
    pub model: String,
    pub messages: Vec<(String, String)>, // (role, content)
    pub temperature_milli: Option<u32>,  // f32 isn't Hash; quantise to milli
    pub top_p_milli: Option<u32>,
    pub max_tokens: Option<u32>,
}

impl CacheKey {
    /// Build a key from the proxy's normalised `ChatFormat`. Streaming
    /// requests are *not* cached at this layer — callers should skip the
    /// cache when `req.is_streaming()`.
    pub fn from_request(req: &ChatFormat) -> Self {
        Self {
            model: req.model.clone(),
            messages: req.messages.iter().map(message_pair).collect(),
            temperature_milli: req.temperature.map(quantise_milli),
            top_p_milli: req.top_p.map(quantise_milli),
            max_tokens: req.max_tokens,
        }
    }

    /// Hex-encoded u64 hash, used as the cache backend's lookup key.
    pub fn fingerprint(&self) -> String {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut h);
        format!("{:016x}", h.finish())
    }
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.model.hash(state);
        for (role, content) in &self.messages {
            role.hash(state);
            content.hash(state);
        }
        self.temperature_milli.hash(state);
        self.top_p_milli.hash(state);
        self.max_tokens.hash(state);
    }
}

fn message_pair(m: &ChatMessage) -> (String, String) {
    (role_str(m.role).to_string(), m.content.clone())
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

    #[test]
    fn extras_and_request_id_do_not_affect_fingerprint() {
        let mut a = req("m", vec![ChatMessage::user("hi")], None);
        a.extra
            .insert("anything".into(), serde_json::json!({"x": 1}));
        let b = req("m", vec![ChatMessage::user("hi")], None);
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

    #[test]
    fn quantise_handles_pathological_floats() {
        assert_eq!(quantise_milli(f32::NAN), 0);
        assert_eq!(quantise_milli(-1.0), 0);
        assert_eq!(quantise_milli(0.0), 0);
        assert_eq!(quantise_milli(0.5), 500);
        assert_eq!(quantise_milli(f32::INFINITY), u32::MAX);
    }
}
