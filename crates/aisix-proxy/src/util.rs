//! Small shared helpers for the proxy layer.

/// Borrow a request object when its only outbound rewrite is `model`.
/// Earlier effort/billing/thinking rewrites are gated by the caller's `Cow`;
/// signatures and provider body overrides still need the owned rewrite path.
pub(crate) fn model_only_fields<'a>(
    body: &'a serde_json::Value,
    pk: &aisix_core::ProviderKey,
    signature_field: Option<&str>,
) -> Option<&'a serde_json::Map<String, serde_json::Value>> {
    let fields = body.as_object()?;
    let signatures = signature_field
        .and_then(|field| fields.get(field))
        .is_some_and(aisix_gateway::tool_calls::body_has_thought_signatures);
    let overrides = pk.request.as_ref().is_some_and(|r| {
        !r.param_renames.is_empty()
            || r.param_constraints.is_some()
            || !r.default_body_fields.is_empty()
    });
    (!signatures && !overrides).then_some(fields)
}

/// The same bytes as cloning an object, replacing an existing `model`,
/// and serializing it, without the intermediate owned object.
pub(crate) struct ModelRenamed<'a> {
    pub fields: &'a serde_json::Map<String, serde_json::Value>,
    pub model: &'a str,
}

impl serde::Serialize for ModelRenamed<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(self.fields.len()))?;
        for (key, value) in self.fields {
            if key == "model" {
                map.serialize_entry(key, self.model)?;
            } else {
                map.serialize_entry(key, value)?;
            }
        }
        map.end()
    }
}

/// Wire bytes of a request body a proxy route sends upstream itself,
/// bypassing the provider bridges. Build the outbound `Value` inside a block
/// that ends here, so only these bytes — which reqwest holds until the
/// response head — outlive it; see [`aisix_gateway::json_body`]. A reader
/// that needs the outbound form after the upstream answered uses
/// [`outbound_text`] / [`outbound_value`] instead of a kept copy.
pub(crate) fn outbound_json<T: serde::Serialize + ?Sized>(
    body: &T,
) -> Result<bytes::Bytes, crate::error::ProxyError> {
    aisix_gateway::json_body(body).map_err(|e| {
        crate::error::ProxyError::Bridge(aisix_gateway::BridgeError::Config(format!(
            "serialize request body: {e}"
        )))
    })
}

/// The outbound request as text, for content capture: the wire bytes are
/// exactly what `serde_json::to_string` of the body would produce.
pub(crate) fn outbound_text(outbound: &[u8]) -> String {
    String::from_utf8_lossy(outbound).into_owned()
}

/// The outbound request as a `Value`, rebuilt from its wire bytes for a
/// reader that runs after the upstream has answered.
pub(crate) fn outbound_value(outbound: &[u8]) -> serde_json::Value {
    serde_json::from_slice(outbound).unwrap_or_default()
}
