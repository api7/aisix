//! Small shared helpers for the proxy layer.

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
