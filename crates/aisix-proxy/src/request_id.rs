use uuid::Uuid;

/// Gateway correlation IDs may be written to telemetry fields backed by UUID
/// columns, so keep handler request IDs as plain UUID strings.
pub(crate) fn new_request_id() -> String {
    Uuid::new_v4().to_string()
}
