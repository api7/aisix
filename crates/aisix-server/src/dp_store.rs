//! [`SharedStore`] over the dp-manager's store API (`/dp/store/v1`), the
//! backend of a managed gateway.
//!
//! The mTLS client is the heartbeat's. The environment comes from the client
//! certificate, so no request names it. Every response decodes loosely: the
//! dp-manager may be many releases newer than this gateway.

use std::collections::BTreeMap;
use std::time::Duration;

use aisix_mcp::{PutCondition, SharedStore, StoreError, StoredValue};
use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use reqwest::{Method, RequestBuilder, Response, StatusCode};
use serde::{Deserialize, Serialize};

use crate::heartbeat::MtlsBundle;

pub const STORE_PATH: &str = "/dp/store/v1";

pub struct DpManagerStore {
    /// `None` when the client could not be built: every call then answers
    /// [`StoreError::Unavailable`], the same as a dp-manager that is down.
    client: Option<reqwest::Client>,
    base_url: String,
}

impl DpManagerStore {
    /// `origin` is the dp-manager's scheme and authority.
    pub fn new(origin: &str, mtls: &MtlsBundle) -> Self {
        let client = crate::heartbeat::build_client(mtls)
            .inspect_err(|error| {
                tracing::error!(error = %format_args!("{error:#}"), "per-user MCP token storage is unavailable: \
                    could not build the dp-manager client");
            })
            .ok();
        Self::with_client(origin, client)
    }

    fn with_client(origin: &str, client: Option<reqwest::Client>) -> Self {
        Self {
            client,
            base_url: format!("{}{STORE_PATH}", origin.trim_end_matches('/')),
        }
    }

    fn request(
        &self,
        method: Method,
        namespace: &str,
        key: &str,
    ) -> Result<RequestBuilder, StoreError> {
        let client = self.client.as_ref().ok_or(StoreError::Unavailable)?;
        Ok(client.request(method, format!("{}/{namespace}/{key}", self.base_url)))
    }
}

async fn send(request: RequestBuilder) -> Result<Response, StoreError> {
    request.send().await.map_err(|error| {
        tracing::warn!(
            error = %aisix_gateway::transport_error_message(&error.without_url()),
            "dp-manager store request failed",
        );
        StoreError::Unavailable
    })
}

fn unusable_body(error: reqwest::Error) -> StoreError {
    aisix_gateway::log_body_read_causes(&error.without_url());
    StoreError::Unavailable
}

/// A status none of the operations below define: a dp-manager that does not
/// serve this API yet, a namespace it does not know, or a failure.
fn unexpected(status: StatusCode) -> StoreError {
    tracing::warn!(%status, "dp-manager store answered unexpectedly");
    StoreError::Unavailable
}

#[derive(Deserialize)]
struct GetResponse {
    value: String,
    version: i64,
    #[serde(default)]
    expires_at: Option<String>,
}

#[derive(Serialize)]
struct PutRequest<'a> {
    value: String,
    ttl_seconds: u64,
    labels: &'a BTreeMap<&'a str, &'a str>,
}

#[derive(Deserialize)]
struct PutResponse {
    version: i64,
}

#[async_trait]
impl SharedStore for DpManagerStore {
    async fn get(&self, namespace: &str, key: &str) -> Result<Option<StoredValue>, StoreError> {
        let response = send(self.request(Method::GET, namespace, key)?).await?;
        match response.status() {
            StatusCode::OK => {}
            StatusCode::NOT_FOUND => return Ok(None),
            status => return Err(unexpected(status)),
        }
        let body: GetResponse = response.json().await.map_err(unusable_body)?;
        let value = STANDARD
            .decode(body.value)
            .map_err(|_| StoreError::Unavailable)?;
        Ok(Some(StoredValue {
            value,
            version: body.version,
            expires_at: body
                .expires_at
                .as_deref()
                .and_then(|at| chrono::DateTime::parse_from_rfc3339(at).ok())
                .map(|at| at.timestamp()),
        }))
    }

    async fn put(
        &self,
        namespace: &str,
        key: &str,
        value: Vec<u8>,
        ttl: Duration,
        labels: &BTreeMap<&str, &str>,
        condition: PutCondition,
    ) -> Result<i64, StoreError> {
        let request = self.request(Method::PUT, namespace, key)?;
        let request = match condition {
            PutCondition::CreateOnly => request.header("if-none-match", "*"),
            PutCondition::IfVersion(version) => request.header("if-match", version.to_string()),
        };
        let response = send(request.json(&PutRequest {
            value: STANDARD.encode(value),
            ttl_seconds: ttl.as_secs(),
            labels,
        }))
        .await?;
        match response.status() {
            StatusCode::OK => {
                let body: PutResponse = response.json().await.map_err(unusable_body)?;
                Ok(body.version)
            }
            StatusCode::PRECONDITION_FAILED => Err(StoreError::PreconditionFailed),
            StatusCode::PAYLOAD_TOO_LARGE => Err(StoreError::TooLarge),
            StatusCode::TOO_MANY_REQUESTS => Err(StoreError::OverQuota),
            status => Err(unexpected(status)),
        }
    }

    async fn delete(
        &self,
        namespace: &str,
        key: &str,
        if_version: Option<i64>,
    ) -> Result<(), StoreError> {
        let mut request = self.request(Method::DELETE, namespace, key)?;
        if let Some(version) = if_version {
            request = request.header("if-match", version.to_string());
        }
        let response = send(request).await?;
        match response.status() {
            status if status.is_success() => Ok(()),
            StatusCode::NOT_FOUND | StatusCode::PRECONDITION_FAILED => Ok(()),
            status => Err(unexpected(status)),
        }
    }
}

#[cfg(test)]
pub(crate) mod test_logs {
    /// Run `f` under a capturing subscriber; returns its result and what it logged.
    pub(crate) async fn capture_logs<T>(f: impl std::future::Future<Output = T>) -> (T, String) {
        #[derive(Clone, Default)]
        struct Buf(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Buf {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let buf = Buf::default();
        let writer = buf.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);
        // A callsite first hit where no subscriber was set has cached "disabled".
        tracing::callsite::rebuild_interest_cache();
        let out = f.await;
        let logged = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
        (out, logged)
    }
}

#[cfg(test)]
mod tests {
    use super::test_logs::capture_logs;
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    async fn store(server: &MockServer) -> DpManagerStore {
        DpManagerStore::with_client(&server.uri(), Some(reqwest::Client::new()))
    }

    const KEY_PATH: &str = "/dp/store/v1/mcp_oauth_token/abc";

    #[tokio::test]
    async fn get_decodes_the_value_and_ignores_unknown_fields() {
        let server = MockServer::builder().start().await;
        Mock::given(method("GET"))
            .and(path(KEY_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "value": "aGk=", "version": 7, "expires_at": "2030-01-01T00:00:00Z", "new": 1
            })))
            .mount(&server)
            .await;
        let stored = store(&server)
            .await
            .get("mcp_oauth_token", "abc")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.value, b"hi");
        assert_eq!(stored.version, 7);
        assert_eq!(stored.expires_at, Some(1_893_456_000));
    }

    #[tokio::test]
    async fn get_of_nothing_is_none_and_a_failure_is_not() {
        let server = MockServer::builder().start().await;
        let store = store(&server).await;
        Mock::given(method("GET"))
            .and(path("/dp/store/v1/mcp_oauth_token/missing"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path(KEY_PATH))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;
        assert_eq!(store.get("mcp_oauth_token", "missing").await, Ok(None));
        assert_eq!(
            store.get("mcp_oauth_token", "abc").await,
            Err(StoreError::Unavailable)
        );
    }

    #[tokio::test]
    async fn put_sends_the_wire_shape_for_each_precondition() {
        let server = MockServer::builder().start().await;
        let labels = BTreeMap::from([("mcp_server_id", "srv"), ("subject", "member:u")]);
        let body = serde_json::json!({
            "value": "aGk=",
            "ttl_seconds": 60,
            "labels": { "mcp_server_id": "srv", "subject": "member:u" },
        });
        Mock::given(method("PUT"))
            .and(path(KEY_PATH))
            .and(header("if-none-match", "*"))
            .and(body_json(&body))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"version": 1})),
            )
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("PUT"))
            .and(path(KEY_PATH))
            .and(header("if-match", "1"))
            .and(body_json(&body))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"version": 2})),
            )
            .expect(1)
            .mount(&server)
            .await;
        let store = store(&server).await;
        let put = |condition| {
            store.put(
                "mcp_oauth_token",
                "abc",
                b"hi".to_vec(),
                Duration::from_secs(60),
                &labels,
                condition,
            )
        };
        assert_eq!(put(PutCondition::CreateOnly).await, Ok(1));
        assert_eq!(put(PutCondition::IfVersion(1)).await, Ok(2));
    }

    #[tokio::test]
    async fn put_statuses_map_to_their_errors() {
        for (status, expected) in [
            (412, StoreError::PreconditionFailed),
            (413, StoreError::TooLarge),
            (429, StoreError::OverQuota),
            (400, StoreError::Unavailable),
            (404, StoreError::Unavailable),
            (503, StoreError::Unavailable),
        ] {
            let server = MockServer::builder().start().await;
            Mock::given(method("PUT"))
                .respond_with(ResponseTemplate::new(status))
                .mount(&server)
                .await;
            let got = store(&server)
                .await
                .put(
                    "mcp_oauth_token",
                    "abc",
                    Vec::new(),
                    Duration::from_secs(1),
                    &BTreeMap::new(),
                    PutCondition::CreateOnly,
                )
                .await;
            assert_eq!(got, Err(expected), "HTTP {status}");
        }
    }

    #[tokio::test]
    async fn delete_is_idempotent_and_names_the_version_it_was_given() {
        let server = MockServer::builder().start().await;
        Mock::given(method("DELETE"))
            .and(path(KEY_PATH))
            .and(header("if-match", "3"))
            .respond_with(ResponseTemplate::new(204))
            .expect(1)
            .mount(&server)
            .await;
        store(&server)
            .await
            .delete("mcp_oauth_token", "abc", Some(3))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn no_client_means_unavailable() {
        let store = DpManagerStore::with_client("https://dpm.example", None);
        assert_eq!(
            store.get("mcp_oauth_token", "abc").await,
            Err(StoreError::Unavailable)
        );
    }

    /// A loopback address nothing listens on, so a connect is refused at once.
    fn refused_url(path: &str) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}{path}", listener.local_addr().unwrap())
    }

    #[tokio::test]
    async fn a_failed_request_logs_its_cause_without_the_url() {
        let store = DpManagerStore::with_client(&refused_url(""), Some(reqwest::Client::new()));
        let (result, logged) = capture_logs(store.get("mcp_oauth_token", "abc")).await;
        assert!(matches!(result, Err(StoreError::Unavailable)));
        assert!(logged.to_lowercase().contains("refused"), "{logged}");
        assert!(!logged.contains("mcp_oauth_token"), "{logged}");
    }
}
