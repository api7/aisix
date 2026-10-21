//! How a request to AWS Bedrock authenticates: a Bedrock API key sent as a
//! bearer token, or SigV4 with credentials that are either configured
//! explicitly, taken from the gateway's own AWS environment (the default
//! credential chain), and optionally exchanged for a role through
//! `sts:AssumeRole`.
//!
//! Temporary credentials refresh on their own: [`resolve`] hands every
//! caller with the same configuration one shared provider, which caches
//! what it fetched until shortly before it expires.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime};

use aws_config::default_provider::credentials::DefaultCredentialsChain;
use aws_config::provider_config::ProviderConfig;
use aws_config::{BehaviorVersion, Region, SdkConfig};
use aws_credential_types::provider::{
    error::CredentialsError, future, ProvideCredentials, SharedCredentialsProvider,
};
use aws_credential_types::Credentials;

/// Credentials are refreshed this long before they expire, so a request
/// never signs with credentials that die while it is in flight.
const EXPIRY_MARGIN: Duration = Duration::from_secs(60);

/// Credentials that carry no expiry (long-lived keys from the environment or
/// a credentials file) are fetched again after this long, so a rotated
/// credentials file is picked up.
const UNEXPIRING_REFRESH: Duration = Duration::from_secs(300);

/// The longest one credential fetch (default chain, STS) may take. Requests
/// on the same configuration wait for a fetch in progress, so it is bounded
/// on its own rather than by any one request's deadline.
const FETCH_TIMEOUT: Duration = Duration::from_secs(10);

/// Distinct fetched-credential configurations kept at once. Rotating a
/// configuration's secret leaves the old entry behind; past this many the
/// cache starts over, which costs each configuration one fresh fetch.
const MAX_PROVIDERS: usize = 256;

const PROVIDER_NAME: &str = "aisix-aws-auth";

/// Where the base credentials come from.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum AwsAuthMode {
    /// Explicitly configured access keys.
    AccessKey {
        access_key_id: String,
        secret_access_key: String,
        session_token: Option<String>,
    },
    /// A Bedrock API key, sent as `Authorization: Bearer`.
    ApiKey(String),
    /// The gateway's own AWS environment: environment variables, web
    /// identity (IRSA), the ECS container endpoint, the instance profile.
    DefaultChain,
}

/// A role the base credentials are exchanged for.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct AssumeRole {
    pub role_arn: String,
    pub external_id: Option<String>,
    pub role_session_name: Option<String>,
}

/// One configured way of authenticating to AWS in one region.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct AwsAuthSpec {
    pub mode: AwsAuthMode,
    /// Never set together with [`AwsAuthMode::ApiKey`]; the parsers of
    /// each configuration surface reject that combination.
    pub assume_role: Option<AssumeRole>,
    pub region: String,
}

impl std::fmt::Debug for AwsAuthSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AwsAuthSpec")
            .field("mode", &self.mode_name())
            .field("role_arn", &self.assume_role.as_ref().map(|r| &r.role_arn))
            .field("region", &self.region)
            .finish()
    }
}

impl AwsAuthSpec {
    /// The `auth_type` this specification was configured with.
    pub fn mode_name(&self) -> &'static str {
        match self.mode {
            AwsAuthMode::AccessKey { .. } => "access_key",
            AwsAuthMode::ApiKey(_) => "api_key",
            AwsAuthMode::DefaultChain => "default_chain",
        }
    }
}

/// What a request carries to authenticate.
#[derive(Clone)]
pub enum AwsAuth {
    /// `Authorization: Bearer <token>`, no SigV4.
    Bearer(String),
    /// SigV4 with credentials from this provider.
    SigV4(SharedCredentialsProvider),
}

/// Whether a credential provider's error says the configured credentials
/// are missing or refused — the configuration's to fix — rather than that
/// a credential service failed and may answer the next attempt. Every
/// Bedrock consumer classifies a credential failure through this, so they
/// cannot drift.
pub fn credentials_refused(err: &CredentialsError) -> bool {
    matches!(
        err,
        CredentialsError::CredentialsNotLoaded(_) | CredentialsError::InvalidConfiguration(_)
    )
}

/// The authentication for `spec`. Credentials that are fetched rather than
/// configured come from one provider per distinct `spec`, shared by every
/// caller, so a refresh is paid once per configuration rather than once per
/// request.
pub fn resolve(spec: &AwsAuthSpec) -> AwsAuth {
    match (&spec.mode, &spec.assume_role) {
        (AwsAuthMode::ApiKey(key), _) => AwsAuth::Bearer(key.clone()),
        (
            AwsAuthMode::AccessKey {
                access_key_id,
                secret_access_key,
                session_token,
            },
            None,
        ) => AwsAuth::SigV4(SharedCredentialsProvider::new(Credentials::new(
            access_key_id,
            secret_access_key,
            session_token.clone(),
            None,
            PROVIDER_NAME,
        ))),
        _ => {
            static PROVIDERS: OnceLock<Mutex<HashMap<AwsAuthSpec, SharedCredentialsProvider>>> =
                OnceLock::new();
            let mut providers = PROVIDERS
                .get_or_init(Default::default)
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(provider) = providers.get(spec) {
                return AwsAuth::SigV4(provider.clone());
            }
            if providers.len() >= MAX_PROVIDERS {
                providers.clear();
            }
            let provider = SharedCredentialsProvider::new(RefreshingCredentials {
                spec: spec.clone(),
                cached: tokio::sync::Mutex::new(None),
                sdk_config: tokio::sync::OnceCell::new(),
            });
            providers.insert(spec.clone(), provider.clone());
            AwsAuth::SigV4(provider)
        }
    }
}

/// Fetches credentials for the default chain and for AssumeRole, and keeps
/// them until [`EXPIRY_MARGIN`] before they expire.
struct RefreshingCredentials {
    spec: AwsAuthSpec,
    /// The last credentials fetched, and when.
    cached: tokio::sync::Mutex<Option<(Credentials, SystemTime)>>,
    sdk_config: tokio::sync::OnceCell<SdkConfig>,
}

impl std::fmt::Debug for RefreshingCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RefreshingCredentials")
            .field("spec", &self.spec)
            .finish()
    }
}

impl ProvideCredentials for RefreshingCredentials {
    fn provide_credentials<'a>(&'a self) -> future::ProvideCredentials<'a>
    where
        Self: 'a,
    {
        future::ProvideCredentials::new(self.credentials())
    }
}

impl RefreshingCredentials {
    async fn credentials(&self) -> Result<Credentials, CredentialsError> {
        let mut cached = self.cached.lock().await;
        let now = SystemTime::now();
        if let Some((c, fetched_at)) = cached.as_ref() {
            let fresh = match c.expiry() {
                Some(exp) => exp > now + EXPIRY_MARGIN,
                None => now < *fetched_at + UNEXPIRING_REFRESH,
            };
            if fresh {
                return Ok(c.clone());
            }
        }
        let fetched = tokio::time::timeout(FETCH_TIMEOUT, self.fetch())
            .await
            .unwrap_or_else(|_| Err(CredentialsError::provider_timed_out(FETCH_TIMEOUT)));
        match fetched {
            Ok(c) => {
                *cached = Some((c.clone(), now));
                Ok(c)
            }
            Err(e) => {
                tracing::warn!(
                    auth_type = self.spec.mode_name(),
                    role_arn = self.spec.assume_role.as_ref().map(|r| r.role_arn.as_str()),
                    region = %self.spec.region,
                    error = %aws_smithy_types::error::display::DisplayErrorContext(&e),
                    "could not obtain AWS credentials"
                );
                // A refresh that fails inside the margin keeps serving the
                // credentials it was meant to replace until they expire.
                match cached.as_ref() {
                    Some((c, _)) if c.expiry().is_none_or(|exp| exp > now) => Ok(c.clone()),
                    _ => Err(e),
                }
            }
        }
    }

    /// The SDK configuration the gateway's AWS environment describes:
    /// the default credential chain, and the STS endpoint
    /// (`AWS_ENDPOINT_URL_STS` / `AWS_ENDPOINT_URL` when set).
    async fn sdk_config(&self) -> &SdkConfig {
        self.sdk_config
            .get_or_init(|| async {
                let mut timeouts = aws_smithy_types::timeout::TimeoutConfig::builder();
                match crate::upstream_http::config().connect_timeout {
                    Some(d) => timeouts = timeouts.connect_timeout(d),
                    None => timeouts = timeouts.disable_connect_timeout(),
                }
                let region = Region::new(self.spec.region.clone());
                // The chain's instance-metadata calls never go through the
                // proxy environment, wherever aws-config resolves the
                // endpoint (the default, the IPv6 endpoint mode, or a
                // configured endpoint); everything else on the chain —
                // its STS calls included — uses the shared client.
                let imds = aws_config::imds::Client::builder()
                    .configure(&ProviderConfig::default().with_http_client(
                        crate::upstream_tls::aws_credential_endpoint_http_client(),
                    ))
                    .build();
                let chain = DefaultCredentialsChain::builder()
                    .configure(
                        ProviderConfig::default()
                            .with_http_client(crate::upstream_tls::aws_http_client())
                            .with_region(Some(region.clone())),
                    )
                    .region(region.clone())
                    .imds_client(imds)
                    .build()
                    .await;
                aws_config::defaults(BehaviorVersion::latest())
                    .region(region)
                    .http_client(crate::upstream_tls::aws_http_client())
                    .credentials_provider(chain)
                    .timeout_config(timeouts.build())
                    .load()
                    .await
            })
            .await
    }

    async fn fetch(&self) -> Result<Credentials, CredentialsError> {
        let base = match &self.spec.mode {
            AwsAuthMode::AccessKey {
                access_key_id,
                secret_access_key,
                session_token,
            } => Credentials::new(
                access_key_id,
                secret_access_key,
                session_token.clone(),
                None,
                PROVIDER_NAME,
            ),
            AwsAuthMode::DefaultChain => {
                let chain = self
                    .sdk_config()
                    .await
                    .credentials_provider()
                    .ok_or_else(|| {
                        CredentialsError::not_loaded(
                            "the default AWS credential chain is unavailable",
                        )
                    })?;
                chain.provide_credentials().await?
            }
            AwsAuthMode::ApiKey(_) => {
                return Err(CredentialsError::invalid_configuration(
                    "a Bedrock API key does not sign requests",
                ))
            }
        };
        let Some(role) = &self.spec.assume_role else {
            return Ok(base);
        };
        let sts_conf = aws_sdk_sts::config::Builder::from(self.sdk_config().await)
            .credentials_provider(SharedCredentialsProvider::new(base.clone()))
            .build();
        let sts = aws_sdk_sts::Client::from_conf(sts_conf);
        if self.already_is_role(&sts, &role.role_arn).await {
            tracing::debug!(
                role_arn = %role.role_arn,
                "the current AWS identity already is the configured role; using its credentials"
            );
            return Ok(base);
        }
        let session_name = role.role_session_name.clone().unwrap_or_else(|| {
            let secs = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or_default();
            format!("aisix-session-{secs}")
        });
        let out = sts
            .assume_role()
            .role_arn(&role.role_arn)
            .role_session_name(session_name)
            .set_external_id(role.external_id.clone())
            .send()
            .await
            .map_err(|e| {
                let msg = format!(
                    "sts:AssumeRole for {} failed: {}",
                    role.role_arn,
                    aws_sdk_sts::error::DisplayErrorContext(&e)
                );
                // A refusal (access denied, malformed role) is the
                // configuration's to fix; anything else may pass.
                let status = e.raw_response().map(|r| r.status().as_u16());
                let throttled = e.as_service_error().and_then(|s| {
                    use aws_sdk_sts::error::ProvideErrorMetadata as _;
                    s.code().map(|c| c.contains("Throttl"))
                }) == Some(true);
                match status {
                    Some(400..=499) if !throttled => CredentialsError::invalid_configuration(msg),
                    _ => CredentialsError::provider_error(msg),
                }
            })?;
        let c = out.credentials().ok_or_else(|| {
            CredentialsError::unhandled("sts:AssumeRole answered without credentials")
        })?;
        let expiry = SystemTime::try_from(*c.expiration()).map_err(|_| {
            CredentialsError::unhandled("sts:AssumeRole answered with an unusable expiration")
        })?;
        Ok(Credentials::new(
            c.access_key_id(),
            c.secret_access_key(),
            Some(c.session_token().to_string()),
            Some(expiry),
            PROVIDER_NAME,
        ))
    }

    /// Whether the base credentials already are `role_arn`, in which case
    /// there is nothing to assume — a role is often not trusted to assume
    /// itself. Under web identity (IRSA) the role is named in the
    /// environment (unless environment access keys, which the chain
    /// prefers, are set); otherwise `sts:GetCallerIdentity` says who the
    /// credentials belong to. Asked once per credential refresh. An
    /// identity that cannot be determined is treated as a different one,
    /// so the role is assumed.
    async fn already_is_role(&self, sts: &aws_sdk_sts::Client, role_arn: &str) -> bool {
        let Some(target) = role_identity(role_arn) else {
            return false;
        };
        if self.spec.mode == AwsAuthMode::DefaultChain
            && std::env::var_os("AWS_ACCESS_KEY_ID").is_none()
        {
            if let (Ok(web_role), Ok(_)) = (
                std::env::var("AWS_ROLE_ARN"),
                std::env::var("AWS_WEB_IDENTITY_TOKEN_FILE"),
            ) {
                return role_identity(&web_role) == Some(target);
            }
        }
        match sts.get_caller_identity().send().await {
            Ok(out) => out.arn().and_then(role_identity) == Some(target),
            Err(e) => {
                tracing::debug!(
                    error = %aws_sdk_sts::error::DisplayErrorContext(&e),
                    "sts:GetCallerIdentity failed; assuming the configured role"
                );
                false
            }
        }
    }
}

/// `(partition, account, role name)` of an IAM role ARN
/// (`arn:aws:iam::123:role/path/Name`) or an assumed-role session ARN
/// (`arn:aws:sts::123:assumed-role/Name/session`).
fn role_identity(arn: &str) -> Option<(&str, &str, &str)> {
    let mut parts = arn.splitn(6, ':');
    if parts.next()? != "arn" {
        return None;
    }
    let partition = parts.next()?;
    let _service = parts.next()?;
    let _region = parts.next()?;
    let account = parts.next()?;
    let resource = parts.next()?;
    let name = if let Some(path) = resource.strip_prefix("role/") {
        path.rsplit('/').next()?
    } else {
        resource.strip_prefix("assumed-role/")?.split('/').next()?
    };
    (!name.is_empty()).then_some((partition, account, name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_and_assumed_role_arns_name_the_same_identity() {
        assert_eq!(
            role_identity("arn:aws:iam::123456789012:role/team/Bedrock"),
            Some(("aws", "123456789012", "Bedrock"))
        );
        assert_eq!(
            role_identity("arn:aws:sts::123456789012:assumed-role/Bedrock/aisix-session-1"),
            Some(("aws", "123456789012", "Bedrock"))
        );
        assert_eq!(role_identity("arn:aws:iam::123456789012:user/alice"), None);
        assert_eq!(role_identity("not-an-arn"), None);
    }

    #[test]
    fn debug_never_prints_secrets() {
        let spec = AwsAuthSpec {
            mode: AwsAuthMode::AccessKey {
                access_key_id: "AKIA-visible-id".into(),
                secret_access_key: "very-secret-value".into(),
                session_token: Some("very-secret-token".into()),
            },
            assume_role: None,
            region: "us-east-1".into(),
        };
        let printed = format!("{spec:?}");
        assert!(!printed.contains("very-secret"), "{printed}");
    }
}
