//! `AzureOpenAiBridge` — family Bridge for [`Adapter::AzureOpenai`].
//!
//! Skeleton: structure + URL-shape helpers + Hub-registrable shell.
//! Real HTTP dispatch lands in follow-up PRs (see crate-level docs).

use aisix_gateway::{
    Bridge, BridgeContext, BridgeError, ChatChunkStream, ChatFormat, ChatResponse,
};
use async_trait::async_trait;

use crate::wire;

/// Family Bridge for Azure OpenAI Service.
///
/// **Skeleton:** compiles, registers, surfaces a clear
/// `BridgeError::Config` on every call. Real dispatch is wired in
/// follow-up PRs — see [`crate`] docs.
pub struct AzureOpenAiBridge {
    /// Static `name()` returned to the Hub. Kept for metrics-label
    /// stability even though we don't have a transport yet. Different
    /// from the inner OpenAI metric label so dashboards can split
    /// Azure traffic from canonical OpenAI traffic.
    name: &'static str,
}

impl AzureOpenAiBridge {
    /// Construct an Azure OpenAI bridge with the canonical name
    /// `"azure-openai"`. The Hub looks this up via [`Bridge::name`]
    /// when emitting per-request metrics (provider label).
    pub fn new() -> Self {
        Self {
            name: "azure-openai",
        }
    }
}

impl Default for AzureOpenAiBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Parsed Azure upstream reference resolved from a provider_key's
/// `api_base` + the request's upstream model id.
///
/// Azure's chat-completions URL pattern (per
/// <https://learn.microsoft.com/en-us/azure/ai-services/openai/reference>):
///
/// ```text
/// https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=<version>
/// ```
///
/// - `resource` — the Azure resource name, e.g. `acme-prod-west-us`
/// - `deployment` — operator-named deployment, e.g. `gpt4o-prod`
/// - `api_version` — Azure's date-stamped API version, e.g. `2024-08-01-preview`
///
/// Skeleton: returned by [`AzureUpstreamRef::resolve`] for use by the
/// follow-up dispatch PR. The resolver is intentionally cautious —
/// any missing piece produces a clear `BridgeError::Config` so an
/// operator can fix the registration before traffic ever hits Azure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AzureUpstreamRef {
    pub resource: String,
    pub deployment: String,
    pub api_version: String,
}

impl AzureUpstreamRef {
    /// Default Azure REST API version Bridge tests rely on when the
    /// operator hasn't pinned one. Real production deployments
    /// **must** pin a version explicitly via `provider_key.api_base`
    /// — Azure deprecates older versions on a published schedule:
    /// <https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation>
    pub const DEFAULT_API_VERSION: &'static str = "2024-08-01-preview";

    /// Resolve from the deployment name + an optional pre-parsed
    /// `api_base`. Real dispatch will call this from `chat()`; today
    /// it's exercised purely by the publisher-resolution tests.
    pub fn resolve(deployment: &str, api_base: Option<&str>) -> Result<Self, BridgeError> {
        if deployment.trim().is_empty() {
            return Err(BridgeError::Config(
                "azure deployment name is empty (expected a deployment id from \
                 the Azure portal, e.g. \"gpt4o-prod\")"
                    .into(),
            ));
        }

        // Skeleton: the api_base contains the resource. Real parser
        // lands in follow-up PRs; for now we accept either:
        //   - "https://<resource>.openai.azure.com" (canonical)
        //   - "<resource>" (bare resource name shorthand)
        // and require the canonical form for anything else.
        let base = api_base.unwrap_or_default().trim();
        let resource = if base.is_empty() {
            return Err(BridgeError::Config(
                "azure provider_key has no api_base — \
                 expected https://<resource>.openai.azure.com or a bare resource name"
                    .into(),
            ));
        } else if let Some(rest) = base
            .strip_prefix("https://")
            .or_else(|| base.strip_prefix("http://"))
        {
            rest.split('.').next().unwrap_or_default().to_string()
        } else {
            base.to_string()
        };

        if resource.is_empty() {
            return Err(BridgeError::Config(format!(
                "azure resource not resolvable from api_base {base:?}"
            )));
        }

        Ok(Self {
            resource,
            deployment: deployment.to_string(),
            api_version: Self::DEFAULT_API_VERSION.to_string(),
        })
    }

    /// Build the chat-completions URL for this Azure upstream. Used
    /// by the follow-up dispatch PR.
    pub fn chat_completions_url(&self) -> String {
        format!(
            "https://{}.openai.azure.com/openai/deployments/{}/chat/completions?api-version={}",
            self.resource, self.deployment, self.api_version,
        )
    }
}

#[async_trait]
impl Bridge for AzureOpenAiBridge {
    fn name(&self) -> &'static str {
        self.name
    }

    async fn chat(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatResponse, BridgeError> {
        // Skeleton: validate the deployment resolution path so a
        // misconfigured row surfaces a clear error today, even
        // though the actual HTTP call is TODO.
        let _upstream =
            AzureUpstreamRef::resolve(&req.model, ctx.provider_key.api_base.as_deref())?;
        // Reserved-config helpers exercised by tests: keep the wire
        // module reachable from the public surface so a future
        // dispatch PR can drop its body straight in (header / query
        // guards for the eventual default_headers /
        // default_body_fields override apply path).
        let _ = wire::reserved_query_params();
        let _ = wire::reserved_auth_headers();
        Err(BridgeError::Config(
            "azure-openai bridge is not yet implemented — \
             tracked under api7/AISIX-Cloud#302 Phase F (D6)"
                .into(),
        ))
    }

    async fn chat_stream(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatChunkStream, BridgeError> {
        let _upstream =
            AzureUpstreamRef::resolve(&req.model, ctx.provider_key.api_base.as_deref())?;
        Err(BridgeError::Config(
            "azure-openai bridge is not yet implemented — \
             tracked under api7/AISIX-Cloud#302 Phase F (D6)"
                .into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_accepts_canonical_https_resource() {
        let r = AzureUpstreamRef::resolve("gpt4o-prod", Some("https://acme-west.openai.azure.com"))
            .unwrap();
        assert_eq!(r.resource, "acme-west");
        assert_eq!(r.deployment, "gpt4o-prod");
        assert_eq!(r.api_version, AzureUpstreamRef::DEFAULT_API_VERSION);
    }

    #[test]
    fn resolve_accepts_bare_resource_name() {
        // Convenience: operator pastes just the resource name as
        // api_base. We let it through — the URL builder synthesizes
        // the canonical host.
        let r = AzureUpstreamRef::resolve("dep", Some("acme-east")).unwrap();
        assert_eq!(r.resource, "acme-east");
    }

    #[test]
    fn resolve_rejects_empty_deployment() {
        let err = AzureUpstreamRef::resolve("", Some("https://acme.openai.azure.com")).unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("deployment name is empty"),
                    "must call out empty deployment; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn resolve_rejects_missing_api_base() {
        let err = AzureUpstreamRef::resolve("dep", None).unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("no api_base"),
                    "must call out missing api_base; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn resolve_rejects_empty_api_base() {
        let err = AzureUpstreamRef::resolve("dep", Some("   ")).unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("no api_base"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn chat_completions_url_matches_azure_api_path() {
        // Tight pin on the URL fragment Azure expects — a typo here
        // would surface as a 404 from every Azure dispatch.
        let r = AzureUpstreamRef {
            resource: "acme-west".into(),
            deployment: "gpt4o-prod".into(),
            api_version: "2024-08-01-preview".into(),
        };
        assert_eq!(
            r.chat_completions_url(),
            "https://acme-west.openai.azure.com/openai/deployments/gpt4o-prod/chat/completions?api-version=2024-08-01-preview",
        );
    }

    #[test]
    fn bridge_name_is_stable() {
        // Metrics label is part of the public contract — a rename
        // would silently break customer dashboards. `"azure-openai"`
        // is the canonical name used in the Adapter enum's
        // `kebab-case` rename.
        assert_eq!(AzureOpenAiBridge::new().name(), "azure-openai");
    }

    use aisix_core::{Model, ProviderKey};
    use aisix_gateway::ChatMessage;
    use std::sync::Arc;

    fn sample_model() -> Arc<Model> {
        // Note: the Model.provider field still uses the legacy 6-value
        // Provider enum (openai/anthropic/google/deepseek/cohere/jina).
        // The Adapter enum's `azure-openai` variant is on
        // ProviderKey.adapter, not Model.provider — see issue #302
        // §3. For the skeleton tests we keep the Model on a valid
        // legacy provider; the bridge resolves the actual Azure
        // upstream from ProviderKey.api_base + Model.model_name.
        Arc::new(
            serde_json::from_str(
                r#"{
                    "display_name": "my-azure-gpt4",
                    "provider": "openai",
                    "model_name": "gpt4o-prod",
                    "provider_key_id": "11111111-1111-1111-1111-111111111111"
                }"#,
            )
            .unwrap(),
        )
    }

    fn sample_pk(api_base: Option<&str>) -> Arc<ProviderKey> {
        let api_base_json = match api_base {
            Some(b) => format!(r#", "api_base": "{b}""#),
            None => String::new(),
        };
        Arc::new(
            serde_json::from_str(&format!(
                r#"{{"display_name": "azure-prod", "secret": "az-key"{}}}"#,
                api_base_json
            ))
            .unwrap(),
        )
    }

    #[tokio::test]
    async fn chat_surfaces_clear_not_implemented_error() {
        let bridge = AzureOpenAiBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model(),
            sample_pk(Some("https://acme-west.openai.azure.com")),
        );
        let req = ChatFormat::new("gpt4o-prod", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("azure-openai bridge is not yet implemented"),
                    "error message must call out the WIP status; got {msg}"
                );
                assert!(
                    msg.contains("#302"),
                    "error message must link to the tracking issue; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_with_missing_api_base_errors_before_dispatch() {
        // The resolve-time guard fires before the not-implemented
        // stub — proves the bridge will reject malformed
        // registrations early once dispatch lands.
        let bridge = AzureOpenAiBridge::new();
        let ctx = BridgeContext::new("req-1", sample_model(), sample_pk(None));
        let req = ChatFormat::new("gpt4o-prod", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("no api_base"),
                    "must mention missing api_base; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }
}
