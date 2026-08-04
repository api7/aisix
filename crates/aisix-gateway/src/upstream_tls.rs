//! Trust material for every TLS connection the gateway *opens*.
//!
//! One module, because the same operator setting has to reach clients
//! built on four different stacks, and a setting that reaches only some
//! of them fails silently — the request just keeps working against the
//! public providers and keeps failing against the private one:
//!
//! - the workspace `reqwest` (provider bridges, guardrails, MCP
//!   OpenAPI/OAuth, A2A, passthrough, JWKS/OIDC discovery, OTLP export)
//!   via [`reqwest_material`], applied inside
//!   [`crate::upstream_http::client_builder`];
//! - rmcp's own `reqwest` line (MCP streamable-http), which is a
//!   different crate version and so needs the raw PEM, not our parsed
//!   `reqwest::Certificate`;
//! - raw rustls (the Realtime WebSocket) via [`rustls_client_config`];
//! - the AWS SDK (Bedrock) via [`extra_ca_pem`] — see the note on
//!   [`TlsSettings::verify`] for the one knob that stack cannot express.
//!
//! Everything here is *additive to* the platform trust store, which
//! stays whatever `SSL_CERT_FILE` / `SSL_CERT_DIR` and the system bundle
//! make it. Trusting a private CA never removes trust in a public one.

use std::sync::{Arc, OnceLock};

use aisix_core::config::OutboundTlsConfig;
use aisix_core::models::provider_key::ProviderKeyTls;

/// PEM material and verification policy shared by every outbound client.
///
/// Holds PEM *bytes* rather than parsed certificates because the four
/// consumers above parse into three incompatible types. The bytes are
/// validated once at load time so a malformed bundle fails the boot
/// rather than the first request that needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlsSettings {
    /// Extra trust roots, PEM-encoded, added to the built-in root set.
    pub extra_ca_pem: Option<Arc<Vec<u8>>>,
    /// Client certificate and private key presented to peers that
    /// require mutual TLS.
    pub client_identity: Option<Arc<ClientIdentityPem>>,
    /// Whether the peer's certificate is verified.
    ///
    /// `false` accepts any certificate from any peer, which is the
    /// whole of what TLS was protecting on a link that carries prompts,
    /// responses, and upstream API keys. Not expressible on the AWS SDK
    /// stack (Bedrock), whose public configuration surface covers trust
    /// roots only; [`aws_http_client`] logs that rather than letting the
    /// setting look applied.
    pub verify: bool,
}

impl Default for TlsSettings {
    fn default() -> Self {
        Self {
            extra_ca_pem: None,
            client_identity: None,
            verify: true,
        }
    }
}

impl TlsSettings {
    /// Whether this leaves every client exactly as it was built before
    /// the `upstream.tls` block existed.
    pub fn is_default(&self) -> bool {
        self.extra_ca_pem.is_none() && self.client_identity.is_none() && self.verify
    }

    /// Read the configured PEM files off disk and validate them.
    ///
    /// Reading here rather than at first use means an unreadable or
    /// malformed bundle stops the boot with the path in the message,
    /// instead of surfacing later as a generic upstream transport error.
    ///
    /// `block` labels the config section in every error, so an operator
    /// reading the failure knows whether to look at `upstream.tls` or at
    /// one of the `redis` blocks.
    pub fn load(block: &str, cfg: &OutboundTlsConfig) -> Result<Self, String> {
        let extra_ca_pem = match &cfg.ca_file {
            None => None,
            Some(path) => {
                let pem = std::fs::read(path)
                    .map_err(|e| format!("{block}.ca_file: read {path}: {e}"))?;
                let count = rustls_pemfile::certs(&mut pem.as_slice())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| format!("{block}.ca_file: parse {path}: {e}"))?
                    .len();
                // An empty-but-readable file is the shape a bad
                // ConfigMap key or a failed `kubectl cp` leaves behind,
                // and it would otherwise look like a configured CA.
                if count == 0 {
                    return Err(format!(
                        "{block}.ca_file: {path} contains no CERTIFICATE block"
                    ));
                }
                Some(Arc::new(pem))
            }
        };

        let client_identity = match (&cfg.client_cert_file, &cfg.client_key_file) {
            (Some(cert_path), Some(key_path)) => {
                let cert = std::fs::read(cert_path)
                    .map_err(|e| format!("{block}.client_cert_file: read {cert_path}: {e}"))?;
                let key = std::fs::read(key_path)
                    .map_err(|e| format!("{block}.client_key_file: read {key_path}: {e}"))?;
                Some(Arc::new(ClientIdentityPem { key, cert }))
            }
            // The mismatched pairs are rejected by `Config::validate`.
            _ => None,
        };

        Ok(Self {
            extra_ca_pem,
            client_identity,
            verify: cfg.verify,
        })
    }
}

/// A client certificate and the private key that goes with it, both PEM.
///
/// Kept as two halves because the consumers disagree about the shape:
/// reqwest wants one concatenated blob ([`ClientIdentityPem::joined`]),
/// redis-rs wants the two fields separately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientIdentityPem {
    pub key: Vec<u8>,
    pub cert: Vec<u8>,
}

impl ClientIdentityPem {
    /// Key then certificate in one PEM blob, the form
    /// `reqwest::Identity::from_pem` expects.
    ///
    /// The separator matters: PEM written by a deploy script or dumped
    /// out of an env var often has no trailing newline, which would glue
    /// the key's END line onto the certificate's BEGIN line and make the
    /// whole blob unparseable.
    pub fn joined(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.key.len() + self.cert.len() + 1);
        out.extend_from_slice(&self.key);
        if !self.key.ends_with(b"\n") {
            out.push(b'\n');
        }
        out.extend_from_slice(&self.cert);
        out
    }
}

/// The extra trust roots as raw PEM, for the stacks that parse PEM
/// themselves (rmcp's reqwest line, the AWS SDK's `TrustStore`).
pub fn extra_ca_pem() -> Option<Arc<Vec<u8>>> {
    crate::upstream_http::config().tls.extra_ca_pem.clone()
}

// ─── reqwest (workspace line) ────────────────────────────────────────

/// Parsed-once reqwest trust material. `reqwest::Certificate` and
/// `Identity` are both `Clone`, so parsing at boot and cloning per
/// client keeps PEM parsing off every client construction.
#[derive(Default)]
pub struct ReqwestTlsMaterial {
    pub roots: Vec<reqwest::Certificate>,
    pub identity: Option<reqwest::Identity>,
}

static REQWEST_TLS: OnceLock<ReqwestTlsMaterial> = OnceLock::new();

/// Parse `settings` into reqwest's types and install them process-wide.
/// Called from [`crate::upstream_http::init`] during boot; the error is
/// what turns a malformed bundle into a failed boot.
pub(crate) fn init_reqwest_material(settings: &TlsSettings) -> Result<(), String> {
    let roots = match &settings.extra_ca_pem {
        Some(pem) => reqwest::Certificate::from_pem_bundle(pem)
            .map_err(|e| format!("upstream.tls.ca_file: {e}"))?,
        None => Vec::new(),
    };
    let identity = match &settings.client_identity {
        Some(id) => Some(
            reqwest::Identity::from_pem(&id.joined())
                .map_err(|e| format!("upstream.tls.client_cert_file/client_key_file: {e}"))?,
        ),
        None => None,
    };
    let _ = REQWEST_TLS.set(ReqwestTlsMaterial { roots, identity });
    Ok(())
}

/// The installed reqwest trust material, empty when nothing was
/// configured (or in tests, which never call `init`).
pub fn reqwest_material() -> &'static ReqwestTlsMaterial {
    REQWEST_TLS.get_or_init(ReqwestTlsMaterial::default)
}

// ─── per-ProviderKey overrides ───────────────────────────────────────

/// Clients built for a `ProviderKey.tls` override, keyed by the override
/// itself so every key with the same settings shares one connection
/// pool.
///
/// A client is the unit reqwest attaches trust to, so an override cannot
/// be applied per request — it needs its own client, and therefore its
/// own pool. Building one per dispatch would pay a TLS handshake on
/// every call, which is precisely what the shared pool exists to avoid;
/// the cache keeps it to one per distinct override.
///
/// Unbounded on purpose. The key space is the set of distinct TLS
/// settings across the Provider Keys an operator has configured — a
/// handful in the deployments this exists for, and each entry is one
/// idle connection pool.
static PK_CLIENTS: OnceLock<dashmap::DashMap<ProviderKeyTls, reqwest::Client>> = OnceLock::new();

// ─── per-worker pools ────────────────────────────────────────────────

/// The user agent every dispatch-path client is built with.
///
/// A worker's pool stands in for those clients, so it has to present the
/// same identity upstream. `every_dispatch_client_presents_the_same_user_agent`
/// holds them in step.
pub(crate) const DISPATCH_USER_AGENT: &str = "aisix/0.1";

thread_local! {
    /// Whether this thread serves proxy traffic on its own runtime.
    static IS_WORKER_THREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// This worker's upstream pool, built on first dispatch. `None` once
    /// a build failure has been reported for this thread.
    static WORKER_CLIENT: std::cell::OnceCell<Option<reqwest::Client>> =
        const { std::cell::OnceCell::new() };
}

/// Declares the calling thread a proxy worker with its own runtime, so
/// its dispatches use a pool this thread alone polls.
///
/// Called once per worker in thread-per-core serving. Left unset
/// everywhere else — the shared runtime's threads, the blocking pool,
/// background tasks — so those keep dispatching on the process-wide
/// pools exactly as before.
pub fn mark_worker_thread() {
    IS_WORKER_THREAD.set(true);
}

/// This worker's pool, or `None` when the thread is not a worker or the
/// pool could not be built.
///
/// One pool per worker rather than per client: the dispatch clients are
/// all built from the same recipe, so merging them costs nothing but the
/// per-owner split of `pool_max_idle_per_host`, and keeping them split
/// would multiply idle connections by the worker count for no gain.
///
/// A build failure falls back to the shared pool — which carries the
/// deployment's trust settings — and is cached so one broken
/// configuration cannot log per request.
fn worker_client() -> Option<reqwest::Client> {
    if !IS_WORKER_THREAD.get() {
        return None;
    }
    WORKER_CLIENT.with(|cell| {
        cell.get_or_init(|| {
            match crate::upstream_http::client_builder()
                .user_agent(DISPATCH_USER_AGENT)
                .build()
            {
                Ok(client) => Some(client),
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "per-worker upstream pool could not be built; this worker \
                         dispatches on the shared pool"
                    );
                    None
                }
            }
        })
        .clone()
    })
}

/// The client to dispatch this Provider Key's request on.
///
/// Returns `shared` unchanged whenever the key sets no override, which
/// is the overwhelmingly common case and the one that must keep sharing
/// the bridge's pool.
///
/// A malformed `ca_cert` falls back to `shared` with a logged error
/// rather than to a client that trusts less than the operator asked
/// for — the request then fails against the private endpoint, which is
/// the same visible outcome as not having configured anything, and is
/// preferable to quietly proceeding.
pub fn client_for_provider_key(
    shared: &reqwest::Client,
    tls: Option<&ProviderKeyTls>,
) -> reqwest::Client {
    let Some(tls) = tls.filter(|t| !t.is_noop()) else {
        // On a thread-per-core worker, dispatch on that worker's own
        // pool: the upstream connection is then read by the same runtime
        // that is waiting for the response, instead of waking a thread
        // that has to hand it back. Everywhere else this is `None` and
        // the shared pool is used, as it always was.
        return worker_client().unwrap_or_else(|| shared.clone());
    };
    let cache = PK_CLIENTS.get_or_init(dashmap::DashMap::new);
    if let Some(existing) = cache.get(tls) {
        return existing.clone();
    }
    match build_provider_key_client(tls) {
        Ok(client) => cache.entry(tls.clone()).or_insert(client).clone(),
        Err(e) => {
            tracing::error!(
                error = %e,
                "provider_key.tls could not be applied; falling back to the \
                 deployment's trust settings"
            );
            shared.clone()
        }
    }
}

fn build_provider_key_client(tls: &ProviderKeyTls) -> Result<reqwest::Client, String> {
    // Layer the key's override ON TOP of the deployment settings rather
    // than replacing them: a deployment CA and a per-key CA are both
    // trust roots, and a client presenting the deployment's mTLS
    // identity must keep presenting it.
    let mut builder = crate::upstream_http::client_builder().user_agent(PROVIDER_KEY_USER_AGENT);
    if let Some(pem) = tls.ca_cert.as_ref().filter(|p| !p.trim().is_empty()) {
        let roots = reqwest::Certificate::from_pem_bundle(pem.as_bytes())
            .map_err(|e| format!("provider_key.tls.ca_cert: {e}"))?;
        if roots.is_empty() {
            return Err("provider_key.tls.ca_cert contains no certificate".into());
        }
        for root in roots {
            builder = builder.add_root_certificate(root);
        }
    }
    if !tls.verify {
        builder = builder.danger_accept_invalid_certs(true);
    }
    builder.build().map_err(|e| e.to_string())
}

/// Matches the agent every bridge sets on its shared client, so a
/// per-key client is indistinguishable upstream from the shared one.
const PROVIDER_KEY_USER_AGENT: &str = "aisix/0.1";

// ─── raw rustls (Realtime WebSocket) ─────────────────────────────────

/// A `rustls::ClientConfig` carrying the same trust decision the reqwest
/// clients get, for the outbound paths that speak rustls directly.
///
/// Built once: assembling the root store re-reads the system trust
/// store, which costs hundreds of milliseconds on some platforms.
pub fn rustls_client_config() -> Arc<rustls::ClientConfig> {
    static CONFIG: OnceLock<Arc<rustls::ClientConfig>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            Arc::new(build_rustls_client_config(
                &crate::upstream_http::config().tls,
            ))
        })
        .clone()
}

fn build_rustls_client_config(tls: &TlsSettings) -> rustls::ClientConfig {
    // `ClientConfig::builder()` picks the crypto provider implicitly and
    // **panics** when it cannot: more than one provider is compiled into
    // the workspace (redis-rs's rustls feature pulls in ring alongside
    // the aws-lc-rs everything else uses). `main` installs a default
    // before anything touches TLS, but this module must not depend on
    // having been reached after that — a panic here would take down a
    // request path over a configuration detail.
    let provider = rustls::crypto::CryptoProvider::get_default()
        .cloned()
        .unwrap_or_else(|| Arc::new(rustls::crypto::aws_lc_rs::default_provider()));
    let builder = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .expect("the crypto provider supports the default protocol versions");
    let mut config = if tls.verify {
        builder
            .with_root_certificates(root_store(tls))
            .with_no_client_auth()
    } else {
        builder
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(danger::AcceptAnyServerCert(provider)))
            .with_no_client_auth()
    };
    // The WebSocket upstream is HTTP/1.1; advertising anything else
    // would let a peer negotiate a protocol the transport can't speak.
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    config
}

/// Built-in roots (platform store + the compiled-in Mozilla set, which
/// is what reqwest trusts) plus whatever `ca_file` added.
fn root_store(tls: &TlsSettings) -> rustls::RootCertStore {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let native = rustls_native_certs::load_native_certs();
    if !native.errors.is_empty() {
        tracing::warn!(errors = ?native.errors, "some native root certificates failed to load");
    }
    let (_added, _ignored) = roots.add_parsable_certificates(native.certs);
    if let Some(pem) = &tls.extra_ca_pem {
        // Validated in `TlsSettings::load`, so a parse failure here is
        // not reachable from a booted process.
        match rustls_pemfile::certs(&mut pem.as_slice()).collect::<Result<Vec<_>, _>>() {
            Ok(certs) => {
                let (_added, ignored) = roots.add_parsable_certificates(certs);
                if ignored > 0 {
                    tracing::warn!(
                        ignored,
                        "upstream.tls.ca_file: some certificates were not usable as trust anchors"
                    );
                }
            }
            Err(e) => tracing::error!(error = %e, "upstream.tls.ca_file: parse failed"),
        }
    }
    roots
}

// ─── AWS SDK (Bedrock) ───────────────────────────────────────────────

/// The HTTP client every Bedrock SDK client is built on, carrying the
/// deployment's extra trust roots.
///
/// Built once and shared: the AWS SDK otherwise constructs a connector
/// per client, and each construction re-reads the platform trust store.
///
/// Two of the four `upstream.tls` knobs stop here, because the SDK's
/// HTTP stack exposes trust roots and nothing else on its stable
/// surface: a client certificate cannot be presented, and verification
/// cannot be switched off. Both are announced by
/// [`warn_unsupported_for_aws`] rather than silently ignored. The
/// private-CA case this feature exists for — a Bedrock-compatible
/// endpoint behind an enterprise CA, or an intercepting corporate proxy
/// — is fully served by `ca_file`.
#[cfg(feature = "aws")]
pub fn aws_http_client() -> aws_smithy_runtime_api::client::http::SharedHttpClient {
    use aws_smithy_http_client::tls;
    static CLIENT: OnceLock<aws_smithy_runtime_api::client::http::SharedHttpClient> =
        OnceLock::new();
    CLIENT
        .get_or_init(|| {
            let tls_cfg = &crate::upstream_http::config().tls;
            warn_unsupported_for_aws(tls_cfg);

            let mut trust_store = tls::TrustStore::default();
            if let Some(pem) = &tls_cfg.extra_ca_pem {
                trust_store = trust_store.with_pem_certificate(pem.as_slice());
            }
            let context = tls::TlsContext::builder()
                .with_trust_store(trust_store)
                .build()
                .expect("TLS context from a bundle validated at boot");

            aws_smithy_http_client::Builder::new()
                .tls_provider(tls::Provider::rustls(
                    tls::rustls_provider::CryptoMode::AwsLc,
                ))
                .tls_context(context)
                .build_https()
        })
        .clone()
}

/// Say out loud which `upstream.tls` knobs the AWS SDK stack cannot
/// honour. An operator who set one and then sees a Bedrock TLS failure
/// should read the reason in the log rather than conclude the setting is
/// broken.
#[cfg(feature = "aws")]
fn warn_unsupported_for_aws(tls: &TlsSettings) {
    if !tls.verify {
        tracing::warn!(
            "upstream.tls.verify=false does not apply to Bedrock: the AWS SDK's HTTP stack \
             has no way to disable certificate verification, so Bedrock certificates are \
             still checked. Use upstream.tls.ca_file to trust the endpoint's issuer."
        );
    }
    if tls.client_identity.is_some() {
        tracing::warn!(
            "upstream.tls client certificate does not apply to Bedrock: the AWS SDK's HTTP \
             stack has no way to present one, so Bedrock connections are not mutually \
             authenticated."
        );
    }
}

mod danger {
    use std::sync::Arc;

    use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
    use rustls::crypto::{verify_tls12_signature, verify_tls13_signature, CryptoProvider};
    use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
    use rustls::{DigitallySignedStruct, Error, SignatureScheme};

    /// Accepts every server certificate. Installed only when the
    /// operator set `upstream.tls.verify: false`.
    ///
    /// Signature *validation* is still delegated to the crypto provider:
    /// the handshake must still be internally consistent, we just stop
    /// asking who signed the certificate. That keeps the failure mode to
    /// "we cannot tell who the peer is" rather than "the transport is
    /// arbitrary".
    #[derive(Debug)]
    pub struct AcceptAnyServerCert(pub Arc<CryptoProvider>);

    impl ServerCertVerifier for AcceptAnyServerCert {
        fn verify_server_cert(
            &self,
            _end_entity: &CertificateDer<'_>,
            _intermediates: &[CertificateDer<'_>],
            _server_name: &ServerName<'_>,
            _ocsp: &[u8],
            _now: UnixTime,
        ) -> Result<ServerCertVerified, Error> {
            Ok(ServerCertVerified::assertion())
        }

        fn verify_tls12_signature(
            &self,
            message: &[u8],
            cert: &CertificateDer<'_>,
            dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, Error> {
            verify_tls12_signature(
                message,
                cert,
                dss,
                &self.0.signature_verification_algorithms,
            )
        }

        fn verify_tls13_signature(
            &self,
            message: &[u8],
            cert: &CertificateDer<'_>,
            dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, Error> {
            verify_tls13_signature(
                message,
                cert,
                dss,
                &self.0.signature_verification_algorithms,
            )
        }

        fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
            self.0.signature_verification_algorithms.supported_schemes()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ca_pem() -> Vec<u8> {
        let mut params = rcgen::CertificateParams::new(Vec::<String>::new()).unwrap();
        params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        let kp = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).unwrap();
        params.self_signed(&kp).unwrap().pem().into_bytes()
    }

    #[test]
    fn default_settings_leave_every_client_untouched() {
        let settings = TlsSettings::load("upstream.tls", &OutboundTlsConfig::default()).unwrap();
        assert!(settings.is_default());
        assert!(settings.verify);
    }

    #[test]
    fn ca_file_is_read_and_validated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ca.pem");
        std::fs::write(&path, ca_pem()).unwrap();

        let settings = TlsSettings::load(
            "upstream.tls",
            &OutboundTlsConfig {
                ca_file: Some(path.to_string_lossy().into_owned()),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(settings.extra_ca_pem.is_some());
        assert!(!settings.is_default());
    }

    /// A readable-but-not-a-certificate file is the shape a mis-keyed
    /// ConfigMap mount leaves behind. It must fail the boot rather than
    /// look like a configured CA that silently trusts nothing extra.
    #[test]
    fn ca_file_that_holds_no_certificate_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ca.pem");
        std::fs::write(&path, b"not a certificate\n").unwrap();

        let err = TlsSettings::load(
            "upstream.tls",
            &OutboundTlsConfig {
                ca_file: Some(path.to_string_lossy().into_owned()),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(err.contains("no CERTIFICATE block"), "{err}");
    }

    #[test]
    fn missing_ca_file_names_the_path() {
        let err = TlsSettings::load(
            "upstream.tls",
            &OutboundTlsConfig {
                ca_file: Some("/nonexistent/private-ca.pem".into()),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(err.contains("/nonexistent/private-ca.pem"), "{err}");
    }

    /// PEM emitted by a deploy script often lacks the trailing newline;
    /// concatenating naively would fuse the key's END line onto the
    /// cert's BEGIN line and make the identity unparseable.
    #[test]
    fn identity_pem_separates_blocks_without_a_trailing_newline() {
        let identity = ClientIdentityPem {
            key: b"-----END PRIVATE KEY-----".to_vec(),
            cert: b"-----BEGIN CERTIFICATE-----".to_vec(),
        };
        assert_eq!(
            String::from_utf8(identity.joined()).unwrap(),
            "-----END PRIVATE KEY-----\n-----BEGIN CERTIFICATE-----"
        );
    }

    #[test]
    fn reqwest_material_parses_a_real_ca_bundle() {
        let settings = TlsSettings {
            extra_ca_pem: Some(Arc::new(ca_pem())),
            ..Default::default()
        };
        let roots =
            reqwest::Certificate::from_pem_bundle(settings.extra_ca_pem.as_ref().unwrap()).unwrap();
        assert_eq!(roots.len(), 1);
    }

    /// Two certificates in one file is the ordinary shape of an
    /// enterprise bundle (root + intermediate); loading only the first
    /// would leave a chain unverifiable.
    #[test]
    fn a_multi_certificate_bundle_loads_every_entry() {
        let mut bundle = ca_pem();
        bundle.extend_from_slice(&ca_pem());
        let roots = reqwest::Certificate::from_pem_bundle(&bundle).unwrap();
        assert_eq!(roots.len(), 2);
    }

    /// More than one crypto provider is compiled in (redis-rs's rustls
    /// feature adds ring next to the aws-lc-rs everything else uses), so
    /// `ClientConfig::builder()` cannot pick one and panics. These two
    /// deliberately do NOT install a default first: the module has to
    /// stand on its own, or the Realtime WebSocket path panics whenever
    /// it is reached before `main` gets to `install_default`.
    #[test]
    fn verify_false_builds_a_config_that_accepts_any_certificate() {
        let cfg = build_rustls_client_config(&TlsSettings {
            verify: false,
            ..Default::default()
        });
        assert_eq!(cfg.alpn_protocols, vec![b"http/1.1".to_vec()]);
    }

    #[test]
    fn verify_true_builds_a_config_over_the_built_in_roots() {
        let cfg = build_rustls_client_config(&TlsSettings::default());
        assert_eq!(cfg.alpn_protocols, vec![b"http/1.1".to_vec()]);
    }

    fn shared_client() -> reqwest::Client {
        reqwest::Client::builder()
            .user_agent("aisix-test-shared")
            .build()
            .unwrap()
    }

    /// `reqwest::Client` exposes no identity, so "did this get its own
    /// client (and therefore its own connection pool)?" is asserted
    /// through the cache: an entry exists exactly when a dedicated client
    /// was built.
    fn cached(tls: &ProviderKeyTls) -> bool {
        PK_CLIENTS
            .get()
            .is_some_and(|cache| cache.contains_key(tls))
    }

    /// `tls: {}` — every field left at its default — is not an override,
    /// and treating it as one would split the connection pool for
    /// nothing.
    #[test]
    fn an_empty_override_is_treated_as_no_override() {
        assert!(ProviderKeyTls::default().is_noop());
        assert!(ProviderKeyTls {
            ca_cert: Some("   ".into()),
            verify: true,
        }
        .is_noop());
        assert!(!ProviderKeyTls {
            ca_cert: None,
            verify: false,
        }
        .is_noop());
    }

    /// The overwhelmingly common case: a key with no override keeps
    /// dispatching on the bridge's own client, so nothing is cached.
    #[test]
    fn a_key_without_an_override_builds_no_dedicated_client() {
        let noop = ProviderKeyTls::default();
        let _ = client_for_provider_key(&shared_client(), None);
        let _ = client_for_provider_key(&shared_client(), Some(&noop));
        assert!(!cached(&noop));
    }

    /// Two keys configured identically land on one client, so a
    /// deployment with several keys behind the same private CA keeps one
    /// connection pool rather than one per key.
    #[test]
    fn identical_overrides_share_one_client() {
        let tls = ProviderKeyTls {
            ca_cert: Some(String::from_utf8(ca_pem()).unwrap()),
            verify: true,
        };
        assert!(!cached(&tls));
        let _ = client_for_provider_key(&shared_client(), Some(&tls));
        let _ = client_for_provider_key(&shared_client(), Some(&tls.clone()));

        // Counted per-CA rather than over the whole map: the tests in
        // this module share the static and run concurrently, so a total
        // is not a stable number. Each test generates its own CA, which
        // makes this count exactly "clients built for these settings".
        let for_this_ca = PK_CLIENTS
            .get()
            .unwrap()
            .iter()
            .filter(|e| e.key().ca_cert == tls.ca_cert)
            .count();
        assert_eq!(
            for_this_ca, 1,
            "two keys with equal TLS settings must share one client, and one pool"
        );
    }

    /// A `ca_cert` that is not a certificate must not silently become a
    /// client that trusts *less* than was asked for. Falling back to the
    /// shared client makes the request fail exactly as it would have
    /// without any configuration, which is the honest outcome.
    #[test]
    fn a_malformed_ca_cert_falls_back_instead_of_caching_a_weaker_client() {
        let tls = ProviderKeyTls {
            ca_cert: Some("-----BEGIN CERTIFICATE-----\nnope\n-----END CERTIFICATE-----\n".into()),
            verify: true,
        };
        let _ = client_for_provider_key(&shared_client(), Some(&tls));
        assert!(!cached(&tls));
    }

    /// `verify: false` alone is a real override — no CA, but a different
    /// trust decision — so it must get its own client.
    #[test]
    fn verify_false_alone_builds_a_dedicated_client() {
        let tls = ProviderKeyTls {
            ca_cert: None,
            verify: false,
        };
        let _ = client_for_provider_key(&shared_client(), Some(&tls));
        assert!(cached(&tls));
    }
}
