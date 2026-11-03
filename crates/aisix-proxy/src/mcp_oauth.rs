//! The sign-in surface for `auth_type: oauth2_authorization_code` MCP servers.
//!
//! To an MCP client each such server looks like its own OAuth authorization
//! server, reached under `{base}/mcp/{server}`: it discovers the endpoints
//! here, opens the browser, and redeems a code of the gateway's. Behind that,
//! the gateway runs the same flow against the upstream, as an OAuth client of
//! its own, and keeps the tokens (`aisix_mcp::user_oauth`).
//!
//! There are two modes, chosen by the environment's trust providers
//! ([`sign_in_mode`]). In the relay mode the client ends up with a placeholder
//! that means nothing to anyone: on every later request it is the gateway
//! credential beside it that says who is calling. In the interactive mode the
//! client has no gateway credential, so the browser pass signs the user in at
//! a trust provider too and the client gets a session token
//! ([`crate::mcp_oauth_login`]).
//!
//! `base` is `mcp_auth_settings.resource_url` without its trailing `/mcp`, so
//! every URL handed out survives a front gateway that routes on a path prefix
//! and strips it. Nothing here depends on a JWKS provider, unlike the `/mcp`
//! discovery in [`crate::mcp_auth`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use aisix_core::models::{McpServerType, OidcProvider};
use aisix_core::resource::ResourceEntry;
use aisix_core::{AisixSnapshot, McpAuthType};
use aisix_mcp::user_oauth::{RedeemError, RedeemRequest};
use aisix_mcp::{UserOAuthFailure, UserOAuthServer, UserSubject};
use axum::body::{to_bytes, Body};
use axum::extract::{FromRequestParts, Request, State};
use axum::http::{header, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use percent_encoding::{utf8_percent_encode, AsciiSet, NON_ALPHANUMERIC};
use serde_json::json;

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::reject::AisixPath;
use crate::state::ProxyState;

/// What a client advertises and receives as its access token's lifetime. The
/// placeholder never expires in any sense the gateway checks; a year keeps a
/// client from asking for another one.
const PLACEHOLDER_EXPIRES_IN: u64 = 365 * 24 * 3600;
/// Everything these routes read is a form or a registration, and most of
/// them are reachable before anyone has authenticated.
const MAX_BODY_BYTES: usize = 64 * 1024;
/// A request parameter ends up in a stored record, so its size is the
/// caller's to choose only up to a point.
const MAX_QUERY_BYTES: usize = 8 * 1024;

/// Characters a server name keeps when written into a URL path.
const PATH_SEGMENT: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'_')
    .remove(b'.')
    .remove(b'~');

pub(crate) fn scoped_url(base: &str, server: &str) -> String {
    format!("{base}/mcp/{}", utf8_percent_encode(server, PATH_SEGMENT))
}

pub(crate) fn callback_url(base: &str) -> String {
    format!("{base}/mcp-oauth/callback")
}

/// The URL a sign-in challenge points at.
fn challenge_metadata_url(base: &str, server: &str) -> String {
    format!(
        "{base}/.well-known/oauth-protected-resource/mcp/{}",
        utf8_percent_encode(server, PATH_SEGMENT)
    )
}

/// The registered server `name`, if it serves this surface.
pub(crate) fn sign_in_server(snapshot: &AisixSnapshot, name: &str) -> Option<UserOAuthServer> {
    let entry = snapshot.mcp_servers.get_by_name(name)?;
    let server = &entry.value;
    (server.enabled
        && server.server_type == McpServerType::Mcp
        && server.auth_type == McpAuthType::OAuth2AuthorizationCode)
        .then(|| UserOAuthServer::new(&entry.id, server))
}

/// How the environment's `oauth2_authorization_code` servers sign users in.
pub(crate) enum SignInMode {
    /// No enabled trust provider has `login`: the client's credential says who
    /// is calling.
    Relay,
    /// Exactly one does, and can use it.
    Interactive(Arc<ResourceEntry<OidcProvider>>),
    /// More than one does, or the one that does has no authorization server to
    /// send anyone to. A disabled provider is ignored, as it is everywhere.
    Misconfigured,
}

pub(crate) fn sign_in_mode(snapshot: &AisixSnapshot) -> SignInMode {
    let mut with_login = snapshot
        .oidc_providers
        .entries()
        .into_iter()
        .filter(|entry| entry.value.enabled && entry.value.login.is_some());
    let Some(provider) = with_login.next() else {
        return SignInMode::Relay;
    };
    if with_login.next().is_some()
        || !provider.value.is_jwks_mode()
        || provider.value.issuer.is_none()
    {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            tracing::warn!(target: "aisix::mcp_oauth",
                "`login` needs exactly one enabled trust provider, with an issuer and no \
                 hmac_secret; sign-in to MCP servers fails until that is so");
        });
        return SignInMode::Misconfigured;
    }
    SignInMode::Interactive(provider)
}

/// Who a stored token belongs to, or `None` for a caller with no user to keep
/// one for. A JWT names its own subject even when the key it maps to has an
/// owner: claim mappings let many identities share one key.
pub(crate) fn caller_subject(auth: &AuthenticatedKey) -> Option<UserSubject> {
    if let Some(jwt) = &auth.jwt {
        return Some(UserSubject::jwt(
            &jwt.provider,
            jwt.issuer.as_deref(),
            &jwt.identity_claim,
            &jwt.subject,
        ));
    }
    if auth.anonymous {
        return None;
    }
    auth.key()
        .user_id
        .as_deref()
        .filter(|id| !id.is_empty())
        .map(UserSubject::member)
}

/// Point a 401 on `server` at that server's own sign-in, if it signs users in.
/// Replaces whatever the `/mcp` discovery would have said: its document names
/// the trust providers, whose tokens this server has no use for.
pub(crate) fn challenge_for_server(
    snapshot: &AisixSnapshot,
    server: &str,
    response: &mut Response,
) {
    if response.status() != StatusCode::UNAUTHORIZED || sign_in_server(snapshot, server).is_none() {
        return;
    }
    let Some(base) = crate::mcp_auth::public_base(snapshot) else {
        return;
    };
    let value = format!(
        "Bearer resource_metadata=\"{}\"",
        challenge_metadata_url(&base, server)
    );
    if let Ok(value) = HeaderValue::from_str(&value) {
        response
            .extensions_mut()
            .remove::<crate::error::AuthChallenge>();
        response
            .headers_mut()
            .insert(header::WWW_AUTHENTICATE, value);
    }
}

/// The response `/mcp/{server}` gives a request it cannot serve for want of a
/// usable token. Only `AuthorizationRequired` starts a sign-in: every other
/// failure must not, because signing in would not help.
pub(crate) fn failure_response(
    snapshot: &AisixSnapshot,
    server: &str,
    failure: UserOAuthFailure,
) -> Response {
    let mut response = ProxyError::McpUserOAuth {
        failure,
        server: server.to_owned(),
    }
    .into_response();
    challenge_for_server(snapshot, server, &mut response);
    response
}

/// An OAuth error body (RFC 6749 §5.2).
pub(crate) fn oauth_error(status: StatusCode, error: &str, description: &str) -> Response {
    let mut response = (
        status,
        Json(json!({ "error": error, "error_description": description })),
    )
        .into_response();
    no_store(&mut response);
    response
}

pub(crate) fn no_store(response: &mut Response) {
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
}

pub(crate) fn redirect(location: &str) -> Response {
    let Ok(value) = HeaderValue::from_str(location) else {
        return oauth_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "redirect_uri cannot be used in a redirect",
        );
    };
    let mut response = StatusCode::FOUND.into_response();
    response.headers_mut().insert(header::LOCATION, value);
    no_store(&mut response);
    response
}

/// Send the browser back to the client with `params` added to its
/// `redirect_uri`, and its `state` if it sent one.
pub(crate) fn redirect_to_client(
    redirect_uri: &str,
    state: Option<&str>,
    params: &[(&str, &str)],
) -> Response {
    let Ok(mut url) = url::Url::parse(redirect_uri) else {
        return oauth_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "redirect_uri is not a valid URL",
        );
    };
    {
        let mut query = url.query_pairs_mut();
        for (name, value) in params {
            query.append_pair(name, value);
        }
        if let Some(state) = state {
            query.append_pair("state", state);
        }
    }
    redirect(url.as_str())
}

pub(crate) fn query_params(uri: &axum::http::Uri) -> HashMap<String, String> {
    url::form_urlencoded::parse(uri.query().unwrap_or_default().as_bytes())
        .into_owned()
        .collect()
}

/// Whether an MCP client may be sent back to `redirect_uri` after signing in:
/// any https URL, the loopback addresses native clients listen on, and the
/// known private-use URIs of native clients. `Err` names what is wrong with it.
///
/// This is not what keeps a code from a site a user was lured into starting a
/// flow for. The consent page is: it names where the code is going, and nothing
/// is issued until the user allows it.
fn validate_redirect_uri(redirect_uri: &str) -> Result<(), &'static str> {
    // A URL parser reads `\` as `/` in an http authority, which a browser may
    // not agree with about which host that is.
    if redirect_uri.contains('\\') {
        return Err("redirect_uri must not contain a backslash");
    }
    let url = url::Url::parse(redirect_uri).map_err(|_| "redirect_uri is not a valid URL")?;
    if url.fragment().is_some() {
        return Err("redirect_uri must not contain a fragment");
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("redirect_uri must not contain userinfo");
    }
    if !matches!(url.scheme(), "http" | "https") {
        let known = url.query().is_none() && NATIVE_REDIRECT_URIS.contains(&url.as_str());
        return if known {
            Ok(())
        } else {
            Err("redirect_uri must use http, https or a known native client callback")
        };
    }
    let host = url.host().ok_or("redirect_uri must include a host")?;
    if url.scheme() == "https" || is_loopback(&host) {
        return Ok(());
    }
    Err("redirect_uri must use https unless it is a loopback address")
}

/// Answer an authorization request that gets no further. A client on this
/// machine hears about it at its `redirect_uri`, as RFC 6749 §4.1.2.1 has it. Any
/// other address is whatever the link said and nobody has consented to it, so
/// sending the browser there would make the gateway an open redirector (RFC 9700
/// §4.11.2): the browser is told here instead.
pub(crate) fn refuse_to_client(
    redirect_uri: &str,
    state: Option<&str>,
    error: &str,
    description: &str,
) -> Response {
    if may_receive_unconsented_redirect(redirect_uri) {
        return redirect_to_client(
            redirect_uri,
            state,
            &[("error", error), ("error_description", description)],
        );
    }
    let status = match error {
        "temporarily_unavailable" => StatusCode::SERVICE_UNAVAILABLE,
        "server_error" => StatusCode::INTERNAL_SERVER_ERROR,
        _ => StatusCode::BAD_REQUEST,
    };
    oauth_error(status, error, description)
}

fn may_receive_unconsented_redirect(redirect_uri: &str) -> bool {
    url::Url::parse(redirect_uri).is_ok_and(|url| {
        !matches!(url.scheme(), "http" | "https") || url.host().is_some_and(|h| is_loopback(&h))
    })
}

/// Private-use redirect URIs of native MCP clients, compared exactly.
const NATIVE_REDIRECT_URIS: &[&str] = &["cursor://anysphere.cursor-mcp/oauth/callback"];

fn is_loopback(host: &url::Host<&str>) -> bool {
    match host {
        url::Host::Domain(domain) => domain.eq_ignore_ascii_case("localhost"),
        url::Host::Ipv4(ip) => ip.is_loopback(),
        url::Host::Ipv6(ip) => {
            ip.is_loopback() || ip.to_ipv4_mapped().is_some_and(|ip| ip.is_loopback())
        }
    }
}

/// Runs a handler with the bookkeeping every route here shares: no usage row
/// on any outcome, one access-log line, the request families.
pub(crate) async fn observed<F>(
    state: &ProxyState,
    caller: Option<&AuthenticatedKey>,
    request: Request,
    handler: F,
) -> Response
where
    F: AsyncFnOnce(&ProxyState, Request) -> Response,
{
    crate::attribution::note_unmetered_route();
    let started = Instant::now();
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    let request_id = request
        .extensions()
        .get::<crate::request_id::RequestId>()
        .map(|r| r.0.clone())
        .unwrap_or_default();
    let response = handler(state, request).await;
    crate::reject::emit_unrouted_access_log(
        method.as_str(),
        &path,
        &request_id,
        caller.map(|auth| auth.entry.id.as_str()),
        response.status().as_u16(),
        started,
    );
    crate::reject::record_undispatched(
        state,
        &path,
        caller.map_or(
            crate::request_metrics::Caller::unattributed(None),
            crate::request_metrics::Caller::new,
        ),
        response.status().as_u16(),
        crate::attribution::rendered_error_source(response.status().as_u16()),
        started,
    );
    response
}

fn json_document(method: &Method, document: serde_json::Value) -> Response {
    let mut response = Json(document).into_response();
    if method == Method::HEAD {
        *response.body_mut() = Body::empty();
    }
    response
}

/// `GET /.well-known/oauth-protected-resource/mcp/{server}`. The server is
/// its own authorization server, as far as the client can tell.
pub(crate) async fn protected_resource(
    AisixPath(server): AisixPath<String>,
    State(state): State<ProxyState>,
    request: Request,
) -> Response {
    observed(&state, None, request, async |state, request| {
        let snapshot = state.snapshot.load();
        let (Some(oauth), Some(base)) = (
            sign_in_server(&snapshot, &server),
            crate::mcp_auth::public_base(&snapshot),
        ) else {
            return StatusCode::NOT_FOUND.into_response();
        };
        let resource = scoped_url(&base, &server);
        let mut document = json!({
            "resource": resource,
            "authorization_servers": [resource],
            "bearer_methods_supported": ["header"],
        });
        if !oauth.scopes().is_empty() {
            document["scopes_supported"] = json!(oauth.scopes());
        }
        json_document(request.method(), document)
    })
    .await
}

/// RFC 8414 metadata, at `/.well-known/oauth-authorization-server/mcp/{server}`
/// and at `/mcp/{server}/.well-known/openid-configuration`. The second is the
/// form a client can still reach behind a front gateway that only routes a
/// path prefix, since it stays under the prefix.
///
/// Reached there, the document has to pass as OpenID provider metadata: MCP
/// clients validate that form against OpenID Connect Discovery, whose required
/// fields are the last three here even though no ID token is ever issued.
pub(crate) async fn authorization_server(
    AisixPath(server): AisixPath<String>,
    State(state): State<ProxyState>,
    request: Request,
) -> Response {
    observed(&state, None, request, async |state, request| {
        let snapshot = state.snapshot.load();
        let (Some(oauth), Some(base)) = (
            sign_in_server(&snapshot, &server),
            crate::mcp_auth::public_base(&snapshot),
        ) else {
            return StatusCode::NOT_FOUND.into_response();
        };
        let issuer = scoped_url(&base, &server);
        let mut document = json!({
            "issuer": issuer,
            "authorization_endpoint": format!("{issuer}/oauth/authorize"),
            "token_endpoint": format!("{issuer}/oauth/token"),
            "registration_endpoint": format!("{issuer}/oauth/register"),
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
            "jwks_uri": format!("{issuer}/.well-known/jwks.json"),
            "subject_types_supported": ["public"],
            "id_token_signing_alg_values_supported": ["RS256"],
        });
        if !oauth.scopes().is_empty() {
            document["scopes_supported"] = json!(oauth.scopes());
        }
        json_document(request.method(), document)
    })
    .await
}

/// `GET /mcp/{server}/.well-known/jwks.json`, what the metadata's `jwks_uri`
/// names: empty, since the gateway signs nothing here.
pub(crate) async fn jwks(
    AisixPath(server): AisixPath<String>,
    State(state): State<ProxyState>,
    request: Request,
) -> Response {
    observed(&state, None, request, async |state, request| {
        if sign_in_server(&state.snapshot.load(), &server).is_none() {
            return StatusCode::NOT_FOUND.into_response();
        }
        json_document(request.method(), json!({ "keys": [] }))
    })
    .await
}

/// `POST /mcp/{server}/oauth/register`. Nothing is stored: the client id only
/// has to come back on the later requests, and the rules for where a code may
/// go do not depend on anything a client registered.
pub(crate) async fn register(
    AisixPath(server): AisixPath<String>,
    State(state): State<ProxyState>,
    request: Request,
) -> Response {
    observed(&state, None, request, async |state, request| {
        if sign_in_server(&state.snapshot.load(), &server).is_none() {
            return StatusCode::NOT_FOUND.into_response();
        }
        let Ok(body) = to_bytes(request.into_body(), MAX_BODY_BYTES).await else {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_client_metadata",
                "the registration request could not be read",
            );
        };
        let Ok(metadata) = serde_json::from_slice::<serde_json::Value>(&body) else {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_client_metadata",
                "the registration request is not JSON",
            );
        };
        let mut response = (
            StatusCode::CREATED,
            Json(json!({
                "client_id": server,
                "redirect_uris": metadata.get("redirect_uris").cloned().unwrap_or(json!([])),
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
            })),
        )
            .into_response();
        no_store(&mut response);
        response
    })
    .await
}

/// `GET /mcp/{server}/oauth/authorize`. A problem with `redirect_uri` or
/// `client_id` is answered here, since nowhere safe to send the browser is
/// known yet; any other goes back to the client (RFC 6749 §4.1.2.1).
///
/// A client backend may request this URL with no cookie jar before the browser
/// does, so it answers the same both times and writes nothing about the flow:
/// all of it goes into the cookie this sets.
pub(crate) async fn authorize(
    AisixPath(server): AisixPath<String>,
    State(state): State<ProxyState>,
    request: Request,
) -> Response {
    observed(&state, None, request, async |state, request| {
        let snapshot = state.snapshot.load();
        let Some(oauth) = sign_in_server(&snapshot, &server) else {
            return StatusCode::NOT_FOUND.into_response();
        };
        let Some(base) = crate::mcp_auth::public_base(&snapshot) else {
            return failure_response(&snapshot, &server, UserOAuthFailure::Misconfigured);
        };
        let mode = sign_in_mode(&snapshot);
        if matches!(mode, SignInMode::Misconfigured) {
            return failure_response(&snapshot, &server, UserOAuthFailure::Misconfigured);
        }
        if request.uri().query().map_or(0, str::len) > MAX_QUERY_BYTES {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "the authorization request is too large",
            );
        }
        let params = query_params(request.uri());
        let param = |name: &str| {
            params
                .get(name)
                .map(String::as_str)
                .filter(|v| !v.is_empty())
        };
        let Some(client_id) = param("client_id") else {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "client_id is required",
            );
        };
        let Some(redirect_uri) = param("redirect_uri") else {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "redirect_uri is required",
            );
        };
        if let Err(reason) = validate_redirect_uri(redirect_uri) {
            tracing::warn!(target: "aisix::mcp_oauth", server = %server, reason,
                "refused an authorization request's redirect_uri");
            return oauth_error(StatusCode::BAD_REQUEST, "invalid_request", reason);
        }
        let state_param = param("state");
        let back = |error: &str, description: &str| {
            refuse_to_client(redirect_uri, state_param, error, description)
        };
        if param("response_type") != Some("code") {
            return back("unsupported_response_type", "response_type must be code");
        }
        let Some(code_challenge) = param("code_challenge") else {
            return back("invalid_request", "code_challenge is required");
        };
        if param("code_challenge_method") != Some("S256") {
            return back("invalid_request", "code_challenge_method must be S256");
        }
        let client = crate::mcp_oauth_login::ClientRequest {
            client_id,
            redirect_uri,
            state: state_param,
            code_challenge,
        };
        let SignInMode::Interactive(provider) = mode else {
            return crate::mcp_oauth_login::begin_relay(
                state,
                &snapshot,
                &oauth,
                &base,
                &client,
                param("scope"),
            )
            .await;
        };
        if param("resource").is_some_and(|resource| !resource_matches(resource, &base, &server)) {
            return back("invalid_target", "resource does not name this MCP server");
        }
        crate::mcp_oauth_login::begin_login(
            state,
            &snapshot,
            &oauth,
            &provider.value,
            &base,
            &client,
        )
        .await
    })
    .await
}

/// Whether `resource` (RFC 8707) names the server at `server`.
pub(crate) fn resource_matches(resource: &str, base: &str, server: &str) -> bool {
    resource.trim_end_matches('/') == scoped_url(base, server)
}

/// `POST /mcp/{server}/oauth/token`. In the relay mode the caller proves who it
/// is with its gateway credential, and the access token it gets back carries no
/// authority at all: the real tokens are stored here under that caller. In the
/// interactive mode no credential is involved: the code already says who.
pub(crate) async fn token(
    AisixPath(server): AisixPath<String>,
    State(state): State<ProxyState>,
    request: Request,
) -> Response {
    let mode = sign_in_mode(&state.snapshot.load());
    if !matches!(mode, SignInMode::Relay) {
        return observed(&state, None, request, async |state, request| {
            crate::mcp_oauth_login::token(state, &server, mode, request).await
        })
        .await;
    }
    // Denials are recorded by the credential extractor itself.
    let (mut parts, body) = request.into_parts();
    let auth = match AuthenticatedKey::from_request_parts(&mut parts, &state).await {
        Ok(auth) => auth,
        Err(err) => {
            crate::attribution::note_unmetered_route();
            // `invalid_client` tells a client to discard its registration, so
            // it is not for a gateway that merely cannot judge the credential.
            return if err.status().is_server_error() {
                oauth_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "temporarily_unavailable",
                    "the gateway could not verify the credential right now",
                )
            } else {
                oauth_error(
                    err.status(),
                    "invalid_client",
                    "send a valid gateway credential in x-aisix-api-key or Authorization",
                )
            };
        }
    };
    let request = Request::from_parts(parts, body);
    observed(&state, Some(&auth), request, async |state, request| {
        let Some(oauth) = sign_in_server(&state.snapshot.load(), &server) else {
            return StatusCode::NOT_FOUND.into_response();
        };
        let Some(subject) = caller_subject(&auth) else {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "this gateway credential has no user, so it cannot be used with an MCP server \
                 whose auth type signs in each user individually",
            );
        };
        let Ok(body) = to_bytes(request.into_body(), MAX_BODY_BYTES).await else {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "the token request could not be read",
            );
        };
        let form: HashMap<String, String> =
            url::form_urlencoded::parse(&body).into_owned().collect();
        let param = |name: &str| form.get(name).map(String::as_str).filter(|v| !v.is_empty());
        if param("grant_type") != Some("authorization_code") {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "unsupported_grant_type",
                "only authorization_code is supported",
            );
        }
        let (Some(code), Some(client_id), Some(redirect_uri), Some(code_verifier)) = (
            param("code"),
            param("client_id"),
            param("redirect_uri"),
            param("code_verifier"),
        ) else {
            return oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "code, client_id, redirect_uri and code_verifier are required",
            );
        };
        match state
            .mcp_oauth
            .redeem_code(
                &oauth,
                &subject,
                &RedeemRequest {
                    code,
                    client_id,
                    redirect_uri,
                    code_verifier,
                },
            )
            .await
        {
            Ok(()) => {
                let mut response = Json(json!({
                    "access_token": placeholder_token(),
                    "token_type": "Bearer",
                    "expires_in": PLACEHOLDER_EXPIRES_IN,
                }))
                .into_response();
                no_store(&mut response);
                response
            }
            Err(RedeemError::InvalidGrant) => oauth_error(
                StatusCode::BAD_REQUEST,
                "invalid_grant",
                "the authorization code is unknown, expired, already used, or does not match \
                 this request",
            ),
            Err(RedeemError::Unavailable) => oauth_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "temporarily_unavailable",
                "the sign-in could not be finished right now; sign in again",
            ),
            Err(RedeemError::Misconfigured) => oauth_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "this MCP server cannot sign in users as configured",
            ),
        }
    })
    .await
}

fn placeholder_token() -> String {
    format!("aisix-placeholder-{}", uuid::Uuid::new_v4().simple())
}

/// `DELETE /mcp/{server}/oauth/credential`: the caller removes its own
/// connection.
pub(crate) async fn delete_credential(
    AisixPath(server): AisixPath<String>,
    State(state): State<ProxyState>,
    request: Request,
) -> Response {
    let (mut parts, body) = request.into_parts();
    let auth = match AuthenticatedKey::from_request_parts(&mut parts, &state).await {
        Ok(auth) => auth,
        Err(err) => {
            crate::attribution::note_unmetered_route();
            return err.into_response();
        }
    };
    let request = Request::from_parts(parts, body);
    observed(&state, Some(&auth), request, async |state, _request| {
        let snapshot = state.snapshot.load();
        let Some(oauth) = sign_in_server(&snapshot, &server) else {
            return StatusCode::NOT_FOUND.into_response();
        };
        let Some(subject) = caller_subject(&auth) else {
            return failure_response(&snapshot, &server, UserOAuthFailure::NoSubject);
        };
        match state.mcp_oauth.disconnect(&oauth, &subject).await {
            Ok(()) => StatusCode::NO_CONTENT.into_response(),
            Err(failure) => failure_response(&snapshot, &server, failure),
        }
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(uri: &str) -> Result<(), &'static str> {
        validate_redirect_uri(uri)
    }

    #[test]
    fn loopback_on_any_port_and_either_scheme_is_accepted() {
        for uri in [
            "http://localhost:3334/cb",
            "http://LOCALHOST/cb",
            "http://127.0.0.1:9/cb",
            "http://127.9.9.9/cb",
            "https://[::1]:8443/cb",
            "http://[::ffff:127.0.0.1]/cb",
        ] {
            check(uri).unwrap_or_else(|e| panic!("{uri}: {e}"));
        }
    }

    #[test]
    fn any_https_is_accepted_and_http_only_for_loopback() {
        check("https://app.example.org/cb").unwrap();
        check("https://app.example.org:8443/cb?x=1").unwrap();
        check("http://app.example.org/cb").unwrap_err();
        check("http://gw.example.com/ui/callback").unwrap_err();
        check("http://localhost.example.org/cb").unwrap_err();
    }

    #[test]
    fn hygiene_rules_refuse_what_no_policy_should_accept() {
        for uri in [
            "not a url",
            "/relative",
            "http://localhost/cb#frag",
            "https://app.example.org/cb#frag",
            "http://user:pw@localhost/cb",
            "https://user@app.example.org/cb",
            "http://evil.example\\@localhost/cb",
            "javascript:alert(1)",
            "data:text/html,x",
            "myapp://callback",
        ] {
            check(uri).expect_err(uri);
        }
    }

    #[test]
    fn only_a_client_on_this_machine_hears_of_a_refusal_by_redirect() {
        for uri in [
            "http://127.0.0.1:9/cb",
            "https://localhost/cb",
            "cursor://anysphere.cursor-mcp/oauth/callback",
        ] {
            let response = refuse_to_client(uri, None, "invalid_request", "no");
            assert_eq!(response.status(), StatusCode::FOUND, "{uri}");
        }
        let response = refuse_to_client("https://evil.example/", None, "invalid_request", "no");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(!response.headers().contains_key(header::LOCATION));
    }

    #[test]
    fn only_a_known_native_uri_is_accepted_and_only_exactly() {
        check("cursor://anysphere.cursor-mcp/oauth/callback").unwrap();
        check("cursor://anysphere.cursor-mcp/oauth/callback?x=1").unwrap_err();
        check("cursor://anysphere.cursor-mcp/oauth/callback-2").unwrap_err();
    }

    #[test]
    fn sign_in_mode_follows_the_enabled_providers_with_login() {
        let provider = |id: &str, doc: serde_json::Value| {
            ResourceEntry::new(id, serde_json::from_value(doc).unwrap(), 1)
        };
        let usable = |name: &str| {
            json!({"name": name, "issuer": format!("https://{name}"), "audiences": ["a"],
                   "login": {"client_id": "gw"}})
        };
        let mode = |docs: Vec<serde_json::Value>| {
            let snapshot = AisixSnapshot::new();
            for (i, doc) in docs.into_iter().enumerate() {
                snapshot
                    .oidc_providers
                    .insert(provider(&format!("p{i}"), doc));
            }
            match sign_in_mode(&snapshot) {
                SignInMode::Relay => "relay",
                SignInMode::Interactive(_) => "interactive",
                SignInMode::Misconfigured => "misconfigured",
            }
        };
        assert_eq!(mode(vec![]), "relay");
        assert_eq!(
            mode(vec![
                json!({"name": "a", "issuer": "https://a", "audiences": ["a"]})
            ]),
            "relay"
        );
        assert_eq!(mode(vec![usable("a")]), "interactive");
        assert_eq!(mode(vec![usable("a"), usable("b")]), "misconfigured");
        let mut disabled = usable("b");
        disabled["enabled"] = json!(false);
        assert_eq!(mode(vec![usable("a"), disabled.clone()]), "interactive");
        assert_eq!(mode(vec![disabled]), "relay");
        assert_eq!(
            mode(vec![json!({"name": "h", "issuer": "https://h",
                             "hmac_secret": "x".repeat(32),
                             "login": {"client_id": "gw"}})]),
            "misconfigured"
        );
    }

    #[test]
    fn a_placeholder_is_not_shaped_like_a_jwt() {
        let token = placeholder_token();
        assert!(!token.contains('.'));
        assert_ne!(token, placeholder_token());
    }
}
