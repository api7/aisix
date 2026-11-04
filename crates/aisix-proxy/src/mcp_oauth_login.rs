//! The browser half of a sign-in for `auth_type: oauth2_authorization_code`
//! MCP servers, and the session tokens a keyless client ends up with.
//!
//! Nothing about a flow in progress is stored. A flow's state lives in an
//! encrypted cookie named after its handle, the same value the identity
//! provider and the upstream carry as `state`, and every step accepts only the
//! stage before it. Whichever mode a flow runs in, the browser stops at a
//! consent page before a code goes to the client: `redirect_uri` is any https
//! URL, so that page is what stands between a user who followed someone's link
//! and a code delivered to that someone.
//!
//! In the interactive mode (see [`crate::mcp_oauth::sign_in_mode`]) the one
//! browser pass goes:
//!
//! 1. `authorize` to the trust provider's login,
//! 2. back to [`login_callback`], which turns the ID token into an identity by
//!    the same steps as a JWT caller,
//! 3. to the upstream's sign-in unless this identity already has a working
//!    credential there, back through [`callback`],
//! 4. the consent page, and [`consent`] sends a code to the client.
//!
//! The client then holds a session token: the gateway's own, sealed so that no
//! store is read to accept it. [`authenticate_session`] re-runs the checks that
//! depend on configuration against what the token recorded at sign-in, so a
//! key, mapping or provider change applies on the next request, and answers as
//! the same change answers a JWT caller. A key's access to the server is decided
//! at sign-in and, after that, by the MCP layer as for any caller.

// A step's `Err` is the page it ends the flow with.
#![allow(clippy::result_large_err)]

use std::sync::Arc;

use aisix_core::models::OidcProvider;
use aisix_core::resource::ResourceEntry;
use aisix_core::AisixSnapshot;
use aisix_mcp::user_oauth::{
    normalized_issuer, pkce_pair, random_token, FlowStage, FlowState, LoginCode, LoginExchange,
    RedeemError, RedeemRequest, SessionIdentity, SessionKeys, UpstreamLeg, ACCESS_TOKEN_PREFIX,
    ACCESS_TOKEN_TTL_SECS, FLOW_TTL_SECS,
};
use aisix_mcp::{UserOAuthFailure, UserOAuthServer, UserSubject};
use axum::body::to_bytes;
use axum::extract::{Request, State};
use axum::http::request::Parts;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use crate::auth::{AuthenticatedKey, JwtIdentity};
use crate::error::ProxyError;
use crate::mcp_oauth::{
    callback_url, failure_response, no_store, oauth_error, observed, query_params, redirect,
    redirect_to_client, refuse_to_client, resource_matches, sign_in_server, SignInMode,
};
use crate::state::ProxyState;

/// Browsers silently drop a cookie past 4096 bytes, name and attributes
/// included, which would surface only as a sign-in that cannot be finished.
const MAX_FLOW_COOKIE_BYTES: usize = 4000;
const FLOW_COOKIE_PREFIX: &str = "mcp_oauth_flow_";
const MAX_BODY_BYTES: usize = 64 * 1024;

fn login_callback_url(base: &str) -> String {
    format!("{base}/mcp-oauth/login/callback")
}

/// What a client asked for at `authorize`, already validated.
pub(crate) struct ClientRequest<'a> {
    pub client_id: &'a str,
    pub redirect_uri: &'a str,
    pub state: Option<&'a str>,
    pub code_challenge: &'a str,
}

fn expiry() -> i64 {
    chrono::Utc::now().timestamp() + FLOW_TTL_SECS
}

fn new_flow(oauth: &UserOAuthServer, client: &ClientRequest<'_>, stage: FlowStage) -> FlowState {
    FlowState {
        handle: random_token(),
        expires_at: expiry(),
        server_id: oauth.id.clone(),
        server_name: oauth.name.clone(),
        client_id: client.client_id.to_owned(),
        redirect_uri: client.redirect_uri.to_owned(),
        client_state: client.state.map(str::to_owned),
        code_challenge: client.code_challenge.to_owned(),
        csrf: random_token(),
        stage,
    }
}

// ── Cookie ──────────────────────────────────────────────────────────────────

fn cookie_attributes(base: &str) -> String {
    let url = url::Url::parse(base);
    let path = url.as_ref().map_or("/", |url| url.path());
    let secure = url.as_ref().is_ok_and(|url| url.scheme() == "https");
    format!(
        "Path={path}; HttpOnly; SameSite=Lax{}",
        if secure { "; Secure" } else { "" }
    )
}

/// The `Set-Cookie` carrying `flow`, or the page explaining why it cannot be
/// carried.
fn flow_cookie(base: &str, keys: &SessionKeys, flow: &FlowState) -> Result<HeaderValue, Response> {
    let name = format!("{FLOW_COOKIE_PREFIX}{}", flow.handle);
    let cookie = format!(
        "{name}={}; Max-Age={FLOW_TTL_SECS}; {}",
        keys.seal_flow(flow),
        cookie_attributes(base)
    );
    if cookie.len() > MAX_FLOW_COOKIE_BYTES {
        tracing::warn!(target: "aisix::mcp_oauth", server = %flow.server_name,
            bytes = cookie.len(), "a sign-in does not fit in its flow cookie");
        return Err(error_page(
            StatusCode::BAD_REQUEST,
            "This sign-in is too large",
            "What this sign-in has to remember does not fit in a browser cookie. The likely \
             causes are a very long state or redirect address from the MCP client, or a very \
             large identity token.",
        ));
    }
    HeaderValue::from_str(&cookie).map_err(|_| {
        error_page(
            StatusCode::INTERNAL_SERVER_ERROR,
            "This sign-in cannot continue",
            "The gateway could not write the cookie this sign-in depends on.",
        )
    })
}

fn clear_flow_cookie(base: &str, handle: &str, response: &mut Response) {
    let cookie = format!(
        "{FLOW_COOKIE_PREFIX}{handle}=; Max-Age=0; {}",
        cookie_attributes(base)
    );
    if let Ok(value) = HeaderValue::from_str(&cookie) {
        response.headers_mut().append(header::SET_COOKIE, value);
    }
}

fn with_cookie(mut response: Response, cookie: HeaderValue) -> Response {
    response.headers_mut().append(header::SET_COOKIE, cookie);
    response
}

fn cookie_value<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(';'))
        .find_map(|pair| {
            let (key, value) = pair.split_once('=')?;
            (key.trim() == name).then(|| value.trim())
        })
}

/// The flow `handle` names, from this browser's cookie. Everything that is not
/// a live flow of this browser reads the same.
async fn read_flow(
    state: &ProxyState,
    headers: &HeaderMap,
    handle: Option<&str>,
) -> Result<(FlowState, Arc<SessionKeys>), Response> {
    let unknown = || {
        error_page(
            StatusCode::BAD_REQUEST,
            "This sign-in has expired",
            "This browser holds no sign-in in progress for this link. It may have taken longer \
             than ten minutes, or cookies may be blocked. Start again from the MCP client.",
        )
    };
    let handle = handle.ok_or_else(unknown)?;
    let is_handle = handle.len() == 64 && handle.bytes().all(|b| b.is_ascii_hexdigit());
    if !is_handle {
        return Err(unknown());
    }
    let value =
        cookie_value(headers, &format!("{FLOW_COOKIE_PREFIX}{handle}")).ok_or_else(unknown)?;
    let keys = match state.mcp_oauth.session_keys().await {
        Ok(Some(keys)) => keys,
        Ok(None) => return Err(unknown()),
        Err(_) => return Err(unavailable_page()),
    };
    let flow = keys.open_flow(handle, value).ok_or_else(unknown)?;
    Ok((flow, keys))
}

// ── Pages ───────────────────────────────────────────────────────────────────

fn escape_html(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

fn html(status: StatusCode, title: &str, body: &str) -> Response {
    let page = format!(
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>{}</title>\
         <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><style>\
         body{{font-family:system-ui,sans-serif;max-width:32rem;margin:4rem auto;padding:0 1rem;\
         color:#1f2328}}h1{{font-size:1.25rem}}dt{{font-weight:600;margin-top:.75rem}}\
         dd{{margin:0;word-break:break-all}}button{{margin-top:1.5rem;padding:.5rem 1.5rem;\
         font-size:1rem}}button+button{{margin-left:.5rem}}</style></head><body><h1>{}</h1>{body}</body></html>",
        escape_html(title),
        escape_html(title),
    );
    let mut response = (status, page).into_response();
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/html; charset=utf-8"),
    );
    headers.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    no_store(&mut response);
    response
}

fn error_page(status: StatusCode, title: &str, reason: &str) -> Response {
    html(status, title, &format!("<p>{}</p>", escape_html(reason)))
}

/// A sign-in refused for who it turned out to be, naming that identity so the
/// person can act on it. A token with no usable identity claim names nobody.
fn refusal_page(identity: Option<&str>, reason: &str) -> Response {
    const TITLE: &str = "You cannot sign in here";
    let Some(identity) = identity else {
        return error_page(StatusCode::FORBIDDEN, TITLE, reason);
    };
    let body = format!(
        "<p>{}</p><dl><dt>Signed in as</dt><dd id=\"identity\">{}</dd></dl>\
         <p>Contact your administrator with this identity, or, if you used the wrong account, \
         sign out of the identity provider and start again from the MCP client.</p>",
        escape_html(reason),
        escape_html(identity),
    );
    html(StatusCode::FORBIDDEN, TITLE, &body)
}

fn unavailable_page() -> Response {
    error_page(
        StatusCode::SERVICE_UNAVAILABLE,
        "Sign-in is temporarily unavailable",
        "The gateway could not finish this sign-in right now. Try again from the MCP client.",
    )
}

fn misconfigured_page() -> Response {
    error_page(
        StatusCode::INTERNAL_SERVER_ERROR,
        "This MCP server cannot sign in users as configured",
        "An administrator needs to check the identity provider and MCP server settings.",
    )
}

/// Where a code would go: the scheme and host of `redirect_uri`, the part a
/// user can judge.
fn redirect_target(redirect_uri: &str) -> String {
    let Ok(url) = url::Url::parse(redirect_uri) else {
        return redirect_uri.to_owned();
    };
    let host = url.host_str().unwrap_or_default();
    match url.port() {
        Some(port) => format!("{}://{host}:{port}", url.scheme()),
        None => format!("{}://{host}", url.scheme()),
    }
}

fn consent_page(base: &str, flow: &FlowState, identity: Option<&str>) -> Response {
    let identity = identity
        .map(|identity| {
            format!(
                "<dt>Signed in as</dt><dd id=\"identity\">{}</dd>",
                escape_html(identity)
            )
        })
        .unwrap_or_default();
    let body = format!(
        "<p>An application is asking to use this MCP server on your behalf. Allow it only if you \
         started this and you recognise where it is going.</p><dl>\
         <dt>MCP server</dt><dd id=\"server\">{server}</dd>{identity}\
         <dt>Application address</dt><dd id=\"target\">{target}</dd></dl>\
         <form method=\"post\" action=\"{action}\">\
         <input type=\"hidden\" name=\"flow\" value=\"{handle}\">\
         <input type=\"hidden\" name=\"csrf\" value=\"{csrf}\">\
         <button type=\"submit\" name=\"decision\" value=\"allow\">Allow</button>\
         <button type=\"submit\" name=\"decision\" value=\"cancel\">Cancel</button></form>",
        server = escape_html(&flow.server_name),
        target = escape_html(&redirect_target(&flow.redirect_uri)),
        action = escape_html(&format!("{base}/mcp-oauth/consent")),
        handle = escape_html(&flow.handle),
        csrf = escape_html(&flow.csrf),
    );
    html(StatusCode::OK, "Allow access to this MCP server?", &body)
}

/// Move `flow` to the consent stage and show the page.
fn to_consent(
    base: &str,
    keys: &SessionKeys,
    mut flow: FlowState,
    stage: FlowStage,
) -> Result<Response, Response> {
    flow.stage = stage;
    flow.expires_at = expiry();
    let cookie = flow_cookie(base, keys, &flow)?;
    Ok(with_cookie(render_consent(base, &flow), cookie))
}

fn render_consent(base: &str, flow: &FlowState) -> Response {
    let identity = match &flow.stage {
        FlowStage::Consent {
            signed_in: Some(signed_in),
            ..
        } => Some(signed_in.identity.as_str()),
        _ => None,
    };
    consent_page(base, flow, identity)
}

fn back_to_client(base: &str, flow: &FlowState, params: &[(&str, &str)]) -> Response {
    let mut response = redirect_to_client(&flow.redirect_uri, flow.client_state.as_deref(), params);
    clear_flow_cookie(base, &flow.handle, &mut response);
    response
}

fn or_page(result: Result<Response, Response>) -> Response {
    result.unwrap_or_else(|page| page)
}

// ── Starting a flow ─────────────────────────────────────────────────────────

/// The relay mode's `authorize`: straight to the upstream, carrying the
/// client's own PKCE challenge.
pub(crate) async fn begin_relay(
    state: &ProxyState,
    snapshot: &AisixSnapshot,
    oauth: &UserOAuthServer,
    base: &str,
    client: &ClientRequest<'_>,
    scope: Option<&str>,
) -> Response {
    let keys = match state.mcp_oauth.session_keys_for_sign_in().await {
        Ok(keys) => keys,
        Err(failure) => return failure_response(snapshot, &oauth.name, failure),
    };
    let handle = random_token();
    let upstream = state
        .mcp_oauth
        .begin_upstream(
            oauth,
            &callback_url(base),
            &handle,
            client.code_challenge,
            scope,
        )
        .await;
    let (location, leg) = match upstream {
        Ok(started) => started,
        Err(failure) => {
            let (error, description) = if failure == UserOAuthFailure::Unavailable {
                (
                    "temporarily_unavailable",
                    "the gateway could not start the sign-in right now",
                )
            } else {
                (
                    "server_error",
                    "this MCP server cannot sign in users as configured",
                )
            };
            return refuse_to_client(client.redirect_uri, client.state, error, description);
        }
    };
    let mut flow = new_flow(
        oauth,
        client,
        FlowStage::Upstream {
            signed_in: None,
            verifier: None,
            leg,
        },
    );
    flow.handle = handle;
    match flow_cookie(base, &keys, &flow) {
        Ok(cookie) => with_cookie(redirect(&location), cookie),
        Err(page) => page,
    }
}

/// The interactive mode's `authorize`: to the trust provider's login.
pub(crate) async fn begin_login(
    state: &ProxyState,
    snapshot: &AisixSnapshot,
    oauth: &UserOAuthServer,
    provider: &OidcProvider,
    base: &str,
    client: &ClientRequest<'_>,
) -> Response {
    let (Some(login), Some(issuer)) = (&provider.login, &provider.issuer) else {
        return failure_response(snapshot, &oauth.name, UserOAuthFailure::Misconfigured);
    };
    let keys = match state.mcp_oauth.session_keys_for_sign_in().await {
        Ok(keys) => keys,
        Err(failure) => return failure_response(snapshot, &oauth.name, failure),
    };
    let endpoints = match state
        .mcp_oauth
        .login_endpoints(&provider.name, issuer)
        .await
    {
        Ok(endpoints) => endpoints,
        Err(failure) => return failure_response(snapshot, &oauth.name, failure),
    };
    let Some(mut location) = url::Url::parse(&endpoints.authorization_endpoint)
        .ok()
        .filter(|url| matches!(url.scheme(), "http" | "https"))
    else {
        return failure_response(snapshot, &oauth.name, UserOAuthFailure::Misconfigured);
    };
    let (verifier, challenge) = pkce_pair();
    let nonce = random_token();
    let mut scopes = vec!["openid".to_owned()];
    for scope in login.scopes.iter().chain(&provider.required_scopes) {
        if !scopes.contains(scope) {
            scopes.push(scope.clone());
        }
    }
    let flow = new_flow(
        oauth,
        client,
        FlowStage::ProviderLogin {
            provider: provider.name.clone(),
            verifier,
            nonce: nonce.clone(),
        },
    );
    location
        .query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", &login.client_id)
        .append_pair("redirect_uri", &login_callback_url(base))
        .append_pair("scope", &scopes.join(" "))
        .append_pair("state", &flow.handle)
        .append_pair("nonce", &nonce)
        .append_pair("code_challenge", &challenge)
        .append_pair("code_challenge_method", "S256");
    match flow_cookie(base, &keys, &flow) {
        Ok(cookie) => with_cookie(redirect(location.as_str()), cookie),
        Err(page) => page,
    }
}

// ── The trust provider's redirect back ──────────────────────────────────────

/// `GET <base>/mcp-oauth/login/callback`, the URI registered at the trust
/// provider.
pub(crate) async fn login_callback(State(state): State<ProxyState>, request: Request) -> Response {
    observed(&state, None, request, async |state, request| {
        or_page(handle_login_callback(state, request.into_parts().0).await)
    })
    .await
}

async fn handle_login_callback(state: &ProxyState, request: Parts) -> Result<Response, Response> {
    let params = query_params(&request.uri);
    let param = |name: &str| {
        params
            .get(name)
            .map(String::as_str)
            .filter(|v| !v.is_empty())
    };
    let (flow, keys) = read_flow(state, &request.headers, param("state")).await?;
    let snapshot = state.snapshot.load();
    let base = crate::mcp_auth::public_base(&snapshot).ok_or_else(misconfigured_page)?;
    let (provider_name, verifier, nonce) =
        match &flow.stage {
            FlowStage::ProviderLogin {
                provider,
                verifier,
                nonce,
            } => (provider.clone(), verifier.clone(), nonce.clone()),
            // A reload of the page this answered with.
            FlowStage::Consent { .. } => return Ok(render_consent(&base, &flow)),
            FlowStage::Upstream { .. } => return Err(error_page(
                StatusCode::BAD_REQUEST,
                "This sign-in has moved on",
                "This link belongs to an earlier step of the sign-in. Start again from the MCP \
                 client.",
            )),
        };
    let oauth = server_of(&snapshot, &flow).ok_or_else(misconfigured_page)?;
    let provider = snapshot
        .oidc_providers
        .get_by_name(&provider_name)
        .filter(|entry| entry.value.enabled)
        .ok_or_else(misconfigured_page)?;
    let provider = &provider.value;
    let (Some(login), Some(issuer)) = (&provider.login, &provider.issuer) else {
        return Err(misconfigured_page());
    };

    // RFC 9207: a provider that says who it is must be the one this flow went to.
    if param("iss").is_some_and(|iss| !same_issuer(iss, issuer)) {
        tracing::warn!(target: "aisix::mcp_oauth", provider = %provider.name,
            "an authentication response named an unexpected issuer");
        return Err(error_page(
            StatusCode::BAD_REQUEST,
            "This sign-in was refused",
            "The response came from a different identity provider than this sign-in went to.",
        ));
    }
    if let Some(error) = param("error") {
        let mut relayed = vec![("error", error)];
        if let Some(description) = param("error_description") {
            relayed.push(("error_description", description));
        }
        return Ok(back_to_client(&base, &flow, &relayed));
    }
    let code = param("code").ok_or_else(|| {
        error_page(
            StatusCode::BAD_REQUEST,
            "This sign-in was refused",
            "The identity provider's response carried no authorization code.",
        )
    })?;

    let endpoints = state
        .mcp_oauth
        .login_endpoints(&provider.name, issuer)
        .await
        .map_err(failure_page)?;
    let exchanged = state
        .mcp_oauth
        .exchange_login_code(&LoginCode {
            provider: &provider.name,
            token_endpoint: &endpoints.token_endpoint,
            client_id: &login.client_id,
            client_secret: login.client_secret.as_ref().map(|s| s.expose()),
            code,
            redirect_uri: &login_callback_url(&base),
            code_verifier: &verifier,
        })
        .await;
    let (id_token, granted_scope) = match exchanged {
        LoginExchange::Issued { id_token, scope } => (id_token, scope),
        LoginExchange::Rejected(error) => {
            tracing::warn!(target: "aisix::mcp_oauth", provider = %provider.name, %error,
                "the trust provider refused a sign-in's authorization code");
            return Err(error_page(
                StatusCode::BAD_GATEWAY,
                "This sign-in was refused",
                "The identity provider refused the sign-in. Start again from the MCP client, and \
                 if it keeps happening an administrator needs to check the gateway's client at \
                 the identity provider.",
            ));
        }
        LoginExchange::NoIdToken => {
            tracing::warn!(target: "aisix::mcp_oauth", provider = %provider.name,
                "the trust provider answered a sign-in without an ID token");
            return Err(misconfigured_page());
        }
        LoginExchange::Failed(failure) => return Err(failure_page(failure)),
    };
    let refused =
        |reason: &str| error_page(StatusCode::FORBIDDEN, "You cannot sign in here", reason);
    let claims = crate::jwt::verify_login_id_token(provider, &id_token, &login.client_id)
        .await
        .map_err(|reason| {
            state
                .metrics
                .record_auth_decision("mcp_oauth_login", false, reason);
            tracing::warn!(target: "aisix::mcp_oauth", provider = %provider.name, reason,
                "refused a sign-in's ID token");
            if reason == "jwks_unavailable" {
                unavailable_page()
            } else {
                refused("The identity provider's token could not be verified.")
            }
        })?;
    if claims.get("nonce").and_then(|v| v.as_str()) != Some(nonce.as_str()) {
        state
            .metrics
            .record_auth_decision("mcp_oauth_login", false, "nonce_mismatch");
        return Err(refused(
            "The identity provider's token belongs to another sign-in.",
        ));
    }
    // RFC 6749 §5.1: a response without `scope` granted what was asked, and
    // these were asked.
    if let Some(granted) = &granted_scope {
        let granted: Vec<&str> = granted.split_whitespace().collect();
        if !provider
            .required_scopes
            .iter()
            .all(|required| granted.contains(&required.as_str()))
        {
            state
                .metrics
                .record_auth_decision("mcp_oauth_login", false, "jwt_scope_missing");
            return Err(refused(
                "The identity provider did not grant the access this gateway requires.",
            ));
        }
    }

    // On what the session will hold, so a sign-in cannot pass and every later
    // request fail on a claim the session does not keep.
    let session_claims = crate::jwt::session_claims(&claims);
    let signed_in_as = crate::jwt::identity_claim(&session_claims, provider);
    let deny = |reason: &'static str| {
        state
            .metrics
            .record_auth_decision("mcp_oauth_login", false, reason);
        tracing::warn!(target: "aisix::mcp_oauth", provider = %provider.name, reason,
            subject = ?signed_in_as.map(crate::jwt::clip), "refused a sign-in");
        refusal_page(signed_in_as, denial_text(reason))
    };
    let principal = session_principal(state, &snapshot, provider, &session_claims)
        .map_err(|d| deny(d.reason))?;
    let acl = aisix_mcp::ToolAcl::resolve(&snapshot, &state.mcp_servers, principal.auth.key());
    if !acl.may_reach_server(&oauth.name, &oauth.id) {
        return Err(deny("mcp_acl_denied"));
    }
    state
        .metrics
        .record_auth_decision("mcp_oauth_login", true, "");
    let subject = UserSubject::jwt(
        &provider.name,
        provider.issuer.as_deref(),
        &provider.identity_claim,
        &principal.subject,
    );
    let identity = SessionIdentity {
        provider: provider.name.clone(),
        identity: principal.subject,
        claims: session_claims,
    };
    connect_upstream(state, &oauth, &base, &keys, flow, identity, &subject).await
}

/// What a refused person is told, in terms of what they can act on.
fn denial_text(reason: &str) -> &'static str {
    match reason {
        "jwt_bound_claim_mismatch" => {
            "Your account does not meet this gateway's sign-in requirements."
        }
        "mcp_acl_denied" => "Your account is not allowed to use this MCP server.",
        "key_disabled" | "key_expired" => {
            "The gateway access assigned to your account is disabled or has expired."
        }
        _ => "Your account is not mapped to any access on this gateway.",
    }
}

fn failure_page(failure: UserOAuthFailure) -> Response {
    match failure {
        UserOAuthFailure::Unavailable => unavailable_page(),
        _ => misconfigured_page(),
    }
}

fn server_of(snapshot: &AisixSnapshot, flow: &FlowState) -> Option<UserOAuthServer> {
    let entry = snapshot.mcp_servers.get_by_id(&flow.server_id)?;
    sign_in_server(snapshot, &entry.value.name)
}

/// With the identity known, make sure it has a working upstream credential:
/// use it, refresh it, or send the user to get one.
async fn connect_upstream(
    state: &ProxyState,
    oauth: &UserOAuthServer,
    base: &str,
    keys: &SessionKeys,
    mut flow: FlowState,
    identity: SessionIdentity,
    subject: &UserSubject,
) -> Result<Response, Response> {
    match state.mcp_oauth.stored_access_token(oauth, subject).await {
        Ok(_) => {
            return to_consent(
                base,
                keys,
                flow,
                FlowStage::Consent {
                    signed_in: Some(identity),
                    relayed: None,
                },
            )
        }
        Err(UserOAuthFailure::AuthorizationRequired) => {}
        Err(failure) => return Err(failure_page(failure)),
    }
    let (verifier, challenge) = pkce_pair();
    let (location, leg) = state
        .mcp_oauth
        .begin_upstream(oauth, &callback_url(base), &flow.handle, &challenge, None)
        .await
        .map_err(failure_page)?;
    flow.stage = FlowStage::Upstream {
        signed_in: Some(identity),
        verifier: Some(verifier),
        leg,
    };
    flow.expires_at = expiry();
    let cookie = flow_cookie(base, keys, &flow)?;
    Ok(with_cookie(redirect(&location), cookie))
}

// ── The upstream's redirect back ────────────────────────────────────────────

/// `GET <base>/mcp-oauth/callback`, where the upstream sends the browser back.
/// The upstream's code goes no further than this.
pub(crate) async fn callback(State(state): State<ProxyState>, request: Request) -> Response {
    observed(&state, None, request, async |state, request| {
        or_page(handle_callback(state, request.into_parts().0).await)
    })
    .await
}

async fn handle_callback(state: &ProxyState, request: Parts) -> Result<Response, Response> {
    let params = query_params(&request.uri);
    let param = |name: &str| {
        params
            .get(name)
            .map(String::as_str)
            .filter(|v| !v.is_empty())
    };
    let (mut flow, keys) = read_flow(state, &request.headers, param("state")).await?;
    let snapshot = state.snapshot.load();
    let base = crate::mcp_auth::public_base(&snapshot).ok_or_else(misconfigured_page)?;
    let stage = std::mem::replace(
        &mut flow.stage,
        FlowStage::Consent {
            signed_in: None,
            relayed: None,
        },
    );
    let (signed_in, verifier, leg): (_, _, UpstreamLeg) =
        match stage {
            FlowStage::Upstream {
                signed_in,
                verifier,
                leg,
            } => (signed_in, verifier, leg),
            // A reload of the page this answered with.
            stage @ FlowStage::Consent { .. } => {
                flow.stage = stage;
                return Ok(render_consent(&base, &flow));
            }
            FlowStage::ProviderLogin { .. } => return Err(error_page(
                StatusCode::BAD_REQUEST,
                "This sign-in has moved on",
                "This link does not belong to the current step of the sign-in. Start again from \
                 the MCP client.",
            )),
        };
    // RFC 9207: an authorization server that says who it is must be the one
    // this flow was sent to.
    if param("iss").is_some_and(|iss| !leg.issuer_matches(iss)) {
        tracing::warn!(target: "aisix::mcp_oauth", server = %flow.server_name,
            "an authorization response named an unexpected issuer");
        return Ok(back_to_client(
            &base,
            &flow,
            &[
                ("error", "access_denied"),
                (
                    "error_description",
                    "the response came from an unexpected authorization server",
                ),
            ],
        ));
    }
    if let Some(error) = param("error") {
        let mut relayed = vec![("error", error)];
        if let Some(description) = param("error_description") {
            relayed.push(("error_description", description));
        }
        return Ok(back_to_client(&base, &flow, &relayed));
    }
    let Some(upstream_code) = param("code") else {
        return Ok(back_to_client(
            &base,
            &flow,
            &[
                ("error", "server_error"),
                (
                    "error_description",
                    "the authorization response carried no code",
                ),
            ],
        ));
    };
    let (Some(signed_in), Some(verifier)) = (signed_in, verifier) else {
        let stage = FlowStage::Consent {
            signed_in: None,
            relayed: Some(leg.relay(upstream_code)),
        };
        return to_consent(&base, &keys, flow, stage);
    };

    // This person signed in at the provider this name meant then. By now it may
    // mean another one, whose record this credential must not land in.
    let Some(provider) = snapshot
        .oidc_providers
        .get_by_name(&signed_in.provider)
        .filter(|provider| session_issuer_is_current(&provider.value, &signed_in.claims))
    else {
        state
            .metrics
            .record_auth_decision("mcp_oauth_login", false, SESSION_ISSUER_CHANGED);
        tracing::warn!(target: "aisix::mcp_oauth", server = %flow.server_name,
            provider = %signed_in.provider,
            "the identity provider changed during a sign-in; no upstream credential was stored");
        return Ok(back_to_client(
            &base,
            &flow,
            &[
                ("error", "access_denied"),
                (
                    "error_description",
                    "the identity provider changed during the sign-in",
                ),
            ],
        ));
    };
    // Likewise a claim that moved: this value is what the old claim said, and the
    // new claim's record of that value is another person's. Without this the code
    // the flow would go on to issue is refused at the token step anyway, by then
    // with the credential stored.
    if crate::jwt::identity_claim(&signed_in.claims, &provider.value)
        != Some(signed_in.identity.as_str())
    {
        state
            .metrics
            .record_auth_decision("mcp_oauth_login", false, SESSION_IDENTITY_CHANGED);
        tracing::warn!(target: "aisix::mcp_oauth", server = %flow.server_name,
            provider = %signed_in.provider,
            "the identity claim changed during a sign-in; no upstream credential was stored");
        return Ok(back_to_client(
            &base,
            &flow,
            &[
                ("error", "access_denied"),
                (
                    "error_description",
                    "the identity claim changed during the sign-in",
                ),
            ],
        ));
    }
    let oauth = server_of(&snapshot, &flow).ok_or_else(misconfigured_page)?;
    let subject = UserSubject::jwt(
        &signed_in.provider,
        provider.value.issuer.as_deref(),
        &provider.value.identity_claim,
        &signed_in.identity,
    );
    state
        .mcp_oauth
        .exchange_upstream_code(&oauth, &subject, upstream_code, leg, &verifier)
        .await
        .map_err(|error| match error {
            RedeemError::InvalidGrant => error_page(
                StatusCode::BAD_GATEWAY,
                "This sign-in was refused",
                "The MCP server's authorization server refused the sign-in. Start again from the \
                 MCP client.",
            ),
            RedeemError::Unavailable => unavailable_page(),
            RedeemError::Misconfigured => misconfigured_page(),
        })?;
    to_consent(
        &base,
        &keys,
        flow,
        FlowStage::Consent {
            signed_in: Some(signed_in),
            relayed: None,
        },
    )
}

// ── Consent ─────────────────────────────────────────────────────────────────

/// `POST <base>/mcp-oauth/consent`: the user's `Allow` or `Cancel`. Only an
/// `Allow` makes a code exist for the client to receive.
pub(crate) async fn consent(State(state): State<ProxyState>, request: Request) -> Response {
    observed(&state, None, request, async |state, request| {
        or_page(handle_consent(state, request).await)
    })
    .await
}

async fn handle_consent(state: &ProxyState, request: Request) -> Result<Response, Response> {
    let (parts, body) = request.into_parts();
    let body = to_bytes(body, MAX_BODY_BYTES).await.map_err(|_| {
        error_page(
            StatusCode::BAD_REQUEST,
            "This sign-in was refused",
            "The form could not be read.",
        )
    })?;
    let form: std::collections::HashMap<String, String> =
        url::form_urlencoded::parse(&body).into_owned().collect();
    let (flow, _keys) =
        read_flow(state, &parts.headers, form.get("flow").map(String::as_str)).await?;
    let snapshot = state.snapshot.load();
    let base = crate::mcp_auth::public_base(&snapshot).ok_or_else(misconfigured_page)?;
    if form.get("csrf").map(String::as_str) != Some(flow.csrf.as_str()) {
        return Err(error_page(
            StatusCode::FORBIDDEN,
            "This sign-in was refused",
            "This form did not come from the page this browser was shown. Start again from the \
             MCP client.",
        ));
    }
    if !matches!(flow.stage, FlowStage::Consent { .. }) {
        return Err(error_page(
            StatusCode::BAD_REQUEST,
            "This sign-in is not finished",
            "There is nothing to allow yet. Start again from the MCP client.",
        ));
    }
    let redirect_uri = flow.redirect_uri.clone();
    let client_state = flow.client_state.clone();
    let handle = flow.handle.clone();
    let (outcome, value) = match form.get("decision").map(String::as_str) {
        Some("allow") => (
            "code",
            state
                .mcp_oauth
                .issue_code(flow)
                .await
                .map_err(failure_page)?,
        ),
        Some("cancel") => ("error", "access_denied".to_owned()),
        _ => {
            return Err(error_page(
                StatusCode::BAD_REQUEST,
                "This sign-in was refused",
                "The form did not say whether to allow or cancel.",
            ))
        }
    };
    let mut url = url::Url::parse(&redirect_uri).map_err(|_| misconfigured_page())?;
    {
        let mut query = url.query_pairs_mut();
        query.append_pair(outcome, &value);
        if let Some(state) = &client_state {
            query.append_pair("state", state);
        }
    }
    let Ok(location) = HeaderValue::from_str(url.as_str()) else {
        return Err(misconfigured_page());
    };
    let mut response = StatusCode::SEE_OTHER.into_response();
    response.headers_mut().insert(header::LOCATION, location);
    no_store(&mut response);
    clear_flow_cookie(&base, &handle, &mut response);
    Ok(response)
}

// ── Who a session is ────────────────────────────────────────────────────────

const SESSION_ISSUER_CHANGED: &str = "session_issuer_changed";
const SESSION_IDENTITY_CHANGED: &str = "session_identity_changed";

/// Two spellings of one issuer, which may differ by a trailing slash.
fn same_issuer(a: &str, b: &str) -> bool {
    normalized_issuer(a) == normalized_issuer(b)
}

/// Whether the identity provider that signed this session in is still the one
/// its provider name points at. The name alone cannot say: repointed at another
/// identity provider, it can give the same identity value to another person,
/// who must not inherit this session or be reached through it.
fn session_issuer_is_current(provider: &OidcProvider, claims: &serde_json::Value) -> bool {
    let recorded = claims.get("iss").and_then(serde_json::Value::as_str);
    match (recorded, provider.issuer.as_deref()) {
        (Some(recorded), Some(current)) => same_issuer(recorded, current),
        _ => false,
    }
}

struct Principal {
    subject: String,
    auth: AuthenticatedKey,
}

/// The steps after signature verification that every session passes, at
/// sign-in on the ID token's claims and later on the token's record of them:
/// bound claims and the key they map to.
fn session_principal(
    state: &ProxyState,
    snapshot: &AisixSnapshot,
    provider: &OidcProvider,
    claims: &serde_json::Value,
) -> Result<Principal, crate::jwt::IdentityDenial> {
    if !crate::jwt::bound_claims_hold(claims, provider) {
        return Err(crate::jwt::IdentityDenial {
            reason: "jwt_bound_claim_mismatch",
            subject: None,
            claim_mapping: None,
            error: ProxyError::JwtClaimsRejected,
        });
    }
    let identity = crate::jwt::resolve_identity(state, snapshot, provider, claims)?;
    let auth = AuthenticatedKey::for_jwt(
        snapshot,
        identity.entry,
        JwtIdentity::for_session(
            identity.subject.clone(),
            provider.name.clone(),
            provider.issuer.clone(),
            provider.identity_claim.clone(),
            identity.claim_mapping,
        ),
        provider,
        claims,
    );
    Ok(Principal {
        subject: identity.subject,
        auth,
    })
}

/// The session token on this request, when its `Authorization` carries one.
pub(crate) fn presented_session_token(headers: &HeaderMap) -> Option<&str> {
    let value = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let token = aisix_core::forwarded_headers::strip_bearer_scheme(value)?.trim();
    token.starts_with(ACCESS_TOKEN_PREFIX).then_some(token)
}

/// Authenticate a session token presented on `/mcp/{server}`. Costs no store
/// round trip once this process holds the session key.
pub(crate) async fn authenticate_session(
    state: &ProxyState,
    token: &str,
    server: &str,
) -> Result<AuthenticatedKey, ProxyError> {
    let snapshot = state.snapshot.load();
    let note_refusal = |reason: &'static str| {
        state
            .metrics
            .record_auth_decision("mcp_oauth_session", false, reason);
        tracing::debug!(target: "aisix::auth", method = "mcp_oauth_session", reason, server,
            "refused an MCP session token");
    };
    let refused = |reason: &'static str| {
        note_refusal(reason);
        // A server that signs users in tells the client to do that. Any other
        // name reads as what the token is to it: a key it does not know.
        if sign_in_server(&snapshot, server).is_some() {
            ProxyError::McpUserOAuth {
                failure: UserOAuthFailure::AuthorizationRequired,
                server: server.to_owned(),
            }
        } else {
            ProxyError::InvalidApiKey
        }
    };
    // Before anything is read: a name that signs nobody in is no reason to
    // reach the store, and this is reachable with no credential at all.
    let Some(oauth) = sign_in_server(&snapshot, server) else {
        return Err(refused("session_wrong_server"));
    };
    let keys = match state.mcp_oauth.session_keys().await {
        Ok(Some(keys)) => keys,
        Ok(None) => return Err(refused("session_key_absent")),
        // Signing in again could not help, so this is not a challenge.
        Err(failure) => {
            return Err(ProxyError::McpUserOAuth {
                failure,
                server: server.to_owned(),
            })
        }
    };
    let Some(session) = keys.open_access_token(token) else {
        return Err(refused("session_invalid"));
    };
    if oauth.id != session.server {
        return Err(refused("session_wrong_server"));
    }
    let Some(provider) = snapshot
        .oidc_providers
        .get_by_name(&session.provider)
        .filter(|entry| entry.value.enabled)
    else {
        return Err(refused("session_provider_gone"));
    };
    if !session_issuer_is_current(&provider.value, &session.claims) {
        return Err(refused(SESSION_ISSUER_CHANGED));
    }
    let principal = session_principal(state, &snapshot, &provider.value, &session.claims).map_err(
        |denial| {
            note_refusal(denial.reason);
            denial.error
        },
    )?;
    if principal.subject != session.identity {
        return Err(refused(SESSION_IDENTITY_CHANGED));
    }
    state
        .metrics
        .record_auth_decision("mcp_oauth_session", true, "");
    Ok(principal.auth)
}

// ── The token endpoint, interactive mode ────────────────────────────────────

/// `POST /mcp/{server}/oauth/token` when no gateway credential is involved.
pub(crate) async fn token(
    state: &ProxyState,
    server: &str,
    mode: SignInMode,
    request: Request,
) -> Response {
    let snapshot = state.snapshot.load();
    let Some(oauth) = sign_in_server(&snapshot, server) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let (SignInMode::Interactive(_), Some(base)) = (mode, crate::mcp_auth::public_base(&snapshot))
    else {
        return oauth_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "this MCP server cannot sign in users as configured",
        );
    };
    let Ok(body) = to_bytes(request.into_body(), MAX_BODY_BYTES).await else {
        return oauth_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "the token request could not be read",
        );
    };
    let form: std::collections::HashMap<String, String> =
        url::form_urlencoded::parse(&body).into_owned().collect();
    let param = |name: &str| form.get(name).map(String::as_str).filter(|v| !v.is_empty());
    if param("resource").is_some_and(|resource| !resource_matches(resource, &base, server)) {
        return oauth_error(
            StatusCode::BAD_REQUEST,
            "invalid_target",
            "resource does not name this MCP server",
        );
    }
    let keys = match state.mcp_oauth.session_keys_for_sign_in().await {
        Ok(keys) => keys,
        Err(_) => return temporarily_unavailable(),
    };
    match param("grant_type") {
        Some("authorization_code") => {
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
            let identity =
                match state
                    .mcp_oauth
                    .redeem_session_code(
                        &oauth,
                        &RedeemRequest {
                            code,
                            client_id,
                            redirect_uri,
                            code_verifier,
                        },
                    )
                    .await
                {
                    Ok(identity) => identity,
                    Err(RedeemError::Unavailable) => return temporarily_unavailable(),
                    Err(_) => return oauth_error(
                        StatusCode::BAD_REQUEST,
                        "invalid_grant",
                        "the authorization code is unknown, expired, already used, or does not \
                         match this request",
                    ),
                };
            if authorized_provider(state, &snapshot, &identity).is_none() {
                return invalid_grant();
            }
            session_response(&keys, &oauth, client_id, &identity)
        }
        Some("refresh_token") => {
            let Some(refresh_token) = param("refresh_token") else {
                return oauth_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    "refresh_token is required",
                );
            };
            let Some(session) = keys.open_refresh_token(refresh_token) else {
                return invalid_grant();
            };
            if session.server != oauth.id
                || param("client_id").is_some_and(|client_id| client_id != session.client_id)
            {
                return invalid_grant();
            }
            let identity = SessionIdentity {
                provider: session.provider.clone(),
                identity: session.identity.clone(),
                claims: session.claims.clone(),
            };
            let Some(provider) = authorized_provider(state, &snapshot, &identity) else {
                return invalid_grant();
            };
            // With the upstream credential gone a new pair would be refused on
            // its first request, and a client holding a refresh token refreshes
            // before it considers the browser. This is what sends it there.
            let subject = UserSubject::jwt(
                &identity.provider,
                provider.value.issuer.as_deref(),
                &provider.value.identity_claim,
                &identity.identity,
            );
            if state.mcp_oauth.stored_access_token(&oauth, &subject).await
                == Err(UserOAuthFailure::AuthorizationRequired)
            {
                return invalid_grant();
            }
            match state.mcp_oauth.spend_refresh_token(&session).await {
                Ok(true) => session_response(&keys, &oauth, &session.client_id, &identity),
                Ok(false) => invalid_grant(),
                Err(_) => temporarily_unavailable(),
            }
        }
        _ => oauth_error(
            StatusCode::BAD_REQUEST,
            "unsupported_grant_type",
            "only authorization_code and refresh_token are supported",
        ),
    }
}

/// The provider, when configuration still gives this identity a usable key, so
/// a client is told at the token step rather than given a token no request will
/// accept.
fn authorized_provider(
    state: &ProxyState,
    snapshot: &AisixSnapshot,
    identity: &SessionIdentity,
) -> Option<Arc<ResourceEntry<OidcProvider>>> {
    snapshot
        .oidc_providers
        .get_by_name(&identity.provider)
        .filter(|entry| entry.value.enabled)
        .filter(|provider| {
            session_issuer_is_current(&provider.value, &identity.claims)
                && session_principal(state, snapshot, &provider.value, &identity.claims)
                    .is_ok_and(|principal| principal.subject == identity.identity)
        })
}

fn session_response(
    keys: &SessionKeys,
    oauth: &UserOAuthServer,
    client_id: &str,
    identity: &SessionIdentity,
) -> Response {
    let pair = keys.mint_pair(&oauth.id, client_id, identity);
    let mut response = Json(json!({
        "access_token": pair.access_token,
        "token_type": "Bearer",
        "expires_in": ACCESS_TOKEN_TTL_SECS,
        "refresh_token": pair.refresh_token,
        "scope": oauth.scopes().join(" "),
    }))
    .into_response();
    no_store(&mut response);
    response
}

fn invalid_grant() -> Response {
    oauth_error(
        StatusCode::BAD_REQUEST,
        "invalid_grant",
        "this token is unknown, expired, already used, or no longer valid; sign in again",
    )
}

fn temporarily_unavailable() -> Response {
    oauth_error(
        StatusCode::SERVICE_UNAVAILABLE,
        "temporarily_unavailable",
        "the sign-in could not be finished right now; try again",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cookie_is_scoped_to_the_public_path_and_secure_on_https() {
        assert_eq!(
            cookie_attributes("https://gw.example.com/team/gw"),
            "Path=/team/gw; HttpOnly; SameSite=Lax; Secure"
        );
        assert_eq!(
            cookie_attributes("http://127.0.0.1:3000"),
            "Path=/; HttpOnly; SameSite=Lax"
        );
    }

    #[tokio::test]
    async fn the_consent_page_escapes_what_it_shows() {
        let flow = FlowState {
            handle: "h".repeat(64),
            expires_at: 0,
            server_id: "s".to_owned(),
            server_name: "<script>alert(1)</script>".to_owned(),
            client_id: "c".to_owned(),
            redirect_uri: "https://app.example.org:8443/cb?x=<y>".to_owned(),
            client_state: None,
            code_challenge: "x".to_owned(),
            csrf: "z".to_owned(),
            stage: FlowStage::Consent {
                signed_in: None,
                relayed: None,
            },
        };
        let response = consent_page("https://gw.example.com", &flow, Some("a&b"));
        assert_eq!(response.headers()[header::X_FRAME_OPTIONS], "DENY");
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        assert_eq!(response.headers()[header::REFERRER_POLICY], "no-referrer");
        assert!(!response
            .headers()
            .contains_key(header::CONTENT_SECURITY_POLICY));
        let body = to_bytes(response.into_body(), MAX_BODY_BYTES)
            .await
            .unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(
            body.contains("&lt;script&gt;alert(1)&lt;/script&gt;"),
            "{body}"
        );
        assert!(!body.contains("<script>"), "{body}");
        assert!(body.contains("a&amp;b"), "{body}");
    }

    #[test]
    fn a_session_belongs_to_the_issuer_it_recorded() {
        let provider = |issuer: serde_json::Value| -> OidcProvider {
            serde_json::from_value(json!({"name": "corp", "issuer": issuer, "audiences": ["a"]}))
                .unwrap()
        };
        let idp = provider(json!("https://idp.example.com"));
        let current = |claims: serde_json::Value| session_issuer_is_current(&idp, &claims);

        assert!(current(json!({"iss": "https://idp.example.com"})));
        assert!(current(json!({"iss": "https://idp.example.com/"})));
        assert!(session_issuer_is_current(
            &provider(json!("https://idp.example.com/")),
            &json!({"iss": "https://idp.example.com"}),
        ));

        assert!(!current(json!({"iss": "https://other.example.com"})));
        assert!(!current(json!({"sub": "alice"})));
        assert!(!current(json!({"iss": 1})));
        assert!(!session_issuer_is_current(
            &provider(serde_json::Value::Null),
            &json!({"iss": "https://idp.example.com"}),
        ));
    }

    #[test]
    fn a_redirect_target_is_its_scheme_and_host() {
        assert_eq!(
            redirect_target("https://app.example.org:8443/cb?x=1"),
            "https://app.example.org:8443"
        );
        assert_eq!(
            redirect_target("https://app.example.org/cb"),
            "https://app.example.org"
        );
        assert_eq!(
            redirect_target("cursor://anysphere.cursor-mcp/oauth/callback"),
            "cursor://anysphere.cursor-mcp"
        );
    }
}
