//! A byte stream to an upstream for the outbound paths that do not ride
//! reqwest (the Realtime WebSocket), honouring the same proxy
//! environment as every reqwest upstream client.
//!
//! The proxy is chosen by the matcher reqwest itself uses: a `secure`
//! target reads `HTTPS_PROXY`, a plain one `HTTP_PROXY`, each falling
//! back to `ALL_PROXY`, and `NO_PROXY` exempts a host from both. A
//! proxied target is reached through an HTTP CONNECT tunnel carrying the
//! proxy URL's credentials as `Proxy-Authorization`; an `https://` proxy
//! is itself reached over TLS under `upstream.tls`. A SOCKS proxy is
//! refused rather than bypassed: reqwest is built without SOCKS support
//! here, so no upstream path can use one.

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::OnceLock;
use std::task::{Context, Poll};

use http::Uri;
use hyper_util::client::legacy::connect::proxy::Tunnel;
use hyper_util::client::proxy::matcher::Matcher;
use hyper_util::rt::TokioIo;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;
use tower_service::Service;

pub trait Io: AsyncRead + AsyncWrite + Send + Unpin {}
impl<T: AsyncRead + AsyncWrite + Send + Unpin> Io for T {}

pub type UpstreamStream = Box<dyn Io>;

fn proxies() -> &'static Matcher {
    static PROXIES: OnceLock<Matcher> = OnceLock::new();
    PROXIES.get_or_init(Matcher::from_system)
}

/// Open a stream to `host:port`, directly or through the environment
/// proxy that takes it. The caller runs its own TLS over the result.
/// `host` is as an `http::Uri` spells it (an IPv6 literal in brackets).
pub async fn connect(host: &str, port: u16, secure: bool) -> io::Result<UpstreamStream> {
    let scheme = if secure { "https" } else { "http" };
    let target: Uri = format!("{scheme}://{host}:{port}/")
        .parse()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let Some(proxy) = proxies().intercept(&target) else {
        return Ok(Box::new(tcp(host, port).await?));
    };
    let mut tunnel = Tunnel::new(proxy.uri().clone(), ToProxy);
    if let Some(auth) = proxy.basic_auth() {
        tunnel = tunnel.with_auth(auth.clone());
    }
    let stream = tunnel.call(target).await.map_err(io::Error::other)?;
    Ok(stream.into_inner())
}

async fn tcp(host: &str, port: u16) -> io::Result<TcpStream> {
    TcpStream::connect((bare(host), port)).await
}

/// An IPv6 literal without the brackets a URI puts around it.
fn bare(host: &str) -> &str {
    host.trim_start_matches('[').trim_end_matches(']')
}

/// The connector under the CONNECT tunnel: a stream to the proxy itself.
#[derive(Clone, Copy)]
struct ToProxy;

impl Service<Uri> for ToProxy {
    type Response = TokioIo<UpstreamStream>;
    type Error = io::Error;
    type Future = Pin<Box<dyn Future<Output = io::Result<Self::Response>> + Send>>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, proxy: Uri) -> Self::Future {
        Box::pin(async move {
            let host = bare(proxy.host().unwrap_or_default()).to_owned();
            let stream: UpstreamStream = match proxy.scheme_str() {
                Some("http") => Box::new(tcp(&host, proxy.port_u16().unwrap_or(80)).await?),
                Some("https") => {
                    let tcp = tcp(&host, proxy.port_u16().unwrap_or(443)).await?;
                    let name = rustls::pki_types::ServerName::try_from(host)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
                    let tls = tokio_rustls::TlsConnector::from(
                        crate::upstream_tls::rustls_client_config(),
                    );
                    Box::new(tls.connect(name, tcp).await?)
                }
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        format!(
                            "proxy scheme {:?} is not supported; use an http:// or https:// proxy",
                            other.unwrap_or_default()
                        ),
                    ))
                }
            };
            Ok(TokioIo::new(stream))
        })
    }
}
