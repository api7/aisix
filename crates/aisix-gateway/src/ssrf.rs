//! `outbound.ssrf_protection`: whether a request to a URL an API caller
//! supplied may reach a given address.
//!
//! On, every address the request connects to must be publicly routable.
//! A direct connection resolves the name here, checks every address in
//! the answer, and connects to exactly those addresses — never to a
//! second lookup that a rebinding name could answer differently; that
//! holds for `http` and `https` alike because the check lives in the
//! client's resolver, while the host name still rides `Host` and SNI.
//! When an environment proxy applies to the URL, the name is resolved and
//! checked here first and the proxy then resolves it again itself, which
//! cannot be pinned. `allowed_hosts` is the only way past the check, and
//! `enabled: false` removes it.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, OnceLock};

use hyper_util::client::proxy::matcher::Matcher;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::Url;

/// `outbound.ssrf_protection`, parsed once at boot.
#[derive(Debug, Clone)]
pub struct SsrfProtection {
    enabled: bool,
    allowed_hosts: Vec<AllowedHost>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AllowedHost {
    host: String,
    port: Option<u16>,
}

impl SsrfProtection {
    pub fn from_config(cfg: &aisix_core::config::SsrfProtectionConfig) -> Result<Self, String> {
        let allowed_hosts = cfg
            .allowed_hosts
            .iter()
            .map(|entry| {
                parse_allowed_host(entry).ok_or_else(|| {
                    format!(
                        "outbound.ssrf_protection.allowed_hosts: {entry:?} is not a host or \
                         host:port (write an IPv6 address with a port as [addr]:port)"
                    )
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            enabled: cfg.enabled,
            allowed_hosts,
        })
    }

    fn allows(&self, host: &str, port: u16) -> bool {
        self.allowed_hosts
            .iter()
            .any(|a| a.host == host && a.port.is_none_or(|p| p == port))
    }

    /// How a request to `url` is sent, or why it may not be. Async because
    /// a URL an environment proxy takes is resolved and checked here.
    pub async fn route(&self, url: &Url) -> Result<Route, Refusal> {
        let (host, ip) = match url.host() {
            Some(url::Host::Domain(d)) => (d.trim_end_matches('.').to_ascii_lowercase(), None),
            Some(url::Host::Ipv4(a)) => (a.to_string(), Some(IpAddr::V4(a))),
            Some(url::Host::Ipv6(a)) => (a.to_string(), Some(IpAddr::V6(a))),
            None => return Err(Refusal::BadUrl),
        };
        let port = url.port_or_known_default().ok_or(Refusal::BadUrl)?;
        if !self.enabled || self.allows(&host, port) {
            return Ok(Route::Unchecked);
        }
        // The resolver never sees an address literal, so it is checked here.
        if let Some(ip) = ip {
            return if is_public(ip) {
                Ok(Route::Unchecked)
            } else {
                Err(Refusal::Blocked(ip))
            };
        }
        if !proxied(url) {
            return Ok(Route::Pinned);
        }
        check_all(resolve(&host).await?)?;
        Ok(Route::Unchecked)
    }
}

/// The client a permitted request goes out on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// Direct, through the resolver that refuses non-public addresses.
    Pinned,
    /// Nothing further to check: the target is allowed, already checked,
    /// or protection is off. Environment proxies apply.
    Unchecked,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    BadUrl,
    Blocked(IpAddr),
    Unresolvable,
}

impl Refusal {
    /// The error a caller gets for a blocked host, naming the settings an
    /// operator would change.
    pub fn blocked_hint() -> &'static str {
        "To allow it, an operator adds the host to outbound.ssrf_protection.allowed_hosts, \
         or turns the check off with outbound.ssrf_protection.enabled: false"
    }

    /// The refusal a resolver error carries, if any.
    pub fn from_error(err: &(dyn std::error::Error + 'static)) -> Option<Self> {
        let mut source = Some(err);
        while let Some(e) = source {
            if let Some(BlockedAddress(ip)) = e.downcast_ref::<BlockedAddress>() {
                return Some(Self::Blocked(*ip));
            }
            if e.downcast_ref::<Unresolvable>().is_some() {
                return Some(Self::Unresolvable);
            }
            source = e.source();
        }
        None
    }
}

static SETTINGS: OnceLock<SsrfProtection> = OnceLock::new();

/// Install `outbound.ssrf_protection`. Called once during boot; fails on
/// an `allowed_hosts` entry that does not parse.
pub fn init(cfg: &aisix_core::config::SsrfProtectionConfig) -> Result<(), String> {
    let _ = SETTINGS.set(SsrfProtection::from_config(cfg)?);
    Ok(())
}

pub fn settings() -> &'static SsrfProtection {
    SETTINGS.get_or_init(|| {
        SsrfProtection::from_config(&Default::default()).expect("the default config parses")
    })
}

/// Whether an environment proxy takes `url`, by the same rules — and the
/// same matcher — the HTTP clients use.
fn proxied(url: &Url) -> bool {
    static MATCHER: OnceLock<Matcher> = OnceLock::new();
    let Ok(uri) = url.as_str().parse::<http::Uri>() else {
        return false;
    };
    MATCHER
        .get_or_init(Matcher::from_system)
        .intercept(&uri)
        .is_some()
}

/// A resolver for the client of [`Route::Pinned`] (`guarded`), which
/// refuses the whole answer if any address in it is not publicly
/// routable, or for the client of [`Route::Unchecked`], which only
/// reports a name that does not resolve as such. The client connects only
/// to the addresses returned here.
pub fn resolver(guarded: bool) -> Arc<impl Resolve> {
    Arc::new(CheckingResolver { guarded })
}

struct CheckingResolver {
    guarded: bool,
}

impl Resolve for CheckingResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let guarded = self.guarded;
        Box::pin(async move {
            let addrs = resolve(name.as_str())
                .await
                .map_err(|_| Box::new(Unresolvable) as Box<dyn std::error::Error + Send + Sync>)?;
            if guarded {
                if let Err(Refusal::Blocked(ip)) = check_all(addrs.iter().copied()) {
                    return Err(Box::new(BlockedAddress(ip)) as _);
                }
            }
            Ok(Box::new(addrs.into_iter()) as Addrs)
        })
    }
}

async fn resolve(host: &str) -> Result<Vec<SocketAddr>, Refusal> {
    let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host, 0))
        .await
        .map_err(|_| Refusal::Unresolvable)?
        .collect();
    if addrs.is_empty() {
        return Err(Refusal::Unresolvable);
    }
    Ok(addrs)
}

fn check_all(addrs: impl IntoIterator<Item = SocketAddr>) -> Result<(), Refusal> {
    match addrs.into_iter().map(|a| a.ip()).find(|ip| !is_public(*ip)) {
        Some(ip) => Err(Refusal::Blocked(ip)),
        None => Ok(()),
    }
}

#[derive(Debug)]
struct BlockedAddress(IpAddr);

impl std::fmt::Display for BlockedAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} is not a publicly routable address", self.0)
    }
}

impl std::error::Error for BlockedAddress {}

#[derive(Debug)]
struct Unresolvable;

impl std::fmt::Display for Unresolvable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the host does not resolve")
    }
}

impl std::error::Error for Unresolvable {}

fn parse_allowed_host(entry: &str) -> Option<AllowedHost> {
    let entry = entry.trim();
    if entry.is_empty() || entry.contains(['/', '@', '?', '#']) {
        return None;
    }
    let port = |p: &str| p.parse::<u16>().ok();
    if let Some(rest) = entry.strip_prefix('[') {
        let (addr, tail) = rest.split_once(']')?;
        let host = addr.parse::<Ipv6Addr>().ok()?.to_string();
        let port = match tail {
            "" => None,
            tail => Some(port(tail.strip_prefix(':')?)?),
        };
        return Some(AllowedHost { host, port });
    }
    if let Ok(v6) = entry.parse::<Ipv6Addr>() {
        return Some(AllowedHost {
            host: v6.to_string(),
            port: None,
        });
    }
    let (host, port) = match entry.rsplit_once(':') {
        Some((host, p)) => (host, Some(port(p)?)),
        None => (entry, None),
    };
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    (!host.is_empty()).then_some(AllowedHost { host, port })
}

/// Whether an address is one a request may reach without an allow-list
/// entry: globally routable unicast, and not a cloud platform's own
/// service address.
fn is_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_public_v4(v4),
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => is_public_v4(v4),
            None => is_public_v6(v6),
        },
    }
}

fn is_public_v4(a: Ipv4Addr) -> bool {
    let [o0, o1, o2, _] = a.octets();
    let blocked = o0 == 0 // "this network", incl. 0.0.0.0
        || a.is_loopback()
        || a.is_private()
        || (o0 == 100 && (o1 & 0xc0) == 64) // CGNAT 100.64.0.0/10
        || a.is_link_local() // 169.254.0.0/16, incl. instance metadata
        || (o0 == 192 && o1 == 0 && o2 == 0) // IETF protocol assignments
        || a.is_documentation()
        || (o0 == 198 && (o1 & 0xfe) == 18) // benchmarking 198.18.0.0/15
        || (o0 == 192 && o1 == 88 && o2 == 99) // 6to4 relay anycast
        || a.is_multicast()
        || o0 >= 240 // reserved, incl. broadcast
        || a == Ipv4Addr::new(168, 63, 129, 16); // Azure platform endpoint
    !blocked
}

fn is_public_v6(a: Ipv6Addr) -> bool {
    let s = a.segments();
    let embedded = |hi: u16, lo: u16| Ipv4Addr::from(((hi as u32) << 16) | lo as u32);
    if a.is_unspecified() || a.is_loopback() || a.is_multicast() {
        return false;
    }
    match s {
        // IPv4-compatible (deprecated) ::/96
        [0, 0, 0, 0, 0, 0, _, _] => false,
        // NAT64 64:ff9b::/96 reaches the IPv4 address it embeds.
        [0x64, 0xff9b, 0, 0, 0, 0, hi, lo] => is_public_v4(embedded(hi, lo)),
        // Local-use NAT64 64:ff9b:1::/48
        [0x64, 0xff9b, 1, ..] => false,
        // Discard-only 100::/64
        [0x100, 0, 0, 0, ..] => false,
        // IETF protocol assignments 2001::/23 (Teredo included), documentation 2001:db8::/32
        [0x2001, b, ..] if b < 0x200 || b == 0xdb8 => false,
        // 6to4 2002::/16 reaches the IPv4 address it embeds.
        [0x2002, hi, lo, ..] => is_public_v4(embedded(hi, lo)),
        // Documentation 3fff::/20
        [0x3fff, b, ..] if b < 0x1000 => false,
        // Unique-local fc00::/7, link-local fe80::/10, site-local fec0::/10
        [first, ..]
            if (first & 0xfe00) == 0xfc00
                || (first & 0xffc0) == 0xfe80
                || (first & 0xffc0) == 0xfec0 =>
        {
            false
        }
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_allowed(hosts: &[&str]) -> SsrfProtection {
        SsrfProtection::from_config(&aisix_core::config::SsrfProtectionConfig {
            allowed_hosts: hosts.iter().map(|h| h.to_string()).collect(),
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn only_globally_routable_addresses_are_public() {
        for blocked in [
            "0.0.0.0",
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.1",
            "192.168.1.1",
            "100.64.0.1",
            "100.127.255.255",
            "169.254.169.254",
            "192.0.0.8",
            "192.0.2.1",
            "198.18.0.1",
            "198.51.100.1",
            "203.0.113.1",
            "224.0.0.1",
            "255.255.255.255",
            "168.63.129.16",
            "::",
            "::1",
            "::ffff:127.0.0.1",
            "::ffff:169.254.169.254",
            "64:ff9b::a00:1",
            "2002:a00:1::",
            "2001:db8::1",
            "fd00::1",
            "fe80::1",
            "ff02::1",
        ] {
            let ip: IpAddr = blocked.parse().unwrap();
            assert!(!is_public(ip), "{blocked} must be refused");
        }
        for public in [
            "1.1.1.1",
            "8.8.8.8",
            "100.128.0.1",
            "::ffff:8.8.8.8",
            "2606:4700:4700::1111",
            "64:ff9b::808:808",
        ] {
            let ip: IpAddr = public.parse().unwrap();
            assert!(is_public(ip), "{public} must be allowed");
        }
    }

    #[test]
    fn allowed_hosts_match_host_and_optional_port() {
        let settings = with_allowed(&["Media.Internal", "127.0.0.1:9000", "[fd00::1]:8080"]);
        assert!(settings.allows("media.internal", 443));
        assert!(settings.allows("127.0.0.1", 9000));
        assert!(!settings.allows("127.0.0.1", 9001));
        assert!(settings.allows("fd00::1", 8080));
        for bad in ["", "http://x", "a:b", "[::1", "user@host"] {
            assert!(parse_allowed_host(bad).is_none(), "{bad:?}");
        }
        let err = SsrfProtection::from_config(&aisix_core::config::SsrfProtectionConfig {
            allowed_hosts: vec!["http://x".into()],
            ..Default::default()
        })
        .unwrap_err();
        assert!(
            err.contains("outbound.ssrf_protection.allowed_hosts"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn disabled_protection_checks_nothing() {
        let off = SsrfProtection::from_config(&aisix_core::config::SsrfProtectionConfig {
            enabled: false,
            allowed_hosts: vec![],
        })
        .unwrap();
        for url in ["http://127.0.0.1/a", "http://localhost/a", "http://[::1]/a"] {
            let url = Url::parse(url).unwrap();
            assert_eq!(off.route(&url).await, Ok(Route::Unchecked), "{url}");
        }
        let on = with_allowed(&[]);
        assert_eq!(
            on.route(&Url::parse("http://127.0.0.1/a").unwrap()).await,
            Err(Refusal::Blocked("127.0.0.1".parse().unwrap()))
        );
        assert_eq!(
            on.route(&Url::parse("http://localhost/a").unwrap()).await,
            Ok(Route::Pinned)
        );
    }
}
