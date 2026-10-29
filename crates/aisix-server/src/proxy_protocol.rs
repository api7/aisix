//! PROXY protocol v1 / v2 on the proxy listeners (AISIX-Cloud#1915).
//!
//! A layer-4 load balancer that rewrites the source address can prepend a
//! PROXY header naming the original client. The header is detected, not
//! configured: a connection that does not start with either signature is
//! served exactly as before, which is what keeps platform probes (which
//! never send one) working on the same port. A header is honoured only
//! from a socket peer inside `proxy.real_ip.trusted_proxies`; from anyone
//! else it is refused, since honouring it would let any client name its
//! own address.
//!
//! Spec: <https://www.haproxy.org/download/2.9/doc/proxy-protocol.txt>.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use ipnet::IpNet;
use tokio::io::{AsyncRead, AsyncReadExt};

const V2_SIGNATURE: &[u8; 12] = b"\r\n\r\n\0\r\nQUIT\n";
const V1_PREFIX: &[u8; 6] = b"PROXY ";
/// Spec §2.1: the longest v1 line, CRLF included.
const V1_MAX_LEN: usize = 107;
/// The fixed v2 prefix: signature, version/command, family, length.
const V2_FIXED_LEN: usize = 16;
/// Upper bound on a whole v2 header. The address block is at most 216
/// bytes (AF_UNIX); the rest is TLVs, which this gateway ignores, so a
/// length beyond this is refused rather than buffered.
const V2_MAX_LEN: usize = 4096;

/// Why a connection that opened with a PROXY header is refused.
#[derive(Debug)]
pub(crate) enum Refused {
    /// The socket peer is not in `proxy.real_ip.trusted_proxies`.
    Untrusted,
    /// The header is malformed or truncated.
    Malformed(&'static str),
    /// The read itself failed (reset, timeout before a header showed up).
    Io(std::io::Error),
}

#[derive(Debug, PartialEq, Eq)]
enum Detected {
    NeedMore,
    None,
    V1,
    V2,
}

fn detect(buf: &[u8]) -> Detected {
    let prefix_of = |sig: &[u8]| {
        let n = buf.len().min(sig.len());
        buf[..n] == sig[..n]
    };
    if buf.starts_with(V2_SIGNATURE) {
        Detected::V2
    } else if buf.starts_with(V1_PREFIX) {
        Detected::V1
    } else if buf.is_empty() || prefix_of(V2_SIGNATURE) || prefix_of(V1_PREFIX) {
        Detected::NeedMore
    } else {
        Detected::None
    }
}

/// Read a PROXY header off the front of `stream` if it carries one.
///
/// Returns the peer the connection should be attributed to — the header's
/// source address, or `socket_peer` when there is no header or the header
/// names none (v2 `LOCAL`, an unspecified family, v1 `UNKNOWN`) — and the
/// bytes read past the header, which the caller must replay ahead of the
/// stream so TLS / HTTP see it from the byte after the header.
///
/// `timeout` bounds the whole read, header included.
pub(crate) async fn read<S: AsyncRead + Unpin>(
    stream: &mut S,
    socket_peer: SocketAddr,
    trusted: &[IpNet],
    timeout: Option<Duration>,
) -> Result<(SocketAddr, Vec<u8>), Refused> {
    let deadline = timeout.map(|d| tokio::time::Instant::now() + d);
    let mut buf = Vec::with_capacity(256);
    let mut chunk = [0u8; 512];

    let version = loop {
        match detect(&buf) {
            Detected::NeedMore => {}
            Detected::None => return Ok((socket_peer, buf)),
            Detected::V1 => break Detected::V1,
            Detected::V2 => break Detected::V2,
        }
        let n = read_by(stream, &mut chunk, deadline)
            .await
            .map_err(Refused::Io)?;
        if n == 0 {
            // Closed before the first bytes could be told apart from a
            // signature: nothing to honour, and whatever follows reports
            // the closed connection exactly as it does without a header.
            return Ok((socket_peer, buf));
        }
        buf.extend_from_slice(&chunk[..n]);
    };

    if !trusted.iter().any(|net| net.contains(&socket_peer.ip())) {
        return Err(Refused::Untrusted);
    }

    loop {
        let parsed = match version {
            Detected::V1 => parse_v1(&buf),
            _ => parse_v2(&buf),
        };
        match parsed {
            Parsed::Done { len, source } => {
                let rest = buf.split_off(len);
                return Ok((source.unwrap_or(socket_peer), rest));
            }
            Parsed::Invalid(reason) => return Err(Refused::Malformed(reason)),
            Parsed::Incomplete => {}
        }
        match read_by(stream, &mut chunk, deadline).await {
            Ok(0) => return Err(Refused::Malformed("connection closed inside the header")),
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
                return Err(Refused::Malformed("timed out inside the header"))
            }
            Err(e) => return Err(Refused::Io(e)),
        }
    }
}

async fn read_by<S: AsyncRead + Unpin>(
    stream: &mut S,
    chunk: &mut [u8],
    deadline: Option<tokio::time::Instant>,
) -> std::io::Result<usize> {
    match deadline {
        None => stream.read(chunk).await,
        Some(at) => tokio::time::timeout_at(at, stream.read(chunk))
            .await
            .unwrap_or_else(|_| {
                Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "downstream opened a connection and sent nothing",
                ))
            }),
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Parsed {
    Incomplete,
    Invalid(&'static str),
    /// `len` header bytes; `source` is `None` when the header names no
    /// client address.
    Done {
        len: usize,
        source: Option<SocketAddr>,
    },
}

fn parse_v1(buf: &[u8]) -> Parsed {
    let window = &buf[..buf.len().min(V1_MAX_LEN)];
    let Some(cr) = window.windows(2).position(|w| w == b"\r\n") else {
        return if buf.len() >= V1_MAX_LEN {
            Parsed::Invalid("v1 line longer than 107 bytes")
        } else {
            Parsed::Incomplete
        };
    };
    let len = cr + 2;
    let Ok(line) = std::str::from_utf8(&buf[V1_PREFIX.len()..cr]) else {
        return Parsed::Invalid("v1 line is not ASCII");
    };
    let mut fields = line.split(' ');
    let family = fields.next().unwrap_or_default();
    if family == "UNKNOWN" {
        return Parsed::Done { len, source: None };
    }
    let fields: Vec<&str> = fields.collect();
    let [src, _dst, sport, _dport] = fields[..] else {
        return Parsed::Invalid("v1 line does not carry four address fields");
    };
    let ip = match family {
        "TCP4" => src.parse::<Ipv4Addr>().map(IpAddr::V4).ok(),
        "TCP6" => src.parse::<Ipv6Addr>().map(IpAddr::V6).ok(),
        _ => return Parsed::Invalid("v1 protocol is not TCP4, TCP6 or UNKNOWN"),
    };
    let (Some(ip), Ok(port)) = (ip, sport.parse::<u16>()) else {
        return Parsed::Invalid("v1 source address or port does not parse");
    };
    Parsed::Done {
        len,
        source: Some(SocketAddr::new(ip, port)),
    }
}

fn parse_v2(buf: &[u8]) -> Parsed {
    if buf.len() < V2_FIXED_LEN {
        return Parsed::Incomplete;
    }
    let ver_cmd = buf[12];
    if ver_cmd >> 4 != 2 {
        return Parsed::Invalid("v2 version is not 2");
    }
    let addr_len = u16::from_be_bytes([buf[14], buf[15]]) as usize;
    let len = V2_FIXED_LEN + addr_len;
    if len > V2_MAX_LEN {
        return Parsed::Invalid("v2 header longer than 4096 bytes");
    }
    let local = match ver_cmd & 0x0f {
        0x0 => true,
        0x1 => false,
        _ => return Parsed::Invalid("v2 command is neither LOCAL nor PROXY"),
    };
    if buf.len() < len {
        return Parsed::Incomplete;
    }
    // LOCAL is the balancer's own health check, whose address block the
    // receiver must ignore; UNSPEC and UNIX carry no IP to attribute the
    // connection to. TLVs are skipped with the rest of the block either way.
    if local {
        return Parsed::Done { len, source: None };
    }
    let family = buf[13] >> 4;
    let transport = buf[13] & 0x0f;
    if family > 3 || transport > 2 {
        return Parsed::Invalid("v2 address family or transport is unknown");
    }
    if transport == 0 || !(family == 1 || family == 2) {
        return Parsed::Done { len, source: None };
    }
    if addr_len < if family == 1 { 12 } else { 36 } {
        return Parsed::Invalid("v2 address block shorter than its family requires");
    }
    let a = &buf[V2_FIXED_LEN..len];
    let source = if family == 1 {
        let ip = Ipv4Addr::new(a[0], a[1], a[2], a[3]);
        SocketAddr::new(IpAddr::V4(ip), u16::from_be_bytes([a[8], a[9]]))
    } else {
        let mut octets = [0u8; 16];
        octets.copy_from_slice(&a[..16]);
        SocketAddr::new(
            IpAddr::V6(Ipv6Addr::from(octets)),
            u16::from_be_bytes([a[32], a[33]]),
        )
    };
    Parsed::Done {
        len,
        source: Some(source),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v2(cmd: u8, fam: u8, block: &[u8]) -> Vec<u8> {
        let mut h = V2_SIGNATURE.to_vec();
        h.push(0x20 | cmd);
        h.push(fam);
        h.extend_from_slice(&(block.len() as u16).to_be_bytes());
        h.extend_from_slice(block);
        h
    }

    fn v4_block(tlv: &[u8]) -> Vec<u8> {
        let mut b = vec![203, 0, 113, 7, 10, 0, 0, 1];
        b.extend_from_slice(&51234u16.to_be_bytes());
        b.extend_from_slice(&443u16.to_be_bytes());
        b.extend_from_slice(tlv);
        b
    }

    fn src(s: &str) -> Option<SocketAddr> {
        Some(s.parse().unwrap())
    }

    #[test]
    fn detection_waits_only_while_the_bytes_could_still_be_a_signature() {
        assert_eq!(detect(b""), Detected::NeedMore);
        assert_eq!(detect(b"\r\n\r"), Detected::NeedMore);
        assert_eq!(detect(b"PROX"), Detected::NeedMore);
        assert_eq!(detect(b"PRI * HTTP/2.0"), Detected::None);
        assert_eq!(detect(b"POST /v1"), Detected::None);
        assert_eq!(detect(b"\x16\x03\x01"), Detected::None);
        assert_eq!(detect(b"\r\n\r\nGET"), Detected::None);
        assert_eq!(detect(b"PROXY TCP4"), Detected::V1);
        assert_eq!(detect(V2_SIGNATURE), Detected::V2);
    }

    #[test]
    fn v2_proxy_over_tcp4_names_the_source_and_skips_tlvs() {
        let mut h = v2(1, 0x11, &v4_block(&[0x04, 0x00, 0x01, 0xaa]));
        let len = h.len();
        h.extend_from_slice(b"GET /");
        assert_eq!(
            parse_v2(&h),
            Parsed::Done {
                len,
                source: src("203.0.113.7:51234"),
            }
        );
    }

    #[test]
    fn v2_proxy_over_udp6_names_the_source() {
        let mut b = vec![0u8; 36];
        b[..16].copy_from_slice(&"2001:db8::7".parse::<Ipv6Addr>().unwrap().octets());
        b[32..34].copy_from_slice(&4000u16.to_be_bytes());
        let h = v2(1, 0x22, &b);
        assert_eq!(
            parse_v2(&h),
            Parsed::Done {
                len: h.len(),
                source: src("[2001:db8::7]:4000"),
            }
        );
    }

    #[test]
    fn v2_local_unspec_and_unix_keep_the_socket_peer() {
        for h in [
            v2(0, 0x11, &v4_block(&[])),
            v2(1, 0x00, &[]),
            v2(1, 0x10, &v4_block(&[])),
            v2(1, 0x31, &[0u8; 216]),
            v2(0, 0xff, &[]),
        ] {
            assert_eq!(
                parse_v2(&h),
                Parsed::Done {
                    len: h.len(),
                    source: None
                }
            );
        }
    }

    #[test]
    fn v2_rejects_what_it_cannot_read() {
        let mut bad_version = v2(1, 0x11, &v4_block(&[]));
        bad_version[12] = 0x11;
        assert!(matches!(parse_v2(&bad_version), Parsed::Invalid(_)));
        assert!(matches!(
            parse_v2(&v2(2, 0x11, &v4_block(&[]))),
            Parsed::Invalid(_)
        ));
        assert!(matches!(parse_v2(&v2(1, 0x41, &[])), Parsed::Invalid(_)));
        assert!(matches!(
            parse_v2(&v2(1, 0x11, &[1, 2, 3])),
            Parsed::Invalid(_)
        ));
        let mut huge = V2_SIGNATURE.to_vec();
        huge.extend_from_slice(&[0x21, 0x11, 0xff, 0xff]);
        assert!(matches!(parse_v2(&huge), Parsed::Invalid(_)));
    }

    #[test]
    fn v2_waits_for_the_declared_length() {
        let h = v2(1, 0x11, &v4_block(&[]));
        assert_eq!(parse_v2(&h[..10]), Parsed::Incomplete);
        assert_eq!(parse_v2(&h[..h.len() - 1]), Parsed::Incomplete);
    }

    #[test]
    fn v1_tcp4_tcp6_and_unknown() {
        let line = b"PROXY TCP4 198.51.100.9 10.0.0.1 40000 443\r\nGET";
        assert_eq!(
            parse_v1(line),
            Parsed::Done {
                len: line.len() - 3,
                source: src("198.51.100.9:40000"),
            }
        );
        let line = b"PROXY TCP6 2001:db8::9 2001:db8::1 40000 443\r\n";
        assert_eq!(
            parse_v1(line),
            Parsed::Done {
                len: line.len(),
                source: src("[2001:db8::9]:40000"),
            }
        );
        let line = b"PROXY UNKNOWN whatever\r\n";
        assert_eq!(
            parse_v1(line),
            Parsed::Done {
                len: line.len(),
                source: None
            }
        );
    }

    #[test]
    fn v1_rejects_what_it_cannot_read() {
        for line in [
            &b"PROXY TCP4 2001:db8::9 10.0.0.1 40000 443\r\n"[..],
            b"PROXY TCP4 198.51.100.9 10.0.0.1 70000 443\r\n",
            b"PROXY TCP4 198.51.100.9 10.0.0.1 40000\r\n",
            b"PROXY UDP4 198.51.100.9 10.0.0.1 40000 443\r\n",
            b"PROXY TCP4 198.51.100.9  10.0.0.1 40000 443\r\n",
        ] {
            assert!(matches!(parse_v1(line), Parsed::Invalid(_)), "{line:?}");
        }
        let mut long = b"PROXY UNKNOWN ".to_vec();
        long.resize(V1_MAX_LEN, b'x');
        assert!(matches!(parse_v1(&long), Parsed::Invalid(_)));
        long.truncate(V1_MAX_LEN - 1);
        assert_eq!(parse_v1(&long), Parsed::Incomplete);
    }

    #[tokio::test]
    async fn the_header_is_honoured_only_from_a_trusted_peer() {
        let trusted: Vec<IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut wire = v2(1, 0x11, &v4_block(&[]));
        wire.extend_from_slice(b"GET / HTTP/1.1\r\n");

        let from_lb: SocketAddr = "10.1.2.3:5000".parse().unwrap();
        let (peer, rest) = read(&mut wire.as_slice(), from_lb, &trusted, None)
            .await
            .expect("trusted");
        assert_eq!(Some(peer), src("203.0.113.7:51234"));
        assert_eq!(rest, b"GET / HTTP/1.1\r\n");

        let stranger: SocketAddr = "192.0.2.1:5000".parse().unwrap();
        assert!(matches!(
            read(&mut wire.as_slice(), stranger, &trusted, None).await,
            Err(Refused::Untrusted)
        ));
        assert!(matches!(
            read(&mut wire.as_slice(), from_lb, &[], None).await,
            Err(Refused::Untrusted)
        ));
    }

    #[tokio::test]
    async fn a_connection_without_a_header_is_replayed_untouched() {
        let trusted: Vec<IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let peer: SocketAddr = "10.1.2.3:5000".parse().unwrap();
        let wire = b"GET /livez HTTP/1.1\r\n\r\n";
        let (got, rest) = read(&mut &wire[..], peer, &trusted, None).await.unwrap();
        assert_eq!(got, peer);
        assert_eq!(rest, wire);
    }

    #[tokio::test]
    async fn a_header_cut_short_by_the_peer_is_malformed() {
        let trusted: Vec<IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let peer: SocketAddr = "10.1.2.3:5000".parse().unwrap();
        let wire = v2(1, 0x11, &v4_block(&[]));
        assert!(matches!(
            read(&mut &wire[..wire.len() - 2], peer, &trusted, None).await,
            Err(Refused::Malformed(_))
        ));
    }

    #[tokio::test]
    async fn a_header_split_across_reads_is_reassembled() {
        use tokio::io::AsyncWriteExt;

        let trusted: Vec<IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let peer: SocketAddr = "10.1.2.3:5000".parse().unwrap();
        let mut wire = v2(1, 0x11, &v4_block(&[]));
        wire.extend_from_slice(b"GET");
        let (mut client, mut server) = tokio::io::duplex(64);
        let writer = tokio::spawn(async move {
            for byte in wire {
                client.write_all(&[byte]).await.unwrap();
                tokio::task::yield_now().await;
            }
            client
        });
        let (got, rest) = read(&mut server, peer, &trusted, None).await.unwrap();
        let _client = writer.await.unwrap();
        assert_eq!(Some(got), src("203.0.113.7:51234"));
        assert!(b"GET".starts_with(&rest), "{rest:?}");
    }

    #[tokio::test(start_paused = true)]
    async fn a_header_that_stalls_midway_is_malformed() {
        use tokio::io::AsyncWriteExt;

        let trusted: Vec<IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let peer: SocketAddr = "10.1.2.3:5000".parse().unwrap();
        let wire = v2(1, 0x11, &v4_block(&[]));
        let (mut client, mut server) = tokio::io::duplex(64);
        client.write_all(&wire[..14]).await.unwrap();
        assert!(matches!(
            read(&mut server, peer, &trusted, Some(Duration::from_secs(5))).await,
            Err(Refused::Malformed("timed out inside the header"))
        ));
    }
}
