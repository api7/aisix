//! pgo-trainer — deterministic training-traffic generator for the PGO release
//! build (api7/aisix#967).
//!
//! One process, two roles:
//!
//! 1. **Mock upstream** — canned OpenAI / Anthropic dialect responses (JSON +
//!    SSE) on a local port; just enough surface for the gateway to dispatch
//!    every training shape against it. Streaming responses are written frame
//!    by frame over chunked transfer encoding so the gateway's SSE relay loop
//!    trains on incremental reads, not one buffered blob.
//! 2. **Load driver** — a fixed number of requests per shape through the
//!    gateway, on a small keep-alive HTTP/1.1 client pool. Deterministic by
//!    construction: fixed bodies, fixed counts, no randomness, no clocks in
//!    the payloads.
//!
//! A PGO profile is a union of hotness, not a traffic-mix replica: each shape
//! is driven in its own gateway process lifetime (see train.sh) and the
//! resulting `.profraw` files are merged. Missing an entire hot path is the
//! failure mode; skewed proportions are not (#967 measured the three-dialect
//! dilution at −1.09%).
//!
//! Std-only on purpose: the release Docker build compiles this tool right
//! before training, and a dependency-free build keeps that phase off the
//! network and off the workspace lockfile.
//!
//! Exit code 0 means every request of the shape succeeded. Anything else is
//! a training failure and MUST fail the calling pipeline — fail-closed, #967
//! hard gate 2. Never weaken a failure here into a warning.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Frames per streaming response. Real LLM streams run tens to hundreds of
/// chunks; a 3-frame canned stream would leave the relay loop body lukewarm.
const STREAM_FRAMES: usize = 48;

/// Pause between streamed frames so the gateway sees separate reads instead
/// of one coalesced buffer. Kept tiny: it paces training, it does not
/// simulate latency.
const FRAME_PACING: Duration = Duration::from_millis(1);

/// Per-socket read timeout. Generous because the instrumented gateway runs
/// several times slower than a release build.
const READ_TIMEOUT: Duration = Duration::from_secs(60);

// ---- shapes -----------------------------------------------------------------

/// Byte-identical to `bench/onthebench/lib.sh` `DEFAULT_BODY` — the public
/// benchmark board's default workload. This shape is load-bearing for the
/// program's published numbers and must NEVER be removed or reworded.
const BOARD_BODY: &str =
    r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}],"max_tokens":16}"#;

struct Shape {
    name: &'static str,
    path: &'static str,
    /// `None` = body built at runtime (the padded mid-size one).
    body: Option<&'static str>,
    anthropic_headers: bool,
}

/// The v1 training matrix (#967, extended at the 2026-08-13 scenario review):
/// 3 dialects × {stream, non-stream} + mid body + quota gate + native
/// anthropic passthrough + routing kind + embeddings. Error paths are
/// deliberately untrained — they are cold in production and PGO treating
/// them as cold is correct. Maintenance rule: this list mirrors the bench
/// suites; when the product grows a new hot path, add a shape in the same PR.
const SHAPES: &[Shape] = &[
    Shape {
        name: "chat-board",
        path: "/v1/chat/completions",
        body: Some(BOARD_BODY),
        anthropic_headers: false,
    },
    Shape {
        name: "chat-stream",
        path: "/v1/chat/completions",
        body: Some(
            r#"{"model":"gpt-4o-mini","stream":true,"messages":[{"role":"user","content":"hello"}],"max_tokens":64}"#,
        ),
        anthropic_headers: false,
    },
    Shape {
        name: "chat-mid",
        path: "/v1/chat/completions",
        body: None,
        anthropic_headers: false,
    },
    Shape {
        name: "bridge-messages",
        path: "/v1/messages",
        body: Some(
            r#"{"model":"gpt-4o-mini","max_tokens":32,"messages":[{"role":"user","content":"hello"}]}"#,
        ),
        anthropic_headers: true,
    },
    Shape {
        name: "bridge-messages-stream",
        path: "/v1/messages",
        body: Some(
            r#"{"model":"gpt-4o-mini","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hello"}]}"#,
        ),
        anthropic_headers: true,
    },
    Shape {
        name: "responses",
        path: "/v1/responses",
        body: Some(r#"{"model":"gpt-4o-mini","input":"hello"}"#),
        anthropic_headers: false,
    },
    Shape {
        name: "responses-stream",
        path: "/v1/responses",
        body: Some(r#"{"model":"gpt-4o-mini","input":"hello","stream":true}"#),
        anthropic_headers: false,
    },
    Shape {
        name: "chat-ratelimit",
        path: "/v1/chat/completions",
        body: Some(
            r#"{"model":"gpt-4o-mini-rl","messages":[{"role":"user","content":"hello"}],"max_tokens":16}"#,
        ),
        anthropic_headers: false,
    },
    Shape {
        name: "native-messages",
        path: "/v1/messages",
        body: Some(
            r#"{"model":"claude-pgo","max_tokens":32,"messages":[{"role":"user","content":"hello"}]}"#,
        ),
        anthropic_headers: true,
    },
    Shape {
        name: "native-messages-stream",
        path: "/v1/messages",
        body: Some(
            r#"{"model":"claude-pgo","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hello"}]}"#,
        ),
        anthropic_headers: true,
    },
    Shape {
        name: "chat-routing",
        path: "/v1/chat/completions",
        body: Some(
            r#"{"model":"gpt-router","messages":[{"role":"user","content":"hello"}],"max_tokens":16}"#,
        ),
        anthropic_headers: false,
    },
    Shape {
        name: "embeddings",
        path: "/v1/embeddings",
        body: Some(r#"{"model":"text-embedding-mock","input":"hello world"}"#),
        anthropic_headers: false,
    },
];

/// ~4 KB user content: the mid-size tier of the body-size axis.
fn mid_body() -> String {
    let filler = "The quick brown fox jumps over the lazy dog. ".repeat(88);
    format!(
        r#"{{"model":"gpt-4o-mini","messages":[{{"role":"user","content":"{filler}"}}],"max_tokens":16}}"#
    )
}

// ---- args -------------------------------------------------------------------

struct Args {
    mock_port: u16,
    gateway: String,
    api_key: String,
    shape: String,
    requests: u64,
    concurrency: u64,
    wait_secs: u64,
}

fn usage() -> ! {
    eprintln!(
        "usage: pgo-trainer --mock-port <p> --gateway <host:port> --api-key <k> \
         --shape <name> [--requests N] [--concurrency C] [--wait-secs S]\n\
         \x20      pgo-trainer --list-shapes"
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut a = Args {
        mock_port: 0,
        gateway: String::new(),
        api_key: String::new(),
        shape: String::new(),
        requests: 3000,
        concurrency: 8,
        wait_secs: 120,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|s| s == "--list-shapes") {
        for s in SHAPES {
            println!("{}", s.name);
        }
        std::process::exit(0);
    }
    let mut i = 0;
    while i < argv.len() {
        let need = |i: usize| argv.get(i + 1).cloned().unwrap_or_else(|| usage());
        match argv[i].as_str() {
            "--mock-port" => a.mock_port = need(i).parse().unwrap_or_else(|_| usage()),
            "--gateway" => a.gateway = need(i),
            "--api-key" => a.api_key = need(i),
            "--shape" => a.shape = need(i),
            "--requests" => a.requests = need(i).parse().unwrap_or_else(|_| usage()),
            "--concurrency" => a.concurrency = need(i).parse().unwrap_or_else(|_| usage()),
            "--wait-secs" => a.wait_secs = need(i).parse().unwrap_or_else(|_| usage()),
            _ => usage(),
        }
        i += 2;
    }
    if a.mock_port == 0 || a.gateway.is_empty() || a.api_key.is_empty() || a.shape.is_empty() {
        usage();
    }
    if a.concurrency == 0 || a.requests == 0 {
        usage();
    }
    a
}

fn main() {
    let args = parse_args();
    let shape = SHAPES
        .iter()
        .find(|s| s.name == args.shape)
        .unwrap_or_else(|| {
            eprintln!("FATAL: unknown shape '{}' (see --list-shapes)", args.shape);
            std::process::exit(2);
        });

    // Bind before anything else: a bind failure means the port is not ours
    // and the profile would be collected against a stranger. Hard exit.
    let listener = TcpListener::bind(("127.0.0.1", args.mock_port)).unwrap_or_else(|e| {
        eprintln!("FATAL: mock bind 127.0.0.1:{} failed: {e}", args.mock_port);
        std::process::exit(2);
    });
    thread::spawn(move || {
        for conn in listener.incoming().flatten() {
            thread::spawn(move || mock_conn(conn));
        }
    });

    if let Err(e) = wait_gateway(&args.gateway, args.wait_secs) {
        eprintln!("FATAL: gateway not ready: {e}");
        std::process::exit(3);
    }

    let body: Arc<String> = Arc::new(match shape.body {
        Some(b) => b.to_string(),
        None => mid_body(),
    });
    let ok = Arc::new(AtomicU64::new(0));
    let started = Instant::now();
    let mut workers = Vec::new();
    for w in 0..args.concurrency {
        // Deterministic split: the first (requests % concurrency) workers
        // take one extra request.
        let n = args.requests / args.concurrency + u64::from(w < args.requests % args.concurrency);
        if n == 0 {
            continue;
        }
        let gateway = args.gateway.clone();
        let api_key = args.api_key.clone();
        let body = Arc::clone(&body);
        let ok = Arc::clone(&ok);
        let path = shape.path;
        let anthropic = shape.anthropic_headers;
        workers.push(thread::spawn(move || -> Result<(), String> {
            drive(&gateway, &api_key, path, anthropic, &body, n, &ok)
        }));
    }

    let mut failures = Vec::new();
    for w in workers {
        match w.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => failures.push(e),
            Err(_) => failures.push("worker panicked".to_string()),
        }
    }
    let done = ok.load(Ordering::Relaxed);
    println!(
        "shape={} ok={}/{} elapsed={:.1}s",
        shape.name,
        done,
        args.requests,
        started.elapsed().as_secs_f64()
    );
    if !failures.is_empty() || done != args.requests {
        for f in failures.iter().take(4) {
            eprintln!("FATAL: {f}");
        }
        std::process::exit(1);
    }
}

// ---- driver (client side) -----------------------------------------------------

fn wait_gateway(addr: &str, secs: u64) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut last = String::from("never connected");
    while Instant::now() < deadline {
        match TcpStream::connect(addr) {
            Ok(mut s) => {
                s.set_read_timeout(Some(Duration::from_secs(5))).ok();
                let req =
                    format!("GET /livez HTTP/1.1\r\nhost: {addr}\r\nconnection: close\r\n\r\n");
                if s.write_all(req.as_bytes()).is_ok() {
                    let mut buf = [0u8; 64];
                    if let Ok(n) = s.read(&mut buf) {
                        let head = String::from_utf8_lossy(&buf[..n]).to_string();
                        if head.starts_with("HTTP/1.1 200") || head.starts_with("HTTP/1.0 200") {
                            return Ok(());
                        }
                        last = head.lines().next().unwrap_or("").to_string();
                    }
                }
            }
            Err(e) => last = e.to_string(),
        }
        thread::sleep(Duration::from_millis(200));
    }
    Err(format!("gave up after {secs}s (last: {last})"))
}

fn connect(gateway: &str) -> Result<(BufReader<TcpStream>, TcpStream), String> {
    let stream = TcpStream::connect(gateway).map_err(|e| format!("connect {gateway}: {e}"))?;
    stream.set_read_timeout(Some(READ_TIMEOUT)).ok();
    stream.set_nodelay(true).ok();
    let write = stream.try_clone().map_err(|e| format!("clone: {e}"))?;
    Ok((BufReader::new(stream), write))
}

fn drive(
    gateway: &str,
    api_key: &str,
    path: &str,
    anthropic: bool,
    body: &str,
    n: u64,
    ok: &AtomicU64,
) -> Result<(), String> {
    let extra = if anthropic {
        "anthropic-version: 2023-06-01\r\n"
    } else {
        ""
    };
    let request = format!(
        "POST {path} HTTP/1.1\r\nhost: {gateway}\r\nauthorization: Bearer {api_key}\r\n\
         {extra}content-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    );
    let (mut reader, mut writer) = connect(gateway)?;
    for i in 0..n {
        writer
            .write_all(request.as_bytes())
            .map_err(|e| format!("{path} req {i}: write: {e}"))?;
        let (status, reused) =
            read_response(&mut reader).map_err(|e| format!("{path} req {i}: {e}"))?;
        if status != 200 {
            return Err(format!("{path} req {i}: HTTP {status}"));
        }
        ok.fetch_add(1, Ordering::Relaxed);
        if !reused {
            let (r, w) = connect(gateway)?;
            reader = r;
            writer = w;
        }
    }
    Ok(())
}

/// Read one HTTP/1.1 response. Returns (status, connection-reusable).
/// Handles both framings the gateway emits: content-length (JSON) and
/// chunked (SSE) — reading to the zero chunk is dialect-agnostic, so one
/// client covers every streaming shape.
fn read_response(r: &mut BufReader<TcpStream>) -> Result<(u16, bool), String> {
    let mut line = String::new();
    r.read_line(&mut line).map_err(|e| format!("status: {e}"))?;
    if line.is_empty() {
        return Err("connection closed before status line".into());
    }
    let status: u16 = line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("bad status line: {line:?}"))?;

    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    let mut close = false;
    loop {
        let mut h = String::new();
        r.read_line(&mut h).map_err(|e| format!("header: {e}"))?;
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        let lower = h.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            content_length = Some(
                v.trim()
                    .parse()
                    .map_err(|_| format!("bad content-length: {h:?}"))?,
            );
        } else if lower.starts_with("transfer-encoding:") && lower.contains("chunked") {
            chunked = true;
        } else if lower.starts_with("connection:") && lower.contains("close") {
            close = true;
        }
    }

    if chunked {
        loop {
            let mut sz = String::new();
            r.read_line(&mut sz)
                .map_err(|e| format!("chunk size: {e}"))?;
            let sz = usize::from_str_radix(sz.trim().split(';').next().unwrap_or(""), 16)
                .map_err(|_| format!("bad chunk size: {sz:?}"))?;
            let mut chunk = vec![0u8; sz + 2]; // data + CRLF
            r.read_exact(&mut chunk)
                .map_err(|e| format!("chunk body: {e}"))?;
            if sz == 0 {
                break;
            }
        }
    } else if let Some(len) = content_length {
        let mut body = vec![0u8; len];
        r.read_exact(&mut body).map_err(|e| format!("body: {e}"))?;
    } else {
        // No length and not chunked: body runs to EOF; connection is done.
        let mut sink = Vec::new();
        r.read_to_end(&mut sink).map_err(|e| format!("body: {e}"))?;
        return Ok((status, false));
    }
    Ok((status, !close))
}

// ---- mock upstream (server side) ----------------------------------------------

fn mock_conn(stream: TcpStream) {
    stream.set_read_timeout(Some(READ_TIMEOUT)).ok();
    stream.set_nodelay(true).ok();
    let write = match stream.try_clone() {
        Ok(w) => w,
        Err(_) => return,
    };
    let mut reader = BufReader::new(stream);
    let mut writer = write;
    loop {
        let (path, body) = match read_mock_request(&mut reader) {
            Ok(Some(v)) => v,
            _ => return,
        };
        let stream_requested =
            body.contains(r#""stream":true"#) || body.contains(r#""stream": true"#);
        let result = match path.as_str() {
            "/v1/chat/completions" => {
                if stream_requested {
                    write_sse(&mut writer, &chat_sse_frames())
                } else {
                    write_json(&mut writer, CHAT_JSON)
                }
            }
            "/v1/responses" => {
                if stream_requested {
                    write_sse(&mut writer, &responses_sse_frames())
                } else {
                    write_json(&mut writer, RESPONSES_JSON)
                }
            }
            "/v1/messages" => {
                if stream_requested {
                    write_sse(&mut writer, &anthropic_sse_frames())
                } else {
                    write_json(&mut writer, ANTHROPIC_JSON)
                }
            }
            "/v1/embeddings" => write_json(&mut writer, EMBEDDINGS_JSON),
            // Anything else is a routing bug in the training setup: answer
            // 404 so the driving request fails loudly instead of training a
            // wrong path.
            _ => {
                let resp = format!(
                    "HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                );
                let _ = writer.write_all(resp.as_bytes());
                return;
            }
        };
        if result.is_err() {
            return;
        }
    }
}

fn read_mock_request(r: &mut BufReader<TcpStream>) -> Result<Option<(String, String)>, String> {
    let mut line = String::new();
    let n = r.read_line(&mut line).map_err(|e| e.to_string())?;
    if n == 0 {
        return Ok(None); // clean EOF between keep-alive requests
    }
    let path = line
        .split_whitespace()
        .nth(1)
        .unwrap_or("")
        .split('?')
        .next()
        .unwrap_or("")
        .to_string();
    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        r.read_line(&mut h).map_err(|e| e.to_string())?;
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        let lower = h.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        r.read_exact(&mut body).map_err(|e| e.to_string())?;
    }
    Ok(Some((path, String::from_utf8_lossy(&body).into_owned())))
}

fn write_json(w: &mut TcpStream, body: &str) -> std::io::Result<()> {
    let resp = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    );
    w.write_all(resp.as_bytes())
}

/// Stream SSE frames over chunked transfer encoding, one chunk per frame with
/// a short pause, so the gateway's relay loop sees many small reads — the
/// production streaming profile, not one buffered blob.
fn write_sse(w: &mut TcpStream, frames: &[String]) -> std::io::Result<()> {
    w.write_all(
        b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
          cache-control: no-cache\r\ntransfer-encoding: chunked\r\n\r\n",
    )?;
    for f in frames {
        let chunk = format!("{:x}\r\n{f}\r\n", f.len());
        w.write_all(chunk.as_bytes())?;
        w.flush()?;
        thread::sleep(FRAME_PACING);
    }
    w.write_all(b"0\r\n\r\n")
}

// ---- canned upstream bodies -----------------------------------------------------

const CHAT_JSON: &str = r#"{"id":"chatcmpl-pgo","object":"chat.completion","created":0,"model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"PGO training reply."},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":12,"total_tokens":21}}"#;

const RESPONSES_JSON: &str = r#"{"id":"resp-pgo","object":"response","status":"completed","model":"gpt-4o-mini","output":[{"type":"message","id":"msg-pgo","role":"assistant","content":[{"type":"output_text","text":"PGO training reply."}]}],"usage":{"input_tokens":9,"output_tokens":12,"total_tokens":21}}"#;

const ANTHROPIC_JSON: &str = r#"{"id":"msg-pgo","type":"message","role":"assistant","model":"claude-pgo","content":[{"type":"text","text":"PGO training reply."}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":9,"output_tokens":12}}"#;

const EMBEDDINGS_JSON: &str = r#"{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.01,-0.02,0.03,-0.04,0.05,-0.06,0.07,-0.08,0.09,-0.1,0.11,-0.12,0.13,-0.14,0.15,-0.16]}],"model":"text-embedding-mock","usage":{"prompt_tokens":3,"total_tokens":3}}"#;

/// OpenAI chat SSE: role delta, N content deltas, a final chunk carrying
/// `finish_reason` + `usage`, then the `[DONE]` sentinel.
fn chat_sse_frames() -> Vec<String> {
    let mut f = Vec::with_capacity(STREAM_FRAMES + 1);
    f.push(concat!(
        r#"data: {"id":"chatcmpl-pgo","object":"chat.completion.chunk","created":0,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
        "\n\n"
    )
    .to_string());
    for _ in 0..STREAM_FRAMES - 2 {
        f.push(concat!(
            r#"data: {"id":"chatcmpl-pgo","object":"chat.completion.chunk","created":0,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"chunk "},"finish_reason":null}]}"#,
            "\n\n"
        )
        .to_string());
    }
    f.push(concat!(
        r#"data: {"id":"chatcmpl-pgo","object":"chat.completion.chunk","created":0,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":46,"total_tokens":55}}"#,
        "\n\n"
    )
    .to_string());
    f.push("data: [DONE]\n\n".to_string());
    f
}

/// OpenAI Responses SSE: text deltas, then the terminal `response.completed`
/// event whose `usage` block the gateway parses in-flight (#808), then
/// `[DONE]`.
fn responses_sse_frames() -> Vec<String> {
    let mut f = Vec::with_capacity(STREAM_FRAMES + 1);
    for _ in 0..STREAM_FRAMES - 1 {
        f.push(
            concat!(
                "event: response.output_text.delta\n",
                r#"data: {"type":"response.output_text.delta","delta":"chunk "}"#,
                "\n\n"
            )
            .to_string(),
        );
    }
    f.push(concat!(
        "event: response.completed\n",
        r#"data: {"type":"response.completed","response":{"id":"resp-pgo","object":"response","status":"completed","model":"gpt-4o-mini","output":[{"type":"message","id":"msg-pgo","role":"assistant","content":[{"type":"output_text","text":"PGO training reply."}]}],"usage":{"input_tokens":9,"output_tokens":47,"total_tokens":56}}}"#,
        "\n\n"
    )
    .to_string());
    f.push("data: [DONE]\n\n".to_string());
    f
}

/// Anthropic Messages SSE: message_start → content_block deltas →
/// message_delta (usage) → message_stop, per the Messages streaming format.
fn anthropic_sse_frames() -> Vec<String> {
    let mut f = Vec::with_capacity(STREAM_FRAMES);
    f.push(concat!(
        "event: message_start\n",
        r#"data: {"type":"message_start","message":{"id":"msg-pgo","type":"message","role":"assistant","model":"claude-pgo","content":[],"stop_reason":null,"usage":{"input_tokens":9,"output_tokens":1}}}"#,
        "\n\n"
    )
    .to_string());
    f.push(concat!(
        "event: content_block_start\n",
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        "\n\n"
    )
    .to_string());
    for _ in 0..STREAM_FRAMES - 5 {
        f.push(concat!(
            "event: content_block_delta\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"chunk "}}"#,
            "\n\n"
        )
        .to_string());
    }
    f.push(
        concat!(
            "event: content_block_stop\n",
            r#"data: {"type":"content_block_stop","index":0}"#,
            "\n\n"
        )
        .to_string(),
    );
    f.push(concat!(
        "event: message_delta\n",
        r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":43}}"#,
        "\n\n"
    )
    .to_string());
    f.push("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string());
    f
}
