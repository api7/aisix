//! Periodic `POST /dp/heartbeat` so cp-api knows the DP is alive.
//!
//! Protocol: prd-09 §9.3.5 and §9.10. The DP sends `{ dp_id,
//! uptime_seconds, version }` over HTTPS with `Authorization:
//! Bearer <dp_id>` — Phase 1 auth (Phase 2 upgrades to mTLS client
//! cert once cp-api terminates mTLS).
//!
//! Shape:
//!   - spawned once from `main` after registration/cert load is
//!     complete
//!   - ticks at the interval returned by the register response
//!     (default 15s)
//!   - individual heartbeats fail fast on network errors; the
//!     ticker keeps running so a transient outage doesn't stop the
//!     DP from being seen when the CP comes back
//!   - cancelled via the shared `watch::Receiver<bool>` so graceful
//!     shutdown doesn't leave an in-flight request dangling

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context};
use serde::Serialize;
use tokio::sync::watch;

/// Configuration captured at register time. All three fields come
/// from the register response (see prd-09 §9.3.5) and are immutable
/// for the life of the heartbeat worker.
#[derive(Debug, Clone)]
pub struct HeartbeatConfig {
    pub url: String,
    pub dp_id: String,
    pub interval: Duration,
}

impl HeartbeatConfig {
    /// Clamp the server-suggested interval into a safe band. Defence
    /// against a buggy CP config that returns 0 or a week.
    pub fn sanitised(url: String, dp_id: String, interval: Duration) -> Self {
        const MIN: Duration = Duration::from_secs(5);
        const MAX: Duration = Duration::from_secs(300);
        let interval = interval.clamp(MIN, MAX);
        Self {
            url,
            dp_id,
            interval,
        }
    }
}

/// Spawn the heartbeat worker. Returns the JoinHandle so `main` can
/// await it at shutdown. Errors during individual heartbeats are
/// logged, not propagated — a heartbeat that can't reach the CP is
/// noisy, not fatal.
pub fn spawn(
    cfg: HeartbeatConfig,
    mut cancel: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        run(cfg, &mut cancel).await;
    })
}

async fn run(cfg: HeartbeatConfig, cancel: &mut watch::Receiver<bool>) {
    let client = match build_client() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            tracing::error!(error = %e, "heartbeat: build reqwest client failed; disabled");
            return;
        }
    };
    let started = Instant::now();
    let mut ticker = tokio::time::interval(cfg.interval);
    // Skip the catch-up fire — we want the first beat to happen
    // immediately at spawn but subsequent ones to follow the tick
    // schedule without bursting if we fall behind.
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    tracing::info!(
        url = %cfg.url,
        dp_id = %cfg.dp_id,
        interval_secs = cfg.interval.as_secs(),
        "heartbeat started",
    );

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                let uptime = started.elapsed().as_secs() as i64;
                match send(&client, &cfg, uptime).await {
                    Ok(()) => tracing::debug!("heartbeat ok"),
                    Err(e) => tracing::warn!(error = %e, "heartbeat failed"),
                }
            }
            _ = cancel.changed() => {
                if *cancel.borrow() {
                    tracing::info!("heartbeat shutting down");
                    return;
                }
            }
        }
    }
}

#[derive(Debug, Serialize)]
struct HeartbeatBody<'a> {
    dp_id: &'a str,
    uptime_seconds: i64,
    version: &'a str,
}

async fn send(client: &reqwest::Client, cfg: &HeartbeatConfig, uptime: i64) -> anyhow::Result<()> {
    let resp = client
        .post(&cfg.url)
        .bearer_auth(&cfg.dp_id)
        .json(&HeartbeatBody {
            dp_id: &cfg.dp_id,
            uptime_seconds: uptime,
            version: env!("CARGO_PKG_VERSION"),
        })
        .send()
        .await
        .with_context(|| format!("POST {}", cfg.url))?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(anyhow!(
            "heartbeat {} returned {} — {}",
            cfg.url,
            status,
            body.trim().chars().take(200).collect::<String>()
        ));
    }
    Ok(())
}

fn build_client() -> anyhow::Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .user_agent(format!("aisix-dp/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .context("build reqwest client")
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_string_contains, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg(url: String) -> HeartbeatConfig {
        HeartbeatConfig::sanitised(url, "dp_test_node_42".into(), Duration::from_millis(50))
    }

    #[tokio::test]
    async fn send_posts_dp_id_and_bearer() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/heartbeat"))
            .and(header("Authorization", "Bearer dp_test_node_42"))
            .and(body_string_contains("\"dp_id\":\"dp_test_node_42\""))
            .and(body_string_contains("\"uptime_seconds\":"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true
            })))
            .mount(&server)
            .await;

        let c = build_client().unwrap();
        send(&c, &cfg(format!("{}/dp/heartbeat", server.uri())), 7)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn send_propagates_non_success_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/heartbeat"))
            .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
                "error": {"code": "DP_NOT_FOUND", "message": "no registered DP matches this id"}
            })))
            .mount(&server)
            .await;

        let c = build_client().unwrap();
        let err = send(&c, &cfg(format!("{}/dp/heartbeat", server.uri())), 7)
            .await
            .unwrap_err();
        let s = format!("{err:#}");
        assert!(s.contains("404"), "expected status in error: {s}");
        assert!(s.contains("DP_NOT_FOUND"), "expected body in error: {s}");
    }

    #[tokio::test]
    async fn run_stops_on_cancel() {
        // Start a server that 200s fast enough that the first tick
        // completes, then cancel and make sure the task exits.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/heartbeat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .mount(&server)
            .await;

        let (tx, rx) = watch::channel(false);
        let handle = spawn(cfg(format!("{}/dp/heartbeat", server.uri())), rx);

        tokio::time::sleep(Duration::from_millis(150)).await;
        tx.send(true).unwrap();

        // Runs to completion within a small grace window.
        tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("heartbeat did not stop after cancel")
            .unwrap();
    }

    #[test]
    fn sanitised_interval_clamps_extremes() {
        let a =
            HeartbeatConfig::sanitised("http://x".into(), "id".into(), Duration::from_millis(10));
        assert_eq!(a.interval, Duration::from_secs(5));

        let b =
            HeartbeatConfig::sanitised("http://x".into(), "id".into(), Duration::from_secs(86_400));
        assert_eq!(b.interval, Duration::from_secs(300));

        let c = HeartbeatConfig::sanitised("http://x".into(), "id".into(), Duration::from_secs(30));
        assert_eq!(c.interval, Duration::from_secs(30));
    }
}
