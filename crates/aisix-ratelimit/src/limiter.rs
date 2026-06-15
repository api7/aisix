//! Two-phase limiter keyed on an opaque `key` (the caller's ApiKey id /
//! policy bucket in production), backed by a pluggable [`RateStore`].
//!
//! Phase 1 — **pre-commit**, called before the upstream request fires:
//! - check concurrency (acquire a slot or fail)
//! - check + increment RPS / RPM / RPH / RPD counters
//! - *check-only* TPM / TPD (we don't know the token cost yet)
//!
//! Phase 2 — **post-deduct**, called after the upstream response
//! completes:
//! - add actual `prompt_tokens + completion_tokens` to TPM / TPD
//! - release the concurrency slot
//!
//! The returned [`Reservation`] handle releases the concurrency slot on
//! drop if `commit_tokens` isn't called, so callers can't forget on the
//! error path.
//!
//! The counters live wherever the [`RateStore`] keeps them: the default
//! [`crate::store::local::LocalStore`] is per-process (historical
//! behaviour), while [`crate::store::redis::RedisStore`] shares them
//! across every DP replica so a cluster enforces one global window
//! (api7/AISIX-Cloud#798).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use aisix_core::RateLimit;

use crate::clock::Clock;
use crate::error::RateLimitError;
use crate::store::local::LocalStore;
use crate::store::RateStore;

/// Current window state for a single key, returned by [`Limiter::peek`].
/// Used by the proxy handlers to inject the `x-ratelimit-*` response
/// headers that OpenAI SDK clients expect.
#[derive(Debug, Clone)]
pub struct RateLimitStatus {
    pub rpm_limit: Option<u64>,
    pub rpm_used: u64,
    pub rpm_reset_secs: u64,
    pub tpm_limit: Option<u64>,
    pub tpm_used: u64,
    pub tpm_reset_secs: u64,
    pub concurrency_limit: Option<u32>,
    pub in_flight: u32,
}

impl RateLimitStatus {
    pub fn rpm_remaining(&self) -> Option<u64> {
        self.rpm_limit.map(|lim| lim.saturating_sub(self.rpm_used))
    }
    pub fn tpm_remaining(&self) -> Option<u64> {
        self.tpm_limit.map(|lim| lim.saturating_sub(self.tpm_used))
    }
}

/// Two-phase limiter over a shared or local [`RateStore`].
pub struct Limiter {
    store: Arc<dyn RateStore>,
    /// Process-unique reservation id prefix (`<uuid>:`), so concurrency
    /// members are globally distinct across replicas in the shared store.
    member_prefix: String,
    seq: AtomicU64,
}

impl Limiter {
    /// Default per-process limiter (in-memory `LocalStore`).
    pub fn new() -> Self {
        Self::with_store(Arc::new(LocalStore::new()))
    }

    /// Build over a specific store — the server bootstrap passes a
    /// `RedisStore` when a shared backend is configured.
    pub fn with_store(store: Arc<dyn RateStore>) -> Self {
        Self {
            store,
            member_prefix: format!("{}:", uuid::Uuid::new_v4().simple()),
            seq: AtomicU64::new(0),
        }
    }

    /// Test helper: a local store driven by an injectable clock.
    pub fn local_with_clock<C: Clock>(clock: C) -> Self {
        Self::with_store(Arc::new(LocalStore::with_clock(clock)))
    }

    fn next_member(&self) -> String {
        let n = self.seq.fetch_add(1, Ordering::Relaxed);
        format!("{}{n}", self.member_prefix)
    }

    /// Pre-commit phase. Returns a [`Reservation`] that must be finalised
    /// via [`Reservation::commit_tokens`] or dropped to release the
    /// concurrency slot automatically.
    pub async fn pre_commit(
        &self,
        key: &str,
        limits: &RateLimit,
    ) -> Result<Reservation, RateLimitError> {
        let member = self.next_member();
        self.store.acquire(key, limits, &member).await?;
        Ok(Reservation {
            store: Arc::clone(&self.store),
            key: key.to_string(),
            member,
            committed: false,
        })
    }

    /// Add `tokens` to the post-deduct TPM/TPD counters for `key` without
    /// going through a [`Reservation`]. Used by the streaming chat path:
    /// at pre_commit time the upstream token cost isn't known, so the
    /// concurrency slot is held by a [`StreamConcurrencyGuard`] and the
    /// tokens are accounted here when the terminal SSE usage frame lands
    /// (issue #108). No-op on zero tokens.
    pub fn add_tokens_post_stream(&self, key: &str, tokens: u64) {
        if tokens == 0 {
            return;
        }
        self.store.add_tokens(key, tokens);
    }

    /// Snapshot of the current rate-limit state for a key, used to inject
    /// `x-ratelimit-*` response headers. Returns `None` when there is
    /// nothing meaningful to report. Read-only — affects no counters.
    pub async fn peek(&self, key: &str, limits: &RateLimit) -> Option<RateLimitStatus> {
        self.store.peek(key, limits).await
    }
}

impl Default for Limiter {
    fn default() -> Self {
        Self::new()
    }
}

/// Reservation guard. Dropping without a `commit_tokens` call is still
/// safe — the concurrency slot is released, just no tokens are counted.
pub struct Reservation {
    store: Arc<dyn RateStore>,
    key: String,
    member: String,
    committed: bool,
}

impl std::fmt::Debug for Reservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reservation")
            .field("key", &self.key)
            .field("committed", &self.committed)
            .finish()
    }
}

impl Reservation {
    /// Post-deduct phase. Records the actual token cost against TPM/TPD
    /// and releases the concurrency slot.
    pub async fn commit_tokens(mut self, tokens: u64) {
        self.store.commit(&self.key, tokens, &self.member).await;
        self.committed = true;
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        self.store.release(&self.key, &self.member);
    }
}

/// Wraps multiple [`Reservation`]s across rate-limit layers (api_key,
/// model, team, member). Commits all with the same token count; dropping
/// releases all concurrency slots.
pub struct MultiReservation {
    reservations: Vec<Reservation>,
}

impl MultiReservation {
    pub fn new(reservations: Vec<Reservation>) -> Self {
        Self { reservations }
    }

    /// Commit the actual token cost to every layer.
    pub async fn commit_tokens(self, tokens: u64) {
        for r in self.reservations {
            r.commit_tokens(tokens).await;
        }
    }

    /// Return owned keys for post-stream token accounting.
    pub fn keys(&self) -> Vec<String> {
        self.reservations.iter().map(|r| r.key.clone()).collect()
    }

    /// Convert into an owned [`StreamConcurrencyGuard`] for the streaming
    /// path. The per-layer concurrency slots stay held — they are NOT
    /// released here — and are released only when the returned guard drops,
    /// i.e. at stream completion or cancellation. Token accounting still
    /// happens via [`Limiter::add_tokens_post_stream`].
    ///
    /// A borrow-based reservation couldn't outlive the request handler, so
    /// the pre-fix streaming path dropped it at handler return; that
    /// released the slot before the stream finished, letting a key capped
    /// at N run many more than N simultaneous streams (#450).
    #[must_use = "dropping the returned guard immediately releases the concurrency \
                  slot, recreating the early-release bug this fixes"]
    pub fn into_stream_hold(mut self) -> StreamConcurrencyGuard {
        let holds = self
            .reservations
            .iter_mut()
            .map(|r| {
                // Defuse each reservation's Drop so it doesn't release the
                // slot now; the returned guard owns release from here on.
                r.committed = true;
                (Arc::clone(&r.store), r.key.clone(), r.member.clone())
            })
            .collect();
        StreamConcurrencyGuard {
            holds,
            released: false,
        }
    }
}

impl std::fmt::Debug for MultiReservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiReservation")
            .field("layers", &self.reservations.len())
            .finish()
    }
}

/// Owned concurrency hold for the streaming path. Releases the
/// concurrency slot(s) on drop — i.e. when the stream completes or is
/// cancelled — instead of at handler return. See
/// [`MultiReservation::into_stream_hold`].
pub struct StreamConcurrencyGuard {
    /// `(store, key, member)` per held layer.
    holds: Vec<(Arc<dyn RateStore>, String, String)>,
    released: bool,
}

impl StreamConcurrencyGuard {
    fn release_now(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        for (store, key, member) in &self.holds {
            store.release(key, member);
        }
    }
}

impl std::fmt::Debug for StreamConcurrencyGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamConcurrencyGuard")
            .field("layers", &self.holds.len())
            .field("released", &self.released)
            .finish()
    }
}

impl Drop for StreamConcurrencyGuard {
    fn drop(&mut self) {
        self.release_now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::TestClock;

    fn limits(rpm: Option<u64>, tpm: Option<u64>, concurrency: Option<u32>) -> RateLimit {
        RateLimit {
            rps: None,
            rpm,
            rph: None,
            rpd: None,
            tpm,
            tpd: None,
            concurrency,
        }
    }

    /// Helper for the rps/rph/compensator tests added by #426.
    fn limits_full(
        rps: Option<u64>,
        rpm: Option<u64>,
        rph: Option<u64>,
        rpd: Option<u64>,
    ) -> RateLimit {
        RateLimit {
            rps,
            rpm,
            rph,
            rpd,
            tpm: None,
            tpd: None,
            concurrency: None,
        }
    }

    #[tokio::test]
    async fn rpm_caps_request_count_in_window() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock.clone());
        let l = limits(Some(2), None, None);

        let _r1 = limiter.pre_commit("k1", &l).await.unwrap();
        let _r2 = limiter.pre_commit("k1", &l).await.unwrap();
        let err = limiter.pre_commit("k1", &l).await.unwrap_err();
        match err {
            RateLimitError::Requests {
                retry_after_secs, ..
            } => {
                assert!(retry_after_secs > 0);
            }
            other => panic!("expected Requests, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rpm_resets_after_window_rollover() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock.clone());
        let l = limits(Some(1), None, None);

        let _r1 = limiter.pre_commit("k1", &l).await.unwrap();
        assert!(limiter.pre_commit("k1", &l).await.is_err());

        // Jump past the minute boundary.
        clock.advance(61);
        let _r2 = limiter.pre_commit("k1", &l).await.unwrap();
    }

    #[tokio::test]
    async fn concurrency_limit_blocks_new_reservations() {
        let clock = TestClock::new(0);
        let limiter = Limiter::local_with_clock(clock.clone());
        let l = limits(None, None, Some(2));

        let r1 = limiter.pre_commit("k1", &l).await.unwrap();
        let r2 = limiter.pre_commit("k1", &l).await.unwrap();
        assert!(matches!(
            limiter.pre_commit("k1", &l).await.unwrap_err(),
            RateLimitError::Concurrency,
        ));

        // Drop r1 — concurrency should free up.
        drop(r1);
        let _r3 = limiter.pre_commit("k1", &l).await.unwrap();
        drop(r2);
    }

    #[tokio::test]
    async fn token_commit_updates_post_deduct_counters() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock.clone());
        let l = limits(Some(10), Some(1_000), None);

        let r1 = limiter.pre_commit("k1", &l).await.unwrap();
        r1.commit_tokens(600).await;

        // TPM now at 600. Next pre_commit with a strict TPM should still
        // succeed because 600 <= 1000.
        let _r2 = limiter.pre_commit("k1", &l).await.unwrap();
    }

    #[tokio::test]
    async fn tpm_blocks_next_request_once_previous_exhausted_the_window() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock.clone());
        let l = limits(Some(10), Some(1_000), None);

        let r1 = limiter.pre_commit("k1", &l).await.unwrap();
        r1.commit_tokens(1_500).await; // overshoot — allowed for the in-flight request

        // Next pre_commit sees tpm > 1000 and refuses.
        let err = limiter.pre_commit("k1", &l).await.unwrap_err();
        assert!(matches!(err, RateLimitError::Tokens { .. }));

        clock.advance(61); // roll the window
        let _r2 = limiter.pre_commit("k1", &l).await.unwrap();
    }

    #[tokio::test]
    async fn reservations_for_different_keys_do_not_collide() {
        let clock = TestClock::new(0);
        let limiter = Limiter::local_with_clock(clock);
        let l = limits(Some(1), None, None);

        let _r_a = limiter.pre_commit("alpha", &l).await.unwrap();
        let _r_b = limiter.pre_commit("beta", &l).await.unwrap();
    }

    #[tokio::test]
    async fn drop_without_commit_still_releases_concurrency_permit() {
        let clock = TestClock::new(0);
        let limiter = Limiter::local_with_clock(clock);
        let l = limits(None, None, Some(1));

        {
            let _r = limiter.pre_commit("k1", &l).await.unwrap();
        } // dropped
        let _r2 = limiter.pre_commit("k1", &l).await.unwrap();
    }

    #[tokio::test]
    async fn peek_returns_none_for_unknown_key() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock);
        assert!(limiter
            .peek("unknown", &RateLimit::default())
            .await
            .is_none());
    }

    #[tokio::test]
    async fn peek_reports_current_window_counts() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock.clone());
        let l = limits(Some(60), Some(100_000), Some(10));

        let r = limiter.pre_commit("k1", &l).await.unwrap();
        r.commit_tokens(500).await;

        let status = limiter.peek("k1", &l).await.unwrap();
        assert_eq!(status.rpm_limit, Some(60));
        assert_eq!(status.rpm_used, 1);
        assert_eq!(status.rpm_remaining(), Some(59));
        assert_eq!(status.tpm_limit, Some(100_000));
        assert_eq!(status.tpm_used, 500);
        assert_eq!(status.tpm_remaining(), Some(99_500));
        assert_eq!(status.in_flight, 0); // committed → released
    }

    #[tokio::test]
    async fn peek_reflects_in_flight_count_during_dispatch() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock);
        let l = limits(None, None, Some(5));

        let _r1 = limiter.pre_commit("k1", &l).await.unwrap();
        let _r2 = limiter.pre_commit("k1", &l).await.unwrap();
        let status = limiter.peek("k1", &l).await.unwrap();
        assert_eq!(status.in_flight, 2);
        assert_eq!(status.concurrency_limit, Some(5));
    }

    #[tokio::test]
    async fn no_limits_means_no_rejections() {
        let clock = TestClock::new(0);
        let limiter = Limiter::local_with_clock(clock);
        let l = RateLimit::default();

        for _ in 0..100 {
            let r = limiter.pre_commit("k1", &l).await.unwrap();
            r.commit_tokens(1_000).await;
        }
    }

    // ---- regression coverage for issue #109 -------------------------
    // The previous compensation path overwrote `s.rpm` with a fresh
    // counter, wiping concurrent siblings' increments. The fix replaces
    // the reset with a precise -1 decrement; these tests pin both the
    // "siblings are preserved" and "fresh window is not granted"
    // properties at the level the exploit happens.

    #[tokio::test]
    async fn rpd_rejection_does_not_grant_fresh_rpm_window() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock.clone());
        let l = RateLimit {
            rps: None,
            rpm: Some(10),
            rph: None,
            rpd: Some(20),
            tpm: None,
            tpd: None,
            concurrency: None,
        };
        // Soak up 19 RPM = 19 RPD across two minutes so RPD is at 19.
        for i in 0..19 {
            if i == 10 {
                clock.advance(61); // roll RPM, keep RPD
            }
            let _r = limiter.pre_commit("k1", &l).await.unwrap();
        }
        // RPM in current minute = 9 (after rollover), RPD = 19. One more
        // goes through (RPM 10/10, RPD 20/20).
        let _r = limiter.pre_commit("k1", &l).await.unwrap();
        // The 21st request must fail — RPD is full.
        let err = limiter.pre_commit("k1", &l).await.unwrap_err();
        assert!(
            matches!(err, RateLimitError::Requests { .. }),
            "expected RPD rejection, got {err:?}"
        );
        // The next request must STILL fail RPM — proving RPM wasn't wiped.
        let err2 = limiter.pre_commit("k1", &l).await.unwrap_err();
        assert!(
            matches!(err2, RateLimitError::Requests { .. }),
            "RPM should still be capped after RPD rejection; got {err2:?}"
        );
        let status = limiter.peek("k1", &l).await.unwrap();
        assert_eq!(status.rpm_used, 10, "RPM should not have been reset");
    }

    #[tokio::test]
    async fn rpd_rejection_preserves_concurrent_rpm_increments() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock.clone());
        let l = RateLimit {
            rps: None,
            rpm: Some(100), // very high — RPM never trips here
            rph: None,
            rpd: Some(5),
            tpm: None,
            tpd: None,
            concurrency: None,
        };
        for _ in 0..5 {
            let _r = limiter.pre_commit("k1", &l).await.unwrap();
        }
        // RPM=5, RPD=5/5. Sixth request fails RPD.
        let err = limiter.pre_commit("k1", &l).await.unwrap_err();
        assert!(matches!(err, RateLimitError::Requests { .. }));
        let status = limiter.peek("k1", &l).await.unwrap();
        assert_eq!(
            status.rpm_used, 5,
            "rpd rejection wiped concurrent rpm increments"
        );
    }

    // ---- regression coverage for issue #108 -------------------------

    #[tokio::test]
    async fn add_tokens_post_stream_increments_tpm() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock);
        let l = limits(Some(10), Some(1_000), None);

        {
            let _r = limiter.pre_commit("k1", &l).await.unwrap();
        }
        assert_eq!(
            limiter.peek("k1", &l).await.unwrap().tpm_used,
            0,
            "TPM should be 0 right after pre_commit + drop",
        );

        limiter.add_tokens_post_stream("k1", 750);
        assert_eq!(
            limiter.peek("k1", &l).await.unwrap().tpm_used,
            750,
            "TPM should reflect the post-stream commit",
        );
    }

    #[tokio::test]
    async fn add_tokens_post_stream_zero_is_a_noop() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock);
        limiter.add_tokens_post_stream("never-seen", 0);
        assert!(
            limiter
                .peek("never-seen", &RateLimit::default())
                .await
                .is_none(),
            "add_tokens_post_stream(0) should not lazily-create state",
        );
    }

    #[tokio::test]
    async fn streaming_path_tpm_cap_blocks_next_request_after_post_stream_commit() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock);
        let l = limits(Some(100), Some(1_000), None);

        {
            let _r = limiter.pre_commit("k1", &l).await.unwrap();
        }
        limiter.add_tokens_post_stream("k1", 1_500);

        let err = limiter.pre_commit("k1", &l).await.unwrap_err();
        assert!(
            matches!(err, RateLimitError::Tokens { .. }),
            "TPM cap should block the next request after streaming over-shoot; got {err:?}",
        );
    }

    // --- MultiReservation tests ----------------------------------------

    #[tokio::test]
    async fn multi_reservation_commit_tokens_updates_all_layers() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock.clone());
        let l = limits(None, Some(1000), None);

        let r1 = limiter.pre_commit("api_key:k1", &l).await.unwrap();
        let r2 = limiter.pre_commit("model:gpt-4o", &l).await.unwrap();
        let multi = MultiReservation::new(vec![r1, r2]);

        multi.commit_tokens(500).await;

        let s1 = limiter.peek("api_key:k1", &l).await.unwrap();
        let s2 = limiter.peek("model:gpt-4o", &l).await.unwrap();
        assert_eq!(s1.tpm_used, 500);
        assert_eq!(s2.tpm_used, 500);
    }

    #[tokio::test]
    async fn multi_reservation_drop_releases_all_concurrency() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock.clone());
        let l = limits(None, None, Some(1));

        let r1 = limiter.pre_commit("k1", &l).await.unwrap();
        let r2 = limiter.pre_commit("k2", &l).await.unwrap();
        let multi = MultiReservation::new(vec![r1, r2]);

        assert!(limiter.pre_commit("k1", &l).await.is_err());
        assert!(limiter.pre_commit("k2", &l).await.is_err());

        drop(multi);

        assert!(limiter.pre_commit("k1", &l).await.is_ok());
        assert!(limiter.pre_commit("k2", &l).await.is_ok());
    }

    #[tokio::test]
    async fn stream_hold_keeps_concurrency_until_guard_drop() {
        // #450: a streaming request must keep its concurrency slot for the
        // stream's full lifetime, not release it at handler return.
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock.clone());
        let l = limits(None, None, Some(1));

        let r = limiter.pre_commit("k", &l).await.unwrap();
        let hold = MultiReservation::new(vec![r]).into_stream_hold();

        // Slot still held while the stream runs — a second concurrent
        // request is rejected.
        assert!(matches!(
            limiter.pre_commit("k", &l).await.unwrap_err(),
            RateLimitError::Concurrency
        ));

        // Stream completes/cancels → guard drops → slot released.
        drop(hold);
        assert!(limiter.pre_commit("k", &l).await.is_ok());
    }

    #[tokio::test]
    async fn multi_reservation_keys_returns_all_held_keys() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock.clone());
        let l = limits(Some(10), None, None);

        let r1 = limiter.pre_commit("api_key:k1", &l).await.unwrap();
        let r2 = limiter.pre_commit("model:m1", &l).await.unwrap();
        let r3 = limiter.pre_commit("team:t1", &l).await.unwrap();
        let multi = MultiReservation::new(vec![r1, r2, r3]);

        let keys = multi.keys();
        assert_eq!(keys, vec!["api_key:k1", "model:m1", "team:t1"]);
    }

    #[tokio::test]
    async fn multi_reservation_partial_failure_releases_acquired_layers() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock.clone());
        let l_key = limits(None, None, Some(1));
        let l_team = limits(None, None, Some(1));
        let l_model = limits(Some(1), None, None);

        // Exhaust model RPM so the third layer will fail.
        let _exhaust = limiter.pre_commit("model:m1", &l_model).await.unwrap();

        let r_key = limiter.pre_commit("k1", &l_key).await.unwrap();
        let r_team = limiter.pre_commit("team:t1", &l_team).await.unwrap();
        let acquired = vec![r_key, r_team];

        assert!(limiter.pre_commit("k1", &l_key).await.is_err());
        assert!(limiter.pre_commit("team:t1", &l_team).await.is_err());

        // Model layer fails — drop the partially-built reservations.
        assert!(limiter.pre_commit("model:m1", &l_model).await.is_err());
        drop(MultiReservation::new(acquired));

        assert!(limiter.pre_commit("k1", &l_key).await.is_ok());
        assert!(limiter.pre_commit("team:t1", &l_team).await.is_ok());
    }

    // ───────────────────────── #426 rps / rph coverage ─────────────────────────

    #[tokio::test]
    async fn rps_caps_request_count_within_one_second() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock.clone());
        let l = limits_full(Some(5), None, None, None);

        for i in 0..5 {
            limiter
                .pre_commit("k1", &l)
                .await
                .unwrap_or_else(|e| panic!("request {i}: {e:?}"));
        }
        let err = limiter.pre_commit("k1", &l).await.unwrap_err();
        assert!(
            matches!(err, RateLimitError::Requests { .. }),
            "expected rps rejection, got {err:?}"
        );
    }

    #[tokio::test]
    async fn rps_window_rolls_at_one_second_boundary() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock.clone());
        let l = limits_full(Some(3), None, None, None);

        for _ in 0..3 {
            limiter.pre_commit("k1", &l).await.unwrap();
        }
        assert!(limiter.pre_commit("k1", &l).await.is_err());

        clock.advance(1);
        for _ in 0..3 {
            limiter.pre_commit("k1", &l).await.unwrap();
        }
        assert!(limiter.pre_commit("k1", &l).await.is_err());
    }

    #[tokio::test]
    async fn rph_caps_request_count_within_one_hour() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock.clone());
        let l = limits_full(None, None, Some(10), None);

        for i in 0..10 {
            limiter
                .pre_commit("k1", &l)
                .await
                .unwrap_or_else(|e| panic!("request {i}: {e:?}"));
        }
        let err = limiter.pre_commit("k1", &l).await.unwrap_err();
        assert!(
            matches!(err, RateLimitError::Requests { .. }),
            "expected rph rejection, got {err:?}"
        );

        clock.advance(3601);
        limiter.pre_commit("k1", &l).await.unwrap();
    }

    #[tokio::test]
    async fn rpd_rejection_rolls_back_rps_and_rph_increments() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock.clone());
        let l = limits_full(Some(1000), Some(1000), Some(1000), Some(2));

        limiter.pre_commit("k1", &l).await.unwrap();
        limiter.pre_commit("k1", &l).await.unwrap();
        let err = limiter.pre_commit("k1", &l).await.unwrap_err();
        assert!(matches!(err, RateLimitError::Requests { .. }));

        let status = limiter.peek("k1", &l).await.unwrap();
        assert_eq!(
            status.rpm_used, 2,
            "rpd rejection must roll back rpm by exactly 1, leaving the two earlier accepts"
        );
    }

    #[tokio::test]
    async fn rph_rejection_rolls_back_rps_and_rpm_increments() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock.clone());
        let l = limits_full(Some(1000), Some(1000), Some(2), None);

        limiter.pre_commit("k1", &l).await.unwrap();
        limiter.pre_commit("k1", &l).await.unwrap();
        let err = limiter.pre_commit("k1", &l).await.unwrap_err();
        assert!(matches!(err, RateLimitError::Requests { .. }));
        let status = limiter.peek("k1", &l).await.unwrap();
        assert_eq!(
            status.rpm_used, 2,
            "rph rejection must roll back rpm by exactly 1, leaving the two earlier accepts"
        );
    }

    #[tokio::test]
    async fn rps_layer_disabled_when_field_unset() {
        let clock = TestClock::new(100);
        let limiter = Limiter::local_with_clock(clock.clone());
        let l = limits_full(None, Some(5), None, None);

        for _ in 0..5 {
            limiter.pre_commit("k1", &l).await.unwrap();
        }
        let err = limiter.pre_commit("k1", &l).await.unwrap_err();
        assert!(matches!(err, RateLimitError::Requests { .. }));
    }
}
