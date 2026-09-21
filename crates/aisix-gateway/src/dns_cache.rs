//! One in-process answer per upstream hostname, shared by every outbound
//! client.
//!
//! Without it each new connection resolves its own name. That is invisible
//! in steady state, because a warm pool opens almost no connections, and
//! decisive when the pool is cold: a gateway resuming from a pause finds
//! every queued request without an idle connection, opens one per request,
//! and pays a `getaddrinfo` for each — 314 at once in the case this was
//! built for, each on its own blocking-pool thread, and under a cluster
//! resolver's `ndots:5` search list about ten DNS packets per lookup. The
//! answers were never lost; the stampede itself was the cost, on a core a
//! request worker also wanted.
//!
//! Two things fix that, and both are here: concurrent lookups of one name
//! collapse into one, and its answer is reused briefly afterwards.
//!
//! **The reuse window is wall-clock, not the record's own TTL.** The
//! platform resolver does not report a TTL — `getaddrinfo` has nowhere to
//! put one — so honouring it would mean resolving DNS ourselves instead of
//! asking the system, which would also take over the search list, the
//! hosts file and nsswitch. Mainstream gateways in this space cache the
//! same way, on a fixed window, for the same reason. [`ANSWER_TTL`] is
//! deliberately far shorter than theirs: a provider addressed through a
//! cluster Service changes address when its endpoints rotate, and a long
//! window means dialling a dead address until it expires.

use std::{
    collections::HashMap,
    net::{SocketAddr, ToSocketAddrs},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use tokio::sync::watch;

/// How long a successful answer is reused.
///
/// Short enough that an endpoint rotation behind a cluster Service costs
/// at most this much dialling of the old address, long enough that a
/// burst of connections to one host resolves once.
const ANSWER_TTL: Duration = Duration::from_secs(30);

/// How long a failure is reused. Long enough to collapse a burst that
/// arrives while a name is genuinely broken, short enough that a name
/// coming back is picked up promptly.
const FAILURE_TTL: Duration = Duration::from_secs(1);

/// Names remembered at once. Upstream hostnames come from configured
/// resources, so this is orders of magnitude above what a deployment
/// uses; it exists so a pathological configuration cannot grow the map
/// without bound.
const MAX_NAMES: usize = 1024;

type Answer = Result<Arc<Vec<SocketAddr>>, String>;

enum Entry {
    /// A lookup is running. Every later caller for this name waits for
    /// it instead of starting another.
    Running(watch::Receiver<Option<Answer>>),
    /// A finished lookup, reusable until `until`.
    Settled { answer: Answer, until: Instant },
}

/// The resolution cache itself, independent of any HTTP client crate.
///
/// Cloning shares the cache: every outbound client in the process holds
/// the same one, including the per-ProviderKey and per-guardrail clients
/// that are rebuilt whenever a configuration snapshot lands.
#[derive(Clone, Default)]
pub struct DnsCache {
    names: Arc<Mutex<HashMap<String, Entry>>>,
}

/// What [`DnsCache::claim`] found for a name.
enum Claim {
    /// A reusable answer.
    Ready(Answer),
    /// A lookup is running — this caller's or someone else's.
    Wait(watch::Receiver<Option<Answer>>),
    /// The cache is full of names still in their window; resolve without
    /// remembering rather than evict one in use.
    Unremembered,
}

impl DnsCache {
    /// The addresses for `host`, resolved at most once per name per
    /// window however many callers ask at once.
    pub async fn lookup(&self, host: &str) -> Result<Arc<Vec<SocketAddr>>, String> {
        let mut waiting = match self.claim(host) {
            Claim::Ready(answer) => return answer,
            Claim::Wait(waiting) => waiting,
            Claim::Unremembered => return resolve(host.to_owned()).await,
        };
        loop {
            if let Some(answer) = waiting.borrow_and_update().clone() {
                return answer;
            }
            if waiting.changed().await.is_err() {
                // The lookup died without settling. Clear the name so the
                // next caller starts a new one instead of waiting on a
                // receiver nothing will ever send to.
                self.names.lock().expect("dns cache").remove(host);
                return Err(format!("resolving {host} did not finish"));
            }
        }
    }

    /// Take the answer for `host`, or the right to resolve it.
    fn claim(&self, host: &str) -> Claim {
        let now = Instant::now();
        let mut names = self.names.lock().expect("dns cache");
        match names.get(host) {
            Some(Entry::Settled { answer, until }) if *until > now => {
                return Claim::Ready(answer.clone())
            }
            Some(Entry::Running(waiting)) => return Claim::Wait(waiting.clone()),
            _ => {}
        }
        // Inserting a name is the only thing that grows the map, so it is
        // where names whose window has passed are dropped.
        if names.len() >= MAX_NAMES && !names.contains_key(host) {
            names.retain(|_, entry| match entry {
                Entry::Settled { until, .. } => *until > now,
                Entry::Running(_) => true,
            });
            if names.len() >= MAX_NAMES {
                return Claim::Unremembered;
            }
        }
        let (sender, receiver) = watch::channel(None);
        names.insert(host.to_owned(), Entry::Running(receiver.clone()));
        self.settle(host.to_owned(), sender);
        Claim::Wait(receiver)
    }

    /// Run the lookup for a claimed name and publish it to its waiters.
    fn settle(&self, host: String, sender: watch::Sender<Option<Answer>>) {
        let names = Arc::clone(&self.names);
        tokio::spawn(async move {
            let answer = resolve(host.clone()).await;
            let until = Instant::now()
                + if answer.is_ok() {
                    ANSWER_TTL
                } else {
                    FAILURE_TTL
                };
            names.lock().expect("dns cache").insert(
                host,
                Entry::Settled {
                    answer: answer.clone(),
                    until,
                },
            );
            // After the map, so a waiter this wakes cannot look the name
            // up again and find it still running.
            let _ = sender.send(Some(answer));
        });
    }

    /// Names remembered right now. Test seam.
    #[cfg(test)]
    fn remembered(&self) -> usize {
        self.names.lock().expect("dns cache").len()
    }
}

/// One lookup through the platform resolver, off the async threads.
///
/// The port is the caller's, not ours: reqwest substitutes the URL's or
/// the scheme's port into whatever comes back.
async fn resolve(host: String) -> Answer {
    match tokio::task::spawn_blocking(move || {
        (host.as_str(), 0u16)
            .to_socket_addrs()
            .map(|addrs| addrs.collect::<Vec<_>>())
            .map_err(|error| error.to_string())
    })
    .await
    {
        Ok(Ok(addrs)) => Ok(Arc::new(addrs)),
        Ok(Err(error)) => Err(error),
        Err(error) => Err(error.to_string()),
    }
}

impl std::fmt::Debug for DnsCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DnsCache").finish_non_exhaustive()
    }
}

impl Resolve for DnsCache {
    fn resolve(&self, name: Name) -> Resolving {
        let cache = self.clone();
        Box::pin(async move {
            let addrs = cache.lookup(name.as_str()).await?;
            Ok(Box::new(addrs.iter().copied().collect::<Vec<_>>().into_iter()) as Addrs)
        })
    }
}

/// The one cache every outbound client resolves through.
pub fn shared() -> Arc<DnsCache> {
    static SHARED: std::sync::OnceLock<Arc<DnsCache>> = std::sync::OnceLock::new();
    Arc::clone(SHARED.get_or_init(|| Arc::new(DnsCache::default())))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Localhost resolves through the platform resolver on every machine
    /// the tests run on, which is what makes these cases real lookups
    /// rather than a stubbed table.
    const HOST: &str = "localhost";

    /// The case this exists for: many connections to one host at once —
    /// what a cold pool produces — must cost one lookup. Every caller
    /// holding the SAME answer is what proves there was only one; an
    /// uncached resolver hands each of them its own.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_burst_of_callers_for_one_name_resolves_it_once() {
        let cache = DnsCache::default();
        let callers: Vec<_> = (0..200)
            .map(|_| {
                let cache = cache.clone();
                tokio::spawn(async move { cache.lookup(HOST).await.expect("localhost resolves") })
            })
            .collect();
        let mut answers = Vec::new();
        for caller in callers {
            answers.push(caller.await.unwrap());
        }
        let first = &answers[0];
        assert!(!first.is_empty(), "localhost has addresses");
        assert!(
            answers.iter().all(|answer| Arc::ptr_eq(answer, first)),
            "every caller in the burst must share the one answer",
        );
        assert_eq!(cache.remembered(), 1);
    }

    #[tokio::test]
    async fn an_answer_is_reused_until_it_expires_and_resolved_again_after() {
        let cache = DnsCache::default();
        let first = cache.lookup(HOST).await.expect("localhost resolves");
        let second = cache.lookup(HOST).await.expect("localhost resolves");
        assert!(
            Arc::ptr_eq(&first, &second),
            "a second lookup inside the window must reuse the first answer",
        );

        // Expire it in place: the window is wall-clock and the cases must
        // not wait one out.
        {
            let mut names = cache.names.lock().unwrap();
            let entry = names.get_mut(HOST).unwrap();
            let Entry::Settled { until, .. } = entry else {
                panic!("the answer must have settled");
            };
            *until = Instant::now() - Duration::from_secs(1);
        }
        let third = cache.lookup(HOST).await.expect("localhost resolves");
        assert!(
            !Arc::ptr_eq(&first, &third),
            "an expired answer must be resolved again",
        );
        assert_eq!(*first, *third, "and localhost still has the same addresses");
    }

    /// A name that changes address must be picked up once the window
    /// passes — the property the short window exists for.
    #[tokio::test]
    async fn a_changed_address_is_picked_up_after_the_window() {
        let cache = DnsCache::default();
        let stale: Arc<Vec<SocketAddr>> = Arc::new(vec!["203.0.113.1:0".parse().unwrap()]);
        {
            let mut names = cache.names.lock().unwrap();
            names.insert(
                HOST.to_owned(),
                Entry::Settled {
                    answer: Ok(Arc::clone(&stale)),
                    until: Instant::now() + ANSWER_TTL,
                },
            );
        }
        assert_eq!(*cache.lookup(HOST).await.unwrap(), *stale);
        {
            let mut names = cache.names.lock().unwrap();
            let Some(Entry::Settled { until, .. }) = names.get_mut(HOST) else {
                panic!("the answer must have settled");
            };
            *until = Instant::now() - Duration::from_secs(1);
        }
        let fresh = cache.lookup(HOST).await.unwrap();
        assert_ne!(*fresh, *stale, "the window having passed, resolve again");
    }

    #[tokio::test]
    async fn a_failure_is_remembered_only_briefly() {
        let cache = DnsCache::default();
        let error = cache
            .lookup("not-a-host.invalid")
            .await
            .expect_err(".invalid never resolves");
        assert!(!error.is_empty());
        let names = cache.names.lock().unwrap();
        let Some(Entry::Settled { until, .. }) = names.get("not-a-host.invalid") else {
            panic!("the failure must have settled");
        };
        assert!(
            *until <= Instant::now() + FAILURE_TTL,
            "a failure must not be remembered for an answer's window",
        );
    }
}
