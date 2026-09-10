use std::sync::{Arc, Mutex};

use bytes::Bytes;
use tokio::sync::watch;

type Result = std::result::Result<Bytes, String>;
type Receiver = watch::Receiver<Option<Result>>;

#[derive(Default)]
pub(crate) struct Scrape {
    current: Arc<Mutex<Option<Receiver>>>,
}

impl Scrape {
    pub(crate) async fn render(&self, render: impl FnOnce() -> String + Send + 'static) -> Result {
        let mut receiver = {
            let mut current = self.current.lock().expect("in-flight scrape");
            if let Some(receiver) = current.as_ref().filter(|r| r.has_changed().is_ok()) {
                receiver.clone()
            } else {
                let (sender, receiver) = watch::channel(None);
                *current = Some(receiver.clone());
                let current = Arc::clone(&self.current);
                // The producer outlives any one HTTP request. A cancelled
                // client must not start duplicate renders or discard samples.
                tokio::spawn(async move {
                    let result = tokio::task::spawn_blocking(render)
                        .await
                        .map(Bytes::from)
                        .map_err(|error| format!("metrics render task failed: {error}"));
                    let mut current = current.lock().expect("in-flight scrape");
                    *current = None;
                    let _ = sender.send(Some(result));
                });
                receiver
            }
        };
        loop {
            if let Some(result) = receiver.borrow_and_update().clone() {
                return result;
            }
            if receiver.changed().await.is_err() {
                return Err("metrics render task stopped before completing".to_owned());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test(flavor = "current_thread")]
    async fn shares_inflight_work_survives_cancellation_and_refreshes_after_completion() {
        let scrape = Scrape::default();
        let (started, start) = tokio::sync::oneshot::channel();
        let (release, wait) = std::sync::mpsc::channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&calls);
        let runtime_thread = std::thread::current().id();
        let mut first = Box::pin(scrape.render(move || {
            assert_ne!(std::thread::current().id(), runtime_thread);
            count.fetch_add(1, Ordering::SeqCst);
            started.send(()).unwrap();
            wait.recv().unwrap();
            "shared snapshot".to_owned()
        }));
        assert!(futures::poll!(first.as_mut()).is_pending());
        start.await.unwrap();
        let mut second = Box::pin(scrape.render(|| panic!("duplicate render")));
        let mut third = Box::pin(scrape.render(|| panic!("duplicate render")));
        assert!(futures::poll!(second.as_mut()).is_pending());
        assert!(futures::poll!(third.as_mut()).is_pending());
        drop(first);
        // This task continues on a current-thread runtime while the renderer
        // is blocked, and cancelling the first requester leaves it running.
        tokio::task::yield_now().await;
        release.send(()).unwrap();
        let (second, third) = tokio::join!(second, third);
        let second = second.unwrap();
        let third = third.unwrap();
        assert_eq!(second, "shared snapshot");
        assert_eq!(second.as_ptr(), third.as_ptr());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            scrape.render(|| "fresh snapshot".to_owned()).await.unwrap(),
            "fresh snapshot"
        );
        assert!(scrape.current.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn a_failed_render_does_not_poison_later_scrapes() {
        let scrape = Scrape::default();
        let result = scrape.render(|| panic!("failed renderer")).await;
        assert!(result.unwrap_err().contains("metrics render task failed"));
        assert_eq!(
            scrape.render(|| "recovered".to_owned()).await.unwrap(),
            "recovered"
        );
    }
}
