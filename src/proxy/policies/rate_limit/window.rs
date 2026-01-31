//! Fixed time window counter

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug)]
pub struct WindowCounter {
    window_start: AtomicU64,
    count: AtomicU64,
    window_size_secs: u64,
}

impl WindowCounter {
    pub fn new(window_size_secs: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let window_start = (now / window_size_secs) * window_size_secs;
        Self {
            window_start: AtomicU64::new(window_start),
            count: AtomicU64::new(0),
            window_size_secs,
        }
    }

    fn current_window_start(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        (now / self.window_size_secs) * self.window_size_secs
    }

    pub fn check_and_increment(&self, amount: u64, limit: u64) -> Result<u64, u64> {
        let current_window = self.current_window_start();
        let stored_window = self.window_start.load(Ordering::Acquire);

        // If window has expired, reset counter
        if current_window >= stored_window + self.window_size_secs {
            // Try to reset the window
            if self
                .window_start
                .compare_exchange(
                    stored_window,
                    current_window,
                    Ordering::SeqCst,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                // Successfully reset, clear counter
                self.count.store(0, Ordering::Release);
            }
        }

        // Increment and check
        let new_count = self.count.fetch_add(amount, Ordering::SeqCst) + amount;

        if new_count > limit {
            // Rollback the increment
            self.count.fetch_sub(amount, Ordering::SeqCst);
            Err(new_count - amount)
        } else {
            Ok(new_count)
        }
    }

    pub fn current_count(&self) -> u64 {
        let current_window = self.current_window_start();
        let stored_window = self.window_start.load(Ordering::Acquire);

        // Only return count if we're still in the same window
        if current_window >= stored_window && current_window < stored_window + self.window_size_secs
        {
            self.count.load(Ordering::Acquire)
        } else {
            0
        }
    }
}
