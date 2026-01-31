//! Local in-memory rate limiter implementation

use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use tokio::sync::Semaphore;

use super::window::WindowCounter;
use super::{ConcurrencyGuard, RateLimit, RateLimitError, RateLimiter};

pub struct LocalRateLimiter {
    counters: DashMap<String, Arc<WindowCounter>>,
    semaphores: DashMap<String, Arc<Semaphore>>,
}

impl LocalRateLimiter {
    pub fn new() -> Self {
        Self {
            counters: DashMap::new(),
            semaphores: DashMap::new(),
        }
    }

    fn counter_key(scope: &str, id: &str, metric: &str) -> String {
        format!("{}:{}:{}", scope, id, metric)
    }

    fn get_or_create_counter(&self, key: String, window_size_secs: u64) -> Arc<WindowCounter> {
        self.counters
            .entry(key)
            .or_insert_with(|| Arc::new(WindowCounter::new(window_size_secs)))
            .clone()
    }

    fn get_or_create_semaphore(&self, key: String, permits: u64) -> Arc<Semaphore> {
        self.semaphores
            .entry(key)
            .or_insert_with(|| Arc::new(Semaphore::new(permits as usize)))
            .clone()
    }
}

#[async_trait]
impl RateLimiter for LocalRateLimiter {
    async fn check_and_reserve(
        &self,
        scope: &str,
        id: &str,
        rate_limit: &RateLimit,
    ) -> Result<Option<ConcurrencyGuard>, RateLimitError> {
        // Check request per minute
        if let Some(rpm) = rate_limit.request_per_minute {
            let key = Self::counter_key(scope, id, "rpm");
            let counter = self.get_or_create_counter(key, 60);

            match counter.check_and_increment(1, rpm) {
                Ok(_) => {}
                Err(current) => {
                    return Err(RateLimitError::RequestPerMinuteExceeded {
                        scope: scope.to_string(),
                        id: id.to_string(),
                        current,
                        limit: rpm,
                    });
                }
            }
        }

        // Check request per day
        if let Some(rpd) = rate_limit.request_per_day {
            let key = Self::counter_key(scope, id, "rpd");
            let counter = self.get_or_create_counter(key, 86400);

            match counter.check_and_increment(1, rpd) {
                Ok(_) => {}
                Err(current) => {
                    return Err(RateLimitError::RequestPerDayExceeded {
                        scope: scope.to_string(),
                        id: id.to_string(),
                        current,
                        limit: rpd,
                    });
                }
            }
        }

        // Check current token per minute usage (without incrementing)
        if let Some(tpm) = rate_limit.token_per_minute {
            let key = Self::counter_key(scope, id, "tpm");
            let counter = self.get_or_create_counter(key, 60);
            let current = counter.current_count();

            if current >= tpm {
                return Err(RateLimitError::TokenPerMinuteExceeded {
                    scope: scope.to_string(),
                    id: id.to_string(),
                    current,
                    limit: tpm,
                });
            }
        }

        // Check current token per day usage (without incrementing)
        if let Some(tpd) = rate_limit.token_per_day {
            let key = Self::counter_key(scope, id, "tpd");
            let counter = self.get_or_create_counter(key, 86400);
            let current = counter.current_count();

            if current >= tpd {
                return Err(RateLimitError::TokenPerDayExceeded {
                    scope: scope.to_string(),
                    id: id.to_string(),
                    current,
                    limit: tpd,
                });
            }
        }

        // Check concurrency
        let guard = if let Some(concurrency) = rate_limit.request_concurrency {
            let key = Self::counter_key(scope, id, "concurrency");
            let semaphore = self.get_or_create_semaphore(key, concurrency);

            match semaphore.clone().try_acquire_owned() {
                Ok(permit) => Some(ConcurrencyGuard {
                    permit: Some(permit),
                }),
                Err(_) => {
                    return Err(RateLimitError::ConcurrencyExceeded {
                        scope: scope.to_string(),
                        id: id.to_string(),
                        limit: concurrency,
                    });
                }
            }
        } else {
            None
        };

        Ok(guard)
    }

    async fn record_usage(
        &self,
        scope: &str,
        id: &str,
        rate_limit: &RateLimit,
        tokens: u64,
    ) -> Result<(), RateLimitError> {
        // Record token per minute
        if let Some(tpm) = rate_limit.token_per_minute {
            let key = Self::counter_key(scope, id, "tpm");
            let counter = self.get_or_create_counter(key, 60);

            match counter.check_and_increment(tokens, tpm) {
                Ok(_) => {}
                Err(current) => {
                    return Err(RateLimitError::TokenPerMinuteExceeded {
                        scope: scope.to_string(),
                        id: id.to_string(),
                        current,
                        limit: tpm,
                    });
                }
            }
        }

        // Record token per day
        if let Some(tpd) = rate_limit.token_per_day {
            let key = Self::counter_key(scope, id, "tpd");
            let counter = self.get_or_create_counter(key, 86400);

            match counter.check_and_increment(tokens, tpd) {
                Ok(_) => {}
                Err(current) => {
                    return Err(RateLimitError::TokenPerDayExceeded {
                        scope: scope.to_string(),
                        id: id.to_string(),
                        current,
                        limit: tpd,
                    });
                }
            }
        }

        Ok(())
    }
}
