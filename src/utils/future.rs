use std::time::Duration;

use tokio::time::error::Elapsed;

pub async fn maybe_timeout<F, T>(dur: Option<Duration>, fut: F) -> Result<T, Elapsed>
where
    F: Future<Output = T>,
{
    match dur {
        Some(d) if d.is_zero() => Ok(fut.await),
        Some(d) => tokio::time::timeout(d, fut).await,
        None => Ok(fut.await),
    }
}
