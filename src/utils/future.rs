use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

use fastrace::Span;
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

#[pin_project::pin_project]
pub struct WithSpan<F> {
    #[pin]
    pub inner: F,
    pub span: Option<Span>,
}

impl<F: Future> Future for WithSpan<F> {
    type Output = (F::Output, Span);

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let _guard = this.span.as_ref().map(|s| s.set_local_parent());

        match this.inner.poll(cx) {
            Poll::Ready(val) => Poll::Ready((val, this.span.take().unwrap())),
            Poll::Pending => Poll::Pending,
        }
    }
}
