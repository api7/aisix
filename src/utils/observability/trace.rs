use std::{fmt::Debug, time::Duration};

use async_trait::async_trait;
use opentelemetry_sdk::{
    Resource,
    error::OTelSdkResult,
    trace::{SpanData, SpanExporter},
};

/// Type-erased span exporter trait object.
#[async_trait]
pub trait DynSpanExporter: Send + Sync + Debug {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult;

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult;

    fn force_flush(&self) -> OTelSdkResult;

    fn set_resource(&mut self, resource: &Resource);
}

#[async_trait]
impl<T: SpanExporter> DynSpanExporter for T {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        SpanExporter::export(self, batch).await
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        SpanExporter::shutdown_with_timeout(self, timeout)
    }

    fn force_flush(&self) -> OTelSdkResult {
        SpanExporter::force_flush(self)
    }

    fn set_resource(&mut self, resource: &Resource) {
        SpanExporter::set_resource(self, resource)
    }
}

/// Type-erased span exporter adapter.
#[derive(Debug)]
pub struct BoxedSpanExporter(Box<dyn DynSpanExporter>);

impl BoxedSpanExporter {
    pub fn new<T: SpanExporter + 'static>(span_exporter: T) -> Self {
        Self(Box::new(span_exporter))
    }
}

impl SpanExporter for BoxedSpanExporter {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        self.0.export(batch).await
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.0.shutdown_with_timeout(timeout)
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.0.force_flush()
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.0.set_resource(resource)
    }
}
