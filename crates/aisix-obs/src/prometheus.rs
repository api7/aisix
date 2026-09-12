//! Per-series storage keeps scrape work off the recording path. Labels are
//! escaped once on the first scrape; histogram samples use a concurrent queue
//! and the exporter's existing distribution/quantile implementation.

use std::{
    collections::HashMap,
    fmt::Write,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex, OnceLock,
    },
};

use crossbeam_queue::SegQueue;
use metrics::{
    atomics::AtomicU64, Counter, CounterFn, Gauge, GaugeFn, Histogram, HistogramFn, Key, KeyName,
    Metadata, Recorder as MetricsRecorder, SharedString, Unit,
};
use metrics_exporter_prometheus::{
    formatting::{sanitize_label_key, sanitize_metric_name, write_help_line, write_type_line},
    Distribution, DistributionBuilder,
};
use metrics_util::registry::Registry;
use quanta::Instant;

struct Labels {
    name: String,
    key: Key,
    text: OnceLock<String>,
}

impl Labels {
    fn new(key: &Key) -> Self {
        Self {
            name: sanitize_metric_name(key.name()),
            key: key.clone(),
            text: OnceLock::new(),
        }
    }

    fn write(
        &self,
        output: &mut String,
        suffix: Option<&str>,
        extra: Option<(&str, &str)>,
        value: impl std::fmt::Display,
    ) {
        output.push_str(&self.name);
        if let Some(suffix) = suffix {
            output.push('_');
            output.push_str(suffix);
        }
        let labels = self.text.get_or_init(|| {
            let mut text = String::new();
            for (index, label) in self.key.labels().enumerate() {
                if index != 0 {
                    text.push(',');
                }
                write!(text, "{}=\"", sanitize_label_key(label.key()))
                    .expect("writing to a String");
                // Values are raw, never pre-escaped. Treating a pair of
                // backslashes as an existing escape aliases distinct series.
                for ch in label.value().chars() {
                    match ch {
                        '\\' => text.push_str("\\\\"),
                        '"' => text.push_str("\\\""),
                        '\n' => text.push_str("\\n"),
                        ch => text.push(ch),
                    }
                }
                text.push('"');
            }
            text
        });
        if !labels.is_empty() || extra.is_some() {
            output.push('{');
            output.push_str(labels);
            if let Some((name, value)) = extra {
                if !labels.is_empty() {
                    output.push(',');
                }
                output.push_str(name);
                output.push_str("=\"");
                output.push_str(value);
                output.push('"');
            }
            output.push('}');
        }
        writeln!(output, " {value}").expect("writing to a String");
    }
}

struct Scalar {
    labels: Labels,
    value: AtomicU64,
}

impl Scalar {
    fn new(key: &Key) -> Self {
        Self {
            labels: Labels::new(key),
            value: AtomicU64::new(0),
        }
    }
}

impl CounterFn for Scalar {
    fn increment(&self, value: u64) {
        CounterFn::increment(&self.value, value);
    }
    fn absolute(&self, value: u64) {
        self.value.absolute(value);
    }
}

impl GaugeFn for Scalar {
    fn increment(&self, value: f64) {
        GaugeFn::increment(&self.value, value);
    }
    fn decrement(&self, value: f64) {
        self.value.decrement(value);
    }
    fn set(&self, value: f64) {
        self.value.set(value);
    }
}

struct DistributionSeries {
    labels: Labels,
    kind: &'static str,
    pending: SegQueue<(f64, Instant)>,
    distribution: Mutex<Distribution>,
}

impl HistogramFn for DistributionSeries {
    fn record(&self, value: f64) {
        self.pending.push((value, Instant::now()));
    }
}

impl DistributionSeries {
    fn drain(&self, distribution: &mut Distribution) {
        // Bound this drain to the current backlog so ongoing writers cannot
        // keep a scrape busy indefinitely. New samples stay queued for next time.
        let count = self.pending.len();
        let mut samples = Vec::with_capacity(count.min(64));
        for _ in 0..count {
            let Some(sample) = self.pending.pop() else {
                break;
            };
            samples.push(sample);
            if samples.len() == 64 {
                distribution.record_samples(&samples);
                samples.clear();
            }
        }
        if !samples.is_empty() {
            distribution.record_samples(&samples);
        }
    }

    fn upkeep(&self) {
        if self.pending.is_empty() {
            return;
        }
        let mut distribution = self.distribution.lock().expect("metric distribution");
        self.drain(&mut distribution);
    }

    fn render(&self, output: &mut String) {
        let mut distribution = self.distribution.lock().expect("metric distribution");
        self.drain(&mut distribution);
        let (sum, count) = match &*distribution {
            Distribution::Summary(summary, quantiles, sum) => {
                let snapshot = summary.snapshot(Instant::now());
                for quantile in quantiles.iter() {
                    self.labels.write(
                        output,
                        None,
                        Some(("quantile", &quantile.value().to_string())),
                        snapshot.quantile(quantile.value()).unwrap_or(0.0),
                    );
                }
                (*sum, summary.count() as u64)
            }
            Distribution::Histogram(histogram) => {
                for (bound, count) in histogram.buckets() {
                    self.labels.write(
                        output,
                        Some("bucket"),
                        Some(("le", &bound.to_string())),
                        count,
                    );
                }
                self.labels.write(
                    output,
                    Some("bucket"),
                    Some(("le", "+Inf")),
                    histogram.count(),
                );
                (histogram.sum(), histogram.count())
            }
        };
        self.labels.write(output, Some("sum"), None, sum);
        self.labels.write(output, Some("count"), None, count);
    }
}

struct Storage {
    distributions: DistributionBuilder,
    generation: Arc<AtomicUsize>,
}

impl metrics_util::registry::Storage<Key> for Storage {
    type Counter = Arc<Scalar>;
    type Gauge = Arc<Scalar>;
    type Histogram = Arc<DistributionSeries>;

    fn counter(&self, key: &Key) -> Self::Counter {
        self.generation.fetch_add(1, Ordering::Release);
        Arc::new(Scalar::new(key))
    }
    fn gauge(&self, key: &Key) -> Self::Gauge {
        self.generation.fetch_add(1, Ordering::Release);
        Arc::new(Scalar::new(key))
    }
    fn histogram(&self, key: &Key) -> Self::Histogram {
        let distribution = self.distributions.get_distribution(key.name());
        let kind = match &distribution {
            Distribution::Histogram(_) => "histogram",
            Distribution::Summary(..) => "summary",
        };
        self.generation.fetch_add(1, Ordering::Release);
        Arc::new(DistributionSeries {
            labels: Labels::new(key),
            kind,
            pending: SegQueue::new(),
            distribution: Mutex::new(distribution),
        })
    }
}

#[derive(Default)]
struct Series {
    counters: Vec<Arc<Scalar>>,
    gauges: Vec<Arc<Scalar>>,
    distributions: Vec<Arc<DistributionSeries>>,
}

pub(crate) struct Recorder {
    registry: Registry<Key, Storage>,
    descriptions: Mutex<HashMap<String, SharedString>>,
    previous_render_bytes: AtomicUsize,
    generation: Arc<AtomicUsize>,
    series: Mutex<Option<(usize, Arc<Series>)>>,
}

impl Recorder {
    pub(crate) fn new(distributions: DistributionBuilder) -> Self {
        let generation = Arc::new(AtomicUsize::new(0));
        Self {
            registry: Registry::new(Storage {
                distributions,
                generation: generation.clone(),
            }),
            descriptions: Mutex::new(HashMap::new()),
            previous_render_bytes: AtomicUsize::new(0),
            generation,
            series: Mutex::new(None),
        }
    }

    fn series(&self) -> Arc<Series> {
        let mut cached = self.series.lock().expect("metric series");
        // Read before visiting: a concurrent insertion can be absent from
        // this snapshot, but must invalidate it for the next visit. Storage
        // increments under the registry's insertion lock, which visits read.
        let generation = self.generation.load(Ordering::Acquire);
        if let Some((previous, series)) = &*cached {
            if *previous == generation {
                return series.clone();
            }
        }
        let mut series = Series::default();
        self.registry
            .visit_counters(|_, value| series.counters.push(value.clone()));
        self.registry
            .visit_gauges(|_, value| series.gauges.push(value.clone()));
        self.registry
            .visit_histograms(|_, value| series.distributions.push(value.clone()));
        series
            .counters
            .sort_unstable_by(|a, b| a.labels.name.cmp(&b.labels.name));
        series
            .gauges
            .sort_unstable_by(|a, b| a.labels.name.cmp(&b.labels.name));
        series
            .distributions
            .sort_unstable_by(|a, b| a.labels.name.cmp(&b.labels.name));
        let series = Arc::new(series);
        *cached = Some((generation, series.clone()));
        series
    }

    pub(crate) fn run_upkeep(&self) {
        for value in &self.series().distributions {
            value.upkeep();
        }
    }

    pub(crate) fn render(&self) -> String {
        let descriptions = self
            .descriptions
            .lock()
            .expect("metric descriptions")
            .clone();
        // A warmed high-cardinality scrape can be hundreds of MiB. Leave
        // room for growing counters without copying that buffer on every scrape.
        let previous_bytes = self.previous_render_bytes.load(Ordering::Relaxed);
        let mut output = String::with_capacity(previous_bytes.saturating_add(previous_bytes / 8));
        let series = self.series();
        let mut previous = "";
        for value in &series.counters {
            write_header(
                &mut output,
                &descriptions,
                &mut previous,
                &value.labels.name,
                "counter",
            );
            value
                .labels
                .write(&mut output, None, None, value.value.load(Ordering::Acquire));
        }
        previous = "";
        for value in &series.gauges {
            write_header(
                &mut output,
                &descriptions,
                &mut previous,
                &value.labels.name,
                "gauge",
            );
            value.labels.write(
                &mut output,
                None,
                None,
                f64::from_bits(value.value.load(Ordering::Acquire)),
            );
        }
        previous = "";
        for value in &series.distributions {
            write_header(
                &mut output,
                &descriptions,
                &mut previous,
                &value.labels.name,
                value.kind,
            );
            value.render(&mut output);
        }
        self.previous_render_bytes
            .store(output.len(), Ordering::Relaxed);
        output
    }

    fn describe(&self, name: KeyName, description: SharedString) {
        self.descriptions
            .lock()
            .expect("metric descriptions")
            .entry(sanitize_metric_name(name.as_str()))
            .or_insert(description);
    }
}

fn write_header<'a>(
    output: &mut String,
    descriptions: &HashMap<String, SharedString>,
    previous: &mut &'a str,
    name: &'a str,
    kind: &str,
) {
    if *previous != name {
        if let Some(description) = descriptions.get(name) {
            write_help_line(output, name, description);
        }
        write_type_line(output, name, kind);
        *previous = name;
    }
}

impl MetricsRecorder for Recorder {
    fn describe_counter(&self, name: KeyName, _: Option<Unit>, description: SharedString) {
        self.describe(name, description);
    }
    fn describe_gauge(&self, name: KeyName, _: Option<Unit>, description: SharedString) {
        self.describe(name, description);
    }
    fn describe_histogram(&self, name: KeyName, _: Option<Unit>, description: SharedString) {
        self.describe(name, description);
    }
    fn register_counter(&self, key: &Key, _: &Metadata<'_>) -> Counter {
        self.registry
            .get_or_create_counter(key, |value| Counter::from_arc(Arc::clone(value)))
    }
    fn register_gauge(&self, key: &Key, _: &Metadata<'_>) -> Gauge {
        self.registry
            .get_or_create_gauge(key, |value| Gauge::from_arc(Arc::clone(value)))
    }
    fn register_histogram(&self, key: &Key, _: &Metadata<'_>) -> Histogram {
        self.registry
            .get_or_create_histogram(key, |value| Histogram::from_arc(Arc::clone(value)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrics_exporter_prometheus::{Matcher, PrometheusBuilder};
    use std::time::Duration;

    fn distributions() -> DistributionBuilder {
        DistributionBuilder::new(
            metrics_util::parse_quantiles(&[0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0]),
            None,
            None,
            None,
            Some(HashMap::from([(
                Matcher::Full("latency".to_owned()),
                vec![0.1, 0.5, 1.0],
            )])),
        )
    }

    fn lines(output: &str) -> Vec<&str> {
        let mut lines = output
            .lines()
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>();
        lines.sort_unstable();
        lines
    }

    #[test]
    fn series_catalog_is_reused_while_values_and_membership_stay_live() {
        let recorder = Recorder::new(distributions());
        let metadata = Metadata::new("test", metrics::Level::INFO, None);
        assert!(recorder.render().is_empty());
        let counter = recorder.register_counter(&Key::from_name("requests"), &metadata);
        counter.increment(1);
        assert!(recorder.render().contains("requests 1\n"));
        let gauge = recorder.register_gauge(&Key::from_name("active"), &metadata);
        gauge.set(2.0);
        assert!(recorder.render().contains("active 2\n"));
        let histogram = recorder.register_histogram(&Key::from_name("latency"), &metadata);
        histogram.record(0.5);
        assert!(recorder.render().contains("latency_count 1\n"));

        let catalog = recorder.series();
        counter.increment(3);
        gauge.set(4.0);
        histogram.record(1.0);
        recorder.run_upkeep();
        let output = recorder.render();
        assert!(output.contains("requests 4\n"));
        assert!(output.contains("active 4\n"));
        assert!(output.contains("latency_count 2\n"));
        assert!(output.contains("latency_sum 1.5\n"));
        assert!(Arc::ptr_eq(&catalog, &recorder.series()));

        let _ = recorder.register_counter(&Key::from_name("requests"), &metadata);
        assert!(Arc::ptr_eq(&catalog, &recorder.series()));
        recorder
            .register_counter(&Key::from_name("new_requests"), &metadata)
            .increment(7);
        assert!(recorder.render().contains("new_requests 7\n"));
        assert!(!Arc::ptr_eq(&catalog, &recorder.series()));
    }

    #[test]
    fn concurrent_registration_and_scrapes_do_not_lose_new_series() {
        let recorder = Arc::new(Recorder::new(distributions()));
        recorder.render();
        let writers: Vec<_> = (0..4)
            .map(|worker| {
                let recorder = recorder.clone();
                std::thread::spawn(move || {
                    let metadata = Metadata::new("test", metrics::Level::INFO, None);
                    for i in 0..100 {
                        let key = |name| {
                            Key::from_parts(
                                name,
                                vec![metrics::Label::new("id", format!("{worker}-{i}"))],
                            )
                        };
                        recorder
                            .register_counter(&key("dynamic_counter"), &metadata)
                            .increment(1);
                        recorder
                            .register_gauge(&key("dynamic_gauge"), &metadata)
                            .set(2.0);
                        recorder
                            .register_histogram(&key("dynamic_histogram"), &metadata)
                            .record(0.5);
                    }
                })
            })
            .collect();
        while writers.iter().any(|writer| !writer.is_finished()) {
            recorder.run_upkeep();
            recorder.render();
        }
        for writer in writers {
            writer.join().unwrap();
        }
        let output = recorder.render();
        for worker in 0..4 {
            for i in 0..100 {
                let label = format!("{{id=\"{worker}-{i}\"}}");
                assert!(output.contains(&format!("dynamic_counter{label} 1\n")));
                assert!(output.contains(&format!("dynamic_gauge{label} 2\n")));
                assert!(output.contains(&format!("dynamic_histogram_count{label} 1\n")));
                assert!(output.contains(&format!("dynamic_histogram_sum{label} 0.5\n")));
            }
        }
    }

    #[test]
    fn preserves_exposition_and_rolling_summary_windows() {
        let (clock, time) = quanta::Clock::mock();
        time.increment(Duration::from_secs(3600));
        quanta::with_clock(&clock, || {
            let recorder = Recorder::new(distributions());
            let reference = PrometheusBuilder::new()
                .set_buckets_for_metric(Matcher::Full("latency".to_owned()), &[0.1, 0.5, 1.0])
                .unwrap()
                .build_recorder();
            let metadata = Metadata::new("test", metrics::Level::INFO, None);
            let labels = vec![
                metrics::Label::new("model", "a\\b\"c\nd"),
                metrics::Label::new("member", "成员"),
            ];
            let key = |name| Key::from_parts(name, labels.clone());
            for sink in [&recorder as &dyn MetricsRecorder, &reference] {
                sink.describe_histogram(
                    "latency".into(),
                    Some(Unit::Seconds),
                    "seconds\\and\nnewlines".into(),
                );
                sink.register_counter(&key("requests_total"), &metadata)
                    .increment(7);
                sink.register_counter(&key("requests_total"), &metadata)
                    .absolute(10);
                sink.register_counter(&key("requests_total"), &metadata)
                    .absolute(8);
                sink.register_gauge(&key("remaining"), &metadata).set(5.5);
                sink.register_gauge(&key("remaining"), &metadata)
                    .decrement(0.5);
                sink.register_gauge(&key("retired"), &metadata)
                    .set(f64::NAN);
                let _ = sink.register_histogram(&key("empty_summary"), &metadata);
                let _ = sink.register_histogram(&key("latency"), &metadata);
            }
            assert_eq!(
                lines(&recorder.render()),
                lines(&reference.handle().render())
            );
            for delay in [0, 21, 21, 81] {
                time.increment(Duration::from_secs(delay));
                for sink in [&recorder as &dyn MetricsRecorder, &reference] {
                    for value in [0.05, 0.1, 0.5, 1.0, 2.0] {
                        sink.register_histogram(&key("latency"), &metadata)
                            .record(value);
                        sink.register_histogram(&key("duration"), &metadata)
                            .record(value);
                    }
                }
                recorder.run_upkeep();
                reference.handle().run_upkeep();
                assert_eq!(
                    lines(&recorder.render()),
                    lines(&reference.handle().render())
                );
            }
            time.increment(Duration::from_secs(81));
            assert_eq!(
                lines(&recorder.render()),
                lines(&reference.handle().render())
            );
            assert!(recorder
                .render()
                .contains("duration_count{model=\"a\\\\b\\\"c\\nd\",member=\"成员\"} 20"));
        });
    }

    #[test]
    fn distinct_raw_label_values_remain_distinct_in_every_metric_type() {
        let recorder = Recorder::new(distributions());
        let metadata = Metadata::new("test", metrics::Level::INFO, None);
        let values = [
            (r"model\x", r"model\\x"),
            (r"model\\x", r"model\\\\x"),
            ("line\nnext", r"line\nnext"),
            (r"line\nnext", r"line\\nnext"),
            ("quote\\\"x", r#"quote\\\"x"#),
            ("末尾\\", r"末尾\\"),
        ];
        for (index, (raw, _)) in values.iter().enumerate() {
            let key = |name| Key::from_parts(name, vec![metrics::Label::new("model", *raw)]);
            let count = (index + 1) as u64;
            recorder
                .register_counter(&key("requests_total"), &metadata)
                .increment(count);
            recorder
                .register_gauge(&key("remaining"), &metadata)
                .set(count as f64);
            for name in ["duration", "latency"] {
                let histogram = recorder.register_histogram(&key(name), &metadata);
                for _ in 0..count {
                    histogram.record(0.1);
                }
            }
        }
        let output = recorder.render();
        for family in [
            "requests_total",
            "remaining",
            "duration_count",
            "latency_count",
        ] {
            for (index, (_, escaped)) in values.iter().enumerate() {
                let prefix = format!("{family}{{model=\"{escaped}\"}} ");
                let samples = output
                    .lines()
                    .filter_map(|line| line.strip_prefix(&prefix))
                    .collect::<Vec<_>>();
                assert_eq!(samples, vec![(index + 1).to_string()], "{prefix}");
            }
        }
    }

    #[test]
    fn concurrent_records_upkeep_and_scrapes_preserve_all_samples() {
        let recorder = Arc::new(Recorder::new(distributions()));
        let metadata = Metadata::new("test", metrics::Level::INFO, None);
        let histogram = recorder.register_histogram(&Key::from_name("latency"), &metadata);
        let summary = recorder.register_histogram(&Key::from_name("duration"), &metadata);
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let maintenance = {
            let recorder = Arc::clone(&recorder);
            let done = Arc::clone(&done);
            std::thread::spawn(move || {
                while !done.load(Ordering::Acquire) {
                    recorder.run_upkeep();
                    std::thread::yield_now();
                }
            })
        };
        let writers = (0..4)
            .map(|_| {
                let histogram = histogram.clone();
                let summary = summary.clone();
                std::thread::spawn(move || {
                    for _ in 0..10_000 {
                        histogram.record(0.5);
                        summary.record(0.5);
                    }
                })
            })
            .collect::<Vec<_>>();
        let sample = |output: &str, prefix: &str| -> f64 {
            output
                .lines()
                .find_map(|line| line.strip_prefix(prefix))
                .unwrap()
                .parse()
                .unwrap()
        };
        for _ in 0..10 {
            let output = recorder.render();
            let count = sample(&output, "latency_count ");
            assert_eq!(sample(&output, "latency_bucket{le=\"+Inf\"} "), count);
            assert_eq!(sample(&output, "latency_bucket{le=\"0.5\"} "), count);
            assert_eq!(sample(&output, "latency_sum "), count * 0.5);
            let count = sample(&output, "duration_count ");
            assert_eq!(sample(&output, "duration_sum "), count * 0.5);
        }
        for writer in writers {
            writer.join().unwrap();
        }
        done.store(true, Ordering::Release);
        maintenance.join().unwrap();
        let output = recorder.render();
        assert_eq!(sample(&output, "latency_count "), 40_000.0);
        assert_eq!(sample(&output, "latency_sum "), 20_000.0);
        assert_eq!(sample(&output, "duration_count "), 40_000.0);
        assert_eq!(sample(&output, "duration_sum "), 20_000.0);
    }
}
