use metrics::{Key, Label, Recorder};
#[cfg(not(feature = "optimized"))]
use metrics_exporter_prometheus::PrometheusBuilder;
use std::time::Instant;

#[cfg(feature = "optimized")]
#[path = "../../../crates/aisix-obs/src/prometheus.rs"]
mod optimized;

fn cpu() -> f64 {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    assert_eq!(
        unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut t) },
        0
    );
    t.tv_sec as f64 + t.tv_nsec as f64 * 1e-9
}

fn measured<T>(phase: &str, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let c = cpu();
    let result = f();
    println!(
        "{}",
        serde_json::json!({"phase":phase,"wall_s":start.elapsed().as_secs_f64(),"cpu_s":cpu()-c})
    );
    result
}

fn main() {
    let (clock, time) = quanta::Clock::mock();
    time.increment(std::time::Duration::from_secs(3600));
    quanta::with_clock(&clock, run);
}

fn run() {
    let args = std::env::args().collect::<Vec<_>>();
    let n: usize = args.get(1).map_or("53260", String::as_str).parse().unwrap();
    let reduced = args.get(2).is_some_and(|v| v == "reduced");
    #[cfg(not(feature = "optimized"))]
    let recorder = PrometheusBuilder::new().build_recorder();
    #[cfg(not(feature = "optimized"))]
    let handle = recorder.handle();
    #[cfg(feature = "optimized")]
    let recorder = optimized::Recorder::new(metrics_exporter_prometheus::DistributionBuilder::new(
        metrics_util::parse_quantiles(&[0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0]),
        None,
        None,
        None,
        None,
    ));
    #[cfg(feature = "optimized")]
    let handle = &recorder;
    let metadata = metrics::Metadata::new("benchmark", metrics::Level::INFO, None);
    let histograms = measured("register_and_observe", || {
        let mut hs = Vec::with_capacity(n * 2);
        for i in 0..n {
            let labels = [
                ("endpoint", "/v1/chat/completions".to_owned()),
                ("inbound_protocol", "openai".to_owned()),
                ("upstream_protocol", "openai".to_owned()),
                ("provider", "openai".to_owned()),
                ("model", format!("benchmark-model-{:03}", i % 72)),
                (
                    "upstream_model",
                    format!("benchmark-upstream-model-{:03}", i % 72),
                ),
                (
                    "provider_key_id",
                    format!("00000000-0000-0000-0000-{:012}", i % 21),
                ),
                (
                    "provider_key_name",
                    format!("benchmark-provider-{:02}", i % 21),
                ),
                ("api_key_id", format!("00000000-0000-0000-0000-{i:012}")),
                ("team_id", format!("00000000-0000-0000-0000-{:012}", i % 20)),
                ("user_id", format!("00000000-0000-0000-0000-{i:012}")),
                ("user_name", format!("benchmark-synthetic-member-{i}")),
                ("stream", "false".to_owned()),
                ("status", "200".to_owned()),
                ("outcome", "success".to_owned()),
            ]
            .into_iter()
            .filter(|(k, _)| {
                !reduced || !matches!(*k, "api_key_id" | "team_id" | "user_id" | "user_name")
            })
            .map(|(k, v)| Label::new(k, v))
            .collect::<Vec<_>>();
            for name in [
                "aisix_proxy_request_duration_seconds",
                "aisix_llm_request_duration_seconds",
            ] {
                let h =
                    recorder.register_histogram(&Key::from_parts(name, labels.clone()), &metadata);
                h.record(0.001 + (i % 100) as f64 / 10000.0);
                hs.push(h);
            }
            for name in [
                "aisix_proxy_requests_total",
                "aisix_llm_requests_total",
                "aisix_llm_input_tokens_total",
                "aisix_llm_output_tokens_total",
                "aisix_llm_total_tokens_total",
                "aisix_usage_events_emitted_total",
            ] {
                recorder
                    .register_counter(&Key::from_parts(name, labels.clone()), &metadata)
                    .increment(1);
            }
        }
        hs
    });
    measured("first_upkeep", || handle.run_upkeep());
    for _ in 0..5 {
        measured("idle_upkeep", || handle.run_upkeep());
    }
    for _ in 0..3 {
        let output = measured("render", || handle.render());
        println!(
            "{}",
            serde_json::json!({"phase":"output","bytes":output.len(),"sample_lines":output.lines().filter(|l| !l.starts_with('#') && !l.is_empty()).count()})
        );
        drop(output);
    }
    for _ in 0..3 {
        histograms[0].record(0.025);
        measured("one_active_series_upkeep", || handle.run_upkeep());
    }
    measured("observe_all", || {
        for (i, h) in histograms.iter().enumerate() {
            h.record(0.001 + ((i / 2) % 100) as f64 / 10000.0);
        }
    });
    measured("all_active_upkeep", || handle.run_upkeep());
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
    assert_eq!(unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) }, 0);
    println!(
        "{}",
        serde_json::json!({"phase":"memory","peak_rss_bytes":usage.ru_maxrss as u64 * 1024})
    );
    println!(
        "{}",
        serde_json::json!({"phase":"done","keys":n,"labels":if reduced {"reduced"} else {"full"}})
    );
}
