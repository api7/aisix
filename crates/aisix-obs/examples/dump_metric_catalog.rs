fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "variables": aisix_obs::metric_labels::METRIC_VARIABLES,
            "metrics": aisix_obs::metric_labels::METRIC_DEFINITIONS,
        }))
        .expect("metric catalog serializes")
    );
}
