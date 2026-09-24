# Metrics recorder benchmark

Linux component benchmark comparing the service recorder with the previous
exporter's storage and rendering path. It uses synthetic labels, two summaries
and six counters per identity. It builds only the recorder dependencies; it does
not measure HTTP, the label-selection adapter, CP synchronization or proxy traffic.

```sh
cargo build --release --locked --manifest-path bench/metrics-scale/Cargo.toml
# Pick a CPU with no competing benchmark and use the same CPU for both runs.
taskset -c 2 bench/metrics-scale/target/release/metrics-scale-bench 53260 full
cargo build --release --locked --manifest-path bench/metrics-scale/Cargo.toml --features optimized
taskset -c 2 bench/metrics-scale/target/release/metrics-scale-bench 53260 full
```

Use `reduced` instead of `full` for a synthetic label-cardinality comparison
without API key, team and member labels. This is a comparison of recorder inputs,
not a ready-to-use configuration for every metric family.

Output is JSON Lines: process CPU seconds, elapsed seconds, payload bytes, sample
lines and peak RSS. Registration, first scrape, warm scrapes, empty maintenance,
one active series and all active series are separate phases. Run each executable
in a fresh process. There is no fixed timing assertion: CPU and allocator costs
vary by host.

The metric timestamp clock is frozen so both implementations occupy the same
rolling-summary bucket. Otherwise a slower phase can cross a 20-second bucket
boundary and compare different sketch allocations. The process and elapsed
measurement clocks remain real. Rolling-window expiration and concurrent sample
preservation are covered by the recorder tests, separately from this benchmark.
